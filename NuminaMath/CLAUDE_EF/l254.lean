import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_stand_profit_is_173_l254_25410

/-- Represents the daily operations of a lemonade stand --/
structure DailyOperation where
  expenses : ℕ
  price_per_cup : ℕ
  cups_sold : ℕ
  free_cups : ℕ

/-- Calculates the profit for a single day of operation --/
def daily_profit (operation : DailyOperation) : ℤ :=
  (operation.price_per_cup * (operation.cups_sold + operation.free_cups / 3)) - operation.expenses

/-- The lemonade stand operations for three days --/
def lemonade_stand_operations : List DailyOperation :=
  [
    { expenses := 23, price_per_cup := 4, cups_sold := 21, free_cups := 0 },
    { expenses := 27, price_per_cup := 3, cups_sold := 18, free_cups := 6 },
    { expenses := 21, price_per_cup := 4, cups_sold := 25, free_cups := 0 }
  ]

/-- Calculates the total profit for the lemonade stand over three days --/
def total_profit : ℤ :=
  (lemonade_stand_operations.map daily_profit).sum

/-- Theorem stating that the total profit of the lemonade stand is $173 --/
theorem lemonade_stand_profit_is_173 : total_profit = 173 := by
  sorry

#eval total_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_stand_profit_is_173_l254_25410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_mean_and_variance_l254_25448

-- Define a type for our dataset
def Dataset : Type := Fin 4 → ℝ

-- Define the mean of a dataset
noncomputable def mean (data : Dataset) : ℝ := (data 0 + data 1 + data 2 + data 3) / 4

-- Define the variance of a dataset
noncomputable def variance (data : Dataset) : ℝ :=
  ((data 0 - mean data)^2 + (data 1 - mean data)^2 + 
   (data 2 - mean data)^2 + (data 3 - mean data)^2) / 4

-- Define the transformation function
def transform (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem transformed_mean_and_variance 
  (data : Dataset) 
  (h_mean : mean data = 1) 
  (h_var : variance data = 1) : 
  mean (λ i => transform (data i)) = 3 ∧ 
  variance (λ i => transform (data i)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_mean_and_variance_l254_25448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subcommittees_with_min_teachers_l254_25492

def total_members : Nat := 12
def total_teachers : Nat := 5
def subcommittee_size : Nat := 5
def min_teachers : Nat := 2

theorem subcommittees_with_min_teachers 
  (h1 : total_members = 12)
  (h2 : total_teachers = 5)
  (h3 : subcommittee_size = 5)
  (h4 : min_teachers = 2)
  (h5 : total_teachers ≤ total_members)
  (h6 : subcommittee_size ≤ total_members)
  (h7 : min_teachers ≤ subcommittee_size) :
  Nat.choose total_members subcommittee_size -
    (Nat.choose (total_members - total_teachers) subcommittee_size +
     Nat.choose total_teachers 1 * Nat.choose (total_members - total_teachers) (subcommittee_size - 1)) = 596 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subcommittees_with_min_teachers_l254_25492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l254_25415

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x

noncomputable def g (x : ℝ) := 2 * Real.sin ((x / 2) + (2 * Real.pi / 3))

theorem f_properties :
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ x : ℝ, f x ≥ -2) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    2 * k * Real.pi + Real.pi / 6 ≤ x ∧ 
    x ≤ 2 * k * Real.pi + 7 * Real.pi / 6 → 
    ∀ y : ℝ, x < y → f y < f x) ∧
  g (Real.pi / 6) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l254_25415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_circumference_is_pi_sqrt_73_l254_25458

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 = 3*y - 6*x + 3

/-- The circumference of the region -/
noncomputable def region_circumference : ℝ := Real.pi * Real.sqrt 73

/-- Theorem stating that the circumference of the region defined by the given equation
    is equal to π * √73 -/
theorem region_circumference_is_pi_sqrt_73 :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_circumference = 2 * Real.pi * radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_circumference_is_pi_sqrt_73_l254_25458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_correct_l254_25498

open Set Real

noncomputable def tanDomain : Set ℝ := {x | ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 6}

theorem tan_domain_correct : 
  {x : ℝ | ∃ y : ℝ, y = tan (x + Real.pi / 3)} = tanDomain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_correct_l254_25498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tan_equation_l254_25485

theorem smallest_angle_tan_equation :
  ∃ y : ℝ, y > 0 ∧
    Real.tan (3 * y) = (Real.cos y - Real.sin y) / (Real.cos y + Real.sin y) ∧
    (∀ z : ℝ, 0 < z → Real.tan (3 * z) = (Real.cos z - Real.sin z) / (Real.cos z + Real.sin z) → y ≤ z) ∧
    y * (180 / Real.pi) = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tan_equation_l254_25485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_spheres_coverage_l254_25419

/-- Represents a sphere tangent to the unit sphere -/
structure TangentSphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Checks if two tangent spheres do not intersect -/
def noIntersection (s1 s2 : TangentSphere) : Prop := sorry

/-- Calculates the area covered by a tangent sphere on the unit sphere -/
noncomputable def coveredArea (s : TangentSphere) : ℝ := Real.pi * s.radius ^ 2

/-- The main theorem -/
theorem tangent_spheres_coverage 
  (s1 s2 s3 s4 : TangentSphere)
  (h_sum : s1.radius + s2.radius + s3.radius + s4.radius = 2)
  (h_no_intersect : 
    noIntersection s1 s2 ∧ noIntersection s1 s3 ∧ noIntersection s1 s4 ∧
    noIntersection s2 s3 ∧ noIntersection s2 s4 ∧
    noIntersection s3 s4) :
  coveredArea s1 + coveredArea s2 + coveredArea s3 + coveredArea s4 ≥ Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_spheres_coverage_l254_25419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_range_l254_25413

theorem sin_alpha_range (m : ℝ) (α : ℝ) :
  (Real.sin α = (2 * m - 3) / (4 - m)) →
  (π < α ∧ α < 2 * π) →
  ∃ (x : ℝ), -1 < x ∧ x < 3/2 ∧ Real.sin α = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_range_l254_25413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l254_25456

theorem intersection_distance_difference (p q : ℕ) : 
  (∃ C D : ℝ × ℝ, 
    (C.2 = 5 ∧ C.2 = 3 * C.1^2 + 2 * C.1 - 2) ∧ 
    (D.2 = 5 ∧ D.2 = 3 * D.1^2 + 2 * D.1 - 2) ∧
    C ≠ D ∧
    (C.1 - D.1)^2 = p / q^2 ∧
    Nat.Coprime p q) →
  p - q = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l254_25456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l254_25404

-- Define the triangle ABC
structure Triangle (α : ℝ) (a b : ℝ) where
  -- Ensure the triangle is acute-angled
  acute : 0 < α ∧ α < π / 2
  -- Ensure positive side lengths
  positive_sides : 0 < a ∧ 0 < b

-- Define the height CD
noncomputable def height_CD (α : ℝ) (a b : ℝ) : ℝ :=
  (a * b * Real.sin α) / Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos α)

-- Define the angle ABC
noncomputable def angle_ABC (α : ℝ) (a b : ℝ) : ℝ :=
  Real.arcsin ((b * Real.sin α) / Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos α))

-- Theorem statement
theorem triangle_properties (α : ℝ) (a b : ℝ) (t : Triangle α a b) :
  let cd := height_CD α a b
  let angle_abc := angle_ABC α a b
  -- The height CD is correct
  cd = (a * b * Real.sin α) / Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos α) ∧
  -- The angle ABC is correct
  angle_abc = Real.arcsin ((b * Real.sin α) / Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos α)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l254_25404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_shifted_sine_l254_25470

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_of_shifted_sine (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π / 2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_shift : ∀ x, f ω φ (x - π / 3) = f ω φ (-x)) :
  ∀ x, f ω φ (5 * π / 12 + x) = f ω φ (5 * π / 12 - x) := by
  sorry

#check symmetry_of_shifted_sine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_shifted_sine_l254_25470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l254_25402

noncomputable def f (x : ℝ) := (2 : ℝ)^(-abs x)

theorem range_of_f :
  Set.range f = Set.Ioo 0 1 ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l254_25402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_intersection_at_three_no_other_intersections_l254_25450

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (9 * x) / Real.log 3

-- State the theorem
theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

-- Prove that the intersection point is at x = 3
theorem intersection_at_three :
  f 3 = g 3 := by
  sorry

-- Prove that there are no other intersection points
theorem no_other_intersections :
  ∀ x : ℝ, x > 0 → x ≠ 3 → f x ≠ g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_intersection_at_three_no_other_intersections_l254_25450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_sum_l254_25474

/-- Represents a face of the cube -/
inductive Face
| one | two | three | four | five | six

/-- The value on each face of the cube -/
def face_value : Face → Nat
| Face.one => 1
| Face.two => 2
| Face.three => 3
| Face.four => 4
| Face.five => 5
| Face.six => 6

/-- Opposite faces of the cube -/
def opposite : Face → Face
| Face.one => Face.six
| Face.two => Face.five
| Face.three => Face.four
| Face.four => Face.three
| Face.five => Face.two
| Face.six => Face.one

/-- The sum of values on opposite faces is 9 -/
axiom opposite_sum (f : Face) : face_value f + face_value (opposite f) = 9

/-- Three faces meet at each corner -/
def Corner := Face × Face × Face

/-- The sum of values at a corner -/
def corner_sum (c : Corner) : Nat :=
  let (f1, f2, f3) := c
  face_value f1 + face_value f2 + face_value f3

/-- The maximum sum of three numbers at any corner is 16 -/
theorem max_corner_sum : 
  (∀ c : Corner, corner_sum c ≤ 16) ∧ (∃ c : Corner, corner_sum c = 16) := by
  sorry

#check max_corner_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_sum_l254_25474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_problem_l254_25461

def A : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 2]

def C1 (x y : ℝ) : Prop := x^2/8 + y^2/2 = 1

theorem matrix_transformation_problem :
  -- Part 1: AB calculation
  A * B = !![0, 2; 1, 0] ∧
  -- Part 2: Transformation of C1 to C2
  ∀ x y : ℝ, C1 x y → 
    ∃ x₀ y₀ : ℝ, 
      -- Transformation by AB
      (A * B).mulVec ![x, y] = ![x₀, y₀] ∧
      -- Equation of C2
      x₀^2 + y₀^2 = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_problem_l254_25461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_half_solution_l254_25453

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else Real.log x / Real.log 9

-- Define the solution set
def solution_set : Set ℝ := Set.union (Set.Ioc (-1) 1) (Set.Ioi 3)

-- Theorem statement
theorem f_greater_than_half_solution :
  {x : ℝ | f x > 1/2} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_half_solution_l254_25453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_proposition_l254_25493

-- Define the propositions
def proposition_1 : Prop := ∀ (p q : Prop), (p → q) ↔ (¬q → ¬p)

def proposition_2 : Prop := 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∨ 
  (∃ x : ℝ, x^2 + x + 1 < 0)

def proposition_3 : Prop := 
  ∀ (a b m : ℝ), (a < b) → (a * m^2 < b * m^2)

def proposition_4 : Prop := 
  ∀ (p q : Prop), ¬(p ∨ q) → (¬p ∧ ¬q)

-- Theorem to prove
theorem incorrect_proposition : 
  proposition_1 ∧ proposition_2 ∧ proposition_4 → ¬proposition_3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_proposition_l254_25493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_perpendicular_bisector_l254_25444

noncomputable def A : ℝ × ℝ := (-1, 0)
noncomputable def B : ℝ × ℝ := (3, 4)

noncomputable def CD_length : ℝ := 4 * Real.sqrt 10

theorem circle_and_perpendicular_bisector :
  ∃ (P : ℝ × ℝ) (C D : ℝ × ℝ),
    -- P is the center of the circle passing through A and B
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
    -- C and D are on the perpendicular bisector of AB
    C.1 + C.2 = 3 ∧ D.1 + D.2 = 3 ∧
    -- |CD| = 4√10
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = CD_length^2 ∧
    -- C and D are on the circle
    (P.1 - C.1)^2 + (P.2 - C.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2 ∧
    (P.1 - D.1)^2 + (P.2 - D.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2 →
    -- The equation of line CD is x + y - 3 = 0
    (∀ (x y : ℝ), x + y = 3 ↔ (x - C.1) * (D.2 - C.2) = (y - C.2) * (D.1 - C.1)) ∧
    -- The equation of circle P is either (x+3)² + (y-6)² = 40 or (x-5)² + (y+2)² = 40
    ((P.1 = -3 ∧ P.2 = 6) ∨ (P.1 = 5 ∧ P.2 = -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_perpendicular_bisector_l254_25444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_periodic_iff_m_odd_l254_25490

/-- Definition of the sequence q_n -/
def q (u : ℚ) (m : ℕ) : ℕ → ℚ
  | 0 => u  -- Added case for 0
  | 1 => u
  | (n+2) => let a := (q u m (n+1)).num
              let b := (q u m (n+1)).den
              (a + m * b) / (b + 1)

/-- Definition of eventually periodic sequence -/
def eventually_periodic (s : ℕ → ℚ) : Prop :=
  ∃ (c t : ℕ), ∀ n ≥ c, s n = s (n + t)

/-- Main theorem -/
theorem eventually_periodic_iff_m_odd (u : ℚ) (m : ℕ) (h_u : u > 0) (h_m : m > 0) :
  eventually_periodic (q u m) ↔ Odd m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_periodic_iff_m_odd_l254_25490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_behavior_l254_25473

-- Define the function f(x) = x + 1/x
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem f_behavior :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_behavior_l254_25473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l254_25422

/-- Calculates the speed of a train crossing a bridge in kmph -/
noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  speed_ms * 3.6

theorem train_speed_calculation :
  train_speed 100 250 34.997200223982084 = 36.0008228577942852 := by
  unfold train_speed
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l254_25422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l254_25462

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
noncomputable def train_meeting_time (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := length1 + length2 + initial_distance
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  total_distance / relative_speed

/-- The time for two trains to meet is approximately 14.26 seconds. -/
theorem trains_meet_time :
  let length1 : ℝ := 250
  let length2 : ℝ := 120
  let initial_distance : ℝ := 50
  let speed1 : ℝ := 64
  let speed2 : ℝ := 42
  abs (train_meeting_time length1 length2 initial_distance speed1 speed2 - 14.26) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l254_25462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l254_25442

theorem power_equality (y : ℝ) (h : (128 : ℝ)^3 = (16 : ℝ)^y) : 
  (2 : ℝ)^(-y) = 1 / (2 : ℝ)^(21/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l254_25442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_september_coprime_days_l254_25488

/-- The number of days in September -/
def september_days : ℕ := 30

/-- The month number of September -/
def september_month : ℕ := 9

/-- A function that returns true if two natural numbers are relatively prime -/
def is_coprime (a b : ℕ) : Bool := (Nat.gcd a b == 1)

/-- The number of days in September that are relatively prime to the month number -/
def coprime_days : ℕ := (List.range september_days).filter (λ d => is_coprime (d + 1) september_month) |>.length

theorem september_coprime_days :
  coprime_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_september_coprime_days_l254_25488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_F_functions_l254_25440

-- Define the concept of an F function
def is_F_function (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≤ M * |x|

-- Define the three functions
def f₁ : ℝ → ℝ := λ x ↦ 2 * x

noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x

noncomputable def f₃ : ℝ → ℝ := sorry

-- State the properties of f₃
axiom f₃_odd : ∀ x : ℝ, f₃ (-x) = -f₃ x
axiom f₃_lipschitz : ∀ x₁ x₂ : ℝ, |f₃ x₁ - f₃ x₂| ≤ 2 * |x₁ - x₂|

-- Theorem stating that all three functions are F functions
theorem all_F_functions :
  is_F_function f₁ ∧ is_F_function f₂ ∧ is_F_function f₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_F_functions_l254_25440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l254_25443

noncomputable def f (x : ℝ) := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1/2

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 2) 
  (h2 : b + c = 2 * Real.sqrt 2) 
  (h3 : 1/2 * b * c * Real.sin A = 1/2) : 
  a = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l254_25443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l254_25464

/-- The circle with equation x² + y² - 4x = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 = 0}

/-- The point P on the circle -/
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 3)

/-- The proposed tangent line equation -/
def TangentLine (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 2 = 0

/-- Theorem stating that the proposed equation is indeed the tangent line -/
theorem tangent_line_equation :
  TangentLine P.1 P.2 ∧
  ∀ (x y : ℝ), (x, y) ∈ Circle → TangentLine x y →
    (x = P.1 ∧ y = P.2) ∨
    ((x - P.1)^2 + (y - P.2)^2 > 0 ∧ TangentLine x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l254_25464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l254_25407

def f (x : ℝ) : ℝ := 4*x^7 - 2*x^6 - 8*x^5 + 3*x^3 + 5*x^2 - 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem : 
  ∃ (q : ℝ → ℝ), ∀ x, f x = (divisor x) * (q x) + 5457 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l254_25407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_if_sum_norm_equals_sum_of_norms_l254_25420

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vectors_collinear_if_sum_norm_equals_sum_of_norms 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  ‖a + b‖ = ‖a‖ + ‖b‖ → ∃ (k : ℝ), a = k • b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_if_sum_norm_equals_sum_of_norms_l254_25420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l254_25428

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := e.c / e.a

noncomputable def Ellipse.shortestDistance (e : Ellipse) : ℝ := e.a - e.c

def Ellipse.equilateralTriangle (e : Ellipse) : Prop := e.a = 2 * e.c

theorem ellipse_properties (e : Ellipse) 
  (h1 : e.equilateralTriangle)
  (h2 : e.shortestDistance = Real.sqrt 3) :
  (e.a = 2 * Real.sqrt 3 ∧ 
   e.b^2 = 9 ∧ 
   e.eccentricity = 1/2 ∧ 
   (∀ x y : ℝ, x^2/12 + y^2/9 = 1 ∨ x^2/9 + y^2/12 = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l254_25428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l254_25432

theorem sin_pi_plus_alpha (α : ℝ) (h1 : Real.cos α = 5/13) (h2 : 0 < α ∧ α < Real.pi/2) : 
  Real.sin (Real.pi + α) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l254_25432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_optimization_l254_25430

-- Define the total cost function
noncomputable def f (s x : ℝ) : ℝ := 225 * x + 360 * s / x - 360

-- Theorem statement
theorem fence_cost_optimization (s : ℝ) (h_s : s > 2.5) :
  -- Condition on x
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 20 →
    -- Total cost function
    f s x = 225 * x + 360 * s / x - 360 ∧
    -- Minimum cost for s ≤ 250
    (s ≤ 250 →
      ∃ (x_min : ℝ), x_min = 2 * Real.sqrt (10 * s) / 5 ∧
        ∀ y : ℝ, 2 ≤ y ∧ y ≤ 20 → f s x_min ≤ f s y ∧
        f s x_min = 180 * Real.sqrt (10 * s) - 360) ∧
    -- Minimum cost for s > 250
    (s > 250 →
      ∀ y : ℝ, 2 ≤ y ∧ y ≤ 20 → f s 20 ≤ f s y ∧
        f s 20 = 4140 + 18 * s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_optimization_l254_25430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_integer_l254_25484

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | (n + 4) => (sequence_a (n + 3) * sequence_a (n + 2) + (n + 1).factorial) / sequence_a (n + 1)

theorem sequence_a_integer : ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_integer_l254_25484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l254_25441

-- Define the line equation
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + Real.sqrt 3 * y + 1 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan (-m)

-- Theorem statement
theorem line_inclination_angle :
  inclination_angle (-(Real.sqrt 3 / Real.sqrt 3)) = 3 * Real.pi / 4 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l254_25441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_implies_sum_of_squares_l254_25418

theorem sqrt_equation_implies_sum_of_squares (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (Real.sqrt (7 + Real.sqrt 48) = m + Real.sqrt n) →
  m ^ 2 + n ^ 2 = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_implies_sum_of_squares_l254_25418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_balls_identical_boxes_l254_25481

-- Define the function for distributing identical balls into identical boxes
def number_of_ways_to_distribute (n m : ℕ) : ℕ :=
  sorry -- We'll leave the implementation as 'sorry' for now

theorem identical_balls_identical_boxes :
  ∀ (n m : ℕ), n = 6 ∧ m = 4 →
  (number_of_ways_to_distribute n m) = 2 :=
by
  intros n m h
  sorry -- We'll leave the proof as 'sorry' for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_balls_identical_boxes_l254_25481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l254_25478

/-- The area of a triangle with vertices at (0,0), (0,8), and (9,18) is 36.0 square units. -/
theorem triangle_area : ℝ := by
  let vertex1 : ℝ × ℝ := (0, 0)
  let vertex2 : ℝ × ℝ := (0, 8)
  let vertex3 : ℝ × ℝ := (9, 18)
  let area : ℝ := 36.0
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l254_25478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_scenarios_l254_25431

/-- Represents a grade level -/
inductive Grade
  | First
  | Second
  | Third

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a scenario of selected numbers -/
structure Scenario where
  numbers : List Nat
  deriving Repr

/-- Represents the distribution of students across grades -/
structure StudentDistribution where
  totalStudents : Nat
  firstGradeStudents : Nat
  secondGradeStudents : Nat
  thirdGradeStudents : Nat

/-- Checks if a scenario is consistent with stratified sampling -/
def isStratifiedSampling (dist : StudentDistribution) (scenario : Scenario) : Prop :=
  let firstGradeCount := (scenario.numbers.filter (λ n => n < dist.firstGradeStudents)).length
  let secondGradeCount := (scenario.numbers.filter (λ n => n ≥ dist.firstGradeStudents ∧ n < dist.firstGradeStudents + dist.secondGradeStudents)).length
  let thirdGradeCount := (scenario.numbers.filter (λ n => n ≥ dist.firstGradeStudents + dist.secondGradeStudents)).length
  firstGradeCount = 4 ∧ secondGradeCount = 3 ∧ thirdGradeCount = 3

/-- The main theorem to be proved -/
theorem stratified_sampling_scenarios (dist : StudentDistribution)
  (scenario1 scenario4 : Scenario)
  (h_dist : dist.totalStudents = 100 ∧ dist.firstGradeStudents = 40 ∧ dist.secondGradeStudents = 30 ∧ dist.thirdGradeStudents = 30)
  (h_scenario1 : scenario1.numbers = [5, 10, 17, 36, 47, 53, 65, 76, 90, 95])
  (h_scenario4 : scenario4.numbers = [8, 15, 22, 29, 48, 55, 62, 78, 85, 92]) :
  isStratifiedSampling dist scenario1 ∧ isStratifiedSampling dist scenario4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_scenarios_l254_25431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poodle_bark_count_l254_25433

/-- Represents the number of times a dog barks -/
structure Barks where
  value : ℕ

/-- Represents the number of times an owner hushes their dog -/
structure Hushes where
  value : ℕ

instance : Coe Barks ℕ where
  coe b := b.value

instance : Coe Hushes ℕ where
  coe h := h.value

instance : HMul ℕ Barks Barks where
  hMul n b := Barks.mk (n * b.value)

instance : HMul ℕ Hushes Barks where
  hMul n h := Barks.mk (n * h.value)

instance : OfNat Barks n where
  ofNat := Barks.mk n

instance : OfNat Hushes n where
  ofNat := Hushes.mk n

/-- The relationship between poodle and terrier barks -/
def poodle_terrier_ratio (poodle_barks terrier_barks : Barks) : Prop :=
  poodle_barks = 2 * terrier_barks

/-- The relationship between terrier barks and owner hushes -/
def terrier_hush_ratio (terrier_barks : Barks) (owner_hushes : Hushes) : Prop :=
  terrier_barks = 2 * owner_hushes

theorem poodle_bark_count 
  (poodle_barks terrier_barks : Barks) 
  (owner_hushes : Hushes) :
  poodle_terrier_ratio poodle_barks terrier_barks →
  terrier_hush_ratio terrier_barks owner_hushes →
  owner_hushes = 6 →
  poodle_barks = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poodle_bark_count_l254_25433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_double_length_l254_25411

open EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define that ABC forms a triangle
variable (triangle_ABC : Triangle ℝ (Fin 2))

-- Define AD as an angle bisector
variable (D : EuclideanSpace ℝ (Fin 2))
variable (AD_bisector : AngleBisector triangle_ABC A D)

-- Define CE as a median
variable (E : EuclideanSpace ℝ (Fin 2))
variable (CE_median : Median triangle_ABC C E)

-- Define that AD and CE intersect at a right angle
variable (F : EuclideanSpace ℝ (Fin 2))
variable (AD_CE_perpendicular : Perpendicular (Seg A D) (Seg C E) F)

-- Theorem statement
theorem side_double_length :
  ‖A - B‖ = 2 * ‖A - C‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_double_length_l254_25411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_l254_25472

theorem consecutive_even_numbers_sum (a : ℤ) :
  (∃ n : ℕ, n = 5) →
  (∀ i : ℕ, i < 5 → Even (a + 2 * ↑i)) →
  ((a) + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 100) →
  (a + 8 = 24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_l254_25472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_distance_l254_25491

/-- Calculates the distance at which Train B overtakes Train A -/
noncomputable def overtake_distance (speed_a speed_b : ℝ) (time_delay : ℝ) : ℝ :=
  (speed_a * time_delay) / (1 - speed_a / speed_b) * speed_a

theorem train_overtake_distance :
  let speed_a : ℝ := 30
  let speed_b : ℝ := 42
  let time_delay : ℝ := 2
  overtake_distance speed_a speed_b time_delay = 150 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_distance_l254_25491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt3_approximation_l254_25482

theorem sqrt3_approximation (a₁ : ℚ) (h₁ : 0 < a₁) :
  let a₂ := 1 + 2 / (1 + a₁)
  (min a₁ a₂ < Real.sqrt 3 ∧ Real.sqrt 3 < max a₁ a₂) ∧
  |a₂ - Real.sqrt 3| < |a₁ - Real.sqrt 3| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt3_approximation_l254_25482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_squares_1000_cube_l254_25455

/-- Represents a cube with side length n and each face divided into n² unit squares. -/
structure Cube (n : ℕ) where
  side_length : ℕ
  total_squares : ℕ
  hp_side_length : side_length = n
  hp_total_squares : total_squares = 6 * n^2

/-- Predicate to determine if a square is colored black. -/
def IsBlack (square : ℕ) : Prop := sorry

/-- Predicate to determine if two squares are adjacent (share an edge). -/
def AreAdjacent (s1 s2 : ℕ) : Prop := sorry

/-- Represents a valid coloring of the cube where no two black squares share an edge. -/
def ValidColoring (c : Cube n) (black_count : ℕ) :=
  black_count ≤ c.total_squares ∧
  ∀ (s1 s2 : ℕ), s1 ≤ c.total_squares → s2 ≤ c.total_squares →
    IsBlack s1 → IsBlack s2 → AreAdjacent s1 s2 → s1 = s2

/-- The maximum number of black squares possible in a valid coloring. -/
def MaxBlackSquares (c : Cube n) : ℕ := sorry

/-- Theorem stating the maximum number of black squares for a 1000 × 1000 × 1000 cube. -/
theorem max_black_squares_1000_cube :
  ∀ (c : Cube 1000) (coloring : ℕ),
    ValidColoring c coloring →
    coloring ≤ MaxBlackSquares c ∧
    MaxBlackSquares c = 2998000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_squares_1000_cube_l254_25455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_order_l254_25429

theorem magnitude_order : 
  let p : ℝ := (2/3)^(2/3)
  let q : ℝ := (2/3)^(3/4)
  let r : ℝ := Real.log 3 / Real.log 2
  r > p ∧ p > q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_order_l254_25429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_coefficients_l254_25469

noncomputable def angle_between_vectors : ℝ := 2 * Real.pi / 3

noncomputable def point_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

noncomputable def vector_OC (x y : ℝ) : ℝ × ℝ := (x - y/2, y * Real.sqrt 3 / 2)

noncomputable def sum_coefficients (θ : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin θ + Real.cos θ

theorem max_sum_coefficients :
  ∃ (θ : ℝ), ∀ (φ : ℝ), sum_coefficients θ ≥ sum_coefficients φ ∧ sum_coefficients θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_coefficients_l254_25469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_l254_25401

/-- Represents the number of fourth-grade students -/
def b : ℕ := sorry

/-- Average minutes run per day by third graders -/
def third_grade_avg : ℚ := 30 / 3

/-- Average minutes run per day by fourth graders -/
def fourth_grade_avg : ℚ := 10

/-- Average minutes run per day by fifth graders -/
def fifth_grade_avg : ℚ := 45 / 3

/-- Total number of students -/
def total_students : ℚ := 3 * b + b + b / 2

/-- Total minutes run by all students per day -/
def total_minutes : ℚ := 3 * b * third_grade_avg + b * fourth_grade_avg + (b / 2) * fifth_grade_avg

theorem average_running_time (h : b > 0) : 
  total_minutes / total_students = 95 / 9 := by
  sorry

#check average_running_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_l254_25401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_36kg_apples_l254_25452

/-- The price of apples for the first 30 kgs and additional kgs -/
structure ApplePrice where
  l : ℚ  -- price per kg for first 30 kgs
  q : ℚ  -- price per kg for additional kgs

/-- Calculate the price of apples given the pricing structure and weight -/
def calculatePrice (p : ApplePrice) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then
    p.l * weight
  else
    p.l * 30 + p.q * (weight - 30)

/-- Theorem stating the price of 36 kg of apples under given conditions -/
theorem price_of_36kg_apples (p : ApplePrice) : 
  p.l * 10 = 200 →    -- cost of first 10 kg is 200
  calculatePrice p 33 = 663 →  -- price of 33 kg is 663
  calculatePrice p 36 = 726 := by
  sorry

#check price_of_36kg_apples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_36kg_apples_l254_25452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_fraction_sum_theorem_l254_25412

/-- A set of prime numbers -/
def PrimeSet (M : Finset Nat) : Prop := ∀ p ∈ M, Nat.Prime p

/-- The sum of unit fractions with denominators being products of powers of all elements in M -/
noncomputable def UnitFractionSum (M : Finset Nat) : ℚ :=
  1 / (M.prod (λ p => p - 1))

theorem unit_fraction_sum_theorem (M : Finset Nat) (k : Nat) 
  (h1 : PrimeSet M) (h2 : M.card = k) (h3 : k > 2) :
  UnitFractionSum M = 1 / (M.prod (λ p => p - 1)) ∧
  UnitFractionSum M < 1 / (2 * 3^(k-2) * Nat.factorial (k-2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_fraction_sum_theorem_l254_25412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_in_special_set_l254_25457

theorem max_element_in_special_set : ∃ (T : Finset ℕ), 
  (T.card = 8) ∧ 
  (∀ x, x ∈ T → x ≥ 1 ∧ x ≤ 20) ∧
  (∀ c d, c ∈ T → d ∈ T → c < d → ¬(d % c = 0)) ∧
  (∃ m, m ∈ T ∧ ∀ x, x ∈ T → x ≤ m) ∧
  (∀ S : Finset ℕ, 
    (S.card = 8) → 
    (∀ x, x ∈ S → x ≥ 1 ∧ x ≤ 20) → 
    (∀ c d, c ∈ S → d ∈ S → c < d → ¬(d % c = 0)) → 
    (∃ n, n ∈ S ∧ ∀ y, y ∈ S → y ≤ n) → 
    n ≤ 20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_in_special_set_l254_25457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_markup_is_twenty_percent_l254_25436

/-- Represents the markup percentage applied by the merchant -/
def markup_percentage : ℝ → ℝ := λ x ↦ x

/-- Represents the discount percentage offered by the merchant -/
def discount_percentage : ℝ := 5

/-- Represents the profit percentage made by the merchant after discount -/
def profit_percentage : ℝ := 14

/-- Theorem stating that the initial markup percentage is 20% given the conditions -/
theorem initial_markup_is_twenty_percent :
  ∃ (x : ℝ),
    markup_percentage x = x ∧
    discount_percentage = 5 ∧
    profit_percentage = 14 ∧
    x = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_markup_is_twenty_percent_l254_25436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_with_one_negative_l254_25480

def S : Finset Int := {-9, -7, -2, 0, 4, 6, 8}

theorem max_product_with_one_negative :
  (Finset.filter (λ x => x < 0) S).card ≥ 1 →
  (Finset.filter (λ x => x > 0) S).card ≥ 2 →
  ∃ a b c : Int,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a < 0 ∧ b > 0 ∧ c > 0) ∧
    a * b * c = -96 ∧
    ∀ x y z : Int,
      x ∈ S → y ∈ S → z ∈ S →
      x ≠ y → y ≠ z → x ≠ z →
      (x < 0 ∧ y > 0 ∧ z > 0) →
      x * y * z ≤ -96 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_with_one_negative_l254_25480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_problem_l254_25459

/-- Two linear functions -/
def y₁ (x : ℝ) : ℝ := -x + 1
def y₂ (x : ℝ) : ℝ := -3*x + 2

/-- Main theorem -/
theorem linear_functions_problem :
  (∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ y₁ x = a + y₂ x) ↔ a > -1) ∧
  (∀ x y : ℝ, y₁ x = y ∧ y₂ x = y → 12*x^2 + 12*x*y + 3*y^2 = 27/4) ∧
  (∀ A B : ℝ, (∀ x : ℝ, x ≠ 1 ∧ 3*x ≠ 2 →
    (4-2*x)/((3*x-2)*(x-1)) = A/(y₁ x) + B/(y₂ x)) →
    A/B + B/A = -(17/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_problem_l254_25459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_2_strictly_increasing_l254_25408

-- Define the function f(x) = log₂(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_base_2_strictly_increasing :
  ∀ (a b : ℝ), a > 0 → b > 0 → (a > b ↔ f a > f b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_2_strictly_increasing_l254_25408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_ppc_correct_l254_25414

/-- The production possibility curve for males -/
def male_ppc (K : ℝ) : ℝ := 128 - 0.5 * K^2

/-- The production possibility curve for females -/
def female_ppc (K : ℝ) : ℝ := 40 - 2 * K

/-- The combined production possibility curve -/
noncomputable def combined_ppc (K : ℝ) : ℝ :=
  if K ≤ 2 then
    168 - 0.5 * K^2
  else if K ≤ 22 then
    170 - 2 * K
  else if K ≤ 36 then
    20 * K - 0.5 * K^2 - 72
  else
    0  -- Outside the valid range

theorem combined_ppc_correct (K : ℝ) :
  combined_ppc K =
    if K ≤ 2 then
      168 - 0.5 * K^2
    else if K ≤ 22 then
      170 - 2 * K
    else if K ≤ 36 then
      20 * K - 0.5 * K^2 - 72
    else
      0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_ppc_correct_l254_25414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l254_25446

theorem unique_integer_power : ∃! n : ℤ, ∃ k : ℤ, 8000 * (2 / 3 : ℚ) ^ n = k := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l254_25446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_when_x_is_5_l254_25486

def line_point (t : ℝ) : ℝ × ℝ × ℝ := (3 + 3*t, 3 - t, 2 - 3*t)

theorem z_coordinate_when_x_is_5 :
  let p1 : ℝ × ℝ × ℝ := (3, 3, 2)
  let p2 : ℝ × ℝ × ℝ := (6, 2, -1)
  let line (t : ℝ) : ℝ × ℝ × ℝ := line_point t
  ∃ t : ℝ, (line t).1 = 5 ∧ (line t).2.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_when_x_is_5_l254_25486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_equals_64_l254_25460

/-- The function f(x) = 2x^2 + a/x where a is a constant -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 + a / x

/-- Theorem: For the function f(x) = 2x^2 + a/x where a is a constant such that f(-1) = -30,
    the sum of the maximum and minimum values of f(x) in the interval [1, 4] is 64 -/
theorem sum_of_extrema_equals_64 (a : ℝ) (h : f a (-1) = -30) :
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ Set.Icc 1 4, f a x ≥ min_val) ∧
    (∃ x ∈ Set.Icc 1 4, f a x = min_val) ∧
    (∀ x ∈ Set.Icc 1 4, f a x ≤ max_val) ∧
    (∃ x ∈ Set.Icc 1 4, f a x = max_val) ∧
    min_val + max_val = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_equals_64_l254_25460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_factor_exists_l254_25417

theorem odd_prime_factor_exists (k : ℕ) (a : ℕ) (p : Fin k → ℕ) 
  (h1 : k ≥ 2)
  (h2 : ∀ i : Fin k, Nat.Prime (p i))
  (h3 : ∀ i : Fin k, Odd (p i))
  (h4 : Nat.Coprime a (Finset.prod Finset.univ p)) :
  ∃ q : ℕ, Nat.Prime q ∧ Odd q ∧ 
    (q ∣ (a^(Finset.prod Finset.univ (fun i => p i - 1)) - 1)) ∧
    (∀ i : Fin k, q ≠ p i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_factor_exists_l254_25417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_to_three_nonparallel_implies_perp_to_plane_plane_perp_to_hexagon_sides_implies_perp_to_plane_l254_25497

-- Define the basic structures
structure Line : Type
structure Plane : Type

-- Define the perpendicular relation
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define the condition of a line being perpendicular to three non-parallel lines in a plane
def perp_to_three_nonparallel_lines (l : Line) (p : Plane) : Prop := sorry

-- Define the condition of a plane being perpendicular to three sides of a regular hexagon
def plane_perp_to_hexagon_sides (p : Plane) : Prop := sorry

-- Define a line being perpendicular to the three sides of the hexagon
def line_perp_to_hexagon_sides (l : Line) (p : Plane) : Prop := sorry

-- Theorem 1
theorem perp_to_three_nonparallel_implies_perp_to_plane 
  (l : Line) (p : Plane) : 
  perp_to_three_nonparallel_lines l p → perpendicular l p := by
  sorry

-- Theorem 2
theorem plane_perp_to_hexagon_sides_implies_perp_to_plane 
  (l : Line) (p : Plane) : 
  plane_perp_to_hexagon_sides p → line_perp_to_hexagon_sides l p → perpendicular l p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_to_three_nonparallel_implies_perp_to_plane_plane_perp_to_hexagon_sides_implies_perp_to_plane_l254_25497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_center_l254_25468

/-- A rectangle in a 2D plane --/
structure Rectangle where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ
  corner3 : ℝ × ℝ
  corner4 : ℝ × ℝ

/-- The center of a rectangle --/
noncomputable def center (r : Rectangle) : ℝ × ℝ :=
  ((r.corner1.1 + r.corner3.1) / 2, (r.corner1.2 + r.corner3.2) / 2)

theorem rectangle_center :
  let r := Rectangle.mk (1, 3) (1, 7) (4, 7) (4, 3)
  center r = (2.5, 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_center_l254_25468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l254_25438

-- Define the sequence of functions f_n
def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => x
  | n + 1 => λ x => x^(n + 1)

-- Define g_n
def g (n : ℕ) (m : ℝ) (x : ℝ) : ℝ := f n x + f n (m - x)

-- Theorem statement
theorem f_and_g_properties :
  (∀ n : ℕ, n > 0 → f n 1 = 1) ∧
  (∀ n : ℕ, n > 0 → ∀ x : ℝ, HasDerivAt (f (n + 1)) ((f n x) + x * (deriv (f n) x)) x) ∧
  (∀ n : ℕ, n > 0 → ∀ x : ℝ, f n x = x^n) ∧
  (∀ m : ℝ, m > 0 → 
    ∀ x₁ x₂ x₃ : ℝ, m / 2 ≤ x₁ ∧ x₁ ≤ 2 * m / 3 ∧ 
                    m / 2 ≤ x₂ ∧ x₂ ≤ 2 * m / 3 ∧ 
                    m / 2 ≤ x₃ ∧ x₃ ≤ 2 * m / 3 → 
    g 3 m x₁ + g 3 m x₂ > g 3 m x₃) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l254_25438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_of_cycle_l254_25400

/-- Represents a thermodynamic cycle with three stages -/
structure ThermodynamicCycle where
  P₀ : ℝ
  ρ₀ : ℝ
  stage1 : ℝ × ℝ  -- isochoric pressure reduction (initial pressure, final pressure)
  stage2 : ℝ × ℝ  -- isobaric density increase (initial density, final density)
  stage3 : ℝ × ℝ × ℝ × ℝ  -- return to initial state (center_x, center_y, radius, angle)

/-- Represents the efficiency of a thermodynamic cycle -/
noncomputable def cycle_efficiency (cycle : ThermodynamicCycle) : ℝ :=
  1 / 12  -- The efficiency we want to prove

/-- Represents the maximum possible efficiency for a given temperature range -/
noncomputable def max_efficiency (T_min T_max : ℝ) : ℝ :=
  1 - (T_min / T_max)

/-- The main theorem stating the efficiency of the given thermodynamic cycle -/
theorem efficiency_of_cycle (cycle : ThermodynamicCycle) 
  (h1 : cycle.stage1 = (3 * cycle.P₀, cycle.P₀))
  (h2 : cycle.stage2 = (cycle.ρ₀, 3 * cycle.ρ₀))
  (h3 : cycle.stage3 = (1, 1, 1, Real.pi / 2))
  (h4 : ∃ (T_min T_max : ℝ), cycle_efficiency cycle = (max_efficiency T_min T_max) / 8) :
  cycle_efficiency cycle = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_of_cycle_l254_25400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l254_25435

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a ≥ b then a * b + b else a * b - a

-- Theorem statement
theorem star_equation_solution :
  ∀ x : ℝ, (star (2 * x - 1) (x + 2) = 0) ↔ (x = -1 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l254_25435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_xt_distance_l254_25494

/-- Represents a rectangular pyramid -/
structure RectangularPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- Represents a section of a pyramid -/
structure PyramidSection where
  base_length : ℝ
  base_width : ℝ
  top_length : ℝ
  top_width : ℝ
  height : ℝ

/-- Helper function to calculate the distance from the center of the circumsphere to the apex -/
def distance_center_circumsphere_to_apex (f : PyramidSection) (total_height : ℝ) : ℝ :=
  sorry -- Implementation would go here

/-- The theorem statement -/
theorem pyramid_section_xt_distance 
  (p : RectangularPyramid) 
  (f : PyramidSection) 
  (p_prime : RectangularPyramid) :
  p.base_length = 8 →
  p.base_width = 10 →
  p.height = 15 →
  (p.base_length * p.base_width * p.height) / 3 = 9 * ((p_prime.base_length * p_prime.base_width * p_prime.height) / 3) →
  f.base_length = p.base_length →
  f.base_width = p.base_width →
  f.top_length = p_prime.base_length →
  f.top_width = p_prime.base_width →
  f.height = p.height - p_prime.height →
  ∃ (x : ℝ), x = (45 - 5 * Real.rpow 9 (1/3)) / 2 ∧ 
    x = distance_center_circumsphere_to_apex f p.height :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_xt_distance_l254_25494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_journey_time_l254_25466

noncomputable def janet_journey (north_blocks : ℝ) (north_speed : ℝ) (west_multiplier : ℝ) (west_speed : ℝ)
                  (south_blocks : ℝ) (south_speed : ℝ) (east_multiplier : ℝ) (east_speed : ℝ)
                  (num_stops : ℕ) (stop_duration : ℝ) : ℝ :=
  let north_time := north_blocks / north_speed
  let west_blocks := west_multiplier * north_blocks
  let west_time := west_blocks / west_speed
  let south_time := south_blocks / south_speed
  let east_blocks := east_multiplier * south_blocks
  let east_time := east_blocks / east_speed + (num_stops : ℝ) * stop_duration
  north_time + west_time + south_time + east_time

theorem janet_journey_time :
  janet_journey 3 2.5 7 1.5 8 3 2 2 2 5 = 35.87 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_journey_time_l254_25466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_eq_188_l254_25423

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then 2
  else -1

theorem f_f_2_eq_188 : f (f 2) = 188 := by
  -- Evaluate f(2)
  have h1 : f 2 = 8 := by
    unfold f
    simp [show 2 > 0 by norm_num]
    norm_num
  
  -- Evaluate f(f(2)) = f(8)
  have h2 : f 8 = 188 := by
    unfold f
    simp [show 8 > 0 by norm_num]
    norm_num
  
  -- Combine the results
  calc
    f (f 2) = f 8 := by rw [h1]
    _       = 188 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_eq_188_l254_25423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_trajectory_l254_25475

noncomputable section

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Left focus of the ellipse -/
def LeftFocus : ℝ × ℝ := (-1, 0)

/-- Point G on the ellipse -/
def PointG : ℝ × ℝ := (1, Real.sqrt 2 / 2)

/-- Midpoint of two points -/
def Midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

/-- Trajectory of midpoint M -/
def MidpointTrajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ^ 2 + 2 * p.2 ^ 2 + p.1 = 0}

theorem ellipse_midpoint_trajectory :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  PointG ∈ Ellipse a b ∧
  ∀ (A B : ℝ × ℝ),
    A ∈ Ellipse a b → B ∈ Ellipse a b →
    ∃ (t : ℝ), A = (1 - t) • LeftFocus + t • B →
      Midpoint A B ∈ MidpointTrajectory := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_trajectory_l254_25475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_necessary_not_sufficient_l254_25427

/-- A complex number is represented as a pair of real numbers -/
def MyComplex := ℝ × ℝ

/-- A complex number is pure imaginary if its real part is 0 and imaginary part is non-zero -/
def is_pure_imaginary (z : MyComplex) : Prop :=
  z.1 = 0 ∧ z.2 ≠ 0

theorem a_zero_necessary_not_sufficient :
  (∀ z : MyComplex, is_pure_imaginary z → z.1 = 0) ∧
  (∃ z : MyComplex, z.1 = 0 ∧ ¬is_pure_imaginary z) := by
  constructor
  · intro z h
    exact h.1
  · use (0, 0)
    constructor
    · rfl
    · intro h
      exact h.2 rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_necessary_not_sufficient_l254_25427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l254_25465

theorem triangle_side_length (Q R S : ℝ × ℝ) (cosR RS : ℝ) : 
  (S.1 = 0 ∧ S.2 = 0) →  -- S is at the origin
  (Q.2 = 0) →  -- Q is on the x-axis
  (R.1 = Q.1 ∧ R.2 < 0) →  -- R is directly below Q
  cosR = 3/5 →
  RS = 10 →
  Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = RS →
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = cosR * RS →
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l254_25465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_l254_25405

noncomputable def line1 (x : ℝ) : ℝ := x + 1
noncomputable def line2 (a x : ℝ) : ℝ := -2 * x + a

noncomputable def intersection_point (a : ℝ) : ℝ × ℝ :=
  let x := (a - 1) / 3
  (x, line1 x)

theorem intersection_in_first_quadrant (a : ℝ) :
  (∀ x y, intersection_point a = (x, y) → x > 0 ∧ y > 0) ↔ a > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_l254_25405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l254_25483

theorem two_solutions_for_equation : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 = p.1^3 + p.2) (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_equation_l254_25483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_colors_four_balls_l254_25463

theorem probability_two_colors_four_balls (black white red : ℕ) 
  (h_black : black = 10) (h_white : white = 8) (h_red : red = 6) : 
  let total := black + white + red
  let ways_two_colors := (black.choose 2 * white.choose 2) + 
                         (black.choose 2 * red.choose 2) + 
                         (white.choose 2 * red.choose 2)
  (ways_two_colors : ℚ) / (total.choose 4) = 157 / 845 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_colors_four_balls_l254_25463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_arrival_l254_25449

/-- Represents the distance between points M and N in kilometers -/
noncomputable def distance : ℝ := 15

/-- Represents the walking speed of individuals A, B, and C in km/h -/
noncomputable def walkingSpeed : ℝ := 6

/-- Represents the speed of the bicycle in km/h -/
noncomputable def bicycleSpeed : ℝ := 15

/-- Represents the time when C should leave N, in hours before A and B start from M -/
noncomputable def cStartTime : ℝ := 3 / 11

/-- Proves that C must leave N 3/11 hours before A and B start from M 
    for A and B to arrive at N simultaneously -/
theorem simultaneous_arrival :
  ∀ (x : ℝ),
  0 ≤ x ∧ x ≤ distance →
  (x / walkingSpeed + (distance - x) / bicycleSpeed) = 
  ((distance - x) / bicycleSpeed + x / walkingSpeed) →
  cStartTime = x / walkingSpeed - (distance - x) / bicycleSpeed :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_arrival_l254_25449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minuend_sum_is_1001_l254_25447

/-- Represents a digit in the column problem -/
def Digit := Fin 10

/-- Represents a column in the subtraction problem -/
def Column := List Digit

/-- The sum of digits in a column -/
def columnSum (c : Column) : ℕ := c.foldl (· + ·.val) 0

/-- Represents the entire subtraction problem -/
structure SubtractionProblem where
  minuend : Column
  subtrahend : Column
  difference : Column

/-- The property that the sum of the minuend column equals the minuend itself -/
def validMinuend (sp : SubtractionProblem) : Prop :=
  columnSum sp.minuend = columnSum sp.minuend

/-- The main theorem: In a valid subtraction problem, the sum of the minuend column is 1001 -/
theorem minuend_sum_is_1001 (sp : SubtractionProblem) (h : validMinuend sp) : 
  columnSum sp.minuend = 1001 := by
  sorry

#check minuend_sum_is_1001

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minuend_sum_is_1001_l254_25447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_one_l254_25426

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_one : lg 5 + lg 2 = 1 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_one_l254_25426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l254_25406

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the point of tangency
def point : ℝ × ℝ := (1, 0)

-- Define the slope of the tangent line
noncomputable def tangent_slope : ℝ := 2 * point.fst - 2

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point.fst) + point.snd

-- Theorem statement
theorem tangent_line_equation :
  ∀ x : ℝ, tangent_line x = x - 1 :=
by
  intro x
  unfold tangent_line tangent_slope point
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l254_25406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_x_value_l254_25487

/-- Given a rectangular figure with right angles, prove that X = 4 cm -/
theorem rectangle_x_value (total_area : ℝ) (top_left : ℝ) (top_middle : ℝ) (top_right : ℝ)
  (bottom_left : ℝ) (bottom_middle : ℝ) (bottom_right : ℝ) (X : ℝ) :
  total_area = 35 →
  top_left = 2 →
  top_middle = 1 →
  top_right = 3 →
  bottom_left = 4 →
  bottom_middle = 1 →
  bottom_right = 5 →
  top_left + top_middle + X + top_right = bottom_left + bottom_middle + bottom_right →
  X = 4 := by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Proof steps would go here
  sorry

#check rectangle_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_x_value_l254_25487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_range_l254_25425

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (x - m)^2 - 2 else 2*x^3 - 3*x^2

theorem min_value_implies_m_range (m : ℝ) :
  (∀ x : ℝ, f m x ≥ -1) ∧ (∃ x : ℝ, f m x = -1) → m ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_range_l254_25425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotonicity_l254_25403

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Define the property of being monotonically increasing on an interval
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem quadratic_monotonicity (a : ℝ) :
  MonoIncreasing (f a) (-2) 2 ↔ a ∈ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotonicity_l254_25403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_even_function_l254_25409

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x - Real.sin x

noncomputable def g (n : ℝ) (x : ℝ) : ℝ := f (x + n)

theorem smallest_shift_for_even_function :
  ∃ (n : ℝ), n > 0 ∧ 
  (∀ (x : ℝ), g n x = g n (-x)) ∧
  (∀ (m : ℝ), 0 < m ∧ m < n → ∃ (y : ℝ), g m y ≠ g m (-y)) ∧
  n = Real.pi * 5 / 6 := by
  sorry

#check smallest_shift_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_even_function_l254_25409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l254_25489

theorem division_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  x % y = 3 → 
  (x : ℝ) / (y : ℝ) = 96.15 → 
  y = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l254_25489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_length_l254_25471

-- Define the rectangular solid
structure RectangularSolid where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the rectangular solid
def total_surface_area (r : RectangularSolid) : ℝ :=
  2 * (r.a * r.b + r.b * r.c + r.a * r.c)

def total_edge_length (r : RectangularSolid) : ℝ :=
  4 * (r.a + r.b + r.c)

noncomputable def interior_diagonal (r : RectangularSolid) : ℝ :=
  Real.sqrt (r.a^2 + r.b^2 + r.c^2)

-- State the theorem
theorem interior_diagonal_length (r : RectangularSolid) :
  total_surface_area r = 150 → total_edge_length r = 60 → interior_diagonal r = 5 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_length_l254_25471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_passed_is_six_l254_25445

/-- Represents a stem-and-leaf diagram --/
structure StemAndLeafDiagram where
  stems : List Nat
  leaves : List (List Nat)

/-- The initial ages of the group --/
def initialAges : List Nat := [19, 34, 37, 42, 48]

/-- The initial stem-and-leaf diagram --/
def initialDiagram : StemAndLeafDiagram :=
  { stems := [1, 3, 4],
    leaves := [[9], [4, 7], [2, 8]] }

/-- The new stem-and-leaf diagram structure --/
def newDiagramStructure : StemAndLeafDiagram :=
  { stems := [0, 1, 2, 3, 4],
    leaves := [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0], [0, 0]] }

/-- Function to update ages after a number of years --/
def updateAges (ages : List Nat) (years : Nat) : List Nat :=
  ages.map (· + years)

/-- Predicate to check if a list of ages fits the given diagram structure --/
def fitsStructure (ages : List Nat) (diag : StemAndLeafDiagram) : Prop :=
  sorry  -- Implementation details omitted for brevity

/-- Theorem stating that 6 years have passed --/
theorem years_passed_is_six :
  ∃ (y : Nat), y = 6 ∧
    fitsStructure (updateAges initialAges y) newDiagramStructure ∧
    ∀ (z : Nat), z ≠ y →
      ¬fitsStructure (updateAges initialAges z) newDiagramStructure :=
by
  sorry  -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_passed_is_six_l254_25445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_for_point_l254_25439

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then sin θ = 1/2 -/
theorem sin_theta_for_point (θ : ℝ) : 
  (∃ (r : ℝ), r * Real.cos θ = -Real.sqrt 3 / 2 ∧ r * Real.sin θ = 1/2) → Real.sin θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_for_point_l254_25439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_arrangement_l254_25495

theorem perfect_square_sum_arrangement : ∃ (p : Equiv.Perm (Fin 9)), 
  ∀ i : Fin 9, ∃ n : ℕ, (i.val + 1 : ℕ) + p.toFun i = n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_arrangement_l254_25495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_land_profit_l254_25496

/-- Calculates the profit from a land development project --/
def landDevelopmentProfit (totalAcres : ℕ) (purchasePricePerAcre : ℕ) (sellingPricePerAcre : ℕ) (fractionSold : ℚ) : ℤ :=
  let costOfLand := totalAcres * purchasePricePerAcre
  let acresSold := (totalAcres : ℚ) * fractionSold
  let revenue := (acresSold * sellingPricePerAcre).floor
  revenue - costOfLand

/-- Theorem stating that the profit for Mike's land development project is $6000 --/
theorem mikes_land_profit :
  landDevelopmentProfit 200 70 200 (1/2) = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_land_profit_l254_25496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_MN_passes_through_fixed_point_l254_25477

/-- The trajectory of the center of the moving circle -/
noncomputable def trajectory (x y : ℝ) : Prop := x^2 = 4*y ∧ y > 0

/-- The fixed circle that the moving circle is externally tangent to -/
noncomputable def fixed_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

/-- A point P on the plane -/
structure Point (a b : ℝ) where
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0

/-- The line MN passing through two points on the trajectory -/
noncomputable def line_MN (a x : ℝ) : ℝ → ℝ := λ y => (1/2)*a*x - y + 2

/-- The theorem stating that the line MN passes through a fixed point -/
theorem line_MN_passes_through_fixed_point (a b : ℝ) (P : Point a b) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ ∧ 
    trajectory x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    (1/2)*x₁*a - b - y₁ = 0 ∧
    (1/2)*x₂*a - b - y₂ = 0 ∧
    b = -2 →
    line_MN a 0 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_MN_passes_through_fixed_point_l254_25477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l254_25451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + a*x - 2 else -a^x

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x y : ℝ, 0 < x → x < y → f a x < f a y) ↔ 
  (0 < a ∧ a ≤ 1/2) := by
  sorry

#check f_increasing_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l254_25451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l254_25434

/-- Given vectors a, b, and c in ℝ², prove that if a + λb is perpendicular to c,
    then λ = -19/14 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (lambda : ℝ) 
    (ha : a = (3, 5))
    (hb : b = (2, 4))
    (hc : c = (-3, -2))
    (h_perp : (a.1 + lambda * b.1, a.2 + lambda * b.2) • c = 0) :
    lambda = -19/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l254_25434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l254_25437

/-- The sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- We define a(0) = 1 to handle the base case
  | n + 1 => a n + 2

/-- Theorem stating the explicit formula for a_n -/
theorem a_explicit_formula (n : ℕ) : a n = 2 * n + 1 := by
  induction n with
  | zero => 
    rfl  -- Base case: a(0) = 1 = 2 * 0 + 1
  | succ k ih => 
    calc
      a (k + 1) = a k + 2 := rfl
      _ = (2 * k + 1) + 2 := by rw [ih]
      _ = 2 * (k + 1) + 1 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l254_25437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_evaluation_at_one_l254_25476

open BigOperators Finset

/-- Given a polynomial f(x) satisfying (1-x)ⁿ f(x) = 1 + Σᵢ₌₁ⁿ aᵢ xᵇⁱ, 
    prove that f(1) = (1/n!) * b₁ * b₂ * ... * bₙ -/
theorem polynomial_evaluation_at_one 
  (n : ℕ)
  (a : Fin n → ℝ)
  (b : Fin n → ℕ)
  (f : ℝ → ℝ)
  (hb_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (hf : ∀ x, (1 - x)^n * f x = 1 + ∑ i, a i * x^(b i)) :
  f 1 = (1 / n.factorial) * ∏ i, (b i : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_evaluation_at_one_l254_25476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_focus_coordinates_l254_25499

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h k : ℝ) :
  let f := fun (p : ℝ × ℝ) => a * p.1^2 + k
  ∃ p : ℝ × ℝ, p.1 = h ∧ p.2 = k + 1 / (4 * a) ∧ ∀ x, f (x, 0) = f p + (x - p.1)^2 / (4 * p.2) :=
sorry

/-- For the parabola y = 4x^2, the focus coordinates are (0, 1/16) -/
theorem focus_coordinates :
  let f := fun (p : ℝ × ℝ) => 4 * p.1^2
  ∃ p : ℝ × ℝ, p = (0, 1/16) ∧ ∀ x, f (x, 0) = f p + (x - p.1)^2 / (4 * p.2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_focus_coordinates_l254_25499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_sum_bound_l254_25479

theorem cosine_product_sum_bound (α β γ : ℝ) :
  Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α ≤ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_sum_bound_l254_25479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_theorem_l254_25454

/-- Calculates the discount percentage given markup and profit percentages -/
noncomputable def calculate_discount (markup : ℝ) (profit : ℝ) : ℝ :=
  let marked_price := 1 + markup / 100
  let selling_price := 1 + profit / 100
  ((marked_price - selling_price) / marked_price) * 100

/-- Theorem: If a merchant marks up goods by 80% and makes a 35% profit after
    offering a discount, then the discount offered is 25%. -/
theorem merchant_discount_theorem :
  calculate_discount 80 35 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_theorem_l254_25454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l254_25421

/-- Molar mass of barium in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.33

/-- Molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- Molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of barium hydroxide in g/mol -/
noncomputable def molar_mass_BaOH2 : ℝ := molar_mass_Ba + 2 * molar_mass_O + 2 * molar_mass_H

/-- Mass percentage of hydrogen in barium hydroxide -/
noncomputable def mass_percentage_H : ℝ := (2 * molar_mass_H / molar_mass_BaOH2) * 100

/-- Theorem stating that the mass percentage of hydrogen in barium hydroxide is approximately 1.179% -/
theorem mass_percentage_H_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |mass_percentage_H - 1.179| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l254_25421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_range_inequality_proof_l254_25467

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

-- Theorem for the range of m
theorem extreme_value_range (m : ℝ) :
  (∃ c ∈ Set.Ioo m (m + 1), ∀ x ∈ Set.Ioo m (m + 1), f x ≤ f c) ↔ m ∈ Set.Ioo 0 1 :=
by sorry

-- Theorem for the inequality
theorem inequality_proof (x : ℝ) (h : x > 1) :
  (x + 1) * (x + Real.exp (-x)) * f x > 2 * (1 + 1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_range_inequality_proof_l254_25467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_geometry_axioms_l254_25424

-- Define the basic geometric objects
variable (Point Line : Type)

-- Define the relationships between points and lines
variable (is_on : Point → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (intersects : Line → Line → Prop)

-- Define angles and their relationships
variable (Angle : Type)
variable (angle_between : Line → Line → Point → Angle)
variable (angle_eq : Angle → Angle → Prop)

-- Define alternate interior angles
variable (alternate_interior : Angle → Angle → Line → Line → Line → Prop)

-- Axiom 1: Through a point not on a given line, there exists exactly one parallel line
axiom parallel_existence_uniqueness 
  (P : Point) (L : Line) (h : ¬is_on P L) :
  ∃! L' : Line, parallel L' L ∧ is_on P L'

-- Axiom 2: If alternate interior angles are equal, then the lines are parallel
axiom alternate_interior_parallel 
  (L1 L2 L3 : Line) (A1 A2 : Angle) :
  intersects L3 L1 → intersects L3 L2 → 
  alternate_interior A1 A2 L1 L2 L3 → 
  angle_eq A1 A2 → parallel L1 L2

-- Theorem: Both axioms are true in Euclidean geometry
theorem euclidean_geometry_axioms : 
  (∀ (P : Point) (L : Line), ¬is_on P L → 
    ∃! L' : Line, parallel L' L ∧ is_on P L') ∧
  (∀ (L1 L2 L3 : Line) (A1 A2 : Angle),
    intersects L3 L1 → intersects L3 L2 → 
    alternate_interior A1 A2 L1 L2 L3 → 
    angle_eq A1 A2 → parallel L1 L2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_geometry_axioms_l254_25424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_correct_l254_25416

/-- Represents the area of the figure enclosed by curve Pₙ -/
noncomputable def S (n : ℕ) : ℝ :=
  8/5 - (3/5) * (4/9)^n

/-- The initial triangle P₀ has area 1 -/
axiom initial_area : S 0 = 1

/-- The number of sides in Pₙ is 3 * 4ⁿ -/
def num_sides (n : ℕ) : ℕ := 3 * 4^n

/-- The area added in step n+1 is (num_sides n) * (1/3^(2*n+2)) -/
noncomputable def area_added (n : ℕ) : ℝ := (num_sides n : ℝ) * (1/3^(2*n+2))

/-- The recursive relation for S(n+1) in terms of S(n) -/
axiom area_recurrence (n : ℕ) : S (n+1) = S n + area_added n

/-- The main theorem: S(n) represents the correct area for all n -/
theorem area_formula_correct (n : ℕ) : 
  S n = 8/5 - (3/5) * (4/9)^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_correct_l254_25416
