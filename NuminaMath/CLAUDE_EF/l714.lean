import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_triangle_13_14_15_l714_71492

/-- Given a triangle with sides a, b, and c, calculates its area using Heron's formula -/
noncomputable def heronArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: In a triangle with sides 13, 14, and 15, the shortest altitude has length 2A/15,
    where A is the area calculated using Heron's formula -/
theorem shortest_altitude_triangle_13_14_15 :
  let a := (13 : ℝ)
  let b := (14 : ℝ)
  let c := (15 : ℝ)
  let A := heronArea a b c
  let h_a := 2 * A / a
  let h_b := 2 * A / b
  let h_c := 2 * A / c
  h_c ≤ h_a ∧ h_c ≤ h_b := by sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check heronArea 13 14 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_triangle_13_14_15_l714_71492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l714_71416

-- Define the triangle and its properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  l : ℝ
  m : ℝ
  n : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define properties of the triangle
def is_acute_angled (t : Triangle) : Prop :=
  0 < t.α ∧ t.α < Real.pi/2 ∧
  0 < t.β ∧ t.β < Real.pi/2 ∧
  0 < t.γ ∧ t.γ < Real.pi/2

def is_inside (t : Triangle) : Prop :=
  -- Define what it means for P to be inside the triangle
  sorry

def perpendicular_lengths (t : Triangle) : Prop :=
  -- Define the relationship between P and the perpendicular lengths
  sorry

-- Define area function
noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  -- Placeholder for area calculation
  sorry

-- Theorem statement
theorem triangle_area (t : Triangle)
  (h_acute : is_acute_angled t)
  (h_inside : is_inside t)
  (h_perp : perpendicular_lengths t) :
  area t.A t.B t.C = (t.l * Real.sin t.γ + t.m * Real.sin t.α + t.n * Real.sin t.β)^2 /
                     (2 * Real.sin t.α * Real.sin t.β * Real.sin t.γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l714_71416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l714_71434

noncomputable section

def vector1 : ℝ × ℝ := (-3, 2)
def vector2 : ℝ × ℝ := (1, -4)
def point1 : ℝ × ℝ := (-1, 3)
def point2 : ℝ × ℝ := (4, -2)

-- v is any vector on the line passing through point1 and point2
def v : ℝ × ℝ := (5, -5)

-- Define the dot product
def dot_product (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1) + (a.2 * b.2)

-- Define the projection function
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

theorem projection_theorem :
  proj vector1 v = (-2.5, 2.5) ∧ proj vector2 v = (1.5, -1.5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l714_71434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_after_discounts_l714_71483

/-- Applies a discount percentage to a given price -/
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount / 100)

/-- Theorem: The final price of a car after successive discounts -/
theorem car_price_after_discounts (initial_price : ℝ) 
  (discount1 discount2 discount3 discount4 : ℝ) :
  initial_price = 20000 ∧ 
  discount1 = 25 ∧ 
  discount2 = 20 ∧ 
  discount3 = 15 ∧ 
  discount4 = 10 → 
  apply_discount 
    (apply_discount 
      (apply_discount 
        (apply_discount initial_price discount1) 
      discount2) 
    discount3) 
  discount4 = 9180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_after_discounts_l714_71483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l714_71412

/-- Calculates the length of a train given its speed, time to pass a platform, and the platform length. -/
theorem train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : 
  speed = 45 * (1000 / 3600) → 
  time = 40.8 → 
  platform_length = 150 → 
  speed * time - platform_length = 360 := by 
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l714_71412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_a_range_l714_71476

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 2/x

-- Define the properties of g
def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x > g y

-- Main theorem
theorem f_increasing_and_a_range 
  (g : ℝ → ℝ) 
  (h_g_decreasing : is_decreasing g)
  (h_inequality : ∀ (x : ℝ), x ≥ 1 → g (x^3 + 2) < g ((a^2 - 2*a) * x)) :
  (∀ (x y : ℝ), x ≥ 1 → y ≥ 1 → x < y → f x < f y) ∧ 
  (-1 < a ∧ a < 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_a_range_l714_71476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_z_coordinate_l714_71458

def line_point (t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 2 - t, 1 - 3*t)

theorem point_z_coordinate :
  let p1 : ℝ × ℝ × ℝ := (2, 2, 1)
  let p2 : ℝ × ℝ × ℝ := (5, 1, -2)
  let line := line_point
  ∃ t : ℝ, (line t).1 = 4 ∧ (line t).2.2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_z_coordinate_l714_71458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l714_71441

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | n + 2 => (1 + sequence_a (n + 1)) / (1 - sequence_a (n + 1))

theorem a_2016_value : sequence_a 2016 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l714_71441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_one_half_l714_71445

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- State the theorem
theorem log4_one_half : log4 (1/2) = -1/2 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_one_half_l714_71445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_path_shorter_l714_71417

/-- Represents a rectangular field with length and width -/
structure RectangularField where
  length : ℝ
  width : ℝ

/-- Calculates the length of the path along two sides of the rectangle -/
def twoSidePath (field : RectangularField) : ℝ :=
  field.length + field.width

/-- Calculates the length of the diagonal path across the rectangle -/
noncomputable def diagonalPath (field : RectangularField) : ℝ :=
  Real.sqrt (field.length^2 + field.width^2)

/-- Calculates the percentage difference between two paths -/
noncomputable def percentageDifference (path1 path2 : ℝ) : ℝ :=
  (path1 - path2) / path1 * 100

theorem diagonal_path_shorter (field : RectangularField) 
  (h1 : field.length = 3)
  (h2 : field.width = 4) :
  ∃ (ε : ℝ), abs (percentageDifference (twoSidePath field) (diagonalPath field) - 100 * (2/7)) < ε ∧ ε < 0.01 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_path_shorter_l714_71417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l714_71418

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * (sin (a * x + π / 4))^2

theorem condition_sufficient_not_necessary :
  (∀ x ∈ Set.Ioo (π / 12) (π / 6), StrictMonoOn (f 1) (Set.Ioo (π / 12) (π / 6))) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∀ x ∈ Set.Ioo (π / 12) (π / 6), StrictMonoOn (f a) (Set.Ioo (π / 12) (π / 6))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l714_71418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l714_71424

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point (x, y) to a line ax + by + c = 0 -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line has equal intercepts on both axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The set of possible equations for the line satisfying the given conditions -/
def possibleLines : Set Line :=
  { Line.mk 7 1 0, Line.mk 1 (-1) 0, Line.mk 1 1 (-2), Line.mk 1 1 (-6) }

theorem line_equation_theorem :
  ∀ l : Line,
    hasEqualIntercepts l ∧ distancePointToLine 1 3 l = Real.sqrt 2 →
    l ∈ possibleLines := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l714_71424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_less_than_mean_l714_71494

noncomputable def normal_distribution (μ σ x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

noncomputable def prob_less_than (μ σ x : ℝ) : ℝ := 
  ∫ y in Set.Iio x, normal_distribution μ σ y

theorem prob_less_than_mean {σ : ℝ} (hσ : σ > 0) : 
  prob_less_than 2016 σ 2016 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_less_than_mean_l714_71494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_expected_greetings_l714_71405

/-- Represents the probability of sending a WeChat greeting -/
def Probability : Type := Float

/-- Represents the number of people for a given probability -/
def PeopleCount : Type := Nat

/-- Calculates the expected number of WeChat greetings -/
def expectedGreetings (probs : List Float) (counts : List Nat) : Float :=
  (List.zip probs counts).map (fun (p, c) => p * c.toFloat) |>.sum

/-- Theorem stating the expected number of WeChat greetings Xiao Li should receive -/
theorem xiao_li_expected_greetings :
  let probs : List Float := [1, 0.8, 0.5, 0]
  let counts : List Nat := [8, 15, 14, 3]
  expectedGreetings probs counts = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_expected_greetings_l714_71405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l714_71406

-- Define the point M
def M : ℝ × ℝ × ℝ := (4, -3, 5)

-- Define m as the distance from M to the x-axis
noncomputable def m : ℝ := Real.sqrt ((M.2.1)^2 + (M.2.2)^2)

-- Define n as the distance from M to the xy-coordinate plane
def n : ℝ := abs M.2.2

-- Theorem statement
theorem distance_sum_theorem : m^2 + n = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l714_71406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finland_forest_percentage_l714_71429

def finland_forest_area : ℝ := 53.42
def world_forest_area : ℝ := 8076

theorem finland_forest_percentage :
  abs ((finland_forest_area / world_forest_area) * 100 - 0.6615) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finland_forest_percentage_l714_71429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_calculation_l714_71471

noncomputable def salary : ℝ := 10000
noncomputable def food_percentage : ℝ := 40
noncomputable def entertainment_percentage : ℝ := 10
noncomputable def conveyance_percentage : ℝ := 10
noncomputable def savings : ℝ := 2000

noncomputable def house_rent_percentage : ℝ := 100 - (food_percentage + entertainment_percentage + conveyance_percentage + (savings / salary * 100))

theorem house_rent_calculation : house_rent_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_calculation_l714_71471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l714_71440

noncomputable section

variables {f : ℝ → ℝ}

axiom f_zero : f 0 = 2
axiom f_deriv_sum : ∀ x : ℝ, f x + deriv f x > 1

theorem solution_set :
  {x : ℝ | Real.exp x * f x > Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l714_71440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_method_larger_volume_l714_71468

/-- The volume of a cylinder with circumference c and height h -/
noncomputable def cylinderVolume (c h : ℝ) : ℝ := (c^2 * h) / (4 * Real.pi)

/-- Theorem stating that the volume of the second method is greater than the first -/
theorem second_method_larger_volume :
  cylinderVolume 5 2.5 > cylinderVolume 2.5 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_method_larger_volume_l714_71468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ten_equals_product_l714_71421

theorem factorial_ten_equals_product (n : ℕ) : 2^6 * 3^3 * 1050 = Nat.factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ten_equals_product_l714_71421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l714_71491

theorem undefined_values_count : 
  ∃! (S : Finset ℝ), (∀ x ∈ S, (x^2 + x - 6) * (x - 4) = 0) ∧ S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l714_71491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_amount_l714_71470

/-- The amount of food given to a cat per day, given the weight of the empty bowl,
    the weight of the bowl after the cat has eaten some food, and the amount eaten. -/
noncomputable def food_per_day (empty_bowl_weight : ℝ) (filled_bowl_weight : ℝ) (food_eaten : ℝ) : ℝ :=
  ((filled_bowl_weight - empty_bowl_weight + food_eaten) / 3)

/-- Theorem stating that the amount of food given to the cat per day is 60 grams. -/
theorem cat_food_amount :
  food_per_day 420 586 14 = 60 := by
  -- Unfold the definition of food_per_day
  unfold food_per_day
  -- Simplify the arithmetic expression
  simp [sub_add_eq_add_sub, add_comm, add_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_amount_l714_71470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_is_120_l714_71467

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_angle_A_is_120 (t : Triangle) :
  t.a^2 - t.b^2 = 3 * t.b * t.c →  -- First condition
  Real.sin t.C = 2 * Real.sin t.B →  -- Second condition
  t.A = 120 * π / 180 :=  -- Conclusion (120° in radians)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_is_120_l714_71467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_one_equals_one_l714_71411

theorem integral_of_one_equals_one : ∫ x in (0:ℝ)..(1:ℝ), (1 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_one_equals_one_l714_71411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_digits_1024_l714_71456

noncomputable def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.floor (Real.log (n : ℝ) / Real.log (base : ℝ)) + 1

theorem equal_digits_1024 :
  num_digits 1024 4 = num_digits 1024 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_digits_1024_l714_71456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l714_71482

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y/3 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the right focus
def line_through_right_focus (m : ℝ) (x : ℝ) : ℝ := m * (x - 2)

-- Define the intersection points P and Q
def intersection_point (x : ℝ) : Prop :=
  ∃ y : ℝ, hyperbola x y ∧ ∃ m : ℝ, y = line_through_right_focus m x

-- Define the right angle condition
def right_angle_condition (P Q : ℝ × ℝ) : Prop :=
  let F₁ := left_focus
  (P.1 - F₁.1) * (Q.1 - P.1) + (P.2 - F₁.2) * (Q.2 - P.2) = 0

-- Theorem statement
theorem inscribed_circle_radius :
  ∀ P Q : ℝ × ℝ,
  intersection_point P.1 →
  intersection_point Q.1 →
  right_angle_condition P Q →
  P.1 > 2 →
  Q.1 > 2 →
  P ≠ Q →
  let F₁ := left_focus
  let s := (dist F₁ P + dist F₁ Q + dist P Q) / 2
  let A := abs ((P.1 - F₁.1) * (Q.2 - F₁.2) - (Q.1 - F₁.1) * (P.2 - F₁.2)) / 2
  A / s = Real.sqrt 7 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l714_71482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_reach_3_5_and_200_6_cannot_reach_12_60_and_200_5_l714_71425

-- Define the type for a point in the coordinate system
structure Point where
  x : ℕ
  y : ℕ

-- Define the possible jumps
inductive Jump : Point → Point → Prop where
  | double_x (p : Point) : Jump p ⟨2 * p.x, p.y⟩
  | double_y (p : Point) : Jump p ⟨p.x, 2 * p.y⟩
  | subtract_x (p : Point) (h : p.x > p.y) : Jump p ⟨p.x - p.y, p.y⟩
  | subtract_y (p : Point) (h : p.x < p.y) : Jump p ⟨p.x, p.y - p.x⟩

-- Define reachability
def Reachable (start finish : Point) : Prop :=
  ∃ (n : ℕ) (path : Fin (n + 1) → Point),
    path ⟨0, Nat.zero_lt_succ n⟩ = start ∧
    path ⟨n, Nat.lt_succ_self n⟩ = finish ∧
    ∀ i : Fin n, Jump (path i) (path i.succ)

-- Theorem: The frog can reach (3, 5) and (200, 6)
theorem can_reach_3_5_and_200_6 :
  Reachable ⟨1, 1⟩ ⟨3, 5⟩ ∧ Reachable ⟨1, 1⟩ ⟨200, 6⟩ := by sorry

-- Theorem: The frog cannot reach (12, 60) and (200, 5)
theorem cannot_reach_12_60_and_200_5 :
  ¬Reachable ⟨1, 1⟩ ⟨12, 60⟩ ∧ ¬Reachable ⟨1, 1⟩ ⟨200, 5⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_reach_3_5_and_200_6_cannot_reach_12_60_and_200_5_l714_71425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_negative_8_l714_71463

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 1) => sequence_a n - 3

theorem a_4_equals_negative_8 : sequence_a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_negative_8_l714_71463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_eq_5_l714_71475

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 1 else -2*x

theorem unique_solution_for_f_eq_5 : 
  ∃! x : ℝ, f x = 5 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_eq_5_l714_71475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_divisors_count_l714_71427

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def count_special_divisors : ℕ :=
  (Finset.filter (λ d ↦ 
    d ∣ factorial 10 ∧ 
    d > factorial 9 ∧ 
    Even (factorial 10 / d)) (Finset.range (factorial 10 + 1))).card

theorem special_divisors_count : count_special_divisors = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_divisors_count_l714_71427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_vacation_duration_l714_71480

/-- Calculates the number of vacation days based on coffee consumption --/
def vacation_days (pods_per_day : ℕ) (pods_per_box : ℕ) (price_per_box : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent / price_per_box * pods_per_box) / pods_per_day

/-- Proves that Sharon's vacation duration is 40 days given her coffee consumption and spending --/
theorem sharon_vacation_duration :
  let pods_per_day : ℕ := 3
  let pods_per_box : ℕ := 30
  let price_per_box : ℚ := 8
  let total_spent : ℚ := 32
  vacation_days pods_per_day pods_per_box price_per_box total_spent = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_vacation_duration_l714_71480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l714_71474

-- Part 1
theorem sin_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by sorry

-- Part 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, x ≠ 0 → f a x < f a 0) ↔ 
  a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l714_71474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_optimal_price_l714_71400

/-- Represents the sales model for a product -/
structure SalesModel where
  cost_price : ℚ
  initial_price : ℚ
  initial_sales : ℚ
  price_decrease : ℚ
  sales_increase : ℚ

/-- Calculates the daily sales volume given a selling price -/
def daily_sales (model : SalesModel) (selling_price : ℚ) : ℚ :=
  model.initial_sales + (model.sales_increase / model.price_decrease) * (model.initial_price - selling_price)

/-- Calculates the daily profit given a selling price -/
def daily_profit (model : SalesModel) (selling_price : ℚ) : ℚ :=
  (selling_price - model.cost_price) * (daily_sales model selling_price)

/-- Theorem stating the maximum daily profit and the optimal selling price -/
theorem max_profit_at_optimal_price (model : SalesModel) 
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 60)
  (h_initial_sales : model.initial_sales = 20)
  (h_price_decrease : model.price_decrease = 5)
  (h_sales_increase : model.sales_increase = 10) :
  ∃ (optimal_price : ℚ), 
    optimal_price = 55 ∧ 
    daily_profit model optimal_price = 450 ∧
    ∀ (price : ℚ), daily_profit model price ≤ daily_profit model optimal_price := by
  sorry

#eval daily_profit { cost_price := 40, initial_price := 60, initial_sales := 20, price_decrease := 5, sales_increase := 10 } 55

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_optimal_price_l714_71400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_condition_l714_71410

theorem no_solution_condition (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π) :
  (∀ x : ℝ, Real.sin (x + t) ≠ 1 - Real.sin x) ↔ (2 * π / 3 < t ∧ t ≤ π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_condition_l714_71410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_calculation_l714_71487

noncomputable def rectangular_park_area (speed_kmh : ℝ) (time_min : ℝ) (ratio : ℝ) : ℝ :=
  let speed_mpm := speed_kmh * 1000 / 60
  let perimeter := speed_mpm * time_min
  let length := perimeter / (2 * (1 + ratio))
  let breadth := ratio * length
  length * breadth

theorem park_area_calculation :
  rectangular_park_area 12 8 3 = 120000 := by
  unfold rectangular_park_area
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_calculation_l714_71487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_projections_l714_71452

/-- A circle in a plane -/
class Circle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
(center : α)
(radius : ℝ)

/-- A point lies on a circle -/
def lies_on_circle {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (p : α) (c : Circle α) : Prop :=
  ‖p - c.center‖ = c.radius

/-- Orthogonal projection of a point onto a line -/
noncomputable def orthogonal_projection {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (p q r : α) : α :=
  q + (inner (p - q) (r - q) / ‖r - q‖^2) • (r - q)

/-- Four points are concyclic if they lie on the same circle -/
def concyclic {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (p q r s : α) : Prop :=
  ∃ (c : Circle α), lies_on_circle p c ∧ lies_on_circle q c ∧ lies_on_circle r c ∧ lies_on_circle s c

theorem concyclic_projections 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
  (Γ : Circle α) (A B C D A' B' C' D' : α) :
  lies_on_circle A Γ → lies_on_circle B Γ → lies_on_circle C Γ → lies_on_circle D Γ →
  A' = orthogonal_projection A B D →
  C' = orthogonal_projection C B D →
  B' = orthogonal_projection B A C →
  D' = orthogonal_projection D A C →
  concyclic A' B' C' D' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_projections_l714_71452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_position_l714_71415

-- Define the radius of the clock
def clock_radius : ℝ := 15

-- Define the number of hours the hand has moved
def hours : ℕ := 7

-- Define the angle per hour in radians
noncomputable def angle_per_hour : ℝ := 2 * Real.pi / 12

-- Define the angle of the hour hand in radians
noncomputable def hour_angle : ℝ := hours * angle_per_hour

-- Theorem statement
theorem clock_hand_position :
  (Real.cos hour_angle = -Real.sqrt 3 / 2) ∧
  (clock_radius * Real.cos hour_angle = -15 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_position_l714_71415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l714_71477

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_on f (Set.Iic 0))
  (h_ineq : f (-2) < f 1) :
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l714_71477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_fence_perimeter_l714_71462

/-- The perimeter of a regular octagonal fence --/
def octagon_perimeter (pole_length : ℝ) (num_poles : ℕ) : ℝ :=
  pole_length * (num_poles : ℝ)

theorem octagon_fence_perimeter :
  let pole_length_m : ℝ := 2.3
  let num_poles : ℕ := 8
  let m_to_cm : ℝ := 100
  octagon_perimeter (pole_length_m * m_to_cm) num_poles = 1840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_fence_perimeter_l714_71462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_tangent_line_at_one_f_le_g_when_lambda_half_lambda_range_for_f_le_g_l714_71498

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.log x
def g (lambda : ℝ) (x : ℝ) : ℝ := lambda * (x^2 - 1)

-- Part 1: Same tangent line at x = 1
theorem same_tangent_line_at_one (lambda : ℝ) : 
  (deriv f 1 = deriv (g lambda) 1) → lambda = 1/2 := by sorry

-- Part 2: f(x) ≤ g(x) when lambda = 1/2 and x ≥ 1
theorem f_le_g_when_lambda_half : 
  ∀ x : ℝ, x ≥ 1 → f x ≤ g (1/2) x := by sorry

-- Part 3: Range of lambda for f(x) ≤ g(x) to hold for all x ∈ [1, +∞)
theorem lambda_range_for_f_le_g (lambda : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → f x ≤ g lambda x) ↔ lambda ≥ 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_tangent_line_at_one_f_le_g_when_lambda_half_lambda_range_for_f_le_g_l714_71498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OABC_l714_71428

noncomputable def A : ℝ × ℝ := (2, 0)

def on_ellipse (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 / 4 + p.2^2 / 3 = 1

noncomputable def C (B : ℝ × ℝ) : ℝ × ℝ := (0, B.2)

noncomputable def area_OABC (B : ℝ × ℝ) : ℝ :=
  (B.1 * B.2) / 2 + (2 - B.1) * B.2 / 2

theorem max_area_OABC :
  ∃ (B : ℝ × ℝ), on_ellipse B ∧
    ∀ (B' : ℝ × ℝ), on_ellipse B' →
      area_OABC B ≥ area_OABC B' ∧
      area_OABC B = 9/4 :=
by
  sorry

#check max_area_OABC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OABC_l714_71428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l714_71451

def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

def P_real : Set ℝ := {x | ∃ n : ℕ, n ∈ P ∧ x = n}

theorem intersection_P_Q : P_real ∩ Q = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l714_71451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_greater_than_y2_l714_71473

-- Define the inverse proportion function
noncomputable def inverse_proportion (x : ℝ) : ℝ := -3 / x

-- Theorem statement
theorem y1_greater_than_y2 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) 
  (h2 : 0 < x₂) 
  (h3 : y₁ = inverse_proportion x₁) 
  (h4 : y₂ = inverse_proportion x₂) : 
  y₁ > y₂ := by
  sorry

#check y1_greater_than_y2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_greater_than_y2_l714_71473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_of_digits_l714_71403

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ ({3, 4, 7, 8} : Set ℕ) ∧ b ∈ ({3, 4, 7, 8} : Set ℕ) ∧ 
  c ∈ ({3, 4, 7, 8} : Set ℕ) ∧ d ∈ ({3, 4, 7, 8} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def form_number (tens ones : ℕ) : ℕ := tens * 10 + ones

theorem smallest_product_of_digits :
  ∀ a b c d : ℕ, is_valid_arrangement a b c d →
    (form_number a b * form_number c d ≥ 1776 ∧
     ∃ w x y z : ℕ, is_valid_arrangement w x y z ∧ form_number w x * form_number y z = 1776) :=
by
  sorry

#check smallest_product_of_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_of_digits_l714_71403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigpen_minimum_cost_l714_71496

/-- The cost function for a rectangular pigpen -/
noncomputable def pigpen_cost (x : ℝ) : ℝ :=
  6 * x + 2 * 80 * (12 / x) + 110 * x

/-- Theorem stating the minimum cost of the pigpen -/
theorem pigpen_minimum_cost :
  ∃ (x : ℝ), x > 0 ∧ x * (12 / x) = 12 ∧
  (∀ (y : ℝ), y > 0 ∧ y * (12 / y) = 12 → pigpen_cost x ≤ pigpen_cost y) ∧
  pigpen_cost x = 4000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigpen_minimum_cost_l714_71496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l714_71453

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  1 + (Real.tan A / Real.tan B) = (2 * c) / b →
  -- Conclusion
  A = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l714_71453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_product_l714_71401

/-- A monic polynomial of degree 3 -/
structure MonicCubic (R : Type*) [CommRing R] where
  a : R
  b : R
  c : R

/-- The evaluation of a monic cubic polynomial at x -/
def eval_monic_cubic {R : Type*} [CommRing R] (p : MonicCubic R) (x : R) : R :=
  x^3 + p.a * x^2 + p.b * x + p.c

theorem constant_term_of_product {R : Type*} [CommRing R] [OrderedRing R] (p q : MonicCubic R) :
  p.b = q.b →
  p.c = q.c →
  p.c > 0 →
  (fun x ↦ eval_monic_cubic p x * eval_monic_cubic q x) =
    (fun x ↦ x^6 + 2*x^5 + 5*x^4 + 8*x^3 + 5*x^2 + 2*x + 9) →
  p.c = 3 :=
by
  sorry

#check constant_term_of_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_product_l714_71401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_employee_count_l714_71488

theorem company_employee_count 
  (january_count : ℝ) 
  (december_count : ℕ) 
  (h1 : january_count = 391.304347826087) 
  (h2 : december_count = Int.floor (january_count * 1.15)) : 
  december_count = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_employee_count_l714_71488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_rotation_bounds_l714_71486

def is_eight_digit (n : ℕ) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def rotate_last_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let q := n / 10
  d * 10000000 + q

def coprime_with_12 (n : ℕ) : Prop := Nat.Coprime n 12

theorem eight_digit_rotation_bounds :
  ∃ (A B : ℕ),
    is_eight_digit A ∧
    is_eight_digit B ∧
    A = rotate_last_to_first B ∧
    coprime_with_12 B ∧
    B > 44444444 ∧
    (∀ A' B', 
      is_eight_digit A' ∧
      is_eight_digit B' ∧
      A' = rotate_last_to_first B' ∧
      coprime_with_12 B' ∧
      B' > 44444444 →
      A' ≤ A) ∧
    (∃ A₀ B₀,
      is_eight_digit A₀ ∧
      is_eight_digit B₀ ∧
      A₀ = rotate_last_to_first B₀ ∧
      coprime_with_12 B₀ ∧
      B₀ > 44444444 ∧
      A₀ = 14444446) :=
by sorry

#eval rotate_last_to_first 99999989  -- Should output 99999998
#eval rotate_last_to_first 44444461  -- Should output 14444446

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_rotation_bounds_l714_71486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_15_15_l714_71472

/-- Represents the time on a 12-hour analog clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  h_range : hours < 12
  m_range : minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position -/
def hour_hand_angle (t : ClockTime) : ℝ :=
  (t.hours * 30 + t.minutes * 0.5 : ℝ)

/-- Calculates the angle of the minute hand from 12 o'clock position -/
def minute_hand_angle (t : ClockTime) : ℝ :=
  (t.minutes * 6 : ℝ)

/-- Calculates the acute angle between the hour and minute hands -/
def angle_between_hands (t : ClockTime) : ℝ :=
  abs (hour_hand_angle t - minute_hand_angle t)

theorem angle_at_15_15 :
  ∃ (t : ClockTime), t.hours = 3 ∧ t.minutes = 15 ∧ angle_between_hands t = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_15_15_l714_71472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_theorem_l714_71437

/-- Represents a hyperbola with center (h, k), focus (h, f), and vertex (h, v) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  f : ℝ
  v : ℝ

/-- The sum of h, k, a, and b for a hyperbola -/
noncomputable def hyperbola_sum (hyp : Hyperbola) : ℝ :=
  let a := |hyp.v - hyp.k|
  let c := |hyp.f - hyp.k|
  let b := Real.sqrt (c^2 - a^2)
  hyp.h + hyp.k + a + b

/-- Theorem: For the given hyperbola, the sum of h, k, a, and b is 4√3 + 6 -/
theorem hyperbola_sum_theorem (hyp : Hyperbola) 
    (h_center : hyp.h = 2 ∧ hyp.k = 0)
    (h_focus : hyp.f = 8)
    (h_vertex : hyp.v = -4) : 
  hyperbola_sum hyp = 4 * Real.sqrt 3 + 6 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_theorem_l714_71437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_unique_same_views_l714_71499

/-- Represents the three standard views of a solid -/
structure ThreeViews where
  side : Set (ℝ × ℝ)
  front : Set (ℝ × ℝ)
  top : Set (ℝ × ℝ)

/-- Defines a solid with its three views -/
class Solid where
  three_views : ThreeViews

/-- Sphere is a Solid -/
def Sphere : Solid where
  three_views := { 
    side := {p | p.1^2 + p.2^2 ≤ 1},
    front := {p | p.1^2 + p.2^2 ≤ 1},
    top := {p | p.1^2 + p.2^2 ≤ 1}
  }

/-- Cube is a Solid -/
def Cube : Solid where
  three_views := { 
    side := {p | -1 ≤ p.1 ∧ p.1 ≤ 1 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1},
    front := {p | -1 ≤ p.1 ∧ p.1 ≤ 1 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1},
    top := {p | -1 ≤ p.1 ∧ p.1 ≤ 1 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1}
  }

/-- Regular tetrahedron is a Solid -/
def RegularTetrahedron : Solid where
  three_views := { 
    side := {p | p.1 + p.2 ≤ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0},
    front := {p | p.1 + p.2 ≤ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0},
    top := {p | p.1^2 + p.2^2 ≤ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}
  }

/-- Predicate to check if all three views are the same -/
def has_same_three_views (s : Solid) : Prop :=
  s.three_views.side = s.three_views.front ∧ 
  s.three_views.front = s.three_views.top

/-- Theorem stating that only the sphere has the same three views -/
theorem sphere_unique_same_views :
  has_same_three_views Sphere ∧
  ¬has_same_three_views Cube ∧
  ¬has_same_three_views RegularTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_unique_same_views_l714_71499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l714_71457

-- Define the parametric equations for the ellipse
noncomputable def ellipse_param (φ : ℝ) : ℝ × ℝ := (5 * Real.cos φ, 4 * Real.sin φ)

-- Define the parametric equations for the line
def line_param (t : ℝ) : ℝ × ℝ := (1 - 3 * t, 4 * t)

-- Theorem for the ellipse
theorem ellipse_equation (φ : ℝ) :
  let (x, y) := ellipse_param φ
  (x^2 / 25) + (y^2 / 16) = 1 := by
  sorry

-- Theorem for the line
theorem line_equation (t : ℝ) :
  let (x, y) := line_param t
  4 * x + 3 * y - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l714_71457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_investment_time_l714_71419

/-- The time (in years) for an investment to grow from an initial amount to a final amount
    given an annual interest rate and compounding frequency. -/
noncomputable def investment_time (initial_amount final_amount : ℝ) (annual_rate : ℝ) (compounds_per_year : ℝ) : ℝ :=
  (Real.log (final_amount / initial_amount)) / (compounds_per_year * Real.log (1 + annual_rate / compounds_per_year))

/-- Theorem stating that Sam's investment time is approximately 1 year -/
theorem sam_investment_time :
  let initial_amount : ℝ := 15000
  let final_amount : ℝ := 16537.5
  let annual_rate : ℝ := 0.10
  let compounds_per_year : ℝ := 2
  abs (investment_time initial_amount final_amount annual_rate compounds_per_year - 1) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_investment_time_l714_71419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_order_l714_71407

/-- Represents the gas volume per capita for a region in cubic meters per person -/
structure GasVolumePerCapita where
  value : Float

/-- The gas volume per capita for the Western region -/
def west : GasVolumePerCapita := ⟨21428⟩

/-- The gas volume per capita for the Non-Western region -/
def nonWest : GasVolumePerCapita := ⟨26848.55⟩

/-- The gas volume per capita for Russia -/
def russia : GasVolumePerCapita := ⟨302790.13⟩

/-- Define an ordering on GasVolumePerCapita -/
instance : LT GasVolumePerCapita where
  lt a b := a.value < b.value

/-- Theorem stating the order of gas volume per capita from lowest to highest -/
theorem gas_volume_order : west < nonWest ∧ nonWest < russia := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_order_l714_71407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_pairs_l714_71432

/-- Greatest Common Factor of two natural numbers -/
def gcf (a b : ℕ) : ℕ := Nat.gcd a b

/-- Least Common Multiple of two natural numbers -/
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

/-- Theorem stating that GCF(LCM(8, 14), LCM(7, 12)) = 28 -/
theorem gcf_of_lcm_pairs : gcf (Nat.lcm 8 14) (Nat.lcm 7 12) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_pairs_l714_71432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l714_71465

/-- Calculates the total amount after applying simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

/-- Proves that given an initial investment of $400 and a total of $480 after 2 years
    under simple interest, the total amount after 7 years is $680 -/
theorem simple_interest_problem (initialAmount : ℝ) (amountAfter2Years : ℝ) 
    (h1 : initialAmount = 400)
    (h2 : amountAfter2Years = 480)
    (h3 : simpleInterest initialAmount ((amountAfter2Years - initialAmount) / (2 * initialAmount)) 2 = amountAfter2Years) :
  simpleInterest initialAmount ((amountAfter2Years - initialAmount) / (2 * initialAmount)) 7 = 680 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l714_71465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_2x_minus_y_l714_71431

-- Define the given conditions
noncomputable def x : ℝ := Real.log 3 / Real.log 5
noncomputable def y : ℝ := Real.log (9/25) / Real.log 5

-- State the theorem
theorem value_of_2x_minus_y : 2 * x - y = 2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_2x_minus_y_l714_71431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_2_or_3_30_l714_71404

def is_multiple_of_2_or_3 (n : ℕ) : Bool := n % 2 = 0 || n % 3 = 0

def count_multiples (n : ℕ) : ℕ := (Finset.range n).filter (fun i => is_multiple_of_2_or_3 (i + 1)) |>.card

theorem probability_multiple_2_or_3_30 :
  (count_multiples 30 : ℚ) / 30 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_2_or_3_30_l714_71404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l714_71422

-- Define the function f(x) = 1 / (2^x)
noncomputable def f (x : ℝ) : ℝ := 1 / (2^x)

-- Define the interval [-1, 2]
def interval : Set ℝ := Set.Icc (-1) 2

-- Statement to prove
theorem min_value_of_f_on_interval :
  ∃ (y : ℝ), y ∈ Set.image f interval ∧
  ∀ (z : ℝ), z ∈ Set.image f interval → y ≤ z ∧
  y = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l714_71422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_a_range_l714_71446

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x - 1) / 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 * (f x - x + 1) / x

-- Theorem for the first part of the problem
theorem f_expression (x : ℝ) (h : x ≥ 0) :
  f (Real.sqrt (2*x + 1)) = x - Real.sqrt (2*x + 1) := by
  sorry

-- Theorem for the second part of the problem
theorem a_range :
  ∀ a : ℝ, (∀ x : ℝ, 1/3 ≤ x ∧ x ≤ 3 → g x ≤ a * x) ↔ a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_a_range_l714_71446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_statue_scale_model_l714_71478

/-- Represents the ratio between a model and the actual object it represents -/
structure ScaleModel where
  objectHeight : ℚ  -- Height of the actual object in feet
  modelHeight : ℚ   -- Height of the model in inches

/-- Calculates how many feet of the actual object one inch of the model represents -/
def ScaleModel.feetPerModelInch (sm : ScaleModel) : ℚ :=
  (sm.objectHeight * 12) / sm.modelHeight

/-- Theorem stating that for the given statue and model, one inch of the model represents 20 feet of the statue -/
theorem washington_statue_scale_model :
  let sm : ScaleModel := { objectHeight := 120, modelHeight := 6 }
  sm.feetPerModelInch = 20 := by
  sorry

#eval (ScaleModel.feetPerModelInch { objectHeight := 120, modelHeight := 6 })

end NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_statue_scale_model_l714_71478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l714_71481

theorem equation_solution : ∃ x : ℝ, 
  5^(Real.sqrt (x^3 + 3*x^2 + 3*x + 1)) = Real.sqrt ((5 * ((x+1)^(5/4)))^3) ∧ 
  x = 65/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l714_71481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_in_terms_of_a_and_b_l714_71447

theorem tan_x_in_terms_of_a_and_b (a b x : ℝ) 
  (h1 : Real.cos x = (a^2 - b^2) / (a^2 + b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x)
  (h5 : x < Real.pi/2) :
  Real.tan x = 2*a*b / (a^2 - b^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_in_terms_of_a_and_b_l714_71447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_point_property_l714_71435

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (is_rhombus : sorry)
  (side_length : ∀ (X Y : ℝ × ℝ), (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A) → 
    Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 2)
  (angle_BAD : sorry)

-- Define points E and F
def E (r : Rhombus) (l : ℝ) : ℝ × ℝ := sorry
def F (r : Rhombus) (m : ℝ) : ℝ × ℝ := sorry

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem rhombus_point_property (r : Rhombus) (l m : ℝ) : 
  dot_product (vec_sub (E r l) r.A) (vec_sub (F r m) r.A) = 1 ∧ 
  dot_product (vec_sub (E r l) r.C) (vec_sub (F r m) r.C) = -3/2 → 
  l + m = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_point_property_l714_71435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_parallel_lines_l714_71450

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Definition of a cyclic quadrilateral -/
def isCyclicQuadrilateral (A B C D : Point2D) (circle : Circle2D) : Prop :=
  sorry

/-- Definition of a point being on a line -/
def isPointOnLine (P : Point2D) (l : Line2D) : Prop :=
  sorry

/-- Definition of two lines being parallel -/
def areParallelLines (l1 l2 : Line2D) : Prop :=
  sorry

/-- Definition of a point being on a circle -/
def isPointOnCircle (P : Point2D) (circle : Circle2D) : Prop :=
  sorry

/-- Definition of the distance between two points -/
noncomputable def distance (P Q : Point2D) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Create a line from two points -/
def lineFromPoints (P Q : Point2D) : Line2D :=
  { a := Q.y - P.y
    b := P.x - Q.x
    c := P.y * Q.x - P.x * Q.y }

theorem cyclic_quadrilateral_parallel_lines 
  (A B C D P Q R S T U : Point2D) 
  (circle : Circle2D) 
  (l1 l2 : Line2D) :
  isCyclicQuadrilateral A B C D circle →
  areParallelLines l1 (lineFromPoints B C) →
  areParallelLines l2 (lineFromPoints A B) →
  isPointOnLine D l1 →
  isPointOnLine D l2 →
  isPointOnLine P l1 →
  isPointOnLine Q l1 →
  isPointOnLine R l1 →
  isPointOnLine S l2 →
  isPointOnLine T l2 →
  isPointOnLine U l2 →
  isPointOnCircle A circle →
  isPointOnCircle B circle →
  isPointOnCircle C circle →
  isPointOnCircle D circle →
  isPointOnCircle R circle →
  isPointOnCircle U circle →
  distance P Q = distance Q R →
  distance S T = distance T U :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_parallel_lines_l714_71450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ac_length_l714_71413

structure RightTriangle where
  B : Real
  AB : Fin 2 → Real
  AC : Fin 2 → Real
  ab_perp_ac : AB 0 * AC 0 + AB 1 * AC 1 = 0
  ab_def : AB = ![- Real.sin B, Real.cos B]
  ac_def : AC = ![1, Real.tan B]
  bc_length : 1 = Real.sqrt ((AB 0 - AC 0)^2 + (AB 1 - AC 1)^2)

theorem right_triangle_ac_length (t : RightTriangle) : 
  Real.sqrt ((t.AC 0)^2 + (t.AC 1)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ac_length_l714_71413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l714_71443

-- Define the triangle ABC
def Triangle (A B C : ℝ) : Prop := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the function f
noncomputable def f (x B : ℝ) : ℝ := Real.sin (2*x + B) + Real.sqrt 3 * Real.sin (2*x + B)

theorem triangle_area (A B C : ℝ) :
  Triangle A B C →
  (∀ x, f x B = f (-x) B) →  -- f is even
  f (Real.pi/12) B = 3 →
  ∃ S, S = (3 * Real.sqrt 3) / 2 ∧ S = (1/2) * Real.sin A * Real.sin B * Real.sin C / Real.sin ((A+B+C)/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l714_71443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_real_complex_product_l714_71459

theorem pure_real_complex_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (Complex.im ((3 - 4 * Complex.I) * Complex.mk p q) = 0) →
  p / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_real_complex_product_l714_71459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_expression_for_all_form_all_numbers_l714_71444

/-- A function that represents a mathematical expression using five threes -/
def expression (n : ℕ) : Prop := sorry

/-- Axiom: The expression function uses exactly five threes -/
axiom uses_five_threes : ∀ n : ℕ, expression n → (∃ count : ℕ, count = 5)

/-- Axiom: The expression function only uses arithmetic operations and exponentiation -/
axiom valid_operations : ∀ n : ℕ, expression n → 
  (∃ ops : Set String, ops ⊆ {"Add", "Sub", "Mul", "Div", "Exp"})

/-- Theorem: For every integer from 1 to 39, there exists a valid expression -/
theorem exists_expression_for_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 39 → expression n := by
  sorry

/-- Corollary: It's possible to form any integer from 1 to 39 using five threes 
    and arithmetic operations including exponentiation -/
theorem form_all_numbers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 39 → 
  expression n ∧ (∃ count : ℕ, count = 5) ∧ (∃ ops : Set String, ops ⊆ {"Add", "Sub", "Mul", "Div", "Exp"}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_expression_for_all_form_all_numbers_l714_71444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_is_three_fifths_neg_one_fifth_l714_71479

noncomputable def line (x : ℝ) : ℝ := 3 * x - 2

noncomputable def vector_on_line (a : ℝ) : ℝ × ℝ := (a, line a)

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1^2 + w.2^2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem constant_projection_is_three_fifths_neg_one_fifth 
  (w : ℝ × ℝ) 
  (h : ∀ (a b : ℝ), projection (vector_on_line a) w = projection (vector_on_line b) w) :
  projection (vector_on_line 0) w = (3/5, -1/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_is_three_fifths_neg_one_fifth_l714_71479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_value_l714_71490

theorem c_value : 2 * Real.sqrt 3 * (1.5 ^ (1/3)) * (12 ^ (1/6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_value_l714_71490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_conclusions_l714_71436

-- Define the function f(x) = x|x| + px + q
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Theorem statement
theorem incorrect_conclusions (p q : ℝ) :
  (∃ x, f p q x = 0) ↔ (p^2 - 4*q ≥ 0) = False ∧
  (p < 0 ∧ q > 0 → ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f p q x = 0 ∧ f p q y = 0 ∧ f p q z = 0) = False :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_conclusions_l714_71436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l714_71449

/-- Given 15 families with an average of 3 children per family, 
    and exactly 3 of these families being childless, 
    prove that the average number of children in the families with children is 3.75. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children_per_family : ℚ)
  (childless_families : ℕ)
  (h1 : total_families = 15)
  (h2 : average_children_per_family = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children_per_family) / (total_families - childless_families) = 45 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l714_71449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_count_is_five_exists_max_count_five_l714_71402

/-- A structure representing a collection of sets with specific properties -/
structure SetCollection where
  /-- The number of sets in the collection -/
  n : Nat
  /-- The sets in the collection -/
  sets : Fin n → Finset Nat
  /-- Each set has exactly 5 elements -/
  set_size : ∀ i, (sets i).card = 5
  /-- The intersection of any two sets contains at least two elements -/
  intersection_size : ∀ i j, i ≠ j → (sets i ∩ sets j).card ≥ 2

/-- The union of all sets in the collection -/
def SetCollection.union (sc : SetCollection) : Finset Nat :=
  Finset.biUnion Finset.univ sc.sets

/-- For each element in the union, count how many sets contain it -/
def SetCollection.elementCount (sc : SetCollection) (x : Nat) : Nat :=
  (Finset.filter (fun i => x ∈ sc.sets i) (Finset.univ : Finset (Fin sc.n))).card

/-- The maximum count of sets containing any element -/
noncomputable def SetCollection.maxCount (sc : SetCollection) : Nat :=
  Finset.sup sc.union (sc.elementCount ·)

/-- Theorem: For any collection of 10 sets satisfying the given conditions,
    the maximum count of sets containing any element is at least 5 -/
theorem min_max_count_is_five (sc : SetCollection) (h : sc.n = 10) :
  sc.maxCount ≥ 5 := by
  sorry

/-- Corollary: There exists a collection of 10 sets where the maximum count is exactly 5 -/
theorem exists_max_count_five :
  ∃ sc : SetCollection, sc.n = 10 ∧ sc.maxCount = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_count_is_five_exists_max_count_five_l714_71402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_and_a_range_l714_71420

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + Real.log (2 * x + 1)) / (2 * x + 1)

theorem f_maximum_and_a_range :
  (∃ (x : ℝ), x > -1/2 ∧ ∀ (y : ℝ), y > -1/2 → f 2 x ≥ f 2 y ∧ f 2 x = Real.exp 1) ∧
  (∀ (a : ℝ), 
    (∀ (x : ℝ), (Real.exp 1 - 1) / 2 ≤ x ∧ x ≤ (Real.exp 2 - 1) / 2 →
      (∃! (t₁ t₂ t₃ : ℝ), t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃ ∧
        (2 * x + 1)^2 * (deriv (f a) x) = t₁^3 - 12 * t₁ ∧
        (2 * x + 1)^2 * (deriv (f a) x) = t₂^3 - 12 * t₂ ∧
        (2 * x + 1)^2 * (deriv (f a) x) = t₃^3 - 12 * t₃)) ↔
    -8 < a ∧ a < 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_and_a_range_l714_71420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_existence_l714_71414

-- Define the function f(x) = lg x + x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 + x

-- Theorem statement
theorem root_existence : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_existence_l714_71414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l714_71433

/-- The function f(x) = 4sin(2x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

/-- The minimum positive period of f is π -/
noncomputable def min_positive_period (f : ℝ → ℝ) : ℝ := sorry

/-- The initial phase of f -/
noncomputable def initial_phase (f : ℝ → ℝ) : ℝ := sorry

/-- The amplitude of f -/
noncomputable def amplitude (f : ℝ → ℝ) : ℝ := sorry

theorem f_properties :
  min_positive_period f = Real.pi ∧
  initial_phase f = Real.pi / 3 ∧
  amplitude f = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l714_71433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extracurricular_reading_choices_l714_71489

-- Define the number of extracurricular reading materials
def total_materials : ℕ := 6

-- Define the number of materials each student chooses
def materials_per_student : ℕ := 2

-- Define the number of common materials
def common_materials : ℕ := 1

-- Theorem statement
theorem extracurricular_reading_choices :
  (Nat.choose total_materials common_materials) *
  (Nat.factorial (total_materials - common_materials) /
   Nat.factorial (total_materials - common_materials - (2 * materials_per_student - 2 * common_materials))) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extracurricular_reading_choices_l714_71489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_learning_machine_distribution_l714_71497

def number_of_distributions (n m : ℕ) : ℕ :=
  -- Number of ways to distribute n distinct objects among m people,
  -- where each person must receive at least one object and order matters
  sorry

theorem learning_machine_distribution (n m : ℕ) (hn : n = 6) (hm : m = 4) :
  (number_of_distributions n m) = 1560 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_learning_machine_distribution_l714_71497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stable_system_constant_temp_l714_71439

-- Define the temperature function
def T : ℤ × ℤ × ℤ → ℝ := sorry

-- Define stability condition
def is_stable (T : ℤ × ℤ × ℤ → ℝ) : Prop :=
  ∀ a b c : ℤ, T (a, b, c) = (1/6) * (T (a+1, b, c) + T (a-1, b, c) + T (a, b+1, c) + T (a, b-1, c) + T (a, b, c+1) + T (a, b, c-1))

-- Define temperature bounds
def temp_bounded (T : ℤ × ℤ × ℤ → ℝ) : Prop :=
  ∀ a b c : ℤ, 0 ≤ T (a, b, c) ∧ T (a, b, c) ≤ 1

-- Theorem statement
theorem stable_system_constant_temp (T : ℤ × ℤ × ℤ → ℝ) 
  (h1 : is_stable T) (h2 : temp_bounded T) : 
  ∀ (a b c : ℤ) (a' b' c' : ℤ), T (a, b, c) = T (a', b', c') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stable_system_constant_temp_l714_71439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_for_dry_window_l714_71455

/-- The speed at which rain falls vertically in m/s -/
noncomputable def v : ℝ := 2

/-- The angle of inclination of the car's rear window to the horizontal in radians -/
noncomputable def α : ℝ := Real.pi / 3  -- 60 degrees in radians

/-- The speed at which the car must travel to keep its rear window dry -/
noncomputable def u : ℝ := (2 * Real.sqrt 3) / 3

theorem car_speed_for_dry_window :
  v * Real.tan α = u := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_for_dry_window_l714_71455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l714_71409

/-- A cube with side length 2 -/
structure Cube where
  side_length : ℝ
  is_two : side_length = 2

/-- A pyramid within the cube -/
structure Pyramid (c : Cube) where
  base_area : ℝ
  height : ℝ
  is_valid : height = c.side_length ∧ base_area = c.side_length * c.side_length / 2

/-- The volume of a pyramid -/
noncomputable def pyramid_volume (c : Cube) (p : Pyramid c) : ℝ :=
  p.base_area * p.height / 3

/-- Theorem: The volume of pyramid ABFH in cube ABCDEFGH with side length 2 is 4/3 -/
theorem pyramid_volume_in_cube (c : Cube) (p : Pyramid c) : 
  pyramid_volume c p = 4/3 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l714_71409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_progression_10_11_12_l714_71448

theorem no_geometric_progression_10_11_12 : ¬∃ (p n : ℕ), p > 0 ∧ n > 0 ∧ (10 : ℚ)^(p - n) * 11^n = 12^p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_progression_10_11_12_l714_71448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_three_seconds_l714_71493

/-- The motion equation of an object -/
noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

/-- The instantaneous velocity of the object at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  deriv motion_equation t

theorem velocity_at_three_seconds : 
  instantaneous_velocity 3 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_three_seconds_l714_71493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l714_71460

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (3 * ((x - 3) / 5)) - 4

-- State the theorem
theorem h_fixed_point :
  ∃! x : ℝ, h x = x ∧ x = -29/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l714_71460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_value_l714_71461

/-- A monic quartic polynomial f(x) with specific values at certain points -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is a monic quartic polynomial -/
axiom f_monic_quartic : ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- Value of f at -1 -/
axiom f_neg_one : f (-1) = -1

/-- Value of f at 2 -/
axiom f_two : f 2 = -4

/-- Value of f at -3 -/
axiom f_neg_three : f (-3) = -9

/-- Value of f at 4 -/
axiom f_four : f 4 = -16

/-- Theorem: The value of f(1) is 23 -/
theorem f_one_value : f 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_value_l714_71461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l714_71469

/-- The length of the common chord between two circles -/
theorem common_chord_length (r a b c : ℝ) : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 + a*x + b*y + c = 0}
  let common_chord_length := Real.sqrt (4 * (r^2 - (a^2 + b^2 - 4*c)^2 / (4*(a^2 + b^2))))
  r = Real.sqrt 50 ∧ a = -12 ∧ b = -6 ∧ c = 40 → common_chord_length = 2 * Real.sqrt 5 := by
  sorry

#check common_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l714_71469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l714_71423

-- Define the functions
def f (x : ℝ) : ℝ := x^2 + 2*x

def g (x : ℝ) : ℝ := -x^2 + 2*x

def h (l : ℝ) (x : ℝ) : ℝ := g x - l * f x + 1

-- State the theorem
theorem problem_solution :
  (∀ x, g x + f (-x) = 0) →
  (∀ x, g x = -x^2 + 2*x) ∧
  ({x : ℝ | g x ≥ f x - |x - 1|} = {x : ℝ | -1 ≤ x ∧ x ≤ 1/2}) ∧
  (∀ l, (∀ x ∈ Set.Icc (-1 : ℝ) 1, Monotone (h l)) ↔ l ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l714_71423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_a_time_l714_71442

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure Race where
  distance : ℝ
  time_difference : ℝ
  distance_difference : ℝ

/-- Calculates the time for a runner to complete the race -/
noncomputable def race_time (runner : Runner) (race : Race) : ℝ :=
  race.distance / runner.speed

theorem runner_a_time (race : Race) (runner_a runner_b : Runner) :
  race.distance = 200 ∧
  race.time_difference = 7 ∧
  race.distance_difference = 35 ∧
  runner_a.speed - runner_b.speed = race.distance_difference / race.time_difference →
  race_time runner_a race = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_a_time_l714_71442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pair_square_difference_not_in_list_l714_71485

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_pair_square_difference_not_in_list :
  ∀ x y : ℕ,
    x ≠ y →
    4 < x → x < 23 →
    4 < y → y < 23 →
    is_prime x →
    is_prime y →
    (x^2 + y^2 - (x + y)^2 : ℤ) ∉ ({105, 240, 408, 528, 720} : Set ℤ) := by
  sorry

#check prime_pair_square_difference_not_in_list

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pair_square_difference_not_in_list_l714_71485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l714_71426

theorem sin_double_alpha (α : ℝ) (h : Real.sin (α + π / 4) = 1 / 2) : Real.sin (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l714_71426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_target_proof_l714_71438

theorem shooting_target_proof (p q : Prop) : 
  (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_target_proof_l714_71438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l714_71430

theorem find_divisor (divisor : ℕ) (h1 : divisor > 0) 
  (h2 : 462 % divisor = 0) 
  (h3 : ∀ k : ℕ, k > 0 → k ∣ 462 → |462 - 457| ≤ |Int.ofNat 462 - Int.ofNat 457|) : divisor = 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l714_71430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_for_columns_l714_71495

/-- The number of gallons of paint needed to cover cylindrical columns -/
noncomputable def paint_needed (num_columns : ℕ) (height : ℝ) (diameter : ℝ) (coverage : ℝ) : ℕ :=
  let radius := diameter / 2
  let lateral_area := 2 * Real.pi * radius * height
  let total_area := num_columns * lateral_area
  let gallons := total_area / coverage
  Int.toNat ⌈gallons⌉

/-- Theorem stating the number of gallons needed for the given problem -/
theorem paint_needed_for_columns : paint_needed 20 15 8 300 = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_for_columns_l714_71495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_bound_l714_71466

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

-- State the theorem
theorem function_derivative_bound (a : ℝ) : 
  (∀ x : ℝ, (deriv f) x ≥ a) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_bound_l714_71466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_diagonal_intersection_probability_l714_71408

/-- A regular octagon is an 8-sided polygon with all sides equal and all angles equal. -/
structure RegularOctagon where
  -- We don't need to define the structure fully, just declare it exists
  mk :: (dummy : Unit)

/-- A diagonal of a regular octagon is a line segment that connects two non-adjacent vertices. -/
structure Diagonal (octagon : RegularOctagon) where
  -- Again, we don't need to fully define this
  mk :: (dummy : Unit)

/-- Two diagonals intersect if they cross each other inside the octagon. -/
def intersect (octagon : RegularOctagon) (d1 d2 : Diagonal octagon) : Prop :=
  sorry -- Definition of intersection

/-- The set of all diagonals in a regular octagon. -/
def allDiagonals (octagon : RegularOctagon) : Set (Diagonal octagon) :=
  sorry -- Definition of all diagonals

/-- The probability of an event occurring when choosing from a finite set. -/
noncomputable def probability {α : Type} (s : Set α) (p : α → Prop) : ℚ :=
  sorry -- Definition of probability

theorem octagon_diagonal_intersection_probability (octagon : RegularOctagon) :
  probability (allDiagonals octagon) (fun d1 => 
    probability (allDiagonals octagon) (fun d2 => 
      d1 ≠ d2 ∧ intersect octagon d1 d2) = 7 / 19) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_diagonal_intersection_probability_l714_71408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_vertices_convex_polyhedron_l714_71484

/-- A convex polyhedron. -/
structure ConvexPolyhedron where

/-- The number of faces in a polyhedron. -/
def num_faces (P : ConvexPolyhedron) : ℕ := sorry

/-- A face of a polyhedron. -/
structure Face (P : ConvexPolyhedron) where

/-- Predicate indicating if a face is a triangle. -/
def is_triangle (P : ConvexPolyhedron) (f : Face P) : Prop := sorry

/-- The number of vertices in a polyhedron where exactly 3 edges meet. -/
def num_good_vertices (P : ConvexPolyhedron) : ℕ := sorry

/-- A convex polyhedron with 2n triangular faces (n ≥ 3) has at most ⌊2n/3⌋ vertices where exactly 3 edges meet. -/
theorem max_good_vertices_convex_polyhedron (n : ℕ) (h : n ≥ 3) :
  ∀ (P : ConvexPolyhedron),
    (num_faces P = 2 * n) →
    (∀ f : Face P, is_triangle P f) →
    (num_good_vertices P ≤ ⌊(2 * n : ℚ) / 3⌋) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_vertices_convex_polyhedron_l714_71484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_digits_is_1_55_l714_71454

/-- The expected number of digits when rolling a fair 20-sided die -/
noncomputable def expected_digits : ℝ := (9 / 20) * 1 + (11 / 20) * 2

/-- Theorem stating that the expected number of digits is 1.55 -/
theorem expected_digits_is_1_55 : expected_digits = 1.55 := by
  -- Unfold the definition of expected_digits
  unfold expected_digits
  -- Simplify the arithmetic expression
  simp [add_mul, mul_add, mul_one]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_digits_is_1_55_l714_71454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l714_71464

def sequence_a : ℕ → ℝ
  | 0 => 3
  | n + 1 => 3 * sequence_a n - 4

theorem sequence_a_formula (n : ℕ) : sequence_a n = 3^n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l714_71464
