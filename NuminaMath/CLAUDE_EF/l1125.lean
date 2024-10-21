import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l1125_112509

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define line l_2
def l2 (x y : ℝ) : Prop := x + 2*y + 2 = 0

-- Define a line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the theorem
theorem constant_product (k : ℝ) : 
  ∃ (P Q M N : ℝ × ℝ),
    -- P and Q are on the circle and the line through A
    circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
    line_through_A k P.1 P.2 ∧ line_through_A k Q.1 Q.2 ∧
    -- M is the midpoint of PQ
    M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2 ∧
    -- N is on l2 and the line through A
    l2 N.1 N.2 ∧ line_through_A k N.1 N.2 →
    -- The product AM * AN is constant
    Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) * 
    Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l1125_112509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_5001_l1125_112534

/-- The sum of the alternating series from -1 to 10002 -/
def alternatingSum : ℕ → ℤ
  | 0 => 0
  | n + 1 => alternatingSum n + (if n % 2 = 0 then -(n + 1 : ℤ) else (n + 1 : ℤ))

/-- The number of terms in the series -/
def numTerms : ℕ := 10002

theorem alternating_sum_equals_5001 : alternatingSum numTerms = 5001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_5001_l1125_112534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_approx_l1125_112523

/-- The length of ribbon needed for a circular gate -/
noncomputable def ribbon_length (area : ℝ) (extra : ℝ) (π_approx : ℝ) : ℝ :=
  let radius := Real.sqrt (area / π_approx)
  let circumference := 2 * π_approx * radius
  circumference + extra

/-- Theorem stating the approximate length of ribbon needed -/
theorem ribbon_length_approx :
  let area := 246
  let extra := 5
  let π_approx := 22 / 7
  abs (ribbon_length area extra π_approx - 60.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_approx_l1125_112523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_dog_food_calculation_l1125_112598

noncomputable def dog_food_bags_needed (cup_weight : ℚ) (num_dogs : ℕ) (cups_per_meal : ℕ) 
  (meals_per_day : ℕ) (days_in_month : ℕ) (bag_weight : ℕ) : ℕ :=
  let daily_cups := cups_per_meal * meals_per_day * num_dogs
  let daily_weight := (daily_cups : ℚ) * cup_weight
  let monthly_weight := daily_weight * days_in_month
  let bags_needed := (monthly_weight / bag_weight).ceil.toNat
  bags_needed

theorem mike_dog_food_calculation : 
  dog_food_bags_needed (1/4) 2 6 2 30 20 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_dog_food_calculation_l1125_112598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_removed_is_correct_l1125_112569

/-- Represents the weight distribution in a suitcase -/
structure Suitcase where
  books : ℚ
  clothes : ℚ
  electronics : ℚ

/-- The initial ratio of books to clothes to electronics -/
def initial_ratio : Fin 3 → ℚ
  | 0 => 7
  | 1 => 4
  | 2 => 3

/-- The weight of electronics in pounds -/
def electronics_weight : ℚ := 9

/-- Calculates the amount of clothing removed to double the ratio of books to clothes -/
def clothing_removed (s : Suitcase) : ℚ :=
  s.clothes - s.books / 2

theorem clothing_removed_is_correct (s : Suitcase) :
  s.books / initial_ratio 0 = s.clothes / initial_ratio 1 ∧
  s.books / initial_ratio 0 = s.electronics / initial_ratio 2 ∧
  s.electronics = electronics_weight →
  clothing_removed s = 3/2 := by
  sorry

#eval clothing_removed { books := 21, clothes := 12, electronics := 9 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_removed_is_correct_l1125_112569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_parallel_l1125_112536

noncomputable def vector_pair1 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) := ((1, -2, 1), (-1, 2, -1))
noncomputable def vector_pair2 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) := ((8, 4, 0), (2, 1, 0))
noncomputable def vector_pair3 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) := ((1, 0, -1), (-3, 0, 3))
noncomputable def vector_pair4 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) := ((-4/3, 1, -1), (4, -3, 3))

def are_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.fst, k * v2.snd.fst, k * v2.snd.snd) ∨ 
            v2 = (k * v1.fst, k * v1.snd.fst, k * v1.snd.snd)

theorem all_pairs_parallel :
  are_parallel vector_pair1.fst vector_pair1.snd ∧
  are_parallel vector_pair2.fst vector_pair2.snd ∧
  are_parallel vector_pair3.fst vector_pair3.snd ∧
  are_parallel vector_pair4.fst vector_pair4.snd :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_parallel_l1125_112536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l1125_112571

/-- The height of a right pyramid with a square base --/
noncomputable def pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) : ℝ :=
  let side_length := base_perimeter / 4
  let half_diagonal := side_length * Real.sqrt 2 / 2
  Real.sqrt (apex_to_vertex^2 - half_diagonal^2)

/-- Theorem stating the height of the specific pyramid --/
theorem specific_pyramid_height :
  pyramid_height 32 10 = 2 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l1125_112571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_orthogonality_sum_of_squares_l1125_112507

/-- Given a 3x3 matrix N, if N^T * N = I, then the sum of squares of its elements is 127/210 -/
theorem matrix_orthogonality_sum_of_squares (x y z : ℝ) : 
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![0, 3*y, z; x, 2*y, -z; 2*x, -y, z]
  N.transpose * N = (1 : Matrix (Fin 3) (Fin 3) ℝ) →
  x^2 + y^2 + z^2 = 127/210 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_orthogonality_sum_of_squares_l1125_112507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequality_l1125_112538

open BigOperators

def a (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / (k + 1 : ℚ)

theorem harmonic_sum_inequality (n : ℕ) (h : n ≥ 2) :
  (a n)^2 > 2 * ∑ k in Finset.range (n - 1), (a (k + 2)) / (k + 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequality_l1125_112538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1125_112539

-- Define the complex logarithm function
noncomputable def complex_log2 (z : ℂ) : ℂ := Complex.log z / Complex.log 2

-- Define the condition function
def condition (x : ℝ) : Prop :=
  (complex_log2 (x^2 - 3*x - 2 : ℂ)).re > 1 ∧ (x^2 + 2*x + 1 : ℝ) = 1

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, condition x ∧ x = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1125_112539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_l1125_112586

/-- The function f(x) = x + 2^x + log_2(x) -/
noncomputable def f (x : ℝ) : ℝ := x + 2^x + Real.log x / Real.log 2

/-- Theorem stating that f(x) has a unique solution for x > 0 -/
theorem unique_solution_f :
  ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_l1125_112586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l1125_112563

/-- Given a circular sector with arc length π and radius 3,
    prove that the central angle is π/3 and the area is 3π/2 -/
theorem sector_properties (l : ℝ) (r : ℝ) 
    (h1 : l = π) 
    (h2 : r = 3) : 
    (l / r = π / 3) ∧ ((1 / 2) * (l / r) * r^2 = 3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l1125_112563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_is_x_equals_zero_l1125_112508

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line defined by its equation ax + by + c = 0 -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The reflection of a point about a vertical line -/
def reflect (p : Point) (l : Line) : Point :=
  { x := 2 * l.c / l.a - p.x, y := p.y }

/-- Theorem: The line of reflection for the given triangle is x = 0 -/
theorem reflection_line_is_x_equals_zero 
  (X Y Z X' Y' Z' : Point)
  (hX : X = ⟨1, 4⟩)
  (hY : Y = ⟨6, 5⟩)
  (hZ : Z = ⟨-3, 2⟩)
  (hX' : X' = ⟨-1, 4⟩)
  (hY' : Y' = ⟨-6, 5⟩)
  (hZ' : Z' = ⟨3, 2⟩)
  (hreflect : ∃ (l : Line), 
    l.b = 0 ∧ 
    X' = reflect X l ∧ 
    Y' = reflect Y l ∧ 
    Z' = reflect Z l) :
  ∃ (l : Line), l.a = 1 ∧ l.b = 0 ∧ l.c = 0 := by
  sorry

#eval toString "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_is_x_equals_zero_l1125_112508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_3pi_implies_n_in_set_l1125_112527

/-- A function with the given properties -/
noncomputable def f (n : ℤ) (x : ℝ) : ℝ := Real.cos (n * x) * Real.sin ((4 / n) * x)

/-- The period of the function is 3π -/
def has_period_3pi (n : ℤ) : Prop := ∀ x, f n (x + 3 * Real.pi) = f n x

/-- The set of possible values for n -/
def possible_n : Set ℤ := {2, -2, 6, -6}

/-- The main theorem -/
theorem period_3pi_implies_n_in_set :
  ∀ n : ℤ, has_period_3pi n → n ∈ possible_n := by
  sorry

#check period_3pi_implies_n_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_3pi_implies_n_in_set_l1125_112527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rattle_belongs_to_first_brother_l1125_112554

-- Define the days of the week
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving Repr, DecidableEq

-- Define the brothers
inductive Brother
| Tweedledee | Tweedledum
deriving Repr, DecidableEq

-- Define a function to determine if a brother tells the truth on a given day
def tellsTruth (b : Brother) (d : Day) : Bool :=
  match b, d with
  | Brother.Tweedledee, Day.Sunday => true
  | Brother.Tweedledee, _ => true
  | Brother.Tweedledum, Day.Sunday => true
  | Brother.Tweedledum, _ => false

-- Define the statements made by the brothers
def firstBrotherStatement (owner : Brother) : Prop :=
  owner = Brother.Tweedledee

def secondBrotherStatement (identity : Brother) : Prop :=
  identity = Brother.Tweedledum

-- Theorem to prove
theorem rattle_belongs_to_first_brother
  (day : Day)
  (h_not_sunday : day ≠ Day.Sunday)
  (first_brother second_brother : Brother)
  (h_different : first_brother ≠ second_brother)
  (h_first_statement : firstBrotherStatement (if tellsTruth first_brother day then Brother.Tweedledee else Brother.Tweedledum))
  (h_second_statement : secondBrotherStatement (if tellsTruth second_brother day then Brother.Tweedledum else Brother.Tweedledee)) :
  first_brother = Brother.Tweedledee := by
  sorry

#check rattle_belongs_to_first_brother

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rattle_belongs_to_first_brother_l1125_112554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_8_neg15_l1125_112522

/-- The distance from the origin to a point in a rectangular coordinate system. -/
noncomputable def distanceFromOrigin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- Theorem: The distance from the origin to the point (8, -15) is 17 units. -/
theorem distance_to_8_neg15 : distanceFromOrigin 8 (-15) = 17 := by
  -- Unfold the definition of distanceFromOrigin
  unfold distanceFromOrigin
  -- Simplify the expression under the square root
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_8_neg15_l1125_112522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_angle_point_coordinates_l1125_112568

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem statement -/
theorem ellipse_max_angle_point_coordinates (e : Ellipse) (f1 f2 : Point)
  (h_foci : distance f1 f2 = 2)
  (h_max_angle : ∀ p : Point, p.x = e.a → angle f1 p f2 ≤ π/4)
  (h_exists_max : ∃ p : Point, p.x = e.a ∧ angle f1 p f2 = π/4) :
  ∃ p : Point, (p.x = Real.sqrt 2 ∧ (p.y = 1 ∨ p.y = -1)) ∧
    p.x = e.a ∧ angle f1 p f2 = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_angle_point_coordinates_l1125_112568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_beta_value_l1125_112514

-- Part 1
theorem cos_alpha_plus_beta (α β : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : Real.cos β = 4/5) 
  (h3 : π/2 < α ∧ α < π) 
  (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -1 := by sorry

-- Part 2
theorem beta_value (α β : ℝ) 
  (h1 : Real.cos α = 1/7) 
  (h2 : Real.cos (α - β) = 13/14) 
  (h3 : 0 < β ∧ β < α ∧ α < π/2) : 
  β = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_beta_value_l1125_112514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_solutions_l1125_112542

theorem parabola_solutions (a b c : ℝ) :
  (∃ (y : ℝ → ℝ), y = λ x => a*x^2 + b*x + c) →
  (λ x => a*x^2 + b*x + c) (-1) = 3 →
  (λ x => a*x^2 + b*x + c) 2 = 3 →
  (∀ x, a*(x-2)^2 - 3 = 2*b - b*x - c ↔ x = 1 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_solutions_l1125_112542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_cyclist_speed_l1125_112549

/-- The problem setup and proof of maximum pedestrian speed and corresponding cyclist speed -/
theorem pedestrian_cyclist_speed :
  ∃ (max_pedestrian_speed : ℕ) (cyclist_speed : ℝ),
  let distance := 24 -- km
  let cyclist_min_time := 2 -- hours
  let return_speed_factor := 2
  let meeting_time := 0.4 -- hours (24 minutes)
  max_pedestrian_speed = 6 ∧ 
  cyclist_speed = 12 ∧
  (∀ pedestrian_speed : ℝ,
    pedestrian_speed > 0 →
    cyclist_speed > 0 →
    distance / cyclist_speed ≥ cyclist_min_time →
    (∃ t : ℝ, t > 0 ∧ pedestrian_speed * t = distance) →
    (∃ t : ℝ, t > 0 ∧ cyclist_speed * t + return_speed_factor * cyclist_speed * meeting_time + pedestrian_speed * (t + meeting_time) = distance) →
    pedestrian_speed ≤ max_pedestrian_speed) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_cyclist_speed_l1125_112549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_l1125_112599

/-- Represents a choir with women and men -/
structure Choir where
  women : ℕ
  women_avg_age : ℚ
  men : ℕ
  men_avg_age : ℚ

/-- Calculates the average age of all people in the choir -/
def average_age (c : Choir) : ℚ :=
  ((c.women : ℚ) * c.women_avg_age + (c.men : ℚ) * c.men_avg_age) / ((c.women + c.men) : ℚ)

/-- Theorem stating the average age of the specific choir -/
theorem choir_average_age :
  let c : Choir := { women := 12, women_avg_age := 25, men := 10, men_avg_age := 40 }
  average_age c = 31.82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_l1125_112599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_in_interval_l1125_112573

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos (x + π/3) + sqrt 3 / 2

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = π :=
sorry

-- Theorem for the maximum value in the given interval
theorem max_value_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-π/6) (π/3) ∧
  f x = 1 - sqrt 3 / 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-π/6) (π/3) → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_in_interval_l1125_112573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_condition_l1125_112550

open Real

/-- The function f(x) = (1/2)x^2 - a*ln(x) + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * log x + 1

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - a / x

theorem minimum_condition (a : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (f a) x) ↔ 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_condition_l1125_112550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_l1125_112510

theorem max_value_of_sum (p q r s : ℕ) : 
  p ∈ ({2, 3, 5, 6} : Set ℕ) ∧ q ∈ ({2, 3, 5, 6} : Set ℕ) ∧ 
  r ∈ ({2, 3, 5, 6} : Set ℕ) ∧ s ∈ ({2, 3, 5, 6} : Set ℕ) ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  p * q + q * r + r * s + s * p ≤ 64 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_l1125_112510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_no_perpendicular_bisector_l1125_112531

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y + 2)^2 = 9

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the distance function from a point to a line
noncomputable def dist_point_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

-- Statement for part I
theorem line_equations :
  (∀ x y, 3 * x + 4 * y - 6 = 0 → dist_point_line x y 3 4 (-6) = 1 ∧ dist_point_line 2 0 3 4 (-6) = 0) ∧
  (∀ y, dist_point_line 2 y 1 0 (-2) = 1 ∧ dist_point_line 2 0 1 0 (-2) = 0) := by
  sorry

-- Statement for part II
theorem no_perpendicular_bisector :
  ¬ ∃ a : ℝ, 
    (∃ x₁ y₁ x₂ y₂, 
      x₁ ≠ x₂ ∧
      circle_C x₁ y₁ ∧ 
      circle_C x₂ y₂ ∧ 
      a * x₁ - y₁ + 1 = 0 ∧ 
      a * x₂ - y₂ + 1 = 0 ∧
      ((y₂ - 0) / (x₂ - 2) = -1 / a) ∧
      ((x₁ + x₂) / 2 - 2) * (-1 / a) = ((y₁ + y₂) / 2 - 0)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_no_perpendicular_bisector_l1125_112531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_12_to_2500_l1125_112530

theorem closest_multiple_of_12_to_2500 : 
  ∀ n : Int, n ≠ 2496 → n % 12 = 0 → |n - 2500| ≥ |2496 - 2500| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_12_to_2500_l1125_112530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extrema_on_interval_l1125_112548

open Set
open Function
open Real

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the open interval (-1, 1)
def I : Set ℝ := Ioo (-1) 1

-- Statement: f has no maximum or minimum on the interval (-1, 1)
theorem no_extrema_on_interval :
  ¬ (∃ (m : ℝ), IsMaxOn f I m) ∧ ¬ (∃ (m : ℝ), IsMinOn f I m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extrema_on_interval_l1125_112548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_half_l1125_112572

theorem reciprocal_of_negative_half :
  (1 : ℚ) / (-1/2 : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_half_l1125_112572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_653_equals_quaternary_12223_l1125_112518

def octal_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_quaternary (n : ℕ) : ℕ := sorry

theorem octal_653_equals_quaternary_12223 :
  octal_to_decimal 653 = 12223 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_653_equals_quaternary_12223_l1125_112518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_small_spheres_l1125_112582

noncomputable def cone_height : ℝ := 12
noncomputable def cone_base_radius : ℝ := 5
noncomputable def slant_height : ℝ := 13
noncomputable def inscribed_sphere_radius : ℝ := cone_base_radius

noncomputable def circumscribed_sphere_radius : ℝ := inscribed_sphere_radius + slant_height
noncomputable def small_sphere_radius : ℝ := slant_height / 3

noncomputable def volume_ratio : ℝ := (64 * slant_height^3) / (81 * (inscribed_sphere_radius + slant_height)^3)

theorem probability_in_small_spheres :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |volume_ratio - (64 * slant_height^3) / (81 * circumscribed_sphere_radius^3)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_small_spheres_l1125_112582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cone_height_for_volume_l1125_112575

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a cone given its volume and radius -/
noncomputable def cone_height (v : ℝ) (r : ℝ) : ℝ := v / ((1/3) * Real.pi * r^2)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ := ⌊x + 0.5⌋

theorem min_cone_height_for_volume :
  let r : ℝ := 3
  let v : ℝ := 93
  let h : ℝ := cone_height v r
  round_to_nearest h = 10 ∧ cone_volume r (↑(round_to_nearest h)) ≥ v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cone_height_for_volume_l1125_112575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1125_112589

/-- Calculates the final value of an investment after two weeks of growth -/
theorem investment_growth (initial_investment : ℝ) (first_week_rate : ℝ) (second_week_rate : ℝ) :
  initial_investment = 400 →
  first_week_rate = 0.25 →
  second_week_rate = 0.50 →
  initial_investment * (1 + first_week_rate) * (1 + second_week_rate) = 750 := by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num

#check investment_growth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1125_112589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_correct_l1125_112557

/-- The number of integral values of a in [1, 300] such that d₁ · d₂ is divisible by 7 -/
def count_divisible_by_seven : ℕ := 257

/-- Definition of d₁ -/
def d₁ (a : ℕ) : ℕ := a^3 + 3^a + a * 3^((a+1)/3)

/-- Definition of d₂ -/
def d₂ (a : ℕ) : ℕ := a^3 + 3^a - a * 3^((a+1)/3)

/-- The main theorem -/
theorem count_divisible_by_seven_correct :
  (Finset.filter (fun a => 1 ≤ a ∧ a ≤ 300 ∧ (d₁ a * d₂ a) % 7 = 0) (Finset.range 301)).card = count_divisible_by_seven :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_correct_l1125_112557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_tax_collection_l1125_112512

/-- The total tax collected from a village given a farmer's tax payment and land proportion -/
theorem village_tax_collection 
  (farmer_tax : ℝ) 
  (farmer_land_proportion : ℝ) 
  (h1 : farmer_tax = 500)
  (h2 : farmer_land_proportion = 0.20833333333333332) : 
  farmer_tax / farmer_land_proportion = 2400 := by
  -- Define the total tax collected from the village
  let total_tax : ℝ := farmer_tax / farmer_land_proportion

  -- Assert that the total tax is equal to $2400
  have h : total_tax = 2400

  -- The proof goes here
  sorry

  -- Return the equality
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_tax_collection_l1125_112512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_range_initial_values_l1125_112526

/-- The function g(x) = 6x - x^2 --/
def g (x : ℝ) : ℝ := 6 * x - x^2

/-- The sequence defined by x_n = g(x_{n-1}) for n ≥ 1 --/
def sequenceG (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => g (sequenceG x₀ n)

/-- A predicate to check if a sequence takes on only finitely many values --/
def has_finite_range (s : ℕ → ℝ) : Prop :=
  ∃ (S : Set ℝ), (Set.Finite S) ∧ (∀ n, s n ∈ S)

/-- The main theorem --/
theorem infinitely_many_finite_range_initial_values :
  {x₀ : ℝ | x₀ ∈ Set.Icc 0 6 ∧ has_finite_range (sequenceG x₀)}.Infinite :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_range_initial_values_l1125_112526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_unit_cube_l1125_112520

/-- The maximum radius of a sphere inscribed in a unit cube and tangent to its main diagonal -/
noncomputable def max_sphere_radius : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 2

/-- Theorem: The maximum radius of a sphere inscribed in a unit cube and tangent to its main diagonal is (√6 - √2) / 2 -/
theorem max_sphere_radius_in_unit_cube :
  let cube_side_length : ℝ := 1
  let main_diagonal_length : ℝ := Real.sqrt 3
  ∃ (center : ℝ × ℝ × ℝ) (radius : ℝ),
    radius = max_sphere_radius ∧
    radius ≤ cube_side_length / 2 ∧
    radius ≤ main_diagonal_length / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_unit_cube_l1125_112520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1125_112543

theorem sin_half_angle (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (-1 + Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1125_112543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_location_l1125_112511

/-- A complex number in the first quadrant and outside the unit circle -/
noncomputable def F : ℂ :=
  sorry

/-- F is in the first quadrant -/
axiom F_in_first_quadrant : F.re > 0 ∧ F.im > 0

/-- F is outside the unit circle -/
axiom F_outside_unit_circle : Complex.abs F > 1

/-- The reciprocal of F -/
noncomputable def F_reciprocal : ℂ := 1 / F

/-- Theorem: The reciprocal of F is inside the unit circle in the fourth quadrant -/
theorem reciprocal_location :
  Complex.abs F_reciprocal < 1 ∧
  F_reciprocal.re > 0 ∧
  F_reciprocal.im < 0 := by
  sorry

#check reciprocal_location

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_location_l1125_112511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_exist_l1125_112544

/-- A line passing through (2,1) that intersects the coordinate axes -/
structure LineThroughPoint where
  slope : ℝ

/-- The x-intercept of the line -/
noncomputable def x_intercept (l : LineThroughPoint) : ℝ := 2 - 1 / l.slope

/-- The y-intercept of the line -/
noncomputable def y_intercept (l : LineThroughPoint) : ℝ := 2 * l.slope + 1

/-- The area of the triangle formed by the line and the coordinate axes -/
noncomputable def triangle_area (l : LineThroughPoint) : ℝ :=
  (1/2) * |x_intercept l| * |y_intercept l|

/-- The theorem stating that there are exactly 3 lines satisfying the conditions -/
theorem three_lines_exist :
  ∃! (s : Finset LineThroughPoint),
    s.card = 3 ∧
    (∀ l ∈ s, triangle_area l = 4) ∧
    (∀ l : LineThroughPoint, triangle_area l = 4 → l ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_exist_l1125_112544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_on_interval_l1125_112553

-- Define the functions
def f (x : ℝ) := x^2 - 4*x
def g (x : ℝ) := 3*x + 1
noncomputable def h (x : ℝ) := (3 : ℝ)^(-x)
noncomputable def t (x : ℝ) := Real.tan x

-- Define the interval (-∞, 0)
def interval := Set.Iio (0 : ℝ)

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem increasing_function_on_interval :
  IsIncreasing g interval ∧
  ¬IsIncreasing f interval ∧
  ¬IsIncreasing h interval ∧
  ¬IsIncreasing t interval :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_on_interval_l1125_112553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cans_collected_difference_l1125_112533

/-- Given that Mike collected cans on Monday and Tuesday, this theorem proves
    the absolute difference between the number of cans collected on these two days. -/
theorem cans_collected_difference (monday_cans tuesday_cans : ℤ) 
    (h1 : monday_cans = 71)
    (h2 : monday_cans + tuesday_cans = 98) :
    |tuesday_cans - monday_cans| = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cans_collected_difference_l1125_112533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_evaporation_rate_l1125_112567

/-- Given a glass of water with initial volume and evaporation rate over a period,
    calculate the daily evaporation rate. -/
theorem daily_evaporation_rate 
  (initial_volume : ℝ) 
  (evaporation_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_volume = 15) 
  (h2 : evaporation_percentage = 0.05) 
  (h3 : days = 15) :
  (initial_volume * evaporation_percentage) / (days : ℝ) = 0.05 := by
  rw [h1, h2, h3]
  norm_num

/-- Evaluate the daily evaporation rate -/
def eval_daily_evaporation : ℚ :=
  (15 * 5 / 100) / 15

#eval eval_daily_evaporation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_evaporation_rate_l1125_112567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_in_expansion_l1125_112515

theorem largest_coefficient_in_expansion :
  ∃ (r : ℕ), r < 7 ∧
    (∀ k : ℕ, k ≤ 7 → (Nat.choose 7 r) * 2^r ≥ (Nat.choose 7 k) * 2^k) ∧
    (∃ s : ℕ, s < 7 ∧
      (Nat.choose 7 s) * 2^s = 2 * (Nat.choose 7 (s-1)) * 2^(s-1) ∧
      (Nat.choose 7 s) * 2^s = (5/6) * (Nat.choose 7 (s+1)) * 2^(s+1)) →
  (Nat.choose 7 5) * 32 = 672 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_in_expansion_l1125_112515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_subtraction_l1125_112590

theorem three_digit_subtraction (a b c : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧  -- a, b, c are single digits
  c - a = 3 ∧  -- condition given
  100 * a + 10 * b + c > 100 * c + 10 * a + b ∧  -- abc > cab
  (100 * a + 10 * b + c) - (100 * c + 10 * a + b) < 100 ∧  -- result is less than 100
  ((100 * a + 10 * b + c) - (100 * c + 10 * a + b)) % 100 / 10 = 0 →  -- tens digit of result is 0
  a = 6 ∧ b = 1 ∧ c = 9 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_subtraction_l1125_112590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_inscribed_equilateral_triangles_l1125_112560

-- Define the necessary structures and functions
def is_equilateral_triangle (T : Set (ℝ × ℝ)) : Prop := sorry

def is_inscribed_in_circle (r : ℝ) (T : Set (ℝ × ℝ)) : Prop := sorry

noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Main theorem
theorem common_area_of_inscribed_equilateral_triangles (r : ℝ) (h : r > 0) :
  ∃ (A : Set (ℝ × ℝ)), 
    (∃ (T₁ T₂ : Set (ℝ × ℝ)), 
      is_equilateral_triangle T₁ ∧ 
      is_equilateral_triangle T₂ ∧ 
      is_inscribed_in_circle r T₁ ∧ 
      is_inscribed_in_circle r T₂ ∧ 
      A = T₁ ∩ T₂) →
    area A ≥ (r^2 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_inscribed_equilateral_triangles_l1125_112560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CP_l1125_112529

-- Define the circle in polar coordinates
noncomputable def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define point P in polar coordinates
noncomputable def point_P : ℝ × ℝ := (4, Real.pi / 3)

-- Define the center of the circle C
noncomputable def center_C : ℝ × ℝ := (2, 0)

-- State the theorem
theorem distance_CP :
  let (ρ, θ) := point_P
  circle_equation ρ θ →
  Real.sqrt ((center_C.1 - ρ * Real.cos θ)^2 + (center_C.2 - ρ * Real.sin θ)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CP_l1125_112529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1125_112528

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x else Real.log (x + 1)

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, f (2 - x^2) > f x ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1125_112528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1125_112579

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x^2 + 1/x)

-- State the theorem
theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1125_112579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_AFB_l1125_112524

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through (4,0)
def line (m : ℝ) (x y : ℝ) : Prop := x - 4 = m*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the area of triangle AFB
noncomputable def triangleArea (m : ℝ) : ℝ := 6 * Real.sqrt (m^2 + 4)

-- Theorem statement
theorem min_area_triangle_AFB :
  ∃ (min_area : ℝ), 
    (∀ m : ℝ, triangleArea m ≥ min_area) ∧
    (∃ m : ℝ, triangleArea m = min_area) ∧
    min_area = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_AFB_l1125_112524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l1125_112570

-- Define the ellipse
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the points
def F : ℝ × ℝ := (-2, 0)
def A : ℝ × ℝ := (3, 0)

-- Define the perpendicularity condition
def is_perpendicular (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (3 - x) * (x + 2) + y * (-y) = 0

-- Define the slope of PF
noncomputable def slope_PF (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  y / (x + 2)

-- Theorem statement
theorem ellipse_slope_theorem (P : ℝ × ℝ) :
  let (x, y) := P
  is_on_ellipse x y → is_perpendicular P →
  (slope_PF P = Real.sqrt 3 ∨ slope_PF P = -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l1125_112570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shoes_theorem_l1125_112581

/-- Represents the number of legs for each participant -/
def Legs := List Nat

/-- Predicate to check if all numbers in the list are positive and even -/
def all_positive_even (legs : Legs) : Prop :=
  legs.all (λ x => x > 0 ∧ x % 2 = 0)

/-- The minimum number of shoes required for all participants to climb -/
def min_shoes_required (legs : Legs) : Nat :=
  match legs with
  | [] => 0
  | [a] => a / 2
  | a :: b :: _ => max ((a + b) / 2) ((legs.maximum?).getD 0 / 2)

/-- Theorem stating the minimum number of shoes required -/
theorem min_shoes_theorem (legs : Legs) (h : all_positive_even legs) :
  min_shoes_required legs = 
    max ((legs.get? 0).getD 0 / 2 + (legs.get? 1).getD 0 / 2) ((legs.maximum?).getD 0 / 2) := by
  sorry

#check min_shoes_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shoes_theorem_l1125_112581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1125_112551

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.sin x + Real.cos x) * (Real.sqrt 3 * Real.cos x - Real.sin x)

-- State the theorem
theorem min_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1125_112551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_speed_l1125_112584

/-- Given a car trip with the following conditions:
  - The trip lasts 6 hours in total
  - The average speed for the first 4 hours is 55 mph
  - The average speed for the entire trip is 60 mph
  This theorem proves that the average speed for the remaining 2 hours is 70 mph. -/
theorem car_trip_speed (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (total_avg_speed : ℝ) 
  (h1 : total_time = 6)
  (h2 : initial_time = 4)
  (h3 : initial_speed = 55)
  (h4 : total_avg_speed = 60) :
  (total_avg_speed * total_time - initial_speed * initial_time) / (total_time - initial_time) = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_speed_l1125_112584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_midpoint_l1125_112564

/-- The vertical coordinate of the midpoint of a line segment intersecting a parabola -/
theorem parabola_intersection_midpoint (C : Set (ℝ × ℝ)) (F A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 = 4*y) →  -- Parabola equation
  F ∈ l →  -- Line passes through focus
  A ∈ C ∧ A ∈ l →  -- A is intersection point
  B ∈ C ∧ B ∈ l →  -- B is intersection point
  A ≠ B →  -- A and B are distinct
  dist A B = 5 →  -- Distance between A and B
  (A.2 + B.2) / 2 = 2 :=  -- Vertical coordinate of midpoint
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_midpoint_l1125_112564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_paper_package_sheets_l1125_112532

/-- Represents the number of sheets in a package of printer paper -/
def sheets_per_package : ℕ := sorry

/-- Represents the number of binders in a package -/
def binders_per_package : ℕ := sorry

/-- Sarah's documents use 2 sheets each -/
def sarah_sheets_per_doc : ℕ := 2

/-- Beth's documents use 5 sheets each -/
def beth_sheets_per_doc : ℕ := 5

/-- Sarah's leftover binders -/
def sarah_leftover_binders : ℕ := 60

/-- Beth's leftover sheets -/
def beth_leftover_sheets : ℕ := 300

theorem printer_paper_package_sheets :
  (sarah_sheets_per_doc * (binders_per_package - sarah_leftover_binders) = sheets_per_package) ∧
  (beth_sheets_per_doc * binders_per_package = sheets_per_package - beth_leftover_sheets) →
  sheets_per_package = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_paper_package_sheets_l1125_112532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1125_112501

-- Define the circle and line
def circle_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0
def line_eq (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the intersection points
def intersection_points (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_eq P.1 P.2 m ∧ line_eq P.1 P.2 ∧
  circle_eq Q.1 Q.2 m ∧ line_eq Q.1 Q.2 ∧
  P ≠ Q

-- Define perpendicularity of OP and OQ
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

theorem circle_line_intersection (m : ℝ) :
  ∃ P Q : ℝ × ℝ, intersection_points P Q m ∧ perpendicular P Q → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1125_112501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1125_112513

noncomputable def point1 : ℝ × ℝ := (3, 4)
noncomputable def point2 : ℝ × ℝ := (0, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_points :
  distance point1 point2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1125_112513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_insulation_thickness_l1125_112558

noncomputable def f (x : ℝ) : ℝ := 600 / (3 * x + 5) + 8 * x

theorem optimal_insulation_thickness :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 10 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 10 → f y ≥ f x) ∧
  x = 10 / 3 ∧ f x = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_insulation_thickness_l1125_112558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1125_112516

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the median AD and its midpoint M
def D (A B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def M (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let D := D A B C
  ((A.1 + D.1) / 2, (A.2 + D.2) / 2)

-- Define the line passing through M
def line_through_M (A B C : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := 
  let M := M A B C
  (M.1 + t, M.2 + t)

-- Define points P and Q
def P (A B C : ℝ × ℝ) : ℝ × ℝ := sorry
def Q (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the functions f and g
def f (x : ℝ) : ℝ := x / (4 * x - 1)
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * a^2 * x + 2 * a

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ ∈ Set.Icc (1/3 : ℝ) 1,
   ∃ x₂ ∈ Set.Icc (0 : ℝ) 1,
   f x₁ = g a x₂) ↔
  a ∈ Set.Iic (-2/3) ∪ Set.Icc 0 (1/6) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1125_112516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_3449_to_hundredth_l1125_112540

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The statement that rounding 2.3449 to the nearest hundredth equals 2.34 -/
theorem round_2_3449_to_hundredth :
  roundToHundredth 2.3449 = 2.34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_3449_to_hundredth_l1125_112540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_C_l1125_112594

def A : Finset ℕ := {1, 3, 5, 7, 9}
def B : Finset ℕ := {2, 4, 6, 8, 10}
def C : Finset ℕ := Finset.biUnion A (fun a => Finset.image (· + a) B)

theorem cardinality_of_C : Finset.card C = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_C_l1125_112594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_geometric_sequence_l1125_112541

theorem min_value_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, ∃ r, a (n + 1) = a n * r) →  -- Geometric sequence
  a 2 = 2 →  -- Given condition
  ∀ x : ℝ, a 1 + 2 * a 3 ≥ x → x ≥ 4 * Real.sqrt 2 → x = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_geometric_sequence_l1125_112541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_l1125_112502

/-- Given the difference between a diagonal and a side of a rectangle,
    and the angle between the diagonals, prove that such a rectangle exists. -/
theorem rectangle_existence (e a : ℝ) (φ : ℝ) 
  (h_positive : e > a ∧ a > 0) 
  (h_angle : 0 < φ ∧ φ < Real.pi) : 
  ∃ (b : ℝ), 
    b > 0 ∧ 
    e^2 = a^2 + b^2 ∧ 
    Real.cos φ = (a^2 + b^2) / (2 * e^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_l1125_112502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_slope_half_line_equation_P_midpoint_l1125_112562

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define a line passing through P with slope m
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Define the intersection points of a line with the ellipse
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ellipse x y ∧ line_through_P m x y}

-- Part 1: Length of AB when slope is 1/2
theorem length_AB_slope_half :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points (1/2) ∧ 
                 B ∈ intersection_points (1/2) ∧ 
                 A ≠ B ∧
                 (A.1 - B.1)^2 + (A.2 - B.2)^2 = 90 := by
  sorry

-- Part 2: Equation of line when P is midpoint of AB
theorem line_equation_P_midpoint :
  ∃ m : ℝ, 
    ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ 
                   B ∈ intersection_points m ∧ 
                   A ≠ B ∧
                   P.1 = (A.1 + B.1) / 2 ∧ 
                   P.2 = (A.2 + B.2) / 2 ∧ 
                   (∀ x y : ℝ, line_through_P m x y ↔ x + 2*y - 8 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_slope_half_line_equation_P_midpoint_l1125_112562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_angle_ACB_l1125_112587

/-- Circle C with center (1, 2) and radius 5 -/
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l passing through points A and B -/
def l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- Point A on both circle C and line l -/
def A (m : ℝ) : {p : ℝ × ℝ // C p.1 p.2 ∧ l m p.1 p.2} := sorry

/-- Point B on both circle C and line l -/
def B (m : ℝ) : {p : ℝ × ℝ // C p.1 p.2 ∧ l m p.1 p.2} := sorry

/-- Angle ACB where C is the center of the circle -/
noncomputable def angle_ACB (m : ℝ) : ℝ := sorry

/-- The maximum value of sin ∠ACB is 4/5 -/
theorem max_sin_angle_ACB : 
  ∃ m₀ : ℝ, ∀ m : ℝ, Real.sin (angle_ACB m) ≤ Real.sin (angle_ACB m₀) ∧ Real.sin (angle_ACB m₀) = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_angle_ACB_l1125_112587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_10_and_5_l1125_112595

def A : Finset ℕ := (Finset.range 90).filter (fun n => (n + 10) % 10 = 0)
def B : Finset ℕ := (Finset.range 90).filter (fun n => (n + 10) % 5 = 0)

def total_two_digit_numbers : ℕ := 90

theorem probability_divisible_by_10_and_5 : 
  (Finset.card (A ∩ B) : ℚ) / total_two_digit_numbers = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_10_and_5_l1125_112595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_less_than_x_less_than_tan_x_l1125_112506

theorem sin_x_less_than_x_less_than_tan_x (x : ℝ) (h : 0 < x ∧ x < π / 2) : 
  Real.sin x < x ∧ x < Real.tan x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_less_than_x_less_than_tan_x_l1125_112506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_head_in_three_tosses_l1125_112547

/-- The probability of getting exactly one head in three tosses of a fair coin -/
theorem probability_one_head_in_three_tosses : (3 : ℝ) / 8 = 
  let p : ℝ := 1/2  -- probability of getting heads in a single toss
  let n : ℕ := 3    -- number of tosses
  let k : ℕ := 1    -- number of desired heads
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k) := by
  sorry

#check probability_one_head_in_three_tosses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_head_in_three_tosses_l1125_112547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transform_l1125_112537

-- Define a type for a list of four real numbers
def FourNumbers := Fin 4 → ℝ

-- Define the mean of four numbers
noncomputable def mean (x : FourNumbers) : ℝ := (x 0 + x 1 + x 2 + x 3) / 4

-- Define the variance of four numbers
noncomputable def variance (x : FourNumbers) : ℝ :=
  ((x 0 - mean x)^2 + (x 1 - mean x)^2 + (x 2 - mean x)^2 + (x 3 - mean x)^2) / 4

-- Define the transformation function
def transform (x : FourNumbers) : FourNumbers :=
  fun i => 3 * (x i) + 5

-- Theorem statement
theorem variance_transform (x : FourNumbers) :
  variance x = 7 → variance (transform x) = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transform_l1125_112537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_rotated_triangle_l1125_112577

theorem surface_area_of_rotated_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a : a = 6) (h_b : b = 8) (h_c : c = 10) : 
  (π * (2 * (a * b) / c) * (a + b)) = (336 / 5) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_rotated_triangle_l1125_112577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_symbol_is_minus_l1125_112566

/-- Represents the types of symbols on the blackboard -/
inductive BoardSymbol
| Plus
| Minus

/-- Represents the state of the blackboard -/
structure Board where
  plus_count : Nat
  minus_count : Nat

/-- Performs one operation on the board -/
def operate (b : Board) : Board :=
  if b.plus_count ≥ 2 then
    { plus_count := b.plus_count - 1, minus_count := b.minus_count }
  else if b.minus_count ≥ 2 then
    { plus_count := b.plus_count + 1, minus_count := b.minus_count - 2 }
  else
    { plus_count := b.plus_count - 1, minus_count := b.minus_count }

/-- The main theorem to prove -/
theorem final_symbol_is_minus (initial_board : Board)
  (h_initial : initial_board.plus_count = 20 ∧ initial_board.minus_count = 35) :
  ∃ (final_board : Board), 
    (∀ b, operate b = b → b = final_board) ∧ 
    final_board.plus_count = 0 ∧ 
    final_board.minus_count = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_symbol_is_minus_l1125_112566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1125_112592

/-- A parabola with equation y^2 = 2px and focus at (1, 0) -/
structure Parabola where
  p : ℝ
  eq : ∀ x y : ℝ, y^2 = 2 * p * x
  focus : (1, 0) = (p / 2, 0)

theorem parabola_properties (para : Parabola) :
  para.p = 2 ∧ (∀ x y : ℝ, x = -1 ↔ y^2 = 2 * para.p * x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1125_112592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_at_pi_l1125_112593

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

noncomputable def g (x : ℝ) := Real.sin (x + Real.pi / 3)

theorem transformed_function_at_pi :
  g Real.pi = -Real.sqrt 3 / 2 :=
by
  -- Expand the definition of g
  unfold g
  -- Simplify the expression
  simp [Real.sin_add, Real.sin_pi, Real.cos_pi]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_at_pi_l1125_112593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1125_112517

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.sin α = 4/5) : 
  Real.sin (2 * α) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1125_112517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1125_112583

noncomputable section

-- Define the curves
def C₁ (θ : ℝ) : ℝ × ℝ := (3 + 3 * Real.cos θ, 3 * Real.sin θ)

def C₂ (θ : ℝ) : ℝ := Real.sqrt 3 * Real.sin θ + Real.cos θ

def C₃ : ℝ := Real.pi / 3

-- Define the intersection points
def A : ℝ × ℝ := C₁ C₃

def B : ℝ × ℝ := (C₂ C₃ * Real.cos C₃, C₂ C₃ * Real.sin C₃)

-- Define the length of segment AB
def AB_length : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem length_of_AB : AB_length = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1125_112583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1125_112574

theorem angle_sum_theorem (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5) (h_cos_β : Real.cos β = 3 * Real.sqrt 10 / 10) :
  Real.sin (α + β) = Real.sqrt 2 / 2 ∧ α + β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1125_112574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_sin_2theta_value_l1125_112546

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_monotonic_increasing (k : ℤ) :
  is_monotonic_increasing f (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6) := by
  sorry

theorem sin_2theta_value (θ : ℝ) (h1 : f θ = 5/6) (h2 : Real.pi / 3 < θ) (h3 : θ < 2 * Real.pi / 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_sin_2theta_value_l1125_112546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insphere_touches_face_centers_l1125_112556

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The faces of the tetrahedron are all equilateral triangles -/
  faces_are_equilateral : Bool

/-- An insphere of a regular tetrahedron -/
structure Insphere (t : RegularTetrahedron)

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The center of an equilateral triangle -/
noncomputable def center_of_equilateral_triangle : Point :=
  { x := 0, y := 0, z := 0 }

/-- The point where an insphere touches a face of a regular tetrahedron -/
noncomputable def insphere_touch_point (t : RegularTetrahedron) (i : Insphere t) (face : Nat) : Point :=
  { x := 0, y := 0, z := 0 }

/-- Theorem: The insphere of a regular tetrahedron touches each face at the center of the equilateral triangle -/
theorem insphere_touches_face_centers (t : RegularTetrahedron) (i : Insphere t) :
  ∀ face, insphere_touch_point t i face = center_of_equilateral_triangle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insphere_touches_face_centers_l1125_112556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_maximum_iff_k_negative_l1125_112552

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x / x - k * x

-- State the theorem
theorem f_has_maximum_iff_k_negative (k : ℝ) :
  (∃ (x₀ : ℝ), ∀ (x : ℝ), f k x ≤ f k x₀) ↔ k < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_maximum_iff_k_negative_l1125_112552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_function_intersection_area_l1125_112561

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a reciprocal function y = k/x -/
noncomputable def ReciprocalFunction (k : ℝ) : ℝ → ℝ := fun x ↦ k / x

/-- Calculates the area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- Theorem stating that if a line through the origin intersects y = k/x at A and C,
    and the area of quadrilateral ABCD is 10, then k = -5 -/
theorem reciprocal_function_intersection_area (k : ℝ) (a c : Point) :
  a.y = ReciprocalFunction k a.x →
  c.y = ReciprocalFunction k c.x →
  a.x * c.y = k →  -- Condition for line through origin
  c.x * a.y = k →  -- Condition for line through origin
  let b : Point := ⟨a.x, 0⟩
  let d : Point := ⟨c.x, 0⟩
  quadrilateralArea a b c d = 10 →
  k = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_function_intersection_area_l1125_112561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_than_cos_at_two_l1125_112503

theorem sin_greater_than_cos_at_two :
  π / 2 < 2 ∧ 2 < π → Real.sin 2 > Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_than_cos_at_two_l1125_112503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_channels_cost_difference_l1125_112559

/-- Represents the monthly cost of cable television services -/
structure CableCost where
  basic : ℕ
  movie : ℕ
  sports : ℕ

/-- The total monthly cost for all services -/
def total_cost (c : CableCost) : ℕ := c.basic + c.movie + c.sports

/-- Theorem: The sports channels cost $3 less per month than the movie channels -/
theorem sports_channels_cost_difference (c : CableCost) 
  (h1 : c.basic = 15) 
  (h2 : c.movie = 12) 
  (h3 : c.sports < c.movie)
  (h4 : total_cost c = 36) : 
  c.movie - c.sports = 3 := by
  sorry

#check sports_channels_cost_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_channels_cost_difference_l1125_112559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1125_112525

theorem division_problem (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℝ) / y = 86.12) : y = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1125_112525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1125_112585

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + Real.sin x^2 = a^2 - a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1125_112585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l1125_112555

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the number of permutations
def permutations (n k : ℕ) : ℕ := sorry

-- Define the first term of the sequence
def a₁ (m : ℕ) : ℤ := (binomial (5 * m) (11 - 2 * m) : ℤ) - (permutations (11 - 3 * m) (2 * m - 2) : ℤ)

-- Define the function for the common difference
noncomputable def common_difference (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define the remainder function
def remainder (a b : ℕ) : ℕ := a % b

-- Theorem statement
theorem arithmetic_sequence_formula :
  ∀ m : ℕ,
  m > 0 →
  let n := remainder (77^77 - 15) 19
  let d := common_difference 1 n  -- We use x = 1 for simplicity
  ∀ k : ℕ,
  arithmetic_sequence (a₁ m) (Int.floor d) k = -4 * k + 104 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l1125_112555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solutions_l1125_112596

theorem sin_cos_equation_solutions (x : ℝ) : 
  (∃ m : ℤ, (x = Real.arccos 0.9114 + 2 * Real.pi * ↑m) ∨ 
            (x = -Real.arccos 0.9114 + 2 * Real.pi * ↑m) ∨ 
            (x = Real.arccos (-0.4114) + 2 * Real.pi * ↑m) ∨ 
            (x = -Real.arccos (-0.4114) + 2 * Real.pi * ↑m)) ↔ 
  Real.sin (Real.pi * Real.cos x) = Real.cos (Real.pi * Real.sin x) := by
  sorry

#check sin_cos_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solutions_l1125_112596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zinc_weight_l1125_112535

-- Define the total weight of the mixture
noncomputable def total_weight : ℚ := 64

-- Define the ratio of zinc to copper
noncomputable def zinc_ratio : ℚ := 1 / 2

-- Theorem to prove the weight of zinc in the mixture
theorem zinc_weight (total : ℚ) (ratio : ℚ) : 
  total * ratio = total * ratio := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zinc_weight_l1125_112535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_inequality_range_l1125_112504

-- Define the operation ⊙
noncomputable def odot (x y : ℝ) : ℝ := x / (2 - y)

-- Define the theorem
theorem odot_inequality_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + 1 - a) ≥ 0 → -2 < x ∧ x < 2) →
  -2 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_inequality_range_l1125_112504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_calculation_l1125_112576

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem floor_calculation : 
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (6.0 : ℝ) = 16.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_calculation_l1125_112576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1125_112500

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the circle
def circleEq (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Theorem statement
theorem parabola_circle_intersection (p : ℝ) :
  ∃ (M N Q : ℝ × ℝ),
    parabola p M.1 M.2 ∧
    parabola p N.1 N.2 ∧
    Q.2 = 0 ∧ Q.1 > 0 ∧
    circleEq (focus p) p M ∧
    circleEq (focus p) p N ∧
    circleEq (focus p) p Q ∧
    (N.1 - Q.1)^2 + (N.2 - Q.2)^2 = 10
  → p = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1125_112500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_minus_pi_6_plus_2_range_l1125_112565

theorem sin_2x_minus_pi_6_plus_2_range (x : ℝ) (h : x ∈ Set.Icc (-π/2) (π/3)) :
  ∃ y ∈ Set.Icc 1 3, y = Real.sin (2*x - π/6) + 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_minus_pi_6_plus_2_range_l1125_112565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sum_product_theorem_l1125_112597

theorem irrational_sum_product_theorem (a : ℝ) (h_a : Irrational a) :
  ∃ (b b' : ℝ), Irrational b ∧ Irrational b' ∧
    (∃ (q r : ℚ), (a + b : ℝ) = q ∧ (a * b' : ℝ) = r) ∧
    (Irrational (a * b) ∧ Irrational (a + b')) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sum_product_theorem_l1125_112597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PMN_l1125_112545

-- Define the circle C
noncomputable def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the line l₂
noncomputable def l₂ (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

-- Define a point P on the circle C
def P_on_C (x y : ℝ) : Prop := C x y

-- Define the maximum area of triangle PMN
noncomputable def max_area_PMN : ℝ := (7 * Real.sqrt 35) / 4

theorem max_area_triangle_PMN :
  ∀ (xM yM xN yN : ℝ),
    C xM yM →
    C xN yN →
    l₂ xM yM →
    l₂ xN yN →
    xM ≠ xN ∨ yM ≠ yN →
    (∀ (xP yP : ℝ), P_on_C xP yP →
      ∃ (S : ℝ), S ≤ max_area_PMN ∧
        S = (1/2) * Real.sqrt ((xN - xM)^2 + (yN - yM)^2) *
          Real.sqrt ((xP - xM)^2 + (yP - yM)^2 - ((xP - xM) * (xN - xM) + (yP - yM) * (yN - yM))^2 /
            ((xN - xM)^2 + (yN - yM)^2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PMN_l1125_112545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_two_l1125_112519

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The sequence of points Qᵢ on CD -/
def Q : ℕ → Point := sorry

/-- The sequence of points Pᵢ on BD -/
def P : ℕ → Point := sorry

/-- The theorem statement -/
theorem sum_of_distances_is_two (ABCD : UnitSquare) : 
  ABCD.A = Point.mk 0 1 →
  ABCD.B = Point.mk 1 1 →
  ABCD.C = Point.mk 0 0 →
  ABCD.D = Point.mk 1 0 →
  Q 1 = Point.mk (2/3) 0 →
  (∀ i : ℕ, P i = Point.mk (ABCD.B.x * (Q i).x / ((Q i).x + ABCD.A.y)) 
                           (ABCD.B.y * (Q i).x / ((Q i).x + ABCD.A.y))) →
  (∀ i : ℕ, (Q (i+1)).x = ABCD.D.x - (1/3)^(i+1)) →
  (∀ i : ℕ, (Q (i+1)).y = 0) →
  (∀ i : ℕ, (P i).y - (Q (i+1)).y = ((P i).x - (Q (i+1)).x) * (ABCD.D.x - ABCD.C.x) / (ABCD.D.y - ABCD.C.y)) →
  (∑' i, ‖(P (i+1)).x - (P i).x‖) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_two_l1125_112519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1125_112588

/-- Calculates the time (in seconds) for a train to pass a person moving in the opposite direction. -/
noncomputable def trainPassingTime (trainLength : ℝ) (trainSpeed : ℝ) (personSpeed : ℝ) : ℝ :=
  let trainSpeedMPS := trainSpeed * 1000 / 3600
  let personSpeedMPS := personSpeed * 1000 / 3600
  let relativeSpeed := trainSpeedMPS + personSpeedMPS
  trainLength / relativeSpeed

/-- The time for a 110 m long train moving at 50 km/h to pass a man moving at 5 km/h 
    in the opposite direction is approximately 7.20 seconds. -/
theorem train_passing_time_approx :
  ∃ ε > 0, |trainPassingTime 110 50 5 - 7.20| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1125_112588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l1125_112591

/-- Given a population that increases by 28% annually, if after 2 years
    the population is 25460.736, then the initial population is approximately 15536. -/
theorem village_population (P : ℝ) (h : P * (1 + 0.28)^2 = 25460.736) :
  abs (P - 15536) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l1125_112591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_x_squared_minus_y_squared_eq_77_l1125_112578

theorem two_solutions_for_x_squared_minus_y_squared_eq_77 :
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    let x := p.1
    let y := p.2
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 77) (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_x_squared_minus_y_squared_eq_77_l1125_112578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_exists_l1125_112580

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = Real.sqrt ((a n)^2 + 1 / (a n))

theorem sequence_bound_exists :
  ∃ (a : ℕ → ℝ) (α : ℝ), sequence_property a ∧ α > 0 ∧ 
    ∀ n : ℕ, n ≥ 1 → (1 / 2 : ℝ) ≤ a n / n^α ∧ a n / n^α ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_exists_l1125_112580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_equals_seven_l1125_112505

theorem combination_sum_equals_seven (n : ℕ) 
  (h1 : 0 ≤ 5 - n ∧ 5 - n ≤ n) 
  (h2 : 0 ≤ 10 - n ∧ 10 - n ≤ n + 1) : 
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 := by
  sorry

#check combination_sum_equals_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_equals_seven_l1125_112505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_water_surface_area_l1125_112521

/-- Represents a cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- Represents the water surface area in the cube -/
def water_surface_area (c : UnitCube) (area : ℝ) : Prop := sorry

/-- The cube is half-filled with water -/
axiom half_filled (c : UnitCube) : ∃ (v : ℝ), v = 1/2 * c.edge_length^3

/-- The water doesn't spill when the cube is rotated -/
axiom no_spill (c : UnitCube) : ∀ (orientation : ℝ → ℝ → ℝ → ℝ), 
  ∃ (area : ℝ), water_surface_area c area

/-- The maximum water surface area is √2 -/
theorem max_water_surface_area (c : UnitCube) : 
  ∃ (max_area : ℝ), (∀ (area : ℝ), water_surface_area c area → area ≤ max_area) ∧ max_area = Real.sqrt 2 := by
  sorry

#check max_water_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_water_surface_area_l1125_112521
