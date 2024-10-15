import Mathlib

namespace NUMINAMATH_CALUDE_catfish_count_l3360_336026

theorem catfish_count (C : ℕ) (total_fish : ℕ) : 
  C + 10 + (3 * C / 2) = total_fish ∧ total_fish = 50 → C = 16 := by
  sorry

end NUMINAMATH_CALUDE_catfish_count_l3360_336026


namespace NUMINAMATH_CALUDE_train_speed_and_length_l3360_336057

/-- Given a train passing a stationary observer in 7 seconds and taking 25 seconds to pass a 378-meter platform at constant speed, prove that the train's speed is 21 m/s and its length is 147 m. -/
theorem train_speed_and_length :
  ∀ (V l : ℝ),
  (7 * V = l) →
  (25 * V = 378 + l) →
  (V = 21 ∧ l = 147) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l3360_336057


namespace NUMINAMATH_CALUDE_xy_problem_l3360_336061

theorem xy_problem (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 6) : 
  ((x - y)^2 = 25) ∧ (x^3 * y + x * y^3 = 222) := by
  sorry

end NUMINAMATH_CALUDE_xy_problem_l3360_336061


namespace NUMINAMATH_CALUDE_actual_speed_proof_l3360_336091

theorem actual_speed_proof (v : ℝ) (h : (v / (v + 10) = 3 / 4)) : v = 30 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_proof_l3360_336091


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_sqrt_four_l3360_336044

theorem sqrt_nine_minus_sqrt_four : Real.sqrt 9 - Real.sqrt 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_sqrt_four_l3360_336044


namespace NUMINAMATH_CALUDE_given_number_equals_scientific_form_l3360_336020

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  is_valid : 1 ≤ mantissa ∧ mantissa < 10

/-- The given number in decimal form -/
def given_number : ℝ := 0.000123

/-- The scientific notation representation of the given number -/
def scientific_form : ScientificNotation := {
  mantissa := 1.23,
  exponent := -4,
  is_valid := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_equals_scientific_form :
  given_number = scientific_form.mantissa * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_given_number_equals_scientific_form_l3360_336020


namespace NUMINAMATH_CALUDE_fixed_point_of_log_function_l3360_336029

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = 1 + logₐ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a x

-- Theorem statement
theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_log_function_l3360_336029


namespace NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l3360_336077

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- State the theorems
theorem tangent_line_at_one (a : ℝ) :
  a = -1 → ∃ m b, ∀ x, f a x = m * (x - 1) + b ∧ m = -Real.log 2 ∧ b = 0 := by sorry

theorem symmetry_condition (a : ℝ) :
  (∀ x > 0, f a (1/x) = f a (1/(-2 * x))) ↔ a = 1/2 := by sorry

theorem extreme_values_condition (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y ∨ f a x ≤ f a y) ↔ 0 < a ∧ a < 1/2 := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l3360_336077


namespace NUMINAMATH_CALUDE_total_apples_l3360_336051

theorem total_apples (marin_apples : ℕ) (david_apples : ℕ) (amanda_apples : ℕ) : 
  marin_apples = 6 →
  david_apples = 2 * marin_apples →
  amanda_apples = david_apples + 5 →
  marin_apples + david_apples + amanda_apples = 35 :=
by sorry

end NUMINAMATH_CALUDE_total_apples_l3360_336051


namespace NUMINAMATH_CALUDE_ellen_painted_ten_roses_l3360_336055

/-- The time it takes to paint different types of flowers and vines --/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  orchid : ℕ
  vine : ℕ

/-- The number of each type of flower and vine painted --/
structure FlowerCounts where
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Calculates the total time spent painting based on the painting times and flower counts --/
def totalPaintingTime (times : PaintingTimes) (counts : FlowerCounts) : ℕ :=
  times.lily * counts.lilies +
  times.rose * counts.roses +
  times.orchid * counts.orchids +
  times.vine * counts.vines

/-- Theorem: Given the painting times and flower counts, prove that Ellen painted 10 roses --/
theorem ellen_painted_ten_roses
  (times : PaintingTimes)
  (counts : FlowerCounts)
  (h1 : times.lily = 5)
  (h2 : times.rose = 7)
  (h3 : times.orchid = 3)
  (h4 : times.vine = 2)
  (h5 : counts.lilies = 17)
  (h6 : counts.orchids = 6)
  (h7 : counts.vines = 20)
  (h8 : totalPaintingTime times counts = 213) :
  counts.roses = 10 :=
sorry

end NUMINAMATH_CALUDE_ellen_painted_ten_roses_l3360_336055


namespace NUMINAMATH_CALUDE_intersection_max_difference_zero_l3360_336047

-- Define the polynomial functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

-- State the theorem
theorem intersection_max_difference_zero :
  (∀ x : ℝ, f x = g x → x = -1) →  -- Given condition: x = -1 is the only intersection
  (∃ x : ℝ, f x = g x) →           -- Ensure at least one intersection exists
  (∀ x y : ℝ, f x = g x ∧ f y = g y → |f x - f y| = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_max_difference_zero_l3360_336047


namespace NUMINAMATH_CALUDE_max_volume_at_five_l3360_336097

def box_volume (x : ℝ) : ℝ := (30 - 2*x)^2 * x

def possible_x : Set ℝ := {4, 5, 6, 7}

theorem max_volume_at_five :
  ∀ x ∈ possible_x, x ≠ 5 → box_volume x ≤ box_volume 5 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_at_five_l3360_336097


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l3360_336076

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space defined by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point satisfies the equation 9x^2 - 25y^2 = 0 -/
def satisfiesEquation (p : Point2D) : Prop :=
  9 * p.x^2 - 25 * p.y^2 = 0

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The two lines that form the solution -/
def line1 : Line2D := { a := 3, b := -5, c := 0 }
def line2 : Line2D := { a := 3, b := 5, c := 0 }

/-- Theorem stating that the equation represents a pair of straight lines -/
theorem equation_represents_pair_of_lines :
  ∀ p : Point2D, satisfiesEquation p ↔ (pointOnLine p line1 ∨ pointOnLine p line2) :=
sorry


end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l3360_336076


namespace NUMINAMATH_CALUDE_dragon_can_be_defeated_l3360_336083

/-- Represents the possible strikes and their corresponding regrowth --/
inductive Strike : Type
| one : Strike
| seventeen : Strike
| twentyone : Strike
| thirtythree : Strike

/-- Returns the number of heads chopped for a given strike --/
def heads_chopped (s : Strike) : ℕ :=
  match s with
  | Strike.one => 1
  | Strike.seventeen => 17
  | Strike.twentyone => 21
  | Strike.thirtythree => 33

/-- Returns the number of heads that grow back for a given strike --/
def heads_regrown (s : Strike) : ℕ :=
  match s with
  | Strike.one => 10
  | Strike.seventeen => 14
  | Strike.twentyone => 0
  | Strike.thirtythree => 48

/-- Represents the state of the dragon --/
structure DragonState :=
  (heads : ℕ)

/-- Applies a strike to the dragon state --/
def apply_strike (state : DragonState) (s : Strike) : DragonState :=
  let new_heads := state.heads - heads_chopped s + heads_regrown s
  ⟨max new_heads 0⟩

/-- Theorem: There exists a sequence of strikes that defeats the dragon --/
theorem dragon_can_be_defeated :
  ∃ (sequence : List Strike), (sequence.foldl apply_strike ⟨2000⟩).heads = 0 :=
sorry

end NUMINAMATH_CALUDE_dragon_can_be_defeated_l3360_336083


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3360_336018

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- First line equation: y = x + 1 -/
def line1 (x y : ℝ) : Prop := y = x + 1

/-- Second line equation: y = -x + 1 -/
def line2 (x y : ℝ) : Prop := y = -x + 1

/-- The theorem stating that the intersection point of the two lines is (0, 1) -/
theorem intersection_point_of_lines : 
  ∃ p : IntersectionPoint, line1 p.x p.y ∧ line2 p.x p.y ∧ p.x = 0 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3360_336018


namespace NUMINAMATH_CALUDE_age_difference_proof_l3360_336046

theorem age_difference_proof (elder_age younger_age : ℕ) : 
  elder_age > younger_age →
  elder_age - 10 = 5 * (younger_age - 10) →
  elder_age = 35 →
  younger_age = 15 →
  elder_age - younger_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3360_336046


namespace NUMINAMATH_CALUDE_elise_remaining_money_l3360_336037

/-- Calculates the remaining money for Elise given her initial amount, savings, and expenses. -/
def remaining_money (initial : ℕ) (savings : ℕ) (comic_expense : ℕ) (puzzle_expense : ℕ) : ℕ :=
  initial + savings - comic_expense - puzzle_expense

/-- Proves that Elise's remaining money is $1 given her initial amount, savings, and expenses. -/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

#eval remaining_money 8 13 2 18

end NUMINAMATH_CALUDE_elise_remaining_money_l3360_336037


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3360_336067

/-- The total surface area of a cylinder with height 15 and radius 2 is 68π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 15
  let r : ℝ := 2
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 68 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3360_336067


namespace NUMINAMATH_CALUDE_tenth_ring_squares_l3360_336017

/-- The number of unit squares in the nth ring around a 3x3 center block -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 10th ring around a 3x3 center block contains 88 unit squares -/
theorem tenth_ring_squares : ring_squares 10 = 88 := by sorry

end NUMINAMATH_CALUDE_tenth_ring_squares_l3360_336017


namespace NUMINAMATH_CALUDE_a6_is_2_in_factorial_base_of_1735_l3360_336039

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem a6_is_2_in_factorial_base_of_1735 :
  factorial_base_coefficient 1735 6 = 2 := by sorry

end NUMINAMATH_CALUDE_a6_is_2_in_factorial_base_of_1735_l3360_336039


namespace NUMINAMATH_CALUDE_f_extrema_l3360_336004

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x^3 - p*x^2 - q*x

-- State the theorem
theorem f_extrema (p q : ℝ) :
  (f p q 1 = 0) →
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f p q x ≤ f p q x₀) ∧
  (f p q x₀ = 4/27) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f p q x ≥ f p q x₁) ∧
  (f p q x₁ = -4) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l3360_336004


namespace NUMINAMATH_CALUDE_power_five_remainder_l3360_336075

theorem power_five_remainder (n : ℕ) : (5^1234 : ℕ) % 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_five_remainder_l3360_336075


namespace NUMINAMATH_CALUDE_rectangle_largest_side_l3360_336062

/-- Given a rectangle with perimeter 240 feet and area equal to fifteen times its perimeter,
    prove that the length of its largest side is 60 feet. -/
theorem rectangle_largest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧                   -- positive dimensions
  2 * (l + w) = 240 ∧               -- perimeter is 240 feet
  l * w = 15 * 240 →                -- area is fifteen times perimeter
  max l w = 60 := by sorry

end NUMINAMATH_CALUDE_rectangle_largest_side_l3360_336062


namespace NUMINAMATH_CALUDE_total_yellow_marbles_l3360_336002

/-- The total number of yellow marbles given the number of marbles each person has -/
def total_marbles (mary_marbles joan_marbles john_marbles : ℕ) : ℕ :=
  mary_marbles + joan_marbles + john_marbles

/-- Theorem stating that the total number of yellow marbles is 19 -/
theorem total_yellow_marbles :
  total_marbles 9 3 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_marbles_l3360_336002


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l3360_336033

open Real

theorem function_inequality_implies_upper_bound (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x * log x) →
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) →
  a ≤ 5 + log 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l3360_336033


namespace NUMINAMATH_CALUDE_doubled_container_volume_l3360_336074

/-- The volume of a container after doubling its dimensions -/
def doubled_volume (original_volume : ℝ) : ℝ := 8 * original_volume

/-- Theorem: Doubling the dimensions of a 3-gallon container results in a 24-gallon container -/
theorem doubled_container_volume : doubled_volume 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_doubled_container_volume_l3360_336074


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l3360_336014

/-- Given four positive prime numbers whose product equals the sum of 55 consecutive positive integers,
    the smallest possible sum of these four primes is 28. -/
theorem smallest_sum_of_four_primes (a b c d : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → Nat.Prime d →
  (∃ x : ℕ, a * b * c * d = (55 : ℕ) * (x + 27)) →
  (∀ w x y z : ℕ, Nat.Prime w → Nat.Prime x → Nat.Prime y → Nat.Prime z →
    (∃ n : ℕ, w * x * y * z = (55 : ℕ) * (n + 27)) →
    a + b + c + d ≤ w + x + y + z) →
  a + b + c + d = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l3360_336014


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_set_l3360_336019

theorem quadratic_inequality_integer_set :
  {x : ℤ | x^2 - 3*x - 4 < 0} = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_set_l3360_336019


namespace NUMINAMATH_CALUDE_june_found_seventeen_eggs_l3360_336000

/-- The number of eggs June found -/
def total_eggs : ℕ :=
  let nest1_eggs := 2 * 5  -- 2 nests with 5 eggs each in 1 tree
  let nest2_eggs := 1 * 3  -- 1 nest with 3 eggs in another tree
  let nest3_eggs := 1 * 4  -- 1 nest with 4 eggs in the front yard
  nest1_eggs + nest2_eggs + nest3_eggs

/-- Theorem stating that June found 17 eggs in total -/
theorem june_found_seventeen_eggs : total_eggs = 17 := by
  sorry

end NUMINAMATH_CALUDE_june_found_seventeen_eggs_l3360_336000


namespace NUMINAMATH_CALUDE_proportion_problem_l3360_336070

theorem proportion_problem (x : ℝ) : 
  (x / 5 = 0.96 / 8) → x = 0.6 := by
sorry

end NUMINAMATH_CALUDE_proportion_problem_l3360_336070


namespace NUMINAMATH_CALUDE_inequality_theorem_l3360_336053

theorem inequality_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (b^2 / a) + (a^2 / b) ≥ a + b := by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3360_336053


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l3360_336092

theorem maximum_marks_calculation (percentage : ℝ) (received_marks : ℝ) (max_marks : ℝ) : 
  percentage = 80 → received_marks = 240 → percentage / 100 * max_marks = received_marks → max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l3360_336092


namespace NUMINAMATH_CALUDE_gcd_372_684_l3360_336089

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l3360_336089


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l3360_336035

theorem cubic_root_ratio (p q r s : ℝ) (h : ∀ x, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) :
  r / s = -5 / 12 := by sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l3360_336035


namespace NUMINAMATH_CALUDE_simplify_expression_l3360_336060

theorem simplify_expression : (625 : ℝ) ^ (1/4 : ℝ) * (256 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3360_336060


namespace NUMINAMATH_CALUDE_amusement_park_admission_l3360_336022

theorem amusement_park_admission (child_fee adult_fee : ℚ) 
  (total_people : ℕ) (total_fees : ℚ) :
  child_fee = 3/2 →
  adult_fee = 4 →
  total_people = 315 →
  total_fees = 810 →
  ∃ (children adults : ℕ),
    children + adults = total_people ∧
    child_fee * children + adult_fee * adults = total_fees ∧
    children = 180 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_l3360_336022


namespace NUMINAMATH_CALUDE_function_properties_l3360_336058

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a * x^2 - 3

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((-1 : ℝ)) 1) →
  (∃ m : ℝ, m = -1 ∧ 
    (∀ x > 0, f (-1) x - m * x ≤ -3) ∧
    (∀ m' < m, ∃ x > 0, f (-1) x - m' * x > -3)) ∧
  (∀ x > 0, x * Real.log x - x^2 - 3 - x * Real.exp x + x^2 < -2 * x - 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3360_336058


namespace NUMINAMATH_CALUDE_p_iff_q_l3360_336036

theorem p_iff_q : ∀ x : ℝ, (x > 1 ∨ x < -1) ↔ |x + 1| + |x - 1| > 2 := by
  sorry

end NUMINAMATH_CALUDE_p_iff_q_l3360_336036


namespace NUMINAMATH_CALUDE_judson_contribution_is_500_l3360_336096

def house_painting_problem (judson_contribution : ℝ) : Prop :=
  let kenny_contribution := 1.2 * judson_contribution
  let camilo_contribution := kenny_contribution + 200
  judson_contribution + kenny_contribution + camilo_contribution = 1900

theorem judson_contribution_is_500 :
  ∃ (judson_contribution : ℝ),
    house_painting_problem judson_contribution ∧ judson_contribution = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_judson_contribution_is_500_l3360_336096


namespace NUMINAMATH_CALUDE_anoop_join_time_l3360_336049

/-- Prove that Anoop joined after 6 months given the investment conditions -/
theorem anoop_join_time (arjun_investment : ℕ) (anoop_investment : ℕ) (total_months : ℕ) :
  arjun_investment = 20000 →
  anoop_investment = 40000 →
  total_months = 12 →
  ∃ x : ℕ, 
    (arjun_investment * total_months = anoop_investment * (total_months - x)) ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_anoop_join_time_l3360_336049


namespace NUMINAMATH_CALUDE_total_spent_l3360_336009

def weekend_expenses (adidas nike skechers clothes : ℕ) : Prop :=
  nike = 3 * adidas ∧
  adidas = skechers / 5 ∧
  adidas = 600 ∧
  clothes = 2600

theorem total_spent (adidas nike skechers clothes : ℕ) :
  weekend_expenses adidas nike skechers clothes →
  adidas + nike + skechers + clothes = 8000 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_l3360_336009


namespace NUMINAMATH_CALUDE_equal_intersection_areas_exist_l3360_336045

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  opposite_edges_perpendicular : Bool
  opposite_edges_horizontal : Bool
  vertical_midline : Bool

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of a tetrahedron and a sphere -/
structure Configuration where
  tetrahedron : Tetrahedron
  sphere : Sphere
  sphere_centered_on_midline : Bool

/-- Represents a horizontal plane -/
structure HorizontalPlane where
  height : ℝ

/-- Function to calculate the area of intersection between a horizontal plane and the tetrahedron -/
def tetrahedron_intersection_area (t : Tetrahedron) (p : HorizontalPlane) : ℝ := sorry

/-- Function to calculate the area of intersection between a horizontal plane and the sphere -/
def sphere_intersection_area (s : Sphere) (p : HorizontalPlane) : ℝ := sorry

/-- Theorem stating that there exists a configuration where all horizontal plane intersections have equal areas -/
theorem equal_intersection_areas_exist : 
  ∃ (c : Configuration), ∀ (p : HorizontalPlane), 
    tetrahedron_intersection_area c.tetrahedron p = sphere_intersection_area c.sphere p :=
sorry

end NUMINAMATH_CALUDE_equal_intersection_areas_exist_l3360_336045


namespace NUMINAMATH_CALUDE_line_parallel_perp_plane_l3360_336065

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perp_plane
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perp m α) :
  perp n α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perp_plane_l3360_336065


namespace NUMINAMATH_CALUDE_initial_percentage_chemical_x_l3360_336068

/-- Given an 80-liter mixture and adding 20 liters of pure chemical x resulting in a 100-liter mixture that is 44% chemical x, prove that the initial percentage of chemical x was 30%. -/
theorem initial_percentage_chemical_x : 
  ∀ (initial_percentage : ℝ),
  initial_percentage ≥ 0 ∧ initial_percentage ≤ 1 →
  (80 * initial_percentage + 20) / 100 = 0.44 →
  initial_percentage = 0.3 := by
sorry

end NUMINAMATH_CALUDE_initial_percentage_chemical_x_l3360_336068


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3360_336064

/-- A circle defined by x^2 + y^2 + 2x + 4y + m = 0 has exactly two points at a distance of √2
    from the line x + y + 1 = 0 if and only if m ∈ (-3, 5) -/
theorem circle_line_intersection (m : ℝ) :
  (∃! (p q : ℝ × ℝ),
    p ≠ q ∧
    (p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + m = 0) ∧
    (q.1^2 + q.2^2 + 2*q.1 + 4*q.2 + m = 0) ∧
    (p.1 + p.2 + 1 ≠ 0) ∧
    (q.1 + q.2 + 1 ≠ 0) ∧
    ((p.1 + p.2 + 1)^2 / 2 = 2) ∧
    ((q.1 + q.2 + 1)^2 / 2 = 2))
  ↔
  (-3 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3360_336064


namespace NUMINAMATH_CALUDE_smallest_number_negative_l3360_336042

theorem smallest_number_negative (a : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) ↔ a < -7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_negative_l3360_336042


namespace NUMINAMATH_CALUDE_outfits_count_l3360_336041

/-- The number of possible outfits with different colored shirt and hat -/
def number_of_outfits (red_shirts green_shirts pants green_hats red_hats : ℕ) : ℕ :=
  (red_shirts * green_hats + green_shirts * red_hats) * pants

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfits_count : number_of_outfits 6 4 7 10 9 = 672 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3360_336041


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l3360_336088

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the closed interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

theorem extreme_values_and_tangent_lines :
  -- Part 1: Extreme values
  (∃ x ∈ I, f x = 2 ∧ ∀ y ∈ I, f y ≤ 2) ∧
  (∃ x ∈ I, f x = -2 ∧ ∀ y ∈ I, f y ≥ -2) ∧
  -- Part 2: Range of m for three tangent lines
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (f x₁ - m) / (x₁ - 2) = 3 * x₁^2 - 3 ∧
    (f x₂ - m) / (x₂ - 2) = 3 * x₂^2 - 3 ∧
    (f x₃ - m) / (x₃ - 2) = 3 * x₃^2 - 3) ↔
  -6 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l3360_336088


namespace NUMINAMATH_CALUDE_estimate_value_l3360_336016

theorem estimate_value : 
  3 < (2 * Real.sqrt 2 + Real.sqrt 6) * Real.sqrt (1/2) ∧ 
  (2 * Real.sqrt 2 + Real.sqrt 6) * Real.sqrt (1/2) < 4 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l3360_336016


namespace NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l3360_336032

theorem multiples_of_six_ending_in_four (n : ℕ) : 
  (∃ m : ℕ, m = 10) ↔ 
  (∀ k : ℕ, (6 * k < 600 ∧ (6 * k) % 10 = 4) → k ≤ n) ∧ 
  (∃ (k₁ k₂ : ℕ), k₁ ≤ n ∧ k₂ ≤ n ∧ k₁ ≠ k₂ ∧ 
    6 * k₁ < 600 ∧ (6 * k₁) % 10 = 4 ∧ 
    6 * k₂ < 600 ∧ (6 * k₂) % 10 = 4) :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l3360_336032


namespace NUMINAMATH_CALUDE_sin_x_plus_pi_l3360_336079

theorem sin_x_plus_pi (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.tan x = -4/3) :
  Real.sin (x + π) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_x_plus_pi_l3360_336079


namespace NUMINAMATH_CALUDE_blue_cars_most_l3360_336087

def total_cars : ℕ := 24

def red_cars : ℕ := total_cars / 4

def blue_cars : ℕ := red_cars + 6

def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem blue_cars_most : blue_cars > red_cars ∧ blue_cars > yellow_cars := by
  sorry

end NUMINAMATH_CALUDE_blue_cars_most_l3360_336087


namespace NUMINAMATH_CALUDE_last_number_proof_l3360_336050

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 → 
  (b + c + d) / 3 = 15 → 
  a = 33 → 
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l3360_336050


namespace NUMINAMATH_CALUDE_solve_inequality_part1_solve_inequality_part2_l3360_336021

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

-- Part 1: Solve f(x) < 0 when a = -4
theorem solve_inequality_part1 : 
  ∀ x : ℝ, f (-4) x < 0 ↔ 1 < x ∧ x < 3 :=
sorry

-- Part 2: Find range of a when f(x) > 0 for all real x
theorem solve_inequality_part2 : 
  (∀ x : ℝ, f a x > 0) ↔ -2 * Real.sqrt 3 < a ∧ a < 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_part1_solve_inequality_part2_l3360_336021


namespace NUMINAMATH_CALUDE_smallest_possible_a_l3360_336024

theorem smallest_possible_a (a b c : ℝ) : 
  a > 0 → 
  (∃ n : ℤ, a + 2*b + 3*c = n) →
  (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y = a*(x - 1/2)^2 - 1/2) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ n : ℤ, a' + 2*b' + 3*c' = n) ∧
    (∀ x y : ℝ, y = a'*x^2 + b'*x + c' ↔ y = a'*(x - 1/2)^2 - 1/2)) →
    a ≤ a') →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l3360_336024


namespace NUMINAMATH_CALUDE_existence_of_special_divisibility_pair_l3360_336093

theorem existence_of_special_divisibility_pair : 
  ∃ (a b : ℕ+), 
    a ∣ b^2 ∧ 
    b^2 ∣ a^3 ∧ 
    a^3 ∣ b^4 ∧ 
    b^4 ∣ a^5 ∧ 
    ¬(a^5 ∣ b^6) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_divisibility_pair_l3360_336093


namespace NUMINAMATH_CALUDE_circle_center_correct_l3360_336078

/-- Definition of the circle C -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that circle_center is the center of the circle defined by circle_equation -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3360_336078


namespace NUMINAMATH_CALUDE_smallest_element_mean_l3360_336015

/-- The arithmetic mean of the smallest number in all r-element subsets of {1, 2, ..., n} -/
def f (r n : ℕ+) : ℚ :=
  (n + 1) / (r + 1)

/-- Theorem stating that f(r, n) is the arithmetic mean of the smallest number
    in all r-element subsets of {1, 2, ..., n} -/
theorem smallest_element_mean (r n : ℕ+) (h : r ≤ n) :
  f r n = (Finset.sum (Finset.range (n - r + 1)) (fun a => a * (Nat.choose (n - a) (r - 1)))) /
          (Nat.choose n r) :=
sorry

end NUMINAMATH_CALUDE_smallest_element_mean_l3360_336015


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3360_336056

/-- The number of ways to arrange books on a shelf -/
def arrange_books (num_math_books : Nat) (num_history_books : Nat) : Nat :=
  if num_math_books ≥ 2 then
    num_math_books * (num_math_books - 1) * Nat.factorial (num_math_books + num_history_books - 2)
  else
    0

/-- Theorem: The number of ways to arrange 3 math books and 5 history books with math books on both ends is 4320 -/
theorem book_arrangement_theorem :
  arrange_books 3 5 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3360_336056


namespace NUMINAMATH_CALUDE_f_minimum_f_le_g_iff_exists_three_roots_l3360_336082

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

-- Statement 1: f has a minimum at x = 1/e
theorem f_minimum : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x := by sorry

-- Statement 2: f(x) ≤ g(x) for all x > 0 iff a ≥ 1
theorem f_le_g_iff (a : ℝ) : (∀ x > 0, f x ≤ g a x) ↔ a ≥ 1 := by sorry

-- Statement 3: When a = 1/8, there exists m such that 3f(x)/(4x) + m + g(x) = 0 has three distinct real roots iff 7/8 < m < 15/8 - 3/4 * ln 3
theorem exists_three_roots :
  ∃ (m : ℝ), (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (3 * f x) / (4 * x) + m + g (1/8) x = 0 ∧
    (3 * f y) / (4 * y) + m + g (1/8) y = 0 ∧
    (3 * f z) / (4 * z) + m + g (1/8) z = 0) ↔
  (7/8 < m ∧ m < 15/8 - 3/4 * Real.log 3) := by sorry

end

end NUMINAMATH_CALUDE_f_minimum_f_le_g_iff_exists_three_roots_l3360_336082


namespace NUMINAMATH_CALUDE_probability_two_red_two_blue_l3360_336071

/-- The probability of selecting 2 red and 2 blue marbles from a bag containing 12 red marbles
    and 8 blue marbles, when 4 marbles are selected at random without replacement. -/
theorem probability_two_red_two_blue (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ)
    (selected_marbles : ℕ) :
    total_marbles = red_marbles + blue_marbles →
    total_marbles = 20 →
    red_marbles = 12 →
    blue_marbles = 8 →
    selected_marbles = 4 →
    (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 : ℚ) /
    (Nat.choose total_marbles selected_marbles) = 56 / 147 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_two_blue_l3360_336071


namespace NUMINAMATH_CALUDE_four_number_sequence_l3360_336090

theorem four_number_sequence : ∃ (a b c d : ℝ), 
  (∃ (q : ℝ), b = a * q ∧ c = b * q) ∧  -- Geometric progression
  (∃ (r : ℝ), c = b + r ∧ d = c + r) ∧  -- Arithmetic progression
  a + d = 21 ∧                          -- Sum of first and last
  b + c = 18 := by                      -- Sum of middle two
sorry

end NUMINAMATH_CALUDE_four_number_sequence_l3360_336090


namespace NUMINAMATH_CALUDE_lines_neither_perpendicular_nor_parallel_l3360_336011

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem lines_neither_perpendicular_nor_parallel
  (m n l : Line) (α β : Plane)
  (h1 : contained m α)
  (h2 : contained n β)
  (h3 : perpendicularPlanes α β)
  (h4 : intersect α β l)
  (h5 : ¬ perpendicular m l ∧ ¬ parallel m l)
  (h6 : ¬ perpendicular n l ∧ ¬ parallel n l) :
  ¬ perpendicular m n ∧ ¬ parallel m n :=
by
  sorry

end NUMINAMATH_CALUDE_lines_neither_perpendicular_nor_parallel_l3360_336011


namespace NUMINAMATH_CALUDE_davids_weighted_average_l3360_336048

-- Define the marks and weightages
def english_marks : ℝ := 96
def math_marks : ℝ := 95
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 97
def biology_marks : ℝ := 95

def english_weight : ℝ := 0.1
def math_weight : ℝ := 0.2
def physics_weight : ℝ := 0.3
def chemistry_weight : ℝ := 0.2
def biology_weight : ℝ := 0.2

-- Define the weighted average calculation
def weighted_average : ℝ :=
  english_marks * english_weight +
  math_marks * math_weight +
  physics_marks * physics_weight +
  chemistry_marks * chemistry_weight +
  biology_marks * biology_weight

-- Theorem statement
theorem davids_weighted_average :
  weighted_average = 91.6 := by sorry

end NUMINAMATH_CALUDE_davids_weighted_average_l3360_336048


namespace NUMINAMATH_CALUDE_reservoir_capacity_l3360_336086

theorem reservoir_capacity : 
  ∀ (C : ℝ), 
  (C / 3 + 150 = 3 * C / 4) → 
  C = 360 := by
sorry

end NUMINAMATH_CALUDE_reservoir_capacity_l3360_336086


namespace NUMINAMATH_CALUDE_vector_magnitude_l3360_336038

/-- Given plane vectors a and b with angle π/2 between them, |a| = 1, and |b| = √3, prove |2a - b| = √7 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a • b = 0) → (‖a‖ = 1) → (‖b‖ = Real.sqrt 3) → ‖2 • a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3360_336038


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3360_336052

theorem sphere_radius_ratio : 
  ∀ (r R : ℝ), 
    (4 / 3 * π * r^3 = 36 * π) → 
    (4 / 3 * π * R^3 = 450 * π) → 
    r / R = 1 / Real.rpow 12.5 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3360_336052


namespace NUMINAMATH_CALUDE_gift_wrapping_calculation_l3360_336031

/-- Represents the gift wrapping scenario for Edmund's shop. -/
structure GiftWrapping where
  wrapper_per_day : ℕ        -- inches of gift wrapper per day
  boxes_per_period : ℕ       -- number of gift boxes wrapped in a period
  days_per_period : ℕ        -- number of days in a period
  wrapper_per_box : ℕ        -- inches of gift wrapper per gift box

/-- Theorem stating the relationship between gift wrapper usage and gift boxes wrapped. -/
theorem gift_wrapping_calculation (g : GiftWrapping)
  (h1 : g.wrapper_per_day = 90)
  (h2 : g.boxes_per_period = 15)
  (h3 : g.days_per_period = 3)
  : g.wrapper_per_box = 18 := by
  sorry


end NUMINAMATH_CALUDE_gift_wrapping_calculation_l3360_336031


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3360_336054

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^2 + 2*b*c)/(b^2 + c^2) + (b^2 + 2*a*c)/(c^2 + a^2) + (c^2 + 2*a*b)/(a^2 + b^2) > 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3360_336054


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l3360_336008

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 1 →
  a 2 + a 4 = 5/2 →
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l3360_336008


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3360_336098

theorem trigonometric_simplification :
  (Real.sin (7 * π / 180) + Real.cos (15 * π / 180) * Real.sin (8 * π / 180)) /
  (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin (8 * π / 180)) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3360_336098


namespace NUMINAMATH_CALUDE_triangle_inequality_l3360_336066

theorem triangle_inequality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_triangle : (x + y - z) * (y + z - x) * (z + x - y) > 0) : 
  x * (y + z)^2 + y * (z + x)^2 + z * (x + y)^2 - (x^3 + y^3 + z^3) ≤ 9 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3360_336066


namespace NUMINAMATH_CALUDE_peaches_for_juice_l3360_336081

def total_peaches : ℝ := 7.5

def drying_percentage : ℝ := 0.3

def juice_percentage_of_remainder : ℝ := 0.4

theorem peaches_for_juice :
  let remaining_after_drying := total_peaches * (1 - drying_percentage)
  let juice_amount := remaining_after_drying * juice_percentage_of_remainder
  juice_amount = 2.1 := by sorry

end NUMINAMATH_CALUDE_peaches_for_juice_l3360_336081


namespace NUMINAMATH_CALUDE_daisy_field_count_l3360_336010

theorem daisy_field_count : ∃! n : ℕ,
  (n : ℚ) / 14 + 2 * ((n : ℚ) / 14) + 4 * ((n : ℚ) / 14) + 7000 = n ∧
  n > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_daisy_field_count_l3360_336010


namespace NUMINAMATH_CALUDE_distinct_nonneg_inequality_l3360_336084

theorem distinct_nonneg_inequality (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a^2 + b^2 + c^2 > Real.sqrt (a*b*c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_distinct_nonneg_inequality_l3360_336084


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3360_336073

theorem sqrt_equation_solution : ∃ x : ℝ, x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3360_336073


namespace NUMINAMATH_CALUDE_initial_lions_l3360_336005

/-- Proves that the initial number of lions is 100 given the conditions of the problem -/
theorem initial_lions (net_increase_per_month : ℕ) (total_increase : ℕ) (final_count : ℕ) : 
  net_increase_per_month = 4 → 
  total_increase = 48 → 
  final_count = 148 → 
  final_count - total_increase = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_lions_l3360_336005


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l3360_336095

/-- The range of a for which |x+3| - |x-1| ≤ a^2 - 3a holds for all real x -/
theorem inequality_range_theorem (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ 
  (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l3360_336095


namespace NUMINAMATH_CALUDE_complex_power_problem_l3360_336043

theorem complex_power_problem (z : ℂ) (i : ℂ) (h : i^2 = -1) (eq : (1 + z) / (1 - z) = i) : z^2019 = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l3360_336043


namespace NUMINAMATH_CALUDE_walkers_meet_at_calculated_point_l3360_336027

/-- Two people walking around a loop -/
structure WalkersOnLoop where
  loop_length : ℕ
  speed_ratio : ℕ

/-- The meeting point of two walkers -/
def meeting_point (w : WalkersOnLoop) : ℕ × ℕ :=
  (w.loop_length / (w.speed_ratio + 1), w.speed_ratio * w.loop_length / (w.speed_ratio + 1))

/-- Theorem: Walkers meet at the calculated point -/
theorem walkers_meet_at_calculated_point (w : WalkersOnLoop) 
  (h1 : w.loop_length = 24) 
  (h2 : w.speed_ratio = 3) : 
  meeting_point w = (6, 18) := by
  sorry

#eval meeting_point ⟨24, 3⟩

end NUMINAMATH_CALUDE_walkers_meet_at_calculated_point_l3360_336027


namespace NUMINAMATH_CALUDE_matt_profit_l3360_336040

/-- Represents a baseball card with its value -/
structure Card where
  value : ℕ

/-- Represents a trade of cards -/
structure Trade where
  cardsGiven : List Card
  cardsReceived : List Card

def initialCards : List Card := List.replicate 8 ⟨6⟩

def trade1 : Trade := {
  cardsGiven := [⟨6⟩, ⟨6⟩],
  cardsReceived := [⟨2⟩, ⟨2⟩, ⟨2⟩, ⟨9⟩]
}

def trade2 : Trade := {
  cardsGiven := [⟨2⟩, ⟨6⟩],
  cardsReceived := [⟨5⟩, ⟨5⟩, ⟨8⟩]
}

def trade3 : Trade := {
  cardsGiven := [⟨5⟩, ⟨9⟩],
  cardsReceived := [⟨3⟩, ⟨3⟩, ⟨3⟩, ⟨10⟩, ⟨1⟩]
}

def cardValue (c : Card) : ℕ := c.value

def tradeProfit (t : Trade) : ℤ :=
  (t.cardsReceived.map cardValue).sum - (t.cardsGiven.map cardValue).sum

theorem matt_profit :
  (tradeProfit trade1 + tradeProfit trade2 + tradeProfit trade3 : ℤ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_matt_profit_l3360_336040


namespace NUMINAMATH_CALUDE_total_albums_l3360_336080

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ
  carlos : ℕ

/-- The conditions given in the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina ∧
  a.carlos = 3 * a.miriam

/-- The theorem to prove -/
theorem total_albums (a : Albums) (h : album_conditions a) :
  a.adele + a.bridget + a.katrina + a.miriam + a.carlos = 1935 := by
  sorry

end NUMINAMATH_CALUDE_total_albums_l3360_336080


namespace NUMINAMATH_CALUDE_power_seven_mod_twelve_l3360_336069

theorem power_seven_mod_twelve : 7^93 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_twelve_l3360_336069


namespace NUMINAMATH_CALUDE_equation_solution_l3360_336094

theorem equation_solution : 
  ∃ x : ℝ, (5 * 1.6 - (2 * x) / 1.3 = 4) ∧ (x = 2.6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3360_336094


namespace NUMINAMATH_CALUDE_total_paid_equals_143_l3360_336025

def manicure_cost : ℝ := 30
def pedicure_cost : ℝ := 40
def hair_treatment_cost : ℝ := 50

def manicure_tip_rate : ℝ := 0.25
def pedicure_tip_rate : ℝ := 0.20
def hair_treatment_tip_rate : ℝ := 0.15

def total_cost (service_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  service_cost * (1 + tip_rate)

theorem total_paid_equals_143 :
  total_cost manicure_cost manicure_tip_rate +
  total_cost pedicure_cost pedicure_tip_rate +
  total_cost hair_treatment_cost hair_treatment_tip_rate = 143 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_equals_143_l3360_336025


namespace NUMINAMATH_CALUDE_problem_statement_l3360_336012

theorem problem_statement (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 6) :
  a * b^2 - a^2 * b = -24 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3360_336012


namespace NUMINAMATH_CALUDE_greatest_two_digit_prime_saturated_is_98_l3360_336006

/-- A number is prime saturated if the product of all its different positive prime factors
    is less than its square root -/
def IsPrimeSaturated (n : ℕ) : Prop :=
  (Finset.prod (Nat.factors n).toFinset id) < Real.sqrt (n : ℝ)

/-- The greatest two-digit prime saturated integer -/
def GreatestTwoDigitPrimeSaturated : ℕ := 98

theorem greatest_two_digit_prime_saturated_is_98 :
  IsPrimeSaturated GreatestTwoDigitPrimeSaturated ∧
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ IsPrimeSaturated n → n ≤ GreatestTwoDigitPrimeSaturated :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_prime_saturated_is_98_l3360_336006


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_expression_l3360_336085

def is_integer_expression (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.factorial (n^3 - 1)) = k * (Nat.factorial n)^(n^2)

theorem unique_integer_satisfying_expression :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ is_integer_expression n :=
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_expression_l3360_336085


namespace NUMINAMATH_CALUDE_stone_piles_sum_l3360_336013

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- Conditions for the stone piles problem -/
def validStonePiles (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = p.pile2 / 2

theorem stone_piles_sum (p : StonePiles) (h : validStonePiles p) :
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

#check stone_piles_sum

end NUMINAMATH_CALUDE_stone_piles_sum_l3360_336013


namespace NUMINAMATH_CALUDE_quadratic_root_l3360_336063

theorem quadratic_root (a b c : ℝ) (h_arithmetic : b - a = c - b) 
  (h_a : a = 5) (h_c : c = 1) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_one_root : ∃! x, a * x^2 + b * x + c = 0) : 
  ∃ x, a * x^2 + b * x + c = 0 ∧ x = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_l3360_336063


namespace NUMINAMATH_CALUDE_zachary_cans_l3360_336072

def can_sequence (n : ℕ) : ℕ := 4 + 5 * (n - 1)

theorem zachary_cans : can_sequence 7 = 34 := by
  sorry

end NUMINAMATH_CALUDE_zachary_cans_l3360_336072


namespace NUMINAMATH_CALUDE_drug_efficacy_rate_l3360_336030

/-- Calculates the efficacy rate of a drug based on a survey --/
def efficacyRate (totalSamples : ℕ) (positiveResponses : ℕ) : ℚ :=
  (positiveResponses : ℚ) / (totalSamples : ℚ)

theorem drug_efficacy_rate :
  let totalSamples : ℕ := 20
  let positiveResponses : ℕ := 16
  efficacyRate totalSamples positiveResponses = 4/5 := by
sorry

end NUMINAMATH_CALUDE_drug_efficacy_rate_l3360_336030


namespace NUMINAMATH_CALUDE_total_berets_is_eleven_l3360_336007

def spools_per_beret : ℕ := 3

def red_spools : ℕ := 12
def black_spools : ℕ := 15
def blue_spools : ℕ := 6

def berets_from_spools (spools : ℕ) : ℕ := spools / spools_per_beret

theorem total_berets_is_eleven :
  berets_from_spools red_spools + berets_from_spools black_spools + berets_from_spools blue_spools = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_berets_is_eleven_l3360_336007


namespace NUMINAMATH_CALUDE_matrix_power_property_l3360_336034

theorem matrix_power_property (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A.mulVec (![5, -2]) = ![(-15), 6] →
  (A ^ 5).mulVec (![5, -2]) = ![(-1215), 486] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_property_l3360_336034


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3360_336003

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∃ (x : ℤ), 12 * x^2 - m * x + 432 = 0) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬∃ (y : ℤ), 12 * y^2 - k * y + 432 = 0) ∧
  m = 144 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3360_336003


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l3360_336001

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (chocolate_pies marshmallow_pies cayenne_pies walnut_pies : ℕ) :
  total_pies = 48 →
  chocolate_pies ≥ 16 →
  marshmallow_pies = 24 →
  cayenne_pies = 36 →
  walnut_pies ≥ 6 →
  ∃ (pies_without_ingredients : ℕ),
    pies_without_ingredients ≤ 12 ∧
    pies_without_ingredients + chocolate_pies + marshmallow_pies + cayenne_pies + walnut_pies ≥ total_pies :=
by sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l3360_336001


namespace NUMINAMATH_CALUDE_circle_radius_from_spherical_coords_l3360_336028

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) is √2/2 -/
theorem circle_radius_from_spherical_coords :
  let r : ℝ := Real.sqrt 2 / 2
  ∀ θ : ℝ,
  let x : ℝ := Real.sin (π/4 : ℝ) * Real.cos θ
  let y : ℝ := Real.sin (π/4 : ℝ) * Real.sin θ
  Real.sqrt (x^2 + y^2) = r :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_spherical_coords_l3360_336028


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l3360_336099

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_initial_investment : ℕ
  b_investment : ℕ
  a_investment_duration : ℕ
  b_investment_duration : ℕ

/-- Calculates the effective capital contribution -/
def effective_capital (investment : ℕ) (duration : ℕ) : ℕ :=
  investment * duration

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- Theorem stating that the profit sharing ratio is 2:3 given the conditions -/
theorem profit_sharing_ratio (p : Partnership) 
  (h1 : p.a_initial_investment = 4500)
  (h2 : p.b_investment = 16200)
  (h3 : p.a_investment_duration = 12)
  (h4 : p.b_investment_duration = 5) :
  simplify_ratio 
    (effective_capital p.a_initial_investment p.a_investment_duration)
    (effective_capital p.b_investment p.b_investment_duration) = (2, 3) := by
  sorry


end NUMINAMATH_CALUDE_profit_sharing_ratio_l3360_336099


namespace NUMINAMATH_CALUDE_bus_children_difference_l3360_336059

theorem bus_children_difference (initial : ℕ) (got_off : ℕ) (final : ℕ) :
  initial = 5 → got_off = 63 → final = 14 →
  ∃ (got_on : ℕ), got_on - got_off = 9 ∧ initial - got_off + got_on = final :=
by sorry

end NUMINAMATH_CALUDE_bus_children_difference_l3360_336059


namespace NUMINAMATH_CALUDE_distribute_6_5_l3360_336023

def distribute (n m : ℕ) : ℕ := 
  Nat.choose (m - 1) (n - m)

theorem distribute_6_5 : distribute 6 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_5_l3360_336023
