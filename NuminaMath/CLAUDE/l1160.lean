import Mathlib

namespace NUMINAMATH_CALUDE_parabola_equation_l1160_116089

/-- Given a point A(1,1) and a parabola C: y^2 = 2px (p > 0) whose focus lies on the perpendicular
    bisector of OA, prove that the equation of the parabola C is y^2 = 4x. -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) : 
  let A : ℝ × ℝ := (1, 1)
  let O : ℝ × ℝ := (0, 0)
  let perpendicular_bisector := {(x, y) : ℝ × ℝ | x + y = 1}
  let focus : ℝ × ℝ := (p / 2, 0)
  focus ∈ perpendicular_bisector →
  ∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 4*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1160_116089


namespace NUMINAMATH_CALUDE_dress_ratio_proof_l1160_116016

/-- Proves that the ratio of Melissa's dresses to Emily's dresses is 1:2 -/
theorem dress_ratio_proof (melissa debora emily : ℕ) : 
  debora = melissa + 12 →
  emily = 16 →
  melissa + debora + emily = 44 →
  melissa = emily / 2 := by
sorry

end NUMINAMATH_CALUDE_dress_ratio_proof_l1160_116016


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1160_116047

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 3) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1160_116047


namespace NUMINAMATH_CALUDE_flour_needed_for_cake_l1160_116032

/-- Given a recipe that requires a certain amount of flour and some flour already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- The problem statement -/
theorem flour_needed_for_cake : remaining_flour 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_cake_l1160_116032


namespace NUMINAMATH_CALUDE_cylinder_height_l1160_116045

/-- The height of a right cylinder with radius 3 feet and surface area 36π square feet is 3 feet. -/
theorem cylinder_height (π : ℝ) (h : ℝ) : 
  2 * π * 3^2 + 2 * π * 3 * h = 36 * π → h = 3 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_l1160_116045


namespace NUMINAMATH_CALUDE_max_sides_convex_polygon_l1160_116014

/-- The maximum number of sides in a convex polygon with interior angles in arithmetic sequence -/
theorem max_sides_convex_polygon (n : ℕ) : 
  n > 0 → -- n is positive
  n ≤ 8 ∧ -- n is at most 8
  let interior_angle (i : ℕ) := 100 + (i - 1) * 10 -- Definition of interior angles
  ∀ i, i ≤ n → interior_angle i < 180 ∧ -- All angles are less than 180°
  (∀ i j, i < j → j ≤ n → interior_angle j - interior_angle i = (j - i) * 10) → -- Arithmetic sequence condition
  ¬∃ m : ℕ, m > n ∧ m ≤ 8 ∧ 
    (∀ i, i ≤ m → interior_angle i < 180) ∧
    (∀ i j, i < j → j ≤ m → interior_angle j - interior_angle i = (j - i) * 10) :=
by sorry

end NUMINAMATH_CALUDE_max_sides_convex_polygon_l1160_116014


namespace NUMINAMATH_CALUDE_min_difference_l1160_116004

open Real

noncomputable def f (x : ℝ) : ℝ := exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := 1/2 + log (x/2)

theorem min_difference (a b : ℝ) (h : f a = g b) :
  ∃ (min : ℝ), min = 1 + log 2 ∧ ∀ (a' b' : ℝ), f a' = g b' → b' - a' ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_difference_l1160_116004


namespace NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l1160_116034

theorem percent_to_decimal (p : ℝ) : p / 100 = p * 0.01 := by sorry

theorem three_percent_to_decimal : (3 : ℝ) / 100 = 0.03 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l1160_116034


namespace NUMINAMATH_CALUDE_function_equation_solution_l1160_116003

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) →
  (∀ x : ℝ, f x = (1/3) * (x^2 + 2*x - 1)) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1160_116003


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l1160_116090

/-- Calculates the dividend percentage given investment details and dividend received -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 840.0000000000001) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 7 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l1160_116090


namespace NUMINAMATH_CALUDE_eve_distance_difference_l1160_116033

/-- Eve's running and walking distances problem -/
theorem eve_distance_difference :
  let run_distance : ℝ := 0.7
  let walk_distance : ℝ := 0.6
  run_distance - walk_distance = 0.1 := by sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l1160_116033


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1160_116051

/-- A convex polyhedron inscribed around a sphere -/
structure InscribedPolyhedron where
  -- The radius of the inscribed sphere
  r : ℝ
  -- The surface area of the polyhedron
  surface_area : ℝ
  -- The volume of the polyhedron
  volume : ℝ
  -- Assumption that the polyhedron is inscribed around the sphere
  inscribed : True

/-- 
Theorem: For any convex polyhedron inscribed around a sphere,
the ratio of its volume to its surface area is equal to r/3,
where r is the radius of the inscribed sphere.
-/
theorem volume_to_surface_area_ratio (P : InscribedPolyhedron) :
  P.volume / P.surface_area = P.r / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1160_116051


namespace NUMINAMATH_CALUDE_russian_alphabet_symmetry_partition_l1160_116066

-- Define the set of Russian alphabet letters
inductive RussianLetter
| A | B | V | G | D | E | Zh | Z | I | K | L | M | N | O | P | R | S | T | U | F | Kh | Ts | Ch | Sh | Shch | Eh | Yu | Ya

-- Define symmetry types
inductive SymmetryType
| Vertical
| Horizontal
| Central
| All
| None

-- Define a function that assigns a symmetry type to each letter
def letterSymmetry : RussianLetter → SymmetryType
| RussianLetter.A => SymmetryType.Vertical
| RussianLetter.D => SymmetryType.Vertical
| RussianLetter.M => SymmetryType.Vertical
| RussianLetter.P => SymmetryType.Vertical
| RussianLetter.T => SymmetryType.Vertical
| RussianLetter.Sh => SymmetryType.Vertical
| RussianLetter.V => SymmetryType.Horizontal
| RussianLetter.E => SymmetryType.Horizontal
| RussianLetter.Z => SymmetryType.Horizontal
| RussianLetter.K => SymmetryType.Horizontal
| RussianLetter.S => SymmetryType.Horizontal
| RussianLetter.Eh => SymmetryType.Horizontal
| RussianLetter.Yu => SymmetryType.Horizontal
| RussianLetter.I => SymmetryType.Central
| RussianLetter.Zh => SymmetryType.All
| RussianLetter.N => SymmetryType.All
| RussianLetter.O => SymmetryType.All
| RussianLetter.F => SymmetryType.All
| RussianLetter.Kh => SymmetryType.All
| _ => SymmetryType.None

-- Define the five groups
def group1 := {l : RussianLetter | letterSymmetry l = SymmetryType.Vertical}
def group2 := {l : RussianLetter | letterSymmetry l = SymmetryType.Horizontal}
def group3 := {l : RussianLetter | letterSymmetry l = SymmetryType.Central}
def group4 := {l : RussianLetter | letterSymmetry l = SymmetryType.All}
def group5 := {l : RussianLetter | letterSymmetry l = SymmetryType.None}

-- Theorem: The groups form a partition of the Russian alphabet
theorem russian_alphabet_symmetry_partition :
  (∀ l : RussianLetter, l ∈ group1 ∨ l ∈ group2 ∨ l ∈ group3 ∨ l ∈ group4 ∨ l ∈ group5) ∧
  (group1 ∩ group2 = ∅) ∧ (group1 ∩ group3 = ∅) ∧ (group1 ∩ group4 = ∅) ∧ (group1 ∩ group5 = ∅) ∧
  (group2 ∩ group3 = ∅) ∧ (group2 ∩ group4 = ∅) ∧ (group2 ∩ group5 = ∅) ∧
  (group3 ∩ group4 = ∅) ∧ (group3 ∩ group5 = ∅) ∧
  (group4 ∩ group5 = ∅) :=
sorry

end NUMINAMATH_CALUDE_russian_alphabet_symmetry_partition_l1160_116066


namespace NUMINAMATH_CALUDE_alice_stool_height_l1160_116015

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height bulb_below_ceiling alice_height alice_reach : ℝ) : ℝ :=
  ceiling_height - bulb_below_ceiling - (alice_height + alice_reach)

theorem alice_stool_height :
  let ceiling_height : ℝ := 2.8 * 100  -- in cm
  let bulb_below_ceiling : ℝ := 15     -- in cm
  let alice_height : ℝ := 1.5 * 100    -- in cm
  let alice_reach : ℝ := 30            -- in cm
  stool_height ceiling_height bulb_below_ceiling alice_height alice_reach = 85 := by
  sorry

#eval stool_height (2.8 * 100) 15 (1.5 * 100) 30

end NUMINAMATH_CALUDE_alice_stool_height_l1160_116015


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1160_116057

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1160_116057


namespace NUMINAMATH_CALUDE_bhanu_petrol_expense_l1160_116024

def bhanu_expenditure (total_income : ℝ) : Prop :=
  let petrol_percent : ℝ := 0.30
  let rent_percent : ℝ := 0.30
  let petrol_expense : ℝ := petrol_percent * total_income
  let remaining_after_petrol : ℝ := total_income - petrol_expense
  let rent_expense : ℝ := rent_percent * remaining_after_petrol
  rent_expense = 210 ∧ petrol_expense = 300

theorem bhanu_petrol_expense : 
  ∃ (total_income : ℝ), bhanu_expenditure total_income :=
sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expense_l1160_116024


namespace NUMINAMATH_CALUDE_clarence_oranges_l1160_116005

-- Define the initial number of oranges
def initial_oranges : ℝ := 5.0

-- Define the number of oranges given away
def oranges_given : ℝ := 3.0

-- Define the number of Skittles bought (not used in the calculation, but mentioned in the problem)
def skittles_bought : ℝ := 9.0

-- Define the function to calculate the remaining oranges
def remaining_oranges : ℝ := initial_oranges - oranges_given

-- Theorem to prove
theorem clarence_oranges : remaining_oranges = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_l1160_116005


namespace NUMINAMATH_CALUDE_haley_shirts_l1160_116061

/-- The number of shirts Haley bought -/
def shirts_bought : ℕ := 11

/-- The number of shirts Haley returned -/
def shirts_returned : ℕ := 6

/-- The number of shirts Haley ended up with -/
def shirts_remaining : ℕ := shirts_bought - shirts_returned

theorem haley_shirts : shirts_remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_shirts_l1160_116061


namespace NUMINAMATH_CALUDE_smallest_n_cube_plus_2square_eq_odd_square_l1160_116095

theorem smallest_n_cube_plus_2square_eq_odd_square : 
  (∀ n : ℕ, 0 < n → n < 7 → ¬∃ k : ℕ, k % 2 = 1 ∧ n^3 + 2*n^2 = k^2) ∧
  (∃ k : ℕ, k % 2 = 1 ∧ 7^3 + 2*7^2 = k^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_cube_plus_2square_eq_odd_square_l1160_116095


namespace NUMINAMATH_CALUDE_expected_trait_count_is_forty_l1160_116092

/-- The probability of an individual having the genetic trait -/
def trait_probability : ℚ := 1 / 8

/-- The total number of people in the sample -/
def sample_size : ℕ := 320

/-- The expected number of people with the genetic trait in the sample -/
def expected_trait_count : ℚ := trait_probability * sample_size

theorem expected_trait_count_is_forty : expected_trait_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_expected_trait_count_is_forty_l1160_116092


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l1160_116055

theorem sufficient_not_necessary_negation 
  (p q : Prop) 
  (h1 : ¬p → q)  -- ¬p is sufficient for q
  (h2 : ¬(q → ¬p)) -- ¬p is not necessary for q
  : (¬q → p) ∧ ¬(p → ¬q) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l1160_116055


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_complement_A_union_B_A_intersection_complement_B_complement_A_union_complement_B_l1160_116070

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for complement of A ∪ B in U
theorem complement_union_A_B : (A ∪ B)ᶜ = {x | x < -3 ∨ 3 ≤ x} ∩ U := by sorry

-- Theorem for (complement of A in U) ∪ B
theorem complement_A_union_B : Aᶜ ∪ B = {x | x ≤ 2 ∨ 3 ≤ x} ∩ U := by sorry

-- Theorem for A ∩ (complement of B in U)
theorem A_intersection_complement_B : A ∩ Bᶜ = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for (complement of A in U) ∪ (complement of B in U)
theorem complement_A_union_complement_B : Aᶜ ∪ Bᶜ = {x | x ≤ -2 ∨ 2 < x} ∩ U := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_complement_A_union_B_A_intersection_complement_B_complement_A_union_complement_B_l1160_116070


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1160_116008

theorem cube_equation_solution : ∃ (N : ℕ), N > 0 ∧ 26^3 * 65^3 = 10^3 * N^3 ∧ N = 169 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1160_116008


namespace NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_equals_one_l1160_116028

theorem same_solution_implies_a_plus_b_equals_one 
  (x y a b : ℝ) 
  (h1 : 2*x + 4*y = 20) 
  (h2 : a*x + b*y = 1)
  (h3 : 2*x - y = 5)
  (h4 : b*x + a*y = 6)
  (h5 : 2*x + 4*y = 20 ∧ a*x + b*y = 1 ↔ 2*x - y = 5 ∧ b*x + a*y = 6) : 
  a + b = 1 := by
sorry


end NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_equals_one_l1160_116028


namespace NUMINAMATH_CALUDE_five_athletes_three_events_l1160_116044

/-- The number of different ways athletes can win championships in events -/
def championship_ways (num_athletes : ℕ) (num_events : ℕ) : ℕ :=
  num_athletes ^ num_events

/-- Theorem: 5 athletes winning 3 events results in 5^3 different ways -/
theorem five_athletes_three_events : 
  championship_ways 5 3 = 5^3 := by
  sorry

end NUMINAMATH_CALUDE_five_athletes_three_events_l1160_116044


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l1160_116025

/-- The number of lanterns along the alley -/
def num_lanterns : ℕ := 100

/-- The position where Petya is observed -/
def petya_observed : ℕ := 22

/-- The position where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The function to calculate the meeting point of Petya and Vasya -/
def meeting_point (n l p v : ℕ) : ℕ :=
  ((n - 1) - (l - p)) + 1

theorem petya_vasya_meeting :
  meeting_point num_lanterns petya_observed vasya_observed 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_meeting_l1160_116025


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1160_116043

theorem fractional_equation_solution (x : ℝ) :
  x ≠ 2 → x ≠ 0 → (1 / (x - 2) = 3 / x) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1160_116043


namespace NUMINAMATH_CALUDE_line_equation_proof_l1160_116026

/-- Proves that the equation of a line with a slope angle of 135° and a y-intercept of -1 is y = -x - 1 -/
theorem line_equation_proof (x y : ℝ) : 
  (∃ (k b : ℝ), k = Real.tan (135 * π / 180) ∧ b = -1 ∧ y = k * x + b) ↔ y = -x - 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1160_116026


namespace NUMINAMATH_CALUDE_problem_statement_l1160_116011

theorem problem_statement (a b c : ℝ) :
  (∀ c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1160_116011


namespace NUMINAMATH_CALUDE_infinitely_many_lines_through_lattice_points_l1160_116009

/-- A line passing through the point (10, 1/2) -/
structure LineThrough10Half where
  slope : ℤ
  intercept : ℚ
  eq : intercept = 1/2 - 10 * slope

/-- A lattice point is a point with integer coordinates -/
def LatticePoint (x y : ℤ) : Prop := True

/-- A line passes through a lattice point -/
def PassesThroughLatticePoint (line : LineThrough10Half) (x y : ℤ) : Prop :=
  y = line.slope * x + line.intercept

theorem infinitely_many_lines_through_lattice_points :
  ∃ (f : ℕ → LineThrough10Half),
    (∀ n : ℕ, ∃ (x₁ y₁ x₂ y₂ : ℤ), 
      x₁ ≠ x₂ ∧ 
      LatticePoint x₁ y₁ ∧ 
      LatticePoint x₂ y₂ ∧ 
      PassesThroughLatticePoint (f n) x₁ y₁ ∧ 
      PassesThroughLatticePoint (f n) x₂ y₂) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_lines_through_lattice_points_l1160_116009


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1160_116007

/-- For the quadratic x^2 + 2200x + 4200, when written in the form (x+b)^2 + c, c/b = -1096 -/
theorem quadratic_form_ratio : ∃ (b c : ℝ), 
  (∀ x, x^2 + 2200*x + 4200 = (x + b)^2 + c) ∧ 
  c / b = -1096 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1160_116007


namespace NUMINAMATH_CALUDE_additional_hour_rate_is_ten_l1160_116080

/-- Represents the rental cost structure for a power tool -/
structure RentalCost where
  firstHourRate : ℝ
  additionalHourRate : ℝ
  totalHours : ℕ
  totalCost : ℝ

/-- Theorem stating that given the rental conditions, the additional hour rate is $10 -/
theorem additional_hour_rate_is_ten
  (rental : RentalCost)
  (h1 : rental.firstHourRate = 25)
  (h2 : rental.totalHours = 11)
  (h3 : rental.totalCost = 125)
  : rental.additionalHourRate = 10 := by
  sorry

#check additional_hour_rate_is_ten

end NUMINAMATH_CALUDE_additional_hour_rate_is_ten_l1160_116080


namespace NUMINAMATH_CALUDE_minute_hand_half_circle_time_l1160_116056

/-- Represents the number of small divisions on a clock face -/
def clock_divisions : ℕ := 60

/-- Represents the number of minutes the minute hand moves for each small division -/
def minutes_per_division : ℕ := 1

/-- Represents the number of small divisions in half a circle -/
def half_circle_divisions : ℕ := 30

/-- Represents half an hour in minutes -/
def half_hour_minutes : ℕ := 30

theorem minute_hand_half_circle_time :
  half_circle_divisions * minutes_per_division = half_hour_minutes :=
sorry

end NUMINAMATH_CALUDE_minute_hand_half_circle_time_l1160_116056


namespace NUMINAMATH_CALUDE_base4_to_base10_conversion_l1160_116039

/-- Converts a base 4 number represented as a list of digits to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number -/
def base4Number : List Nat := [2, 1, 0, 1, 2]

theorem base4_to_base10_conversion :
  base4ToBase10 base4Number = 582 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_base10_conversion_l1160_116039


namespace NUMINAMATH_CALUDE_food_drive_mark_cans_l1160_116069

/-- Represents the number of cans brought by each person -/
structure Cans where
  rachel : ℕ
  jaydon : ℕ
  mark : ℕ
  sophie : ℕ

/-- Conditions for the food drive -/
def FoodDriveConditions (c : Cans) : Prop :=
  c.mark = 4 * c.jaydon ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  4 * c.sophie = 3 * c.jaydon ∧
  c.rachel ≥ 5 ∧ c.jaydon ≥ 5 ∧ c.mark ≥ 5 ∧ c.sophie ≥ 5 ∧
  Odd (c.rachel + c.jaydon + c.mark + c.sophie) ∧
  c.rachel + c.jaydon + c.mark + c.sophie ≥ 250

theorem food_drive_mark_cans (c : Cans) (h : FoodDriveConditions c) : c.mark = 148 := by
  sorry

end NUMINAMATH_CALUDE_food_drive_mark_cans_l1160_116069


namespace NUMINAMATH_CALUDE_faster_train_speed_l1160_116058

/-- Proves that the speed of the faster train is 50 km/hr given the problem conditions -/
theorem faster_train_speed 
  (speed_diff : ℝ) 
  (faster_train_length : ℝ) 
  (passing_time : ℝ) :
  speed_diff = 32 →
  faster_train_length = 75 →
  passing_time = 15 →
  ∃ (slower_speed faster_speed : ℝ),
    faster_speed - slower_speed = speed_diff ∧
    faster_train_length / passing_time * 3.6 = speed_diff ∧
    faster_speed = 50 := by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l1160_116058


namespace NUMINAMATH_CALUDE_mistake_correction_l1160_116053

theorem mistake_correction (x : ℤ) : x - 23 = 4 → x * 23 = 621 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l1160_116053


namespace NUMINAMATH_CALUDE_cosine_sum_upper_bound_l1160_116038

theorem cosine_sum_upper_bound (α β γ : Real) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 2) : 
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_upper_bound_l1160_116038


namespace NUMINAMATH_CALUDE_alley_width_equals_height_l1160_116085

/-- Represents a ladder in an alley scenario -/
structure LadderInAlley where
  a : ℝ  -- length of the ladder
  w : ℝ  -- width of the alley
  k : ℝ  -- height at 45° angle
  h : ℝ  -- height at 75° angle

/-- The theorem stating that the width of the alley equals the height h -/
theorem alley_width_equals_height (l : LadderInAlley) : 
  l.w = l.h ∧ 
  Real.cos (45 * π / 180) * l.a = l.k ∧ 
  Real.cos (75 * π / 180) * l.a = l.h :=
by sorry

end NUMINAMATH_CALUDE_alley_width_equals_height_l1160_116085


namespace NUMINAMATH_CALUDE_inequality_properties_l1160_116002

theorem inequality_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1 / a < 1 / b) := by sorry

end NUMINAMATH_CALUDE_inequality_properties_l1160_116002


namespace NUMINAMATH_CALUDE_plant_supplier_money_left_l1160_116035

/-- Represents the plant supplier's business --/
structure PlantSupplier where
  orchids : ℕ
  orchidPrice : ℕ
  moneyPlants : ℕ
  moneyPlantPrice : ℕ
  bonsai : ℕ
  bonsaiPrice : ℕ
  cacti : ℕ
  cactiPrice : ℕ
  airPlants : ℕ
  airPlantPrice : ℕ
  fullTimeWorkers : ℕ
  fullTimeWage : ℕ
  partTimeWorkers : ℕ
  partTimeWage : ℕ
  ceramicPotsCost : ℕ
  plasticPotsCost : ℕ
  fertilizersCost : ℕ
  toolsCost : ℕ
  utilityBill : ℕ
  tax : ℕ

/-- Calculates the total earnings of the plant supplier --/
def totalEarnings (s : PlantSupplier) : ℕ :=
  s.orchids * s.orchidPrice +
  s.moneyPlants * s.moneyPlantPrice +
  s.bonsai * s.bonsaiPrice +
  s.cacti * s.cactiPrice +
  s.airPlants * s.airPlantPrice

/-- Calculates the total expenses of the plant supplier --/
def totalExpenses (s : PlantSupplier) : ℕ :=
  s.fullTimeWorkers * s.fullTimeWage +
  s.partTimeWorkers * s.partTimeWage +
  s.ceramicPotsCost +
  s.plasticPotsCost +
  s.fertilizersCost +
  s.toolsCost +
  s.utilityBill +
  s.tax

/-- Calculates the money left from the plant supplier's earnings --/
def moneyLeft (s : PlantSupplier) : ℕ :=
  totalEarnings s - totalExpenses s

/-- Theorem stating that the money left is $3755 given the specified conditions --/
theorem plant_supplier_money_left :
  ∃ (s : PlantSupplier),
    s.orchids = 35 ∧ s.orchidPrice = 52 ∧
    s.moneyPlants = 30 ∧ s.moneyPlantPrice = 32 ∧
    s.bonsai = 20 ∧ s.bonsaiPrice = 77 ∧
    s.cacti = 25 ∧ s.cactiPrice = 22 ∧
    s.airPlants = 40 ∧ s.airPlantPrice = 15 ∧
    s.fullTimeWorkers = 3 ∧ s.fullTimeWage = 65 ∧
    s.partTimeWorkers = 2 ∧ s.partTimeWage = 45 ∧
    s.ceramicPotsCost = 280 ∧
    s.plasticPotsCost = 150 ∧
    s.fertilizersCost = 100 ∧
    s.toolsCost = 125 ∧
    s.utilityBill = 225 ∧
    s.tax = 550 ∧
    moneyLeft s = 3755 := by
  sorry

end NUMINAMATH_CALUDE_plant_supplier_money_left_l1160_116035


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_half_l1160_116022

theorem angle_sum_is_pi_half (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1) 
  (h_eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) : 
  α + 2 * β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_half_l1160_116022


namespace NUMINAMATH_CALUDE_trigonometric_calculation_and_algebraic_simplification_l1160_116060

theorem trigonometric_calculation_and_algebraic_simplification :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (30 * π / 180) + |(-1/2)| = 1) ∧
  (let a := 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180)
   let b := 3
   1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_and_algebraic_simplification_l1160_116060


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1160_116068

/-- The dimensions of a rectangle satisfying specific conditions -/
theorem rectangle_dimensions :
  ∀ x y : ℝ,
  x > 0 ∧ y > 0 →  -- Ensure positive dimensions
  y = 2 * x →      -- Length is twice the width
  2 * (x + y) = 2 * (x * y) →  -- Perimeter is twice the area
  (x, y) = (3/2, 3) := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1160_116068


namespace NUMINAMATH_CALUDE_sequence_characterization_l1160_116010

theorem sequence_characterization (a : ℕ+ → ℝ) :
  (∀ m n : ℕ+, a (m + n) = a m + a n - (m * n : ℝ)) ∧
  (∀ m n : ℕ+, a (m * n) = (m ^ 2 : ℝ) * a n + (n ^ 2 : ℝ) * a m + 2 * a m * a n) →
  (∀ n : ℕ+, a n = -(n * (n - 1) : ℝ) / 2) ∨
  (∀ n : ℕ+, a n = -(n ^ 2 : ℝ) / 2) := by
sorry

end NUMINAMATH_CALUDE_sequence_characterization_l1160_116010


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1160_116030

theorem fraction_evaluation (x : ℝ) (h : x = 7) :
  (x^6 - 25*x^3 + 144) / (x^3 - 12) = 312 / 331 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1160_116030


namespace NUMINAMATH_CALUDE_range_of_a_l1160_116064

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*x + 3 ≤ a^2 - 2*a - 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem statement
theorem range_of_a : 
  (∀ a : ℝ, solution_set a = ∅) ↔ (∀ a : ℝ, -1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1160_116064


namespace NUMINAMATH_CALUDE_security_breach_likely_and_measures_needed_l1160_116076

/-- Represents the security level of an online transaction -/
inductive SecurityLevel
| Low
| Medium
| High

/-- Represents the actions taken by the user -/
structure UserActions where
  clickedSuspiciousEmail : Bool
  enteredSensitiveInfo : Bool
  usedUnofficialWebsite : Bool
  enteredSMSPassword : Bool

/-- Represents additional security measures -/
structure SecurityMeasures where
  useSecureNetworks : Bool
  useAntivirus : Bool
  updateApplications : Bool
  checkAddressBar : Bool
  useStrongPasswords : Bool
  use2FA : Bool

/-- Determines the security level based on user actions -/
def determineSecurityLevel (actions : UserActions) : SecurityLevel :=
  if actions.clickedSuspiciousEmail && actions.enteredSensitiveInfo && 
     actions.usedUnofficialWebsite && actions.enteredSMSPassword then
    SecurityLevel.Low
  else if actions.clickedSuspiciousEmail || actions.enteredSensitiveInfo || 
          actions.usedUnofficialWebsite || actions.enteredSMSPassword then
    SecurityLevel.Medium
  else
    SecurityLevel.High

/-- Checks if additional security measures are sufficient -/
def areMeasuresSufficient (measures : SecurityMeasures) : Bool :=
  measures.useSecureNetworks && measures.useAntivirus && measures.updateApplications &&
  measures.checkAddressBar && measures.useStrongPasswords && measures.use2FA

/-- Theorem: Given the user's actions, the security level is low and additional measures are necessary -/
theorem security_breach_likely_and_measures_needed 
  (actions : UserActions)
  (measures : SecurityMeasures)
  (h1 : actions.clickedSuspiciousEmail = true)
  (h2 : actions.enteredSensitiveInfo = true)
  (h3 : actions.usedUnofficialWebsite = true)
  (h4 : actions.enteredSMSPassword = true) :
  determineSecurityLevel actions = SecurityLevel.Low ∧ 
  areMeasuresSufficient measures = true :=
by sorry


end NUMINAMATH_CALUDE_security_breach_likely_and_measures_needed_l1160_116076


namespace NUMINAMATH_CALUDE_complex_multiplication_l1160_116048

theorem complex_multiplication :
  (1 + Complex.I) * (2 - Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1160_116048


namespace NUMINAMATH_CALUDE_coin_difference_l1160_116075

def coin_values : List ℕ := [5, 10, 25, 50]

def target_amount : ℕ := 75

def min_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

def max_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

theorem coin_difference :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 13 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l1160_116075


namespace NUMINAMATH_CALUDE_investment_growth_proof_l1160_116018

/-- The initial investment amount that results in $132 after two years with given growth rates and addition --/
def initial_investment : ℝ := 80

/-- The growth rate for the first year --/
def first_year_growth_rate : ℝ := 0.15

/-- The amount added after the first year --/
def added_amount : ℝ := 28

/-- The growth rate for the second year --/
def second_year_growth_rate : ℝ := 0.10

/-- The final portfolio value after two years --/
def final_value : ℝ := 132

theorem investment_growth_proof :
  ((1 + first_year_growth_rate) * initial_investment + added_amount) * 
  (1 + second_year_growth_rate) = final_value := by
  sorry

#eval initial_investment

end NUMINAMATH_CALUDE_investment_growth_proof_l1160_116018


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l1160_116037

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ colored_lipstick : ℕ, colored_lipstick = total_students / 2)
  (h3 : ∃ red_lipstick : ℕ, red_lipstick = (total_students / 2) / 4)
  (h4 : ∃ blue_lipstick : ℕ, blue_lipstick = ((total_students / 2) / 4) / 5) :
  ∃ blue_lipstick : ℕ, blue_lipstick = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l1160_116037


namespace NUMINAMATH_CALUDE_locus_of_centers_l1160_116006

/-- The locus of centers of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)) →
  8 * a^2 + 9 * b^2 - 16 * a - 64 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1160_116006


namespace NUMINAMATH_CALUDE_remaining_pets_count_l1160_116071

/-- Represents the number of pets of each type -/
structure PetCounts where
  puppies : ℕ
  kittens : ℕ
  rabbits : ℕ
  guineaPigs : ℕ
  chameleons : ℕ
  parrots : ℕ

/-- Calculates the total number of pets -/
def totalPets (counts : PetCounts) : ℕ :=
  counts.puppies + counts.kittens + counts.rabbits + counts.guineaPigs + counts.chameleons + counts.parrots

/-- Represents the pet store transactions throughout the day -/
def petStoreTransactions (initial : PetCounts) : PetCounts :=
  { puppies := initial.puppies - 2 - 1 + 3 - 1 - 1,
    kittens := initial.kittens - 1 - 2 + 2 - 1 + 1 - 1,
    rabbits := initial.rabbits - 1 - 1 + 1 - 1 - 1,
    guineaPigs := initial.guineaPigs - 1 - 2 - 1 - 1,
    chameleons := initial.chameleons + 1 + 2 - 1,
    parrots := initial.parrots - 1 }

/-- The main theorem stating that after all transactions, 16 pets remain -/
theorem remaining_pets_count (initial : PetCounts)
    (h_initial : initial = { puppies := 7, kittens := 6, rabbits := 4,
                             guineaPigs := 5, chameleons := 3, parrots := 2 }) :
    totalPets (petStoreTransactions initial) = 16 := by
  sorry


end NUMINAMATH_CALUDE_remaining_pets_count_l1160_116071


namespace NUMINAMATH_CALUDE_brandon_squirrel_count_l1160_116021

/-- The number of squirrels Brandon can catch in an hour -/
def S : ℕ := sorry

/-- The number of rabbits Brandon can catch in an hour -/
def R : ℕ := 2

/-- The calorie content of a squirrel -/
def squirrel_calories : ℕ := 300

/-- The calorie content of a rabbit -/
def rabbit_calories : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits -/
def additional_calories : ℕ := 200

theorem brandon_squirrel_count :
  S * squirrel_calories = R * rabbit_calories + additional_calories ∧ S = 6 := by
  sorry

end NUMINAMATH_CALUDE_brandon_squirrel_count_l1160_116021


namespace NUMINAMATH_CALUDE_column_products_sign_l1160_116073

/-- Represents a 3x3 matrix with elements of type α -/
def Matrix3x3 (α : Type*) := Fin 3 → Fin 3 → α

/-- Given a 3x3 matrix where the product of numbers in each row is negative,
    the products of numbers in the columns must be either
    negative in one column and positive in two columns,
    or negative in all three columns. -/
theorem column_products_sign
  (α : Type*) [LinearOrderedField α]
  (A : Matrix3x3 α)
  (row_products_negative : ∀ i : Fin 3, (A i 0) * (A i 1) * (A i 2) < 0) :
  (∃ j : Fin 3, (A 0 j) * (A 1 j) * (A 2 j) < 0 ∧
    ∀ k : Fin 3, k ≠ j → (A 0 k) * (A 1 k) * (A 2 k) > 0) ∨
  (∀ j : Fin 3, (A 0 j) * (A 1 j) * (A 2 j) < 0) :=
by sorry

end NUMINAMATH_CALUDE_column_products_sign_l1160_116073


namespace NUMINAMATH_CALUDE_log_expression_simplification_l1160_116063

theorem log_expression_simplification :
  Real.log 16 / Real.log 4 / (Real.log (1/16) / Real.log 4) + Real.log 32 / Real.log 4 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l1160_116063


namespace NUMINAMATH_CALUDE_num_multicolor_ducks_l1160_116040

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolored duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def num_white_ducks : ℕ := 3

/-- The number of black ducks -/
def num_black_ducks : ℕ := 7

/-- The total number of fish in the lake -/
def total_fish : ℕ := 157

/-- The theorem stating the number of multicolored ducks -/
theorem num_multicolor_ducks : ℕ := by
  sorry

#check num_multicolor_ducks

end NUMINAMATH_CALUDE_num_multicolor_ducks_l1160_116040


namespace NUMINAMATH_CALUDE_total_bulbs_needed_l1160_116087

def ceiling_lights (medium_count : ℕ) : ℕ × ℕ × ℕ := 
  let large_count := 2 * medium_count
  let small_count := medium_count + 10
  (small_count, medium_count, large_count)

def bulb_count (lights : ℕ × ℕ × ℕ) : ℕ :=
  let (small, medium, large) := lights
  small * 1 + medium * 2 + large * 3

theorem total_bulbs_needed : 
  bulb_count (ceiling_lights 12) = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_bulbs_needed_l1160_116087


namespace NUMINAMATH_CALUDE_largest_possible_b_l1160_116078

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 := by sorry

end NUMINAMATH_CALUDE_largest_possible_b_l1160_116078


namespace NUMINAMATH_CALUDE_production_equation_holds_l1160_116059

/-- Represents the production rate of a factory -/
structure FactoryProduction where
  current_rate : ℝ
  original_rate : ℝ
  h_rate_increase : current_rate = original_rate + 50

/-- The equation representing the production scenario -/
def production_equation (fp : FactoryProduction) : Prop :=
  (450 / fp.original_rate) - (400 / fp.current_rate) = 1

/-- Theorem stating that the production equation holds for the given scenario -/
theorem production_equation_holds (fp : FactoryProduction) :
  production_equation fp := by
  sorry

#check production_equation_holds

end NUMINAMATH_CALUDE_production_equation_holds_l1160_116059


namespace NUMINAMATH_CALUDE_monday_sales_is_five_l1160_116036

/-- Represents the number of crates of eggs sold on each day of the week --/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Defines the conditions for Gabrielle's egg sales --/
def validEggSales (sales : EggSales) : Prop :=
  sales.tuesday = 2 * sales.monday ∧
  sales.wednesday = sales.tuesday - 2 ∧
  sales.thursday = sales.tuesday / 2 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28

/-- Theorem stating that if the egg sales satisfy the given conditions,
    then the number of crates sold on Monday is 5 --/
theorem monday_sales_is_five (sales : EggSales) 
  (h : validEggSales sales) : sales.monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_monday_sales_is_five_l1160_116036


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1160_116029

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 169 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1160_116029


namespace NUMINAMATH_CALUDE_miran_has_least_paper_l1160_116050

def miran_paper : ℕ := 6
def junga_paper : ℕ := 13
def minsu_paper : ℕ := 10

theorem miran_has_least_paper : 
  miran_paper ≤ junga_paper ∧ miran_paper ≤ minsu_paper :=
sorry

end NUMINAMATH_CALUDE_miran_has_least_paper_l1160_116050


namespace NUMINAMATH_CALUDE_line_through_point_l1160_116084

/-- Given a line ax + (a+1)y = a+2 that passes through the point (4, -8), prove that a = -2 -/
theorem line_through_point (a : ℝ) : 
  (∀ x y : ℝ, a * x + (a + 1) * y = a + 2 → x = 4 ∧ y = -8) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_l1160_116084


namespace NUMINAMATH_CALUDE_min_voters_for_tall_win_l1160_116082

/-- Structure representing the giraffe beauty contest voting system -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Definition of the specific contest configuration -/
def contest : GiraffeContest :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

/-- Theorem stating the minimum number of voters needed for Tall to win -/
theorem min_voters_for_tall_win (c : GiraffeContest) 
  (h1 : c.total_voters = c.num_districts * c.precincts_per_district * c.voters_per_precinct)
  (h2 : c = contest) : 
  ∃ (min_voters : Nat), 
    min_voters = 30 ∧ 
    min_voters ≤ c.total_voters ∧
    min_voters = (c.num_districts / 2 + 1) * (c.precincts_per_district / 2 + 1) * (c.voters_per_precinct / 2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_min_voters_for_tall_win_l1160_116082


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1160_116081

def C : Set Nat := {62, 64, 65, 69, 71}

theorem smallest_prime_factor_in_C :
  ∃ x ∈ C, ∀ y ∈ C, ∀ p q : Nat,
    Prime p → Prime q → p ∣ x → q ∣ y → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1160_116081


namespace NUMINAMATH_CALUDE_triangle_side_constraint_l1160_116041

theorem triangle_side_constraint (a : ℝ) : 
  (6 > 0 ∧ 1 - 3*a > 0 ∧ 10 > 0) ∧  -- positive side lengths
  (6 + (1 - 3*a) > 10 ∧ 6 + 10 > 1 - 3*a ∧ 10 + (1 - 3*a) > 6) →  -- triangle inequality
  -5 < a ∧ a < -1 :=
by sorry


end NUMINAMATH_CALUDE_triangle_side_constraint_l1160_116041


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_equality_condition_l1160_116074

theorem min_sum_squares (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (sum_eq : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 := by
sorry

theorem min_sum_squares_equality_condition (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (sum_eq : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 = 1990 ↔ a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_equality_condition_l1160_116074


namespace NUMINAMATH_CALUDE_german_team_goals_l1160_116086

def journalist1_correct (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2_correct (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3_correct (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1_correct x ∧ journalist2_correct x ∧ ¬journalist3_correct x) ∨
  (journalist1_correct x ∧ ¬journalist2_correct x ∧ journalist3_correct x) ∨
  (¬journalist1_correct x ∧ journalist2_correct x ∧ journalist3_correct x)

theorem german_team_goals :
  {x : ℕ | exactly_two_correct x} = {11, 12, 14, 16, 17} := by sorry

end NUMINAMATH_CALUDE_german_team_goals_l1160_116086


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1160_116000

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![0, 5; 0, 10]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; 4, 3]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![10, 15; 20, 6]
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1160_116000


namespace NUMINAMATH_CALUDE_probability_b_speaks_truth_l1160_116031

theorem probability_b_speaks_truth 
  (prob_a : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_a = 0.55) 
  (h2 : prob_both = 0.33) : 
  ∃ (prob_b : ℝ), prob_b = 0.6 ∧ prob_both = prob_a * prob_b :=
sorry

end NUMINAMATH_CALUDE_probability_b_speaks_truth_l1160_116031


namespace NUMINAMATH_CALUDE_diagonal_cubes_140_320_360_l1160_116091

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: The internal diagonal of a 140 × 320 × 360 rectangular solid passes through 760 unit cubes -/
theorem diagonal_cubes_140_320_360 :
  diagonal_cubes 140 320 360 = 760 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cubes_140_320_360_l1160_116091


namespace NUMINAMATH_CALUDE_waiter_customers_l1160_116062

/-- The number of customers who didn't leave a tip -/
def no_tip : ℕ := 34

/-- The number of customers who left a tip -/
def left_tip : ℕ := 15

/-- The number of customers added during the lunch rush -/
def added_customers : ℕ := 20

/-- The number of customers before the lunch rush -/
def customers_before : ℕ := 29

theorem waiter_customers :
  customers_before = (no_tip + left_tip) - added_customers :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l1160_116062


namespace NUMINAMATH_CALUDE_james_coffee_consumption_l1160_116013

/-- Proves that James bought 2 coffees per day before buying a coffee machine -/
theorem james_coffee_consumption
  (machine_cost : ℕ)
  (daily_making_cost : ℕ)
  (previous_coffee_cost : ℕ)
  (payoff_days : ℕ)
  (h1 : machine_cost = 180)
  (h2 : daily_making_cost = 3)
  (h3 : previous_coffee_cost = 4)
  (h4 : payoff_days = 36) :
  ∃ x : ℕ, x = 2 ∧ payoff_days * (previous_coffee_cost * x - daily_making_cost) = machine_cost :=
by sorry

end NUMINAMATH_CALUDE_james_coffee_consumption_l1160_116013


namespace NUMINAMATH_CALUDE_f_lipschitz_implies_m_bounded_l1160_116099

theorem f_lipschitz_implies_m_bounded (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ [-2, 2] → x₂ ∈ [-2, 2] →
    |((fun x => Real.exp (m * x) + x^4 - m * x) x₁) -
     ((fun x => Real.exp (m * x) + x^4 - m * x) x₂)| ≤ Real.exp 4 + 11) →
  m ∈ [-2, 2] := by
sorry

end NUMINAMATH_CALUDE_f_lipschitz_implies_m_bounded_l1160_116099


namespace NUMINAMATH_CALUDE_initial_average_height_l1160_116017

/-- Given a class of students with an incorrect height measurement,
    prove that the initially calculated average height is 174 cm. -/
theorem initial_average_height
  (n : ℕ)  -- number of students
  (incorrect_height correct_height : ℝ)  -- heights of the misrecorded student
  (actual_average : ℝ)  -- actual average height after correction
  (h_n : n = 30)  -- there are 30 students
  (h_incorrect : incorrect_height = 151)  -- incorrectly recorded height
  (h_correct : correct_height = 136)  -- actual height of the misrecorded student
  (h_actual_avg : actual_average = 174.5)  -- actual average height
  : (n * actual_average - (incorrect_height - correct_height)) / n = 174 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_height_l1160_116017


namespace NUMINAMATH_CALUDE_linda_notebooks_count_l1160_116046

/-- The number of notebooks Linda bought -/
def num_notebooks : ℕ := 3

/-- The cost of each notebook in dollars -/
def notebook_cost : ℚ := 6/5

/-- The cost of a box of pencils in dollars -/
def pencil_box_cost : ℚ := 3/2

/-- The cost of a box of pens in dollars -/
def pen_box_cost : ℚ := 17/10

/-- The total amount spent in dollars -/
def total_spent : ℚ := 68/10

theorem linda_notebooks_count :
  (num_notebooks : ℚ) * notebook_cost + pencil_box_cost + pen_box_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_linda_notebooks_count_l1160_116046


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1160_116023

theorem quadratic_inequality (a b c A B C : ℝ) (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4*a*c| ≤ |B^2 - 4*A*C| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1160_116023


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l1160_116027

theorem smallest_multiples_sum (x y : ℕ) : 
  (x ≥ 10 ∧ x < 100 ∧ x % 2 = 0 ∧ ∀ z : ℕ, (z ≥ 10 ∧ z < 100 ∧ z % 2 = 0) → x ≤ z) ∧
  (y ≥ 100 ∧ y < 1000 ∧ y % 5 = 0 ∧ ∀ w : ℕ, (w ≥ 100 ∧ w < 1000 ∧ w % 5 = 0) → y ≤ w) →
  2 * (x + y) = 220 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l1160_116027


namespace NUMINAMATH_CALUDE_staircase_perimeter_l1160_116088

/-- Represents a staircase-shaped region with specific properties -/
structure StaircaseRegion where
  right_angles : Bool
  congruent_segments : ℕ
  segment_length : ℝ
  area : ℝ
  bottom_width : ℝ

/-- Calculates the perimeter of a staircase-shaped region -/
def perimeter (s : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific staircase region -/
theorem staircase_perimeter :
  ∀ (s : StaircaseRegion),
    s.right_angles = true →
    s.congruent_segments = 8 →
    s.segment_length = 1 →
    s.area = 41 →
    s.bottom_width = 7 →
    perimeter s = 128 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l1160_116088


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l1160_116020

/-- The total weight of Marco's and his dad's strawberries is 23 pounds. -/
theorem strawberry_weight_sum : 
  let marco_weight : ℕ := 14
  let dad_weight : ℕ := 9
  marco_weight + dad_weight = 23 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l1160_116020


namespace NUMINAMATH_CALUDE_solve_bones_problem_l1160_116019

def bones_problem (initial_bones final_bones : ℕ) : Prop :=
  let doubled_bones := 2 * initial_bones
  let stolen_bones := doubled_bones - final_bones
  stolen_bones = 2

theorem solve_bones_problem :
  bones_problem 4 6 := by sorry

end NUMINAMATH_CALUDE_solve_bones_problem_l1160_116019


namespace NUMINAMATH_CALUDE_valid_a_value_l1160_116052

-- Define the linear equation
def linear_equation (a x : ℝ) : Prop := (a - 1) * x - 6 = 0

-- State the theorem
theorem valid_a_value : ∃ (a : ℝ), a ≠ 1 ∧ ∀ (x : ℝ), linear_equation a x → True :=
by
  sorry

end NUMINAMATH_CALUDE_valid_a_value_l1160_116052


namespace NUMINAMATH_CALUDE_smallest_multiple_l1160_116079

theorem smallest_multiple (n : ℕ) : n = 3441 ↔ 
  n > 0 ∧ 
  37 ∣ n ∧ 
  n % 103 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 37 ∣ m → m % 103 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1160_116079


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1160_116042

-- Define the angle α
def α : Real := sorry

-- Define the point P₀
def P₀ : ℝ × ℝ := (-3, -4)

-- Theorem statement
theorem cos_pi_half_plus_alpha (h : (Real.cos α * (-3) = Real.sin α * (-4))) : 
  Real.cos (π / 2 + α) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1160_116042


namespace NUMINAMATH_CALUDE_alex_pictures_l1160_116001

/-- The number of pictures Alex has, given processing time per picture and total processing time. -/
def number_of_pictures (minutes_per_picture : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours * 60) / minutes_per_picture

/-- Theorem stating that Alex has 960 pictures. -/
theorem alex_pictures : number_of_pictures 2 32 = 960 := by
  sorry

end NUMINAMATH_CALUDE_alex_pictures_l1160_116001


namespace NUMINAMATH_CALUDE_water_cooler_problem_l1160_116065

/-- Represents the problem of calculating remaining water in coolers after filling cups for a meeting --/
theorem water_cooler_problem (gallons_per_ounce : ℚ) 
  (first_cooler_gallons second_cooler_gallons : ℚ)
  (small_cup_ounces large_cup_ounces : ℚ)
  (rows chairs_per_row : ℕ) :
  first_cooler_gallons = 4.5 →
  second_cooler_gallons = 3.25 →
  small_cup_ounces = 4 →
  large_cup_ounces = 8 →
  rows = 7 →
  chairs_per_row = 12 →
  gallons_per_ounce = 1 / 128 →
  (first_cooler_gallons / gallons_per_ounce) - 
    (↑(rows * chairs_per_row) * small_cup_ounces) = 240 :=
by sorry

end NUMINAMATH_CALUDE_water_cooler_problem_l1160_116065


namespace NUMINAMATH_CALUDE_am_gm_squared_max_value_on_interval_max_value_sqrt_function_l1160_116083

-- Statement 1
theorem am_gm_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a * b ≤ ((a + b) / 2) ^ 2 := by sorry

-- Statement 2
theorem max_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) :
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c := by sorry

theorem max_value_sqrt_function :
  ∃ c ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, x * Real.sqrt (4 - x^2) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_am_gm_squared_max_value_on_interval_max_value_sqrt_function_l1160_116083


namespace NUMINAMATH_CALUDE_actual_plot_area_l1160_116049

/-- Represents the scale of the map --/
def scale : ℝ := 3

/-- Represents the conversion factor from square miles to acres --/
def sq_mile_to_acre : ℝ := 640

/-- Represents the length of the rectangle on the map in cm --/
def map_length : ℝ := 20

/-- Represents the width of the rectangle on the map in cm --/
def map_width : ℝ := 12

/-- Theorem stating that the area of the actual plot is 1,382,400 acres --/
theorem actual_plot_area :
  (map_length * scale) * (map_width * scale) * sq_mile_to_acre = 1382400 := by
  sorry

end NUMINAMATH_CALUDE_actual_plot_area_l1160_116049


namespace NUMINAMATH_CALUDE_distinct_values_x9_mod_999_l1160_116098

theorem distinct_values_x9_mod_999 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 999) ∧ 
  (∀ x : ℕ, ∃ n ∈ S, x^9 ≡ n [ZMOD 999]) ∧
  Finset.card S = 15 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_x9_mod_999_l1160_116098


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l1160_116054

def U : Set Nat := {1, 2, 3, 4}
def P : Set Nat := {1, 2}
def Q : Set Nat := {2, 3}

theorem complement_of_union_equals_four : 
  (U \ (P ∪ Q)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l1160_116054


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1160_116093

theorem quadratic_inequality (a : ℝ) :
  (a > 1 → ∀ x : ℝ, x^2 + 2*x + a > 0) ∧
  (a = 1 → ∀ x : ℝ, x ≠ -1 → x^2 + 2*x + 1 > 0) ∧
  (a < 1 → ∀ x : ℝ, (x > -1 + Real.sqrt (1 - a) ∨ x < -1 - Real.sqrt (1 - a)) ↔ x^2 + 2*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1160_116093


namespace NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l1160_116096

theorem cos_squared_30_minus_2_minus_pi_to_0 :
  Real.cos (30 * π / 180) ^ 2 - (2 - π) ^ 0 = -(1/4) := by sorry

end NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l1160_116096


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1160_116097

/-- A rectangular solid with prime edge lengths and volume 455 has surface area 382 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 455 →
  2 * (a * b + b * c + c * a) = 382 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1160_116097


namespace NUMINAMATH_CALUDE_baseball_team_wins_l1160_116072

theorem baseball_team_wins (total_games : ℕ) (ratio : ℚ) (wins : ℕ) : 
  total_games = 10 → 
  ratio = 2 → 
  ratio = total_games / (total_games - wins) → 
  wins = 5 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_wins_l1160_116072


namespace NUMINAMATH_CALUDE_section_area_formula_l1160_116094

/-- Regular hexagonal prism with base side length a -/
structure HexagonalPrism :=
  (a : ℝ)
  (a_pos : 0 < a)

/-- Plane intersecting the prism -/
structure IntersectingPlane (prism : HexagonalPrism) :=
  (α : ℝ)
  (α_acute : 0 < α ∧ α < π/2)

/-- The area of the section formed by the intersecting plane -/
noncomputable def sectionArea (prism : HexagonalPrism) (plane : IntersectingPlane prism) : ℝ :=
  (3 * prism.a^2 * Real.sqrt 3) / (2 * Real.cos plane.α)

/-- Theorem stating the area of the section -/
theorem section_area_formula (prism : HexagonalPrism) (plane : IntersectingPlane prism) :
  sectionArea prism plane = (3 * prism.a^2 * Real.sqrt 3) / (2 * Real.cos plane.α) :=
by sorry

end NUMINAMATH_CALUDE_section_area_formula_l1160_116094


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1160_116077

theorem complex_expression_simplification (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a + 4*b = 17 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1160_116077


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_for_pyramid_l1160_116067

/-- Minimum surface area of a sphere containing a specific triangular pyramid -/
theorem min_sphere_surface_area_for_pyramid (V : ℝ) (h : ℝ) (angle : ℝ) : 
  V = 8 * Real.sqrt 3 →
  h = 4 →
  angle = π / 3 →
  ∃ (S : ℝ), S = 48 * π ∧ 
    ∀ (S' : ℝ), (∃ (r : ℝ), S' = 4 * π * r^2 ∧ 
      ∃ (a b c : ℝ), 
        a^2 + (h/2)^2 ≤ r^2 ∧
        b^2 + (h/2)^2 ≤ r^2 ∧
        c^2 + h^2 ≤ r^2 ∧
        (1/3) * (1/2) * a * b * Real.sin angle * h = V) → 
    S ≤ S' :=
sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_for_pyramid_l1160_116067


namespace NUMINAMATH_CALUDE_total_age_is_22_l1160_116012

/-- Given three people A, B, and C with the following age relationships:
    - A is two years older than B
    - B is twice as old as C
    - B is 8 years old
    This theorem proves that the sum of their ages is 22 years. -/
theorem total_age_is_22 (a b c : ℕ) : 
  b = 8 → a = b + 2 → b = 2 * c → a + b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_age_is_22_l1160_116012
