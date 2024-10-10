import Mathlib

namespace m_range_l3462_346250

theorem m_range (m : ℝ) : (∀ x > 0, x + 1/x - m > 0) → m < 2 := by
  sorry

end m_range_l3462_346250


namespace stadium_ratio_l3462_346209

theorem stadium_ratio (initial_total : ℕ) (initial_girls : ℕ) (final_total : ℕ) 
  (h1 : initial_total = 600)
  (h2 : initial_girls = 240)
  (h3 : final_total = 480)
  (h4 : (initial_total - initial_girls) / 4 + (initial_girls - (initial_total - final_total - (initial_total - initial_girls) / 4)) = initial_girls) :
  (initial_total - final_total - (initial_total - initial_girls) / 4) / initial_girls = 1 / 8 := by
sorry

end stadium_ratio_l3462_346209


namespace surface_area_increase_l3462_346210

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the original solid -/
def originalSolid : RectangularSolid :=
  { length := 4, width := 3, height := 2 }

/-- Represents the size of the removed cube -/
def cubeSize : ℝ := 1

/-- Theorem stating that removing a 1-foot cube from the center of the original solid
    increases its surface area by 6 square feet -/
theorem surface_area_increase :
  surfaceArea originalSolid + 6 = surfaceArea originalSolid + 6 * cubeSize^2 := by
  sorry

#check surface_area_increase

end surface_area_increase_l3462_346210


namespace lotto_winning_percentage_l3462_346243

theorem lotto_winning_percentage :
  let total_tickets : ℕ := 200
  let cost_per_ticket : ℚ := 2
  let grand_prize : ℚ := 5000
  let profit : ℚ := 4830
  let five_dollar_win_ratio : ℚ := 4/5
  let ten_dollar_win_ratio : ℚ := 1/5
  let five_dollar_prize : ℚ := 5
  let ten_dollar_prize : ℚ := 10
  ∃ (winning_tickets : ℕ),
    (winning_tickets : ℚ) / total_tickets * 100 = 19 ∧
    profit = five_dollar_win_ratio * winning_tickets * five_dollar_prize +
             ten_dollar_win_ratio * winning_tickets * ten_dollar_prize +
             grand_prize -
             (total_tickets * cost_per_ticket) :=
by sorry

end lotto_winning_percentage_l3462_346243


namespace residue_of_11_pow_2010_mod_19_l3462_346267

theorem residue_of_11_pow_2010_mod_19 : (11 : ℤ) ^ 2010 ≡ 3 [ZMOD 19] := by
  sorry

end residue_of_11_pow_2010_mod_19_l3462_346267


namespace derivative_f_at_one_l3462_346259

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem derivative_f_at_one : 
  deriv f 1 = 0 := by sorry

end derivative_f_at_one_l3462_346259


namespace gcd_of_72_120_168_l3462_346293

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l3462_346293


namespace same_sign_range_l3462_346221

theorem same_sign_range (m : ℝ) : 
  ((2 - m) * (|m| - 3) > 0) → (m ∈ Set.Ioo 2 3 ∪ Set.Iio (-3)) := by
  sorry

end same_sign_range_l3462_346221


namespace no_special_primes_l3462_346246

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_swap (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem no_special_primes :
  ∀ n : ℕ, 13 ≤ n → n < 100 →
    is_prime n →
    is_prime (digit_swap n) →
    is_prime (digit_sum n) →
    False :=
sorry

end no_special_primes_l3462_346246


namespace winning_percentage_approx_l3462_346253

def total_votes : ℕ := 2500 + 5000 + 15000

def winning_votes : ℕ := 15000

def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_percentage_approx : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |winning_percentage - 200/3| < ε :=
sorry

end winning_percentage_approx_l3462_346253


namespace base9_85_to_decimal_l3462_346201

/-- Converts a two-digit number in base 9 to its decimal representation -/
def base9ToDecimal (tens : Nat) (ones : Nat) : Nat :=
  tens * 9^1 + ones * 9^0

/-- Theorem stating that 85 in base 9 is equal to 77 in decimal -/
theorem base9_85_to_decimal : base9ToDecimal 8 5 = 77 := by
  sorry

end base9_85_to_decimal_l3462_346201


namespace airplane_fraction_is_one_third_l3462_346281

/-- Represents the travel scenario with given conditions -/
structure TravelScenario where
  driving_time : ℕ
  airport_drive_time : ℕ
  airport_wait_time : ℕ
  post_flight_time : ℕ
  time_saved : ℕ

/-- Calculates the fraction of time spent on the airplane compared to driving -/
def airplane_time_fraction (scenario : TravelScenario) : ℚ :=
  let airplane_time := scenario.driving_time - scenario.airport_drive_time - 
                       scenario.airport_wait_time - scenario.post_flight_time - 
                       scenario.time_saved
  airplane_time / scenario.driving_time

/-- The main theorem stating that the fraction of time spent on the airplane is 1/3 -/
theorem airplane_fraction_is_one_third (scenario : TravelScenario) 
    (h1 : scenario.driving_time = 195)
    (h2 : scenario.airport_drive_time = 10)
    (h3 : scenario.airport_wait_time = 20)
    (h4 : scenario.post_flight_time = 10)
    (h5 : scenario.time_saved = 90) :
    airplane_time_fraction scenario = 1/3 := by
  sorry

end airplane_fraction_is_one_third_l3462_346281


namespace green_beads_count_l3462_346284

/-- The number of green beads in a jewelry pattern -/
def green_beads : ℕ := 3

/-- The number of purple beads in the pattern -/
def purple_beads : ℕ := 5

/-- The number of red beads in the pattern -/
def red_beads : ℕ := 2 * green_beads

/-- The number of times the pattern repeats in a bracelet -/
def bracelet_repeats : ℕ := 3

/-- The number of times the pattern repeats in a necklace -/
def necklace_repeats : ℕ := 5

/-- The total number of beads needed for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

/-- The number of bracelets to be made -/
def num_bracelets : ℕ := 1

/-- The number of necklaces to be made -/
def num_necklaces : ℕ := 10

theorem green_beads_count : 
  num_bracelets * bracelet_repeats * (green_beads + purple_beads + red_beads) + 
  num_necklaces * necklace_repeats * (green_beads + purple_beads + red_beads) = total_beads :=
by sorry

end green_beads_count_l3462_346284


namespace two_problems_without_conditional_statements_l3462_346204

/-- Represents a mathematical problem that may or may not require conditional statements in its algorithm. -/
inductive Problem
| CommonLogarithm
| SquarePerimeter
| MaximumOfThree
| PiecewiseFunction

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requiresConditionalStatements (p : Problem) : Bool :=
  match p with
  | Problem.CommonLogarithm => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.PiecewiseFunction => true

/-- The list of all problems given in the question. -/
def allProblems : List Problem :=
  [Problem.CommonLogarithm, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.PiecewiseFunction]

/-- Theorem stating that the number of problems not requiring conditional statements is 2. -/
theorem two_problems_without_conditional_statements :
  (allProblems.filter (fun p => ¬requiresConditionalStatements p)).length = 2 := by
  sorry

end two_problems_without_conditional_statements_l3462_346204


namespace total_cost_of_pens_and_pencils_l3462_346237

/-- The cost of buying multiple items given their individual prices -/
theorem total_cost_of_pens_and_pencils (x y : ℝ) : 
  5 * x + 3 * y = 5 * x + 3 * y := by sorry

end total_cost_of_pens_and_pencils_l3462_346237


namespace det_A_squared_minus_3A_l3462_346230

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 88 := by
  sorry

end det_A_squared_minus_3A_l3462_346230


namespace gasoline_price_growth_rate_l3462_346266

theorem gasoline_price_growth_rate (initial_price final_price : ℝ) (months : ℕ) (x : ℝ) 
  (h1 : initial_price = 6.2)
  (h2 : final_price = 8.9)
  (h3 : months = 2)
  (h4 : x > 0)
  : initial_price * (1 + x)^months = final_price := by
  sorry

end gasoline_price_growth_rate_l3462_346266


namespace foot_of_perpendicular_l3462_346244

-- Define the point A
def A : ℝ × ℝ := (1, 2)

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the perpendicular line from A to the x-axis
def perp_line : Set (ℝ × ℝ) := {p | p.1 = A.1}

-- Define point M as the intersection of the perpendicular line and the x-axis
def M : ℝ × ℝ := (A.1, 0)

-- Theorem statement
theorem foot_of_perpendicular : M ∈ x_axis ∧ M ∈ perp_line := by sorry

end foot_of_perpendicular_l3462_346244


namespace triangle_problem_l3462_346219

open Real

noncomputable def f (x : ℝ) := 2 * sin x * cos (x - π/3) - sqrt 3 / 2

theorem triangle_problem (A B C : ℝ) (a b c R : ℝ) :
  (0 < A ∧ A < π/2) →
  (0 < B ∧ B < π/2) →
  (0 < C ∧ C < π/2) →
  A + B + C = π →
  a * cos B - b * cos A = R →
  f A = 1 →
  a = 2 * R * sin A →
  b = 2 * R * sin B →
  c = 2 * R * sin C →
  (B = π/4 ∧ -1 < (R - c) / b ∧ (R - c) / b < 0) :=
by sorry

end triangle_problem_l3462_346219


namespace mikes_net_spending_l3462_346254

/-- The net amount Mike spent at the music store -/
def net_amount (trumpet_cost song_book_price : ℚ) : ℚ :=
  trumpet_cost - song_book_price

/-- Theorem stating the net amount Mike spent -/
theorem mikes_net_spending :
  let trumpet_cost : ℚ := 145.16
  let song_book_price : ℚ := 5.84
  net_amount trumpet_cost song_book_price = 139.32 := by
  sorry

end mikes_net_spending_l3462_346254


namespace distribution_schemes_eq_60_l3462_346286

/-- Represents the number of girls in the group. -/
def num_girls : ℕ := 5

/-- Represents the number of boys in the group. -/
def num_boys : ℕ := 2

/-- Represents the number of places for volunteer activities. -/
def num_places : ℕ := 2

/-- Calculates the number of ways to distribute girls and boys to two places. -/
def distribution_schemes : ℕ := sorry

/-- Theorem stating that the number of distribution schemes is 60. -/
theorem distribution_schemes_eq_60 : distribution_schemes = 60 := by sorry

end distribution_schemes_eq_60_l3462_346286


namespace no_integer_solution_l3462_346239

theorem no_integer_solution (P : Int → Int) (a b c : Int) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  P a = 2 ∧ P b = 2 ∧ P c = 2 →
  ∀ k : Int, P k ≠ 3 := by
sorry

end no_integer_solution_l3462_346239


namespace arithmetic_sequence_common_difference_l3462_346257

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is a sequence of real numbers indexed by natural numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- a is an arithmetic sequence
  (h_sum1 : a 2 + a 6 = 8)  -- given condition
  (h_sum2 : a 3 + a 4 = 3)  -- given condition
  : ∃ d, ∀ n, a (n + 1) - a n = d ∧ d = 5 :=
by sorry

end arithmetic_sequence_common_difference_l3462_346257


namespace rectangle_cylinder_volume_ratio_l3462_346278

/-- Given a rectangle with dimensions 6 and 9, prove that the ratio of the volumes of cylinders
    formed by rolling along each side is 3/4, with the larger volume in the numerator. -/
theorem rectangle_cylinder_volume_ratio :
  let rect_width : ℝ := 6
  let rect_height : ℝ := 9
  let volume1 := π * (rect_width / (2 * π))^2 * rect_height
  let volume2 := π * (rect_height / (2 * π))^2 * rect_width
  max volume1 volume2 / min volume1 volume2 = 3 / 4 := by
sorry

end rectangle_cylinder_volume_ratio_l3462_346278


namespace correct_staffing_arrangements_l3462_346215

def total_members : ℕ := 6
def positions_to_fill : ℕ := 4
def restricted_members : ℕ := 2
def restricted_positions : ℕ := 1

def staffing_arrangements (n m k r : ℕ) : ℕ :=
  (n.factorial / (n - m).factorial) - k * ((n - 1).factorial / (n - m).factorial)

theorem correct_staffing_arrangements :
  staffing_arrangements total_members positions_to_fill restricted_members restricted_positions = 240 := by
  sorry

end correct_staffing_arrangements_l3462_346215


namespace eighth_term_of_geometric_sequence_l3462_346218

-- Define a geometric sequence
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Theorem statement
theorem eighth_term_of_geometric_sequence :
  let a := 4
  let a2 := 16
  let r := a2 / a
  geometric_sequence a r 8 = 65536 := by
sorry

end eighth_term_of_geometric_sequence_l3462_346218


namespace largest_root_of_cubic_l3462_346291

theorem largest_root_of_cubic (p q r : ℝ) : 
  p + q + r = 3 → 
  p * q + p * r + q * r = -6 → 
  p * q * r = -8 → 
  ∃ (largest : ℝ), largest = (1 + Real.sqrt 17) / 2 ∧ 
    largest ≥ p ∧ largest ≥ q ∧ largest ≥ r ∧
    largest^3 - 3 * largest^2 - 6 * largest + 8 = 0 := by
  sorry

end largest_root_of_cubic_l3462_346291


namespace rectangular_views_imply_prism_or_cylinder_l3462_346214

/-- A solid object in 3D space -/
structure Solid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Front view of a solid -/
def frontView (s : Solid) : Set (ℝ × ℝ) :=
  sorry

/-- Side view of a solid -/
def sideView (s : Solid) : Set (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set is a rectangle -/
def isRectangle (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a solid is a rectangular prism -/
def isRectangularPrism (s : Solid) : Prop :=
  sorry

/-- Predicate to check if a solid is a cylinder -/
def isCylinder (s : Solid) : Prop :=
  sorry

/-- Theorem: If a solid has rectangular front and side views, it can be either a rectangular prism or a cylinder -/
theorem rectangular_views_imply_prism_or_cylinder (s : Solid) :
  isRectangle (frontView s) → isRectangle (sideView s) →
  isRectangularPrism s ∨ isCylinder s :=
by
  sorry

end rectangular_views_imply_prism_or_cylinder_l3462_346214


namespace product_four_consecutive_integers_l3462_346252

theorem product_four_consecutive_integers (a : ℤ) : 
  a^2 = 1000 * 1001 * 1002 * 1003 + 1 → a = 1002001 := by
  sorry

end product_four_consecutive_integers_l3462_346252


namespace problem_statement_l3462_346229

theorem problem_statement (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : a^2 * b + a * b^2 = -30) : 
  a^2 - a*b + b^2 + 11 = -10 := by
sorry

end problem_statement_l3462_346229


namespace isosceles_right_triangle_area_l3462_346271

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 square units. -/
theorem isosceles_right_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  h = a * Real.sqrt 2 →  -- relationship between hypotenuse and leg in isosceles right triangle
  A = (1/2) * a^2 →      -- area formula for right triangle
  A = 18 := by
    sorry

end isosceles_right_triangle_area_l3462_346271


namespace max_value_inequality_l3462_346255

theorem max_value_inequality (x y : ℝ) : 
  (x + 3*y + 4) / Real.sqrt (x^2 + y^2 + x + 1) ≤ Real.sqrt 26 := by
  sorry

end max_value_inequality_l3462_346255


namespace negation_existence_l3462_346285

theorem negation_existence (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 - a*x + 1 ≥ 0) :=
sorry

end negation_existence_l3462_346285


namespace product_zero_in_special_set_l3462_346296

theorem product_zero_in_special_set (n : ℕ) (h : n = 1997) (S : Finset ℝ) 
  (hS : S.card = n) 
  (hSum : ∀ x ∈ S, (S.sum id - x) ∈ S) : 
  S.prod id = 0 := by
sorry

end product_zero_in_special_set_l3462_346296


namespace problem_solution_l3462_346212

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {a, 2, 2*a - 1}

-- State the theorem
theorem problem_solution :
  ∃ (a : ℝ), A ⊆ B a ∧ A = {2, 3} ∧ a = 3 := by
  sorry

end problem_solution_l3462_346212


namespace rod_length_proof_l3462_346247

/-- Given that a 6-meter rod weighs 14.04 kg, prove that a rod weighing 23.4 kg is 10 meters long. -/
theorem rod_length_proof (weight_per_meter : ℝ) (h1 : weight_per_meter = 14.04 / 6) :
  23.4 / weight_per_meter = 10 := by
sorry

end rod_length_proof_l3462_346247


namespace fathers_seedlings_count_l3462_346207

/-- The number of seedlings Remi planted on the first day -/
def first_day_seedlings : ℕ := 200

/-- The number of seedlings Remi planted on the second day -/
def second_day_seedlings : ℕ := 2 * first_day_seedlings

/-- The total number of seedlings transferred to the farm on both days -/
def total_seedlings : ℕ := 1200

/-- The number of seedlings Remi's father planted -/
def fathers_seedlings : ℕ := total_seedlings - (first_day_seedlings + second_day_seedlings)

theorem fathers_seedlings_count : fathers_seedlings = 600 := by
  sorry

end fathers_seedlings_count_l3462_346207


namespace absolute_value_equality_l3462_346295

theorem absolute_value_equality (x : ℝ) (h : x < -2) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end absolute_value_equality_l3462_346295


namespace solution_set_of_inequality_l3462_346208

theorem solution_set_of_inequality (x : ℝ) :
  (x * (x - 1) < 2) ↔ (-1 < x ∧ x < 2) := by
  sorry

end solution_set_of_inequality_l3462_346208


namespace intersection_A_B_union_complement_B_P_intersection_AB_complement_P_l3462_346248

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for (ᶜB) ∪ P
theorem union_complement_B_P : (Bᶜ : Set ℝ) ∪ P = {x : ℝ | x ≤ 0 ∨ x ≥ 5/2} := by sorry

-- Theorem for (A ∩ B) ∩ (ᶜP)
theorem intersection_AB_complement_P : (A ∩ B) ∩ (Pᶜ : Set ℝ) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end intersection_A_B_union_complement_B_P_intersection_AB_complement_P_l3462_346248


namespace zoo_animal_ratio_l3462_346206

theorem zoo_animal_ratio (initial_animals : ℕ) (final_animals : ℕ)
  (gorillas_sent : ℕ) (hippo_adopted : ℕ) (rhinos_taken : ℕ) (lion_cubs_born : ℕ)
  (h1 : initial_animals = 68)
  (h2 : final_animals = 90)
  (h3 : gorillas_sent = 6)
  (h4 : hippo_adopted = 1)
  (h5 : rhinos_taken = 3)
  (h6 : lion_cubs_born = 8) :
  (final_animals - (initial_animals - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs_born)) / lion_cubs_born = 2 :=
by sorry

end zoo_animal_ratio_l3462_346206


namespace maggies_share_l3462_346262

def total_sum : ℝ := 6000
def debby_percentage : ℝ := 0.25

theorem maggies_share :
  let debby_share := debby_percentage * total_sum
  let maggie_share := total_sum - debby_share
  maggie_share = 4500 := by sorry

end maggies_share_l3462_346262


namespace monotonicity_and_inequality_min_m_value_l3462_346203

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem monotonicity_and_inequality (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → f a x₁ ≥ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ x₁ < x₂ → f a x₁ ≤ f a x₂) ↔
  a = 9 :=
sorry

theorem min_m_value :
  (∀ x : ℝ, x ≥ 1 ∧ x ≤ 4 → x + 9 / x - 6.25 ≤ 0) ∧
  ∀ m : ℝ, m < 6.25 →
    ∃ x : ℝ, x ≥ 1 ∧ x ≤ 4 ∧ x + 9 / x - m > 0 :=
sorry

end monotonicity_and_inequality_min_m_value_l3462_346203


namespace age_difference_l3462_346269

theorem age_difference (patrick michael monica nathan : ℝ) : 
  patrick / michael = 3 / 5 →
  michael / monica = 3 / 4 →
  monica / nathan = 5 / 7 →
  nathan / patrick = 4 / 9 →
  patrick + michael + monica + nathan = 252 →
  nathan - patrick = 66.5 := by
sorry

end age_difference_l3462_346269


namespace min_value_trig_expression_l3462_346272

theorem min_value_trig_expression (α γ : ℝ) :
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2 ≥ 36 := by
  sorry

end min_value_trig_expression_l3462_346272


namespace fraction_simplification_l3462_346274

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3*x^2 - 2*x - 4) / ((x+2)*(x-3)) - (5+x) / ((x+2)*(x-3)) = 3*(x^2-x-3) / ((x+2)*(x-3)) :=
by sorry

end fraction_simplification_l3462_346274


namespace function_identity_l3462_346260

theorem function_identity (f : ℕ → ℕ) (h : ∀ n, f (n + 1) > f (f n)) : ∀ n, f n = n := by
  sorry

end function_identity_l3462_346260


namespace boat_rental_problem_l3462_346225

theorem boat_rental_problem :
  ∀ (big_boats small_boats : ℕ),
    big_boats + small_boats = 12 →
    6 * big_boats + 4 * small_boats = 58 →
    big_boats = 5 ∧ small_boats = 7 := by
  sorry

end boat_rental_problem_l3462_346225


namespace necklace_beads_l3462_346256

theorem necklace_beads (total : ℕ) (blue : ℕ) (h1 : total = 40) (h2 : blue = 5) : 
  let red := 2 * blue
  let white := blue + red
  let silver := total - (blue + red + white)
  silver = 10 := by
  sorry

end necklace_beads_l3462_346256


namespace quality_difference_confidence_l3462_346290

/-- Production data for two machines -/
structure ProductionData :=
  (machine_a_first : ℕ)
  (machine_a_second : ℕ)
  (machine_b_first : ℕ)
  (machine_b_second : ℕ)

/-- Calculate K^2 statistic -/
def calculate_k_squared (data : ProductionData) : ℚ :=
  let n := data.machine_a_first + data.machine_a_second + data.machine_b_first + data.machine_b_second
  let a := data.machine_a_first
  let b := data.machine_a_second
  let c := data.machine_b_first
  let d := data.machine_b_second
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical values for K^2 test -/
def critical_value_99_percent : ℚ := 6635 / 1000
def critical_value_999_percent : ℚ := 10828 / 1000

/-- Theorem stating the confidence level for the difference in quality -/
theorem quality_difference_confidence (data : ProductionData) 
  (h1 : data.machine_a_first = 150)
  (h2 : data.machine_a_second = 50)
  (h3 : data.machine_b_first = 120)
  (h4 : data.machine_b_second = 80) :
  critical_value_99_percent < calculate_k_squared data ∧ 
  calculate_k_squared data < critical_value_999_percent :=
sorry

end quality_difference_confidence_l3462_346290


namespace existence_of_m_l3462_346277

def M : Set ℕ := {n : ℕ | n ≤ 2007}

def arithmetic_progression (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ i : ℕ, k = n + i * (n + 1)}

theorem existence_of_m :
  (∀ n ∈ M, arithmetic_progression n ⊆ M) →
  ∃ m : ℕ, ∀ k > m, k ∈ M :=
sorry

end existence_of_m_l3462_346277


namespace card_selection_count_l3462_346220

/-- Represents a standard deck of cards -/
def StandardDeck : Nat := 52

/-- Number of suits in a standard deck -/
def NumSuits : Nat := 4

/-- Number of ranks in a standard deck -/
def NumRanks : Nat := 13

/-- Number of cards to be chosen -/
def CardsToChoose : Nat := 5

/-- Number of cards that must be of the same suit -/
def SameSuitCards : Nat := 2

/-- Number of cards that must be of different suits -/
def DiffSuitCards : Nat := 3

theorem card_selection_count : 
  (Nat.choose NumSuits 1) * 
  (Nat.choose NumRanks SameSuitCards) * 
  (Nat.choose (NumSuits - 1) DiffSuitCards) * 
  ((Nat.choose (NumRanks - SameSuitCards) 1) ^ DiffSuitCards) = 414384 := by
  sorry

end card_selection_count_l3462_346220


namespace password_combinations_l3462_346241

/-- A digit is either odd or even -/
inductive Digit
| odd
| even

/-- The set of possible digits -/
def digit_set : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- A valid password is a list of four digits satisfying the given conditions -/
def ValidPassword : Type := List Digit

/-- The number of odd digits in the digit set -/
def num_odd_digits : Nat := (digit_set.filter (fun n => n % 2 = 1)).card

/-- The number of even digits in the digit set -/
def num_even_digits : Nat := (digit_set.filter (fun n => n % 2 = 0)).card

/-- The total number of digits in the digit set -/
def total_digits : Nat := digit_set.card

/-- The number of valid passwords -/
def num_valid_passwords : Nat := 
  (num_odd_digits * num_even_digits * total_digits * total_digits) +
  (num_even_digits * num_odd_digits * total_digits * total_digits)

theorem password_combinations : num_valid_passwords = 648 := by
  sorry

end password_combinations_l3462_346241


namespace oranges_to_put_back_correct_l3462_346200

/-- Represents the number of oranges to put back -/
def oranges_to_put_back (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℕ :=
  6

theorem oranges_to_put_back_correct 
  (apple_price : ℚ) 
  (orange_price : ℚ) 
  (total_fruits : ℕ) 
  (initial_avg_price : ℚ) 
  (desired_avg_price : ℚ) 
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : initial_avg_price = 54/100)
  (h5 : desired_avg_price = 45/100) :
  ∃ (A O : ℕ), 
    A + O = total_fruits ∧ 
    (apple_price * A + orange_price * O) / total_fruits = initial_avg_price ∧
    (apple_price * A + orange_price * (O - oranges_to_put_back apple_price orange_price total_fruits initial_avg_price desired_avg_price)) / 
      (total_fruits - oranges_to_put_back apple_price orange_price total_fruits initial_avg_price desired_avg_price) = desired_avg_price :=
by sorry

#check oranges_to_put_back_correct

end oranges_to_put_back_correct_l3462_346200


namespace jimmy_stair_climbing_time_l3462_346227

/-- Represents the time taken to climb stairs with an increasing time for each flight -/
def stair_climbing_time (initial_time : ℕ) (time_increase : ℕ) (num_flights : ℕ) : ℕ :=
  (num_flights * (2 * initial_time + (num_flights - 1) * time_increase)) / 2

/-- Theorem stating the total time Jimmy takes to climb eight flights of stairs -/
theorem jimmy_stair_climbing_time :
  stair_climbing_time 30 10 8 = 520 := by
  sorry

end jimmy_stair_climbing_time_l3462_346227


namespace total_salaries_l3462_346268

/-- The problem of calculating total salaries given specific conditions -/
theorem total_salaries (A_salary B_salary : ℝ) : 
  A_salary = 2250 →
  0.05 * A_salary = 0.15 * B_salary →
  A_salary + B_salary = 3000 := by
  sorry

#check total_salaries

end total_salaries_l3462_346268


namespace f_difference_f_equals_x_plus_3_l3462_346279

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem 1: For any real number a, f(a) - f(a + 1) = -2a - 1
theorem f_difference (a : ℝ) : f a - f (a + 1) = -2 * a - 1 := by
  sorry

-- Theorem 2: If f(x) = x + 3, then x = -1 or x = 2
theorem f_equals_x_plus_3 (x : ℝ) : f x = x + 3 → x = -1 ∨ x = 2 := by
  sorry

end f_difference_f_equals_x_plus_3_l3462_346279


namespace tigers_losses_l3462_346245

theorem tigers_losses (total_games wins : ℕ) (h1 : total_games = 56) (h2 : wins = 38) : 
  ∃ losses ties : ℕ, 
    losses + ties + wins = total_games ∧ 
    ties = losses / 2 ∧
    losses = 12 := by
sorry

end tigers_losses_l3462_346245


namespace no_periodic_sequence_exists_l3462_346273

-- Define a_n as the first non-zero digit from the unit place in n!
def a (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_periodic_sequence_exists :
  ∀ N : ℕ, ¬∃ T : ℕ, T > 0 ∧ ∀ k : ℕ, a (N + k + T) = a (N + k) :=
sorry

end no_periodic_sequence_exists_l3462_346273


namespace trumpet_trombone_difference_l3462_346216

/-- Represents the number of players for each instrument in the school band --/
structure BandComposition where
  flute : Nat
  trumpet : Nat
  trombone : Nat
  drummer : Nat
  clarinet : Nat
  french_horn : Nat

/-- Theorem stating the difference between trumpet and trombone players --/
theorem trumpet_trombone_difference (band : BandComposition) : 
  band.flute = 5 →
  band.trumpet = 3 * band.flute →
  band.trumpet > band.trombone →
  band.drummer = band.trombone + 11 →
  band.clarinet = 2 * band.flute →
  band.french_horn = band.trombone + 3 →
  band.flute + band.trumpet + band.trombone + band.drummer + band.clarinet + band.french_horn = 65 →
  band.trumpet - band.trombone = 8 := by
  sorry


end trumpet_trombone_difference_l3462_346216


namespace base_10_to_base_3_172_l3462_346217

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem base_10_to_base_3_172 :
  toBase3 172 = [2, 0, 1, 0, 1] := by
  sorry

#eval toBase3 172  -- This line is for verification purposes

end base_10_to_base_3_172_l3462_346217


namespace nine_caps_per_box_l3462_346238

/-- Given a total number of bottle caps and a number of boxes, 
    calculate the number of bottle caps in each box. -/
def bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) : ℕ :=
  total_caps / num_boxes

/-- Theorem stating that with 54 total bottle caps and 6 boxes, 
    there are 9 bottle caps in each box. -/
theorem nine_caps_per_box :
  bottle_caps_per_box 54 6 = 9 := by
  sorry

#eval bottle_caps_per_box 54 6

end nine_caps_per_box_l3462_346238


namespace moss_pollen_diameter_scientific_notation_l3462_346264

/-- Expresses a given decimal number in scientific notation -/
def scientificNotation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem moss_pollen_diameter_scientific_notation :
  scientificNotation 0.0000084 = (8.4, -6) := by sorry

end moss_pollen_diameter_scientific_notation_l3462_346264


namespace first_division_percentage_l3462_346211

theorem first_division_percentage (total_students : ℕ) 
  (second_division_percent : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percent = 54 / 100 →
  just_passed = 54 →
  (just_passed : ℚ) / total_students + second_division_percent + 28 / 100 = 1 :=
by sorry

end first_division_percentage_l3462_346211


namespace isosceles_when_negative_one_is_root_roots_of_equilateral_triangle_l3462_346236

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.b - t.c)

theorem isosceles_when_negative_one_is_root (t : Triangle) :
  quadratic t (-1) = 0 → t.a = t.b :=
sorry

theorem roots_of_equilateral_triangle (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (quadratic t 0 = 0 ∧ quadratic t (-1) = 0) :=
sorry

end isosceles_when_negative_one_is_root_roots_of_equilateral_triangle_l3462_346236


namespace correct_num_classes_l3462_346213

/-- The number of classes in a single round-robin basketball tournament -/
def num_classes : ℕ := 10

/-- The total number of games played in the tournament -/
def total_games : ℕ := 45

/-- Theorem stating that the number of classes is correct given the total number of games played -/
theorem correct_num_classes : 
  (num_classes * (num_classes - 1)) / 2 = total_games :=
by sorry

end correct_num_classes_l3462_346213


namespace point_in_second_quadrant_l3462_346261

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (a : ℝ) : second_quadrant (-1) (a^2 + 1) := by
  sorry

end point_in_second_quadrant_l3462_346261


namespace ellipse_line_intersection_slope_l3462_346275

/-- Theorem: For an ellipse and a line intersecting it under specific conditions, the slope of the line is ±1/2. -/
theorem ellipse_line_intersection_slope (k : ℝ) : 
  (∀ x y, x^2/4 + y^2/3 = 1 → y = k*x + 1 → 
    ∃ x₁ x₂ y₁ y₂, 
      x₁^2/4 + y₁^2/3 = 1 ∧ 
      x₂^2/4 + y₂^2/3 = 1 ∧
      y₁ = k*x₁ + 1 ∧ 
      y₂ = k*x₂ + 1 ∧ 
      x₁ = -x₂/2) →
  k = 1/2 ∨ k = -1/2 := by
sorry

end ellipse_line_intersection_slope_l3462_346275


namespace geometric_sequence_value_l3462_346298

/-- A geometric sequence with sum of first n terms S_n = a · 2^n + a - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := fun n ↦ 
  if n = 0 then 0 else (a * 2^n + a - 2) - (a * 2^(n-1) + a - 2)

/-- The sum of the first n terms of the geometric sequence -/
def SumFirstNTerms (a : ℝ) : ℕ → ℝ := fun n ↦ a * 2^n + a - 2

theorem geometric_sequence_value (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → GeometricSequence a (n+1) / GeometricSequence a n = GeometricSequence a 2 / GeometricSequence a 1) →
  a = 1 := by
  sorry


end geometric_sequence_value_l3462_346298


namespace soccer_penalty_kicks_l3462_346205

theorem soccer_penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) 
  (h1 : total_players = 24) 
  (h2 : goalkeepers = 4) 
  (h3 : goalkeepers ≤ total_players) : 
  (total_players - 1) * goalkeepers = 92 := by
  sorry

end soccer_penalty_kicks_l3462_346205


namespace problem_solution_l3462_346202

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y - x - y = -7) : 
  x^2*y + x*y^2 = 196/9 := by
sorry

end problem_solution_l3462_346202


namespace train_journey_time_l3462_346270

theorem train_journey_time (X : ℝ) (h1 : 0 < X) (h2 : X < 60) : 
  (X * 6 - X * 0.5 = 360 - X) → X = 360 / 7 := by
  sorry

end train_journey_time_l3462_346270


namespace all_transylvanians_answer_yes_l3462_346289

-- Define the types of Transylvanians
inductive TransylvanianType
  | SaneHuman
  | InsaneHuman
  | SaneVampire
  | InsaneVampire

-- Define the possible questions
inductive Question
  | ConsiderHuman
  | Reliable

-- Define the function that represents a Transylvanian's answer
def transylvanianAnswer (t : TransylvanianType) (q : Question) : Bool :=
  match q with
  | Question.ConsiderHuman => true
  | Question.Reliable => true

-- Theorem statement
theorem all_transylvanians_answer_yes
  (t : TransylvanianType) (q : Question) :
  transylvanianAnswer t q = true := by sorry

end all_transylvanians_answer_yes_l3462_346289


namespace dot_product_equals_negative_102_l3462_346234

def vector1 : Fin 4 → ℤ := ![4, -5, 6, -3]
def vector2 : Fin 4 → ℤ := ![-2, 8, -7, 4]

theorem dot_product_equals_negative_102 :
  (Finset.univ.sum fun i => (vector1 i) * (vector2 i)) = -102 := by
  sorry

end dot_product_equals_negative_102_l3462_346234


namespace arccos_sin_three_l3462_346222

theorem arccos_sin_three (x : ℝ) : x = Real.arccos (Real.sin 3) → x = 3 - π / 2 := by
  sorry

end arccos_sin_three_l3462_346222


namespace remainder_theorem_l3462_346224

theorem remainder_theorem (z : ℕ) (hz : z > 0) (hz_div : 4 ∣ z) : (z * (2 + 4 + z) + 3) % 2 = 1 := by
  sorry

end remainder_theorem_l3462_346224


namespace complex_power_modulus_l3462_346299

theorem complex_power_modulus : 
  Complex.abs ((1 / 2 : ℂ) + (Complex.I * Real.sqrt 3 / 2)) ^ 12 = 1 := by sorry

end complex_power_modulus_l3462_346299


namespace plum_count_l3462_346294

/-- The number of plums initially in the basket -/
def initial_plums : ℕ := 17

/-- The number of plums added to the basket -/
def added_plums : ℕ := 4

/-- The final number of plums in the basket -/
def final_plums : ℕ := initial_plums + added_plums

theorem plum_count : final_plums = 21 := by sorry

end plum_count_l3462_346294


namespace investment_profit_comparison_l3462_346233

/-- Profit calculation for selling at the beginning of the month -/
def profit_beginning (x : ℝ) : ℝ := 0.265 * x

/-- Profit calculation for selling at the end of the month -/
def profit_end (x : ℝ) : ℝ := 0.3 * x - 700

theorem investment_profit_comparison :
  /- The investment amount where profits are equal is 20,000 yuan -/
  (∃ x : ℝ, x = 20000 ∧ profit_beginning x = profit_end x) ∧
  /- For a 50,000 yuan investment, profit from selling at the end is greater -/
  (profit_end 50000 > profit_beginning 50000) :=
by sorry

end investment_profit_comparison_l3462_346233


namespace more_birds_than_nests_l3462_346288

/-- Given 6 birds and 3 nests, prove that there are 3 more birds than nests. -/
theorem more_birds_than_nests (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) (h2 : nests = 3) : birds - nests = 3 := by
  sorry

end more_birds_than_nests_l3462_346288


namespace rational_root_of_polynomial_l3462_346251

def p (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_of_polynomial :
  (p (-1/3) = 0) ∧ (∀ q : ℚ, p q = 0 → q = -1/3) :=
by sorry

end rational_root_of_polynomial_l3462_346251


namespace zane_picked_62_pounds_l3462_346258

/-- The amount of garbage picked up by Daliah in pounds -/
def daliah_garbage : ℝ := 17.5

/-- The amount of garbage picked up by Dewei in pounds -/
def dewei_garbage : ℝ := daliah_garbage - 2

/-- The amount of garbage picked up by Zane in pounds -/
def zane_garbage : ℝ := 4 * dewei_garbage

/-- Theorem stating that Zane picked up 62 pounds of garbage -/
theorem zane_picked_62_pounds : zane_garbage = 62 := by sorry

end zane_picked_62_pounds_l3462_346258


namespace initial_phone_count_prove_initial_phone_count_l3462_346265

theorem initial_phone_count : ℕ → Prop :=
  fun initial_count =>
    let defective_count : ℕ := 5
    let customer_a_bought : ℕ := 3
    let customer_b_bought : ℕ := 5
    let customer_c_bought : ℕ := 7
    let total_sold := customer_a_bought + customer_b_bought + customer_c_bought
    initial_count - defective_count = total_sold ∧ initial_count = 20

theorem prove_initial_phone_count :
  ∃ (x : ℕ), initial_phone_count x :=
sorry

end initial_phone_count_prove_initial_phone_count_l3462_346265


namespace f_monotonic_and_odd_l3462_346232

def f (x : ℝ) : ℝ := x^3

theorem f_monotonic_and_odd : 
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x) := by sorry

end f_monotonic_and_odd_l3462_346232


namespace inner_cube_surface_area_l3462_346282

/-- Given a cube of volume 64 cubic meters with an inscribed sphere, which in turn has an inscribed cube, 
    the surface area of the inner cube is 32 square meters. -/
theorem inner_cube_surface_area (outer_cube : Real → Real → Real → Bool) 
  (outer_sphere : Real → Real → Real → Bool) (inner_cube : Real → Real → Real → Bool) :
  (∀ x y z, outer_cube x y z ↔ (0 ≤ x ∧ x ≤ 4) ∧ (0 ≤ y ∧ y ≤ 4) ∧ (0 ≤ z ∧ z ≤ 4)) →
  (∀ x y z, outer_sphere x y z ↔ (x - 2)^2 + (y - 2)^2 + (z - 2)^2 ≤ 4) →
  (∀ x y z, inner_cube x y z → outer_sphere x y z) →
  (∃! l : Real, ∀ x y z, inner_cube x y z ↔ 
    (0 ≤ x ∧ x ≤ l) ∧ (0 ≤ y ∧ y ≤ l) ∧ (0 ≤ z ∧ z ≤ l) ∧ l^2 + l^2 + l^2 = 16) →
  (∃ sa : Real, sa = 6 * (4 * Real.sqrt 3 / 3)^2 ∧ sa = 32) :=
by sorry

end inner_cube_surface_area_l3462_346282


namespace reduced_rate_fraction_l3462_346287

def hours_in_week : ℕ := 7 * 24

def weekday_reduced_hours : ℕ := 5 * 12

def weekend_reduced_hours : ℕ := 2 * 24

def total_reduced_hours : ℕ := weekday_reduced_hours + weekend_reduced_hours

theorem reduced_rate_fraction :
  (total_reduced_hours : ℚ) / hours_in_week = 9 / 14 := by
  sorry

end reduced_rate_fraction_l3462_346287


namespace reflect_point_coordinates_l3462_346297

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflect_point_coordinates :
  let P : Point := { x := 4, y := -1 }
  reflectAcrossYAxis P = { x := -4, y := -1 } := by
  sorry

end reflect_point_coordinates_l3462_346297


namespace quadratic_equation_solution_l3462_346231

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ + 10)^2 = (4*x₁ + 6)*(x₁ + 8) ∧ 
  (x₂ + 10)^2 = (4*x₂ + 6)*(x₂ + 8) ∧ 
  (abs (x₁ - 2.131) < 0.001) ∧ 
  (abs (x₂ + 8.131) < 0.001) := by
sorry

end quadratic_equation_solution_l3462_346231


namespace sodium_bisulfite_moles_l3462_346226

-- Define the molecules and their molar quantities
structure Reaction :=
  (NaHSO3 : ℝ)  -- moles of Sodium bisulfite
  (HCl : ℝ)     -- moles of Hydrochloric acid
  (H2O : ℝ)     -- moles of Water

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.NaHSO3 = r.HCl ∧ r.NaHSO3 = r.H2O

-- Theorem statement
theorem sodium_bisulfite_moles :
  ∀ r : Reaction,
  r.HCl = 1 →        -- 1 mole of Hydrochloric acid is used
  r.H2O = 1 →        -- The reaction forms 1 mole of Water
  balanced_equation r →  -- The reaction equation is balanced
  r.NaHSO3 = 1 :=    -- The number of moles of Sodium bisulfite is 1
by
  sorry

end sodium_bisulfite_moles_l3462_346226


namespace lost_bottle_caps_l3462_346228

/-- Represents the number of bottle caps Danny has now -/
def current_bottle_caps : ℕ := 25

/-- Represents the number of bottle caps Danny had at first -/
def initial_bottle_caps : ℕ := 91

/-- Theorem stating that the number of lost bottle caps is the difference between
    the initial number and the current number of bottle caps -/
theorem lost_bottle_caps : 
  initial_bottle_caps - current_bottle_caps = 66 := by
  sorry

end lost_bottle_caps_l3462_346228


namespace angle_A_triangle_area_l3462_346263

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 7 ∧
  t.b = 2 ∧
  Real.sqrt 3 * t.b * Real.cos t.A = t.a * Real.sin t.B

-- Theorem for angle A
theorem angle_A (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 :=
sorry

-- Theorem for area of triangle ABC
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 :=
sorry

end angle_A_triangle_area_l3462_346263


namespace point_order_on_line_l3462_346292

theorem point_order_on_line (m n b : ℝ) : 
  (2 * (-1/2) + b = m) → (2 * 2 + b = n) → m < n := by sorry

end point_order_on_line_l3462_346292


namespace meat_pie_cost_l3462_346235

/-- The cost of a meat pie given Gerald's initial farthings and remaining pfennigs -/
theorem meat_pie_cost
  (initial_farthings : ℕ)
  (farthings_per_pfennig : ℕ)
  (remaining_pfennigs : ℕ)
  (h1 : initial_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7)
  : (initial_farthings / farthings_per_pfennig) - remaining_pfennigs = 2 := by
  sorry

#check meat_pie_cost

end meat_pie_cost_l3462_346235


namespace interest_calculation_l3462_346240

theorem interest_calculation (P : ℝ) : 
  P * (1 + 5/100)^2 - P - (P * 5 * 2 / 100) = 17 → P = 6800 := by
  sorry

end interest_calculation_l3462_346240


namespace prime_sum_gcd_ratio_l3462_346249

theorem prime_sum_gcd_ratio (n : ℕ) (p : ℕ) (hp : Prime p) (h_p : p = 2 * n - 1) 
  (a : Fin n → ℕ+) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j : ℕ) / Nat.gcd (a i) (a j) ≥ p := by
sorry

end prime_sum_gcd_ratio_l3462_346249


namespace weight_of_new_person_l3462_346280

/-- Given a group of 9 people where one person is replaced, this theorem calculates the weight of the new person based on the average weight increase. -/
theorem weight_of_new_person
  (n : ℕ) -- number of people
  (w : ℝ) -- weight of the person being replaced
  (d : ℝ) -- increase in average weight
  (h1 : n = 9)
  (h2 : w = 65)
  (h3 : d = 1.5) :
  w + n * d = 78.5 := by
  sorry

end weight_of_new_person_l3462_346280


namespace particle_position_after_1991_minutes_l3462_346276

-- Define the particle's position type
def Position := ℤ × ℤ

-- Define the starting position
def start_position : Position := (0, 1)

-- Define the movement pattern for a single rectangle
def rectangle_movement (n : ℕ) : Position := 
  if n % 2 = 1 then (n, n + 1) else (-(n + 1), -n)

-- Define the time taken for a single rectangle
def rectangle_time (n : ℕ) : ℕ := 2 * n + 1

-- Define the total time for n rectangles
def total_time (n : ℕ) : ℕ := (n + 1)^2 - 1

-- Define the function to calculate the position after n rectangles
def position_after_rectangles (n : ℕ) : Position :=
  sorry

-- Define the function to calculate the final position
def final_position (time : ℕ) : Position :=
  sorry

-- The theorem to prove
theorem particle_position_after_1991_minutes :
  final_position 1991 = (-45, -32) :=
sorry

end particle_position_after_1991_minutes_l3462_346276


namespace arithmetic_sequence_properties_l3462_346242

/-- Arithmetic sequence a_n with a₁ = 8 and a₃ = 4 -/
def a (n : ℕ) : ℚ :=
  8 - 2 * (n - 1)

/-- Sum of first n terms of a_n -/
def S (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- b_n sequence -/
def b (n : ℕ+) : ℚ :=
  1 / ((n : ℚ) * (12 - a n))

/-- Sum of first n terms of b_n -/
def T (n : ℕ+) : ℚ :=
  (n : ℚ) / (2 * (n + 1))

theorem arithmetic_sequence_properties :
  (∃ n : ℕ, S n = 20 ∧ ∀ m : ℕ, S m ≤ S n) ∧
  (∀ n : ℕ+, T n = (n : ℚ) / (2 * (n + 1))) :=
sorry

end arithmetic_sequence_properties_l3462_346242


namespace onions_remaining_l3462_346283

theorem onions_remaining (initial : Nat) (sold : Nat) (h1 : initial = 98) (h2 : sold = 65) :
  initial - sold = 33 := by
  sorry

end onions_remaining_l3462_346283


namespace remaining_laps_after_sunday_morning_l3462_346223

def total_required_laps : ℕ := 198

def friday_morning_laps : ℕ := 23
def friday_afternoon_laps : ℕ := 12
def friday_evening_laps : ℕ := 28

def saturday_morning_laps : ℕ := 35
def saturday_afternoon_laps : ℕ := 27

def sunday_morning_laps : ℕ := 15

def friday_total : ℕ := friday_morning_laps + friday_afternoon_laps + friday_evening_laps
def saturday_total : ℕ := saturday_morning_laps + saturday_afternoon_laps
def laps_before_sunday_break : ℕ := friday_total + saturday_total + sunday_morning_laps

theorem remaining_laps_after_sunday_morning :
  total_required_laps - laps_before_sunday_break = 58 := by
  sorry

end remaining_laps_after_sunday_morning_l3462_346223
