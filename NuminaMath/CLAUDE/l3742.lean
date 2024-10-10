import Mathlib

namespace large_pots_delivered_l3742_374284

/-- The number of boxes delivered -/
def num_boxes : ℕ := 32

/-- The number of small pots in each box -/
def small_pots_per_box : ℕ := 36

/-- The number of large pots in each box -/
def large_pots_per_box : ℕ := 12

/-- The total number of large pots delivered -/
def total_large_pots : ℕ := num_boxes * large_pots_per_box

/-- The number of boxes used for comparison -/
def comparison_boxes : ℕ := 8

theorem large_pots_delivered :
  total_large_pots = 384 ∧
  total_large_pots = comparison_boxes * (small_pots_per_box + large_pots_per_box) :=
by sorry

end large_pots_delivered_l3742_374284


namespace newspaper_photos_theorem_l3742_374245

/-- Represents the number of photos in a section of the newspaper --/
def photos_in_section (pages : ℕ) (photos_per_page : ℕ) : ℕ :=
  pages * photos_per_page

/-- Calculates the total number of photos in the newspaper for a given day --/
def total_photos_per_day (section_a : ℕ) (section_b : ℕ) (section_c : ℕ) : ℕ :=
  section_a + section_b + section_c

theorem newspaper_photos_theorem :
  let section_a := photos_in_section 25 4
  let section_b := photos_in_section 18 6
  let section_c_monday := photos_in_section 12 5
  let section_c_tuesday := photos_in_section 15 3
  let monday_total := total_photos_per_day section_a section_b section_c_monday
  let tuesday_total := total_photos_per_day section_a section_b section_c_tuesday
  monday_total + tuesday_total = 521 := by
  sorry

end newspaper_photos_theorem_l3742_374245


namespace gcd_lcm_42_30_l3742_374219

theorem gcd_lcm_42_30 :
  (Nat.gcd 42 30 = 6) ∧ (Nat.lcm 42 30 = 210) := by
  sorry

end gcd_lcm_42_30_l3742_374219


namespace gumball_distribution_l3742_374249

theorem gumball_distribution (joanna_initial : Nat) (jacques_initial : Nat) : 
  joanna_initial = 40 →
  jacques_initial = 60 →
  let joanna_final := joanna_initial + 5 * joanna_initial
  let jacques_final := jacques_initial + 3 * jacques_initial
  let total := joanna_final + jacques_final
  let shared := total / 2
  shared = 240 :=
by sorry

end gumball_distribution_l3742_374249


namespace smallest_sum_of_squares_l3742_374260

/-- A system of equations with consecutive non-integer complex solutions -/
structure ConsecutiveComplexSystem where
  x : ℂ
  y : ℂ
  z : ℂ
  eq1 : (x + 5) * (y - 5) = 0
  eq2 : (y + 5) * (z - 5) = 0
  eq3 : (z + 5) * (x - 5) = 0
  consecutive : ∃ (a b : ℝ), x = a + b * Complex.I ∧ 
                              y = a + (b + 1) * Complex.I ∧ 
                              z = a + (b + 2) * Complex.I
  non_integer : x.im ≠ 0 ∧ y.im ≠ 0 ∧ z.im ≠ 0

/-- The smallest possible sum of absolute squares -/
theorem smallest_sum_of_squares (s : ConsecutiveComplexSystem) : 
  Complex.abs s.x ^ 2 + Complex.abs s.y ^ 2 + Complex.abs s.z ^ 2 ≥ 83.75 := by
  sorry

end smallest_sum_of_squares_l3742_374260


namespace solution_exists_l3742_374242

theorem solution_exists : ∃ x : ℚ, 
  (10 / (Real.sqrt (x - 5) - 10) + 
   2 / (Real.sqrt (x - 5) - 5) + 
   9 / (Real.sqrt (x - 5) + 5) + 
   18 / (Real.sqrt (x - 5) + 10) = 0) ∧ 
  (x = 1230 / 121) := by
  sorry


end solution_exists_l3742_374242


namespace equal_cake_distribution_l3742_374255

theorem equal_cake_distribution (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end equal_cake_distribution_l3742_374255


namespace circle_area_from_diameter_endpoints_l3742_374299

/-- The area of a circle with diameter endpoints C(-2,3) and D(4,-1) is 13π square units -/
theorem circle_area_from_diameter_endpoints : 
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_length := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let radius := diameter_length / 2
  let circle_area := π * radius^2
  circle_area = 13 * π := by sorry

end circle_area_from_diameter_endpoints_l3742_374299


namespace largest_quantity_l3742_374276

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l3742_374276


namespace scientific_notation_of_56_99_million_l3742_374256

theorem scientific_notation_of_56_99_million :
  (56.99 * 1000000 : ℝ) = 5.699 * (10 ^ 7) :=
by sorry

end scientific_notation_of_56_99_million_l3742_374256


namespace loss_recording_l3742_374263

/-- Represents the recording of a financial transaction -/
def record (amount : Int) : Int := amount

/-- A profit of $300 is recorded as $+300 -/
axiom profit_recording : record 300 = 300

/-- Theorem: If a profit of $300 is recorded as $+300, then a loss of $300 should be recorded as $-300 -/
theorem loss_recording : record (-300) = -300 := by
  sorry

end loss_recording_l3742_374263


namespace train_speed_l3742_374293

/-- Given a train of length 160 meters that crosses a stationary point in 18 seconds, 
    its speed is 32 km/h. -/
theorem train_speed (length : Real) (time : Real) (speed : Real) : 
  length = 160 ∧ time = 18 → speed = (length / time) * 3.6 → speed = 32 := by
  sorry

end train_speed_l3742_374293


namespace jakes_birdhouse_depth_l3742_374233

/-- Calculates the depth of Jake's birdhouse given the dimensions of both birdhouses and their volume difference --/
theorem jakes_birdhouse_depth
  (sara_width : ℝ) (sara_height : ℝ) (sara_depth : ℝ)
  (jake_width : ℝ) (jake_height : ℝ)
  (volume_difference : ℝ)
  (h1 : sara_width = 1) -- 1 foot
  (h2 : sara_height = 2) -- 2 feet
  (h3 : sara_depth = 2) -- 2 feet
  (h4 : jake_width = 16) -- 16 inches
  (h5 : jake_height = 20) -- 20 inches
  (h6 : volume_difference = 1152) -- 1,152 cubic inches
  : ∃ (jake_depth : ℝ),
    jake_depth = 25.2 ∧
    (jake_width * jake_height * jake_depth) - (sara_width * sara_height * sara_depth * 12^3) = volume_difference :=
by sorry

end jakes_birdhouse_depth_l3742_374233


namespace days_to_eat_candy_correct_l3742_374267

/-- Given the initial number of candies, the number of candies eaten per day for the first week,
    and the number of candies to be eaten per day after the first week,
    calculate the number of additional days Yuna can eat candy. -/
def days_to_eat_candy (initial_candies : ℕ) (candies_per_day_week1 : ℕ) (candies_per_day_after : ℕ) : ℕ :=
  let candies_eaten_week1 := candies_per_day_week1 * 7
  let remaining_candies := initial_candies - candies_eaten_week1
  remaining_candies / candies_per_day_after

theorem days_to_eat_candy_correct (initial_candies : ℕ) (candies_per_day_week1 : ℕ) (candies_per_day_after : ℕ) 
  (h1 : initial_candies = 60)
  (h2 : candies_per_day_week1 = 6)
  (h3 : candies_per_day_after = 3) :
  days_to_eat_candy initial_candies candies_per_day_week1 candies_per_day_after = 6 := by
  sorry

end days_to_eat_candy_correct_l3742_374267


namespace reduced_oil_price_l3742_374232

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction) 
  (h1 : scenario.reduced_price = 0.8 * scenario.original_price) 
  (h2 : scenario.original_quantity * scenario.original_price = scenario.total_cost) 
  (h3 : (scenario.original_quantity + scenario.additional_quantity) * scenario.reduced_price = scenario.total_cost) 
  (h4 : scenario.additional_quantity = 4) 
  (h5 : scenario.total_cost = 600) : 
  scenario.reduced_price = 30 := by
  sorry

end reduced_oil_price_l3742_374232


namespace f_composition_negative_three_eq_pi_l3742_374277

/-- Piecewise function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2
  else if x = 0 then Real.pi
  else 0

/-- Theorem stating that f(f(-3)) = π -/
theorem f_composition_negative_three_eq_pi : f (f (-3)) = Real.pi := by sorry

end f_composition_negative_three_eq_pi_l3742_374277


namespace calculator_square_presses_l3742_374217

def square (x : ℕ) : ℕ := x * x

def exceed_1000 (n : ℕ) : Prop := n > 1000

theorem calculator_square_presses :
  (∃ k : ℕ, exceed_1000 (square (square (square 3)))) ∧
  (∀ m : ℕ, m < 3 → ¬exceed_1000 (Nat.iterate square 3 m)) :=
by sorry

end calculator_square_presses_l3742_374217


namespace derivative_log2_l3742_374220

theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end derivative_log2_l3742_374220


namespace candy_distribution_l3742_374226

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) 
  (h1 : total_candy = 648) 
  (h2 : num_bags = 8) 
  (h3 : candy_per_bag * num_bags = total_candy) :
  candy_per_bag = 81 := by
  sorry

end candy_distribution_l3742_374226


namespace no_consecutive_tails_probability_l3742_374251

/-- Represents the number of ways to toss n coins without getting two consecutive tails -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => a (n + 1) + a n

/-- The probability of not getting two consecutive tails when tossing five fair coins -/
theorem no_consecutive_tails_probability : 
  (a 5 : ℚ) / (2^5 : ℚ) = 13 / 32 := by sorry

end no_consecutive_tails_probability_l3742_374251


namespace extra_apples_l3742_374275

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 43)
  (h2 : green_apples = 32)
  (h3 : students = 2) :
  red_apples + green_apples - students = 73 := by
  sorry

end extra_apples_l3742_374275


namespace count_numbers_mod_three_eq_one_l3742_374210

theorem count_numbers_mod_three_eq_one (n : ℕ) : 
  (Finset.filter (fun x => x % 3 = 1) (Finset.range 50)).card = 17 := by
  sorry

end count_numbers_mod_three_eq_one_l3742_374210


namespace line_l_equation_l3742_374258

/-- A line l passes through point P(-1,2) and has equal distances from points A(2,3) and B(-4,6) -/
def line_l (x y : ℝ) : Prop :=
  (x = -1 ∧ y = 2) ∨ 
  (abs ((2 * x - y + 2) / Real.sqrt (x^2 + 1)) = abs ((-4 * x - y + 2) / Real.sqrt (x^2 + 1)))

/-- The equation of line l is either x+2y-3=0 or x=-1 -/
theorem line_l_equation : 
  ∀ x y : ℝ, line_l x y ↔ (x + 2*y - 3 = 0 ∨ x = -1) :=
by sorry

end line_l_equation_l3742_374258


namespace tunnel_construction_days_l3742_374231

/-- The number of days to complete the tunnel with new equipment -/
def total_days : ℕ := 185

/-- The fraction of the tunnel completed at original speed -/
def original_fraction : ℚ := 1/3

/-- The speed increase factor with new equipment -/
def speed_increase : ℚ := 1.2

/-- The working hours reduction factor with new equipment -/
def hours_reduction : ℚ := 0.8

/-- The effective daily construction rate with new equipment -/
def effective_rate : ℚ := speed_increase * hours_reduction

theorem tunnel_construction_days : 
  ∃ (original_days : ℕ), 
    (original_days : ℚ) * (original_fraction + (1 - original_fraction) / effective_rate) = total_days ∧ 
    original_days = 180 := by
  sorry

end tunnel_construction_days_l3742_374231


namespace abcd_sum_l3742_374270

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = 0) :
  a * b + c * d = -31 := by
  sorry

end abcd_sum_l3742_374270


namespace range_of_sum_l3742_374294

theorem range_of_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 2 * x + y + 4 * x * y = 15 / 2) : 2 * x + y ≥ 3 := by
  sorry

end range_of_sum_l3742_374294


namespace vector_subtraction_and_scalar_multiplication_l3742_374282

theorem vector_subtraction_and_scalar_multiplication :
  (⟨3, -8⟩ : ℝ × ℝ) - 3 • (⟨-2, 6⟩ : ℝ × ℝ) = (⟨9, -26⟩ : ℝ × ℝ) := by
  sorry

end vector_subtraction_and_scalar_multiplication_l3742_374282


namespace quadrilateral_sum_l3742_374292

/-- A quadrilateral ABCD with specific side lengths and angles -/
structure Quadrilateral :=
  (BC : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (angleA : ℝ)
  (angleB : ℝ)
  (p : ℕ)
  (q : ℕ)
  (h_BC : BC = 10)
  (h_CD : CD = 15)
  (h_AD : AD = 12)
  (h_angleA : angleA = 60)
  (h_angleB : angleB = 120)
  (h_AB : p + Real.sqrt q = AD + BC)

/-- The sum of p and q in the quadrilateral ABCD is 17 -/
theorem quadrilateral_sum (ABCD : Quadrilateral) : ABCD.p + ABCD.q = 17 := by
  sorry

end quadrilateral_sum_l3742_374292


namespace function_equality_exists_l3742_374287

theorem function_equality_exists (a : ℕ+) : ∃ (b c : ℕ+), a^2 + 3*a + 2 = b^2 - b + 3*c^2 + 3*c := by
  sorry

end function_equality_exists_l3742_374287


namespace no_primes_divisible_by_45_l3742_374213

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factorization of 45
axiom factorization_of_45 : 45 = 3 * 3 * 5

-- Theorem statement
theorem no_primes_divisible_by_45 :
  ∀ p : ℕ, is_prime p → ¬(45 ∣ p) :=
sorry

end no_primes_divisible_by_45_l3742_374213


namespace fixed_ray_exists_l3742_374211

/-- Represents a circle with a color -/
structure ColoredCircle where
  center : ℝ × ℝ
  radius : ℝ
  color : Bool

/-- Represents an angle with colored sides -/
structure ColoredAngle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop
  color1 : Bool
  color2 : Bool

/-- Represents a configuration of circles and an angle -/
structure Configuration where
  circle1 : ColoredCircle
  circle2 : ColoredCircle
  angle : ColoredAngle

/-- Predicate to check if circles are non-overlapping -/
def non_overlapping (c1 c2 : ColoredCircle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 > (c1.radius + c2.radius) ^ 2

/-- Predicate to check if a point is outside an angle -/
def outside_angle (p : ℝ × ℝ) (a : ColoredAngle) : Prop :=
  ¬a.side1 p ∧ ¬a.side2 p

/-- Predicate to check if a side touches a circle -/
def touches (side : ℝ × ℝ → Prop) (c : ColoredCircle) : Prop :=
  ∃ p : ℝ × ℝ, side p ∧ (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

/-- Main theorem statement -/
theorem fixed_ray_exists (config : Configuration) 
  (h1 : non_overlapping config.circle1 config.circle2)
  (h2 : config.circle1.color ≠ config.circle2.color)
  (h3 : config.angle.color1 = config.circle1.color)
  (h4 : config.angle.color2 = config.circle2.color)
  (h5 : outside_angle config.circle1.center config.angle)
  (h6 : outside_angle config.circle2.center config.angle)
  (h7 : touches config.angle.side1 config.circle1)
  (h8 : touches config.angle.side2 config.circle2)
  (h9 : config.angle.vertex ≠ config.circle1.center)
  (h10 : config.angle.vertex ≠ config.circle2.center) :
  ∃ (ray : ℝ × ℝ → Prop), ∀ (config' : Configuration), 
    (config'.circle1 = config.circle1 ∧ 
     config'.circle2 = config.circle2 ∧
     config'.angle.vertex = config.angle.vertex ∧
     touches config'.angle.side1 config'.circle1 ∧
     touches config'.angle.side2 config'.circle2) →
    ∃ p : ℝ × ℝ, ray p ∧ 
      (∃ t : ℝ, t > 0 ∧ p = (config'.angle.vertex.1 + t * (p.1 - config'.angle.vertex.1),
                             config'.angle.vertex.2 + t * (p.2 - config'.angle.vertex.2))) :=
sorry

end fixed_ray_exists_l3742_374211


namespace yellow_paint_calculation_l3742_374204

/-- Given a ratio of red:yellow:blue paint and the amount of blue paint,
    calculate the amount of yellow paint required. -/
def yellow_paint_amount (red yellow blue : ℚ) (blue_amount : ℚ) : ℚ :=
  (yellow / blue) * blue_amount

/-- Prove that for the given ratio and blue paint amount, 
    the required yellow paint amount is 9 quarts. -/
theorem yellow_paint_calculation :
  let red : ℚ := 5
  let yellow : ℚ := 3
  let blue : ℚ := 7
  let blue_amount : ℚ := 21
  yellow_paint_amount red yellow blue blue_amount = 9 := by
  sorry

#eval yellow_paint_amount 5 3 7 21

end yellow_paint_calculation_l3742_374204


namespace select_five_from_eight_l3742_374272

/-- The number of ways to select k items from n items without considering order -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: Selecting 5 books from 8 books without order consideration yields 56 ways -/
theorem select_five_from_eight : combination 8 5 = 56 := by
  sorry

end select_five_from_eight_l3742_374272


namespace sequence_theorem_l3742_374295

def sequence_condition (a : ℕ → Fin 2) : Prop :=
  (∀ n : ℕ, n > 0 → a n + a (n + 1) ≠ a (n + 2) + a (n + 3)) ∧
  (∀ n : ℕ, n > 0 → a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_theorem (a : ℕ → Fin 2) (h : sequence_condition a) (h₁ : a 1 = 0) :
  a 2020 = 1 := by
  sorry

end sequence_theorem_l3742_374295


namespace amanda_notebooks_l3742_374285

/-- Represents the number of notebooks Amanda ordered -/
def ordered_notebooks : ℕ := 6

/-- Amanda's initial number of notebooks -/
def initial_notebooks : ℕ := 10

/-- Number of notebooks Amanda lost -/
def lost_notebooks : ℕ := 2

/-- Amanda's final number of notebooks -/
def final_notebooks : ℕ := 14

theorem amanda_notebooks :
  initial_notebooks + ordered_notebooks - lost_notebooks = final_notebooks :=
by sorry

end amanda_notebooks_l3742_374285


namespace largest_solution_reciprocal_sixth_power_l3742_374236

noncomputable def largest_solution (x : ℝ) : Prop :=
  (Real.log 10 / Real.log (10 * x^3)) + (Real.log 10 / Real.log (100 * x^4)) = -1 ∧
  ∀ y, (Real.log 10 / Real.log (10 * y^3)) + (Real.log 10 / Real.log (100 * y^4)) = -1 → y ≤ x

theorem largest_solution_reciprocal_sixth_power (x : ℝ) :
  largest_solution x → 1 / x^6 = 1000 := by
  sorry

end largest_solution_reciprocal_sixth_power_l3742_374236


namespace range_of_a_l3742_374290

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≥ a}
def B : Set ℝ := {x | |x - 1| < 1}

-- Define the property of A being a necessary but not sufficient condition for B
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B)

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) :
  necessary_not_sufficient a → a ≤ 0 :=
sorry

end range_of_a_l3742_374290


namespace fifteen_equation_system_solution_l3742_374224

theorem fifteen_equation_system_solution (x : Fin 15 → ℝ) :
  (∀ i : Fin 14, 1 - x i * x (i + 1) = 0) ∧
  (1 - x 15 * x 1 = 0) →
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) := by
  sorry

end fifteen_equation_system_solution_l3742_374224


namespace negative_three_is_square_mod_p_l3742_374212

theorem negative_three_is_square_mod_p (p q : ℕ) (h_prime : Nat.Prime p) (h_form : p = 3 * q + 1) :
  ∃ x : ZMod p, x^2 = -3 := by
  sorry

end negative_three_is_square_mod_p_l3742_374212


namespace bananas_purchased_is_96_l3742_374291

/-- The number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 96

/-- The purchase price in dollars for 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- The selling price in dollars for 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- The total profit in dollars -/
def total_profit : ℝ := 8.00

/-- Theorem stating that the number of pounds of bananas purchased is 96 -/
theorem bananas_purchased_is_96 :
  bananas_purchased = 96 ∧
  purchase_price = 0.50 ∧
  selling_price = 1.00 ∧
  total_profit = 8.00 ∧
  (selling_price / 4 - purchase_price / 3) * bananas_purchased = total_profit :=
by sorry

end bananas_purchased_is_96_l3742_374291


namespace solve_for_d_l3742_374215

theorem solve_for_d (n c b d : ℝ) (h : n = (d * c * b) / (c - d)) :
  d = (n * c) / (c * b + n) :=
by sorry

end solve_for_d_l3742_374215


namespace inequality_proof_l3742_374230

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) + a + b + c ≤ 3 + (1 / 3) * (a * b + b * c + c * a) :=
by sorry

end inequality_proof_l3742_374230


namespace prop_3_prop_4_l3742_374227

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersectionPP : Plane → Plane → Line)
variable (intersectionPL : Plane → Line → Prop)

-- Proposition ③
theorem prop_3 (α β γ : Plane) (m : Line) :
  perpendicularPP α β →
  perpendicularPP α γ →
  intersectionPP β γ = m →
  perpendicularPL α m :=
sorry

-- Proposition ④
theorem prop_4 (α β : Plane) (m n : Line) :
  perpendicularPL α m →
  perpendicularPL β n →
  perpendicular m n →
  perpendicularPP α β :=
sorry

end prop_3_prop_4_l3742_374227


namespace correlation_coefficient_properties_l3742_374234

/-- Linear correlation coefficient between two variables -/
def linear_correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Positive correlation between two variables -/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Perfect linear relationship between two variables -/
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop := sorry

theorem correlation_coefficient_properties
  (x y : ℝ → ℝ) (r : ℝ) (h : r = linear_correlation_coefficient x y) :
  ((r > 0 → positively_correlated x y) ∧
   (r = 1 ∨ r = -1 → perfect_linear_relationship x y)) := by sorry

end correlation_coefficient_properties_l3742_374234


namespace min_sphere_surface_area_l3742_374225

theorem min_sphere_surface_area (a b c : ℝ) (h1 : a * b * c = 4) (h2 : a * b = 1) :
  let r := (3 * Real.sqrt 2) / 2
  4 * Real.pi * r^2 = 18 * Real.pi := by
  sorry

end min_sphere_surface_area_l3742_374225


namespace polynomial_remainder_theorem_l3742_374248

def polynomial_remainder_problem (p : ℝ → ℝ) (r : ℝ → ℝ) : Prop :=
  (p (-1) = 2) ∧ 
  (p 3 = -2) ∧ 
  (p (-4) = 5) ∧ 
  (∃ q : ℝ → ℝ, ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * q x + r x) ∧
  (r (-5) = 6)

theorem polynomial_remainder_theorem :
  ∃ p r : ℝ → ℝ, polynomial_remainder_problem p r :=
sorry

end polynomial_remainder_theorem_l3742_374248


namespace andrews_age_l3742_374235

theorem andrews_age (carlos_age bella_age andrew_age : ℕ) : 
  carlos_age = 20 →
  bella_age = carlos_age + 4 →
  andrew_age = bella_age - 5 →
  andrew_age = 19 :=
by
  sorry

end andrews_age_l3742_374235


namespace smallest_of_five_consecutive_even_numbers_l3742_374280

theorem smallest_of_five_consecutive_even_numbers (x : ℤ) : 
  (∀ i : ℕ, i < 5 → 2 ∣ (x + 2*i)) →  -- x and the next 4 numbers are even
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) →  -- sum is 200
  x = 36 :=  -- smallest number is 36
by
  sorry

end smallest_of_five_consecutive_even_numbers_l3742_374280


namespace power_equality_l3742_374283

theorem power_equality (q : ℕ) (h : (81 : ℕ)^6 = 3^q) : q = 24 := by
  sorry

end power_equality_l3742_374283


namespace range_of_a_l3742_374223

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 + Complex.I) * (a + 2 * Complex.I^3)

-- Define the condition for z to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (in_fourth_quadrant (z a)) ↔ (-1 < a ∧ a < 4) :=
sorry

end range_of_a_l3742_374223


namespace power_product_equals_sum_of_exponents_l3742_374278

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end power_product_equals_sum_of_exponents_l3742_374278


namespace f_inequality_implies_b_geq_one_l3742_374228

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

-- State the theorem
theorem f_inequality_implies_b_geq_one :
  ∀ b : ℝ,
  (∀ a : ℝ, a ≤ 0 → ∀ x : ℝ, x ≥ 0 → f a x ≤ b * Real.log (x + 1)) →
  b ≥ 1 := by
  sorry

end

end f_inequality_implies_b_geq_one_l3742_374228


namespace range_of_a_l3742_374296

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a^2 - 3*a ≤ |x + 3| + |x - 1|) → 
  -1 < a ∧ a < 4 :=
by sorry

end range_of_a_l3742_374296


namespace kaplan_bobby_slice_ratio_l3742_374200

/-- Represents the number of pizzas Bobby has -/
def bobby_pizzas : ℕ := 2

/-- Represents the number of slices per pizza -/
def slices_per_pizza : ℕ := 6

/-- Represents the number of slices Mrs. Kaplan has -/
def kaplan_slices : ℕ := 3

/-- Calculates the total number of slices Bobby has -/
def bobby_slices : ℕ := bobby_pizzas * slices_per_pizza

/-- Represents the ratio of Mrs. Kaplan's slices to Bobby's slices -/
def slice_ratio : Rat := kaplan_slices / bobby_slices

theorem kaplan_bobby_slice_ratio :
  slice_ratio = 1 / 4 := by sorry

end kaplan_bobby_slice_ratio_l3742_374200


namespace parcera_triples_l3742_374271

def isParcera (p q r : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  p ∣ (q^2 - 4) ∧ q ∣ (r^2 - 4) ∧ r ∣ (p^2 - 4)

theorem parcera_triples :
  ∀ p q r : Nat, isParcera p q r ↔ 
    ((p, q, r) = (2, 2, 2) ∨ 
     (p, q, r) = (5, 3, 7) ∨ 
     (p, q, r) = (7, 5, 3) ∨ 
     (p, q, r) = (3, 7, 5)) :=
by sorry

end parcera_triples_l3742_374271


namespace class_mean_calculation_l3742_374274

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ) 
  (group2_students : ℕ) (group2_mean : ℚ) : 
  total_students = 50 →
  group1_students = 45 →
  group2_students = 5 →
  group1_mean = 85 / 100 →
  group2_mean = 90 / 100 →
  let overall_mean := (group1_students * group1_mean + group2_students * group2_mean) / total_students
  overall_mean = 855 / 1000 := by
sorry

end class_mean_calculation_l3742_374274


namespace consecutive_points_distance_l3742_374214

/-- Given 5 consecutive points on a straight line, if certain conditions are met, 
    then the distance between the first two points is 5. -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b) = 2 * (d - c) →  -- bc = 2cd
  (e - d) = 4 →            -- de = 4
  (c - a) = 11 →           -- ac = 11
  (e - a) = 18 →           -- ae = 18
  (b - a) = 5 :=           -- ab = 5
by sorry

end consecutive_points_distance_l3742_374214


namespace pudding_distribution_l3742_374221

theorem pudding_distribution (pudding_cups : ℕ) (students : ℕ) 
  (h1 : pudding_cups = 4752) (h2 : students = 3019) : 
  let additional_cups := (students * ((pudding_cups + students - 1) / students)) - pudding_cups
  additional_cups = 1286 := by
sorry

end pudding_distribution_l3742_374221


namespace sample_size_theorem_l3742_374239

/-- Represents a population of students -/
structure Population where
  size : Nat

/-- Represents a sample of students -/
structure Sample where
  size : Nat
  population : Population

/-- Theorem: Given a population of 5000 students and a selection of 250 students,
    the 250 students form a sample of the population with a sample size of 250. -/
theorem sample_size_theorem (pop : Population) (sam : Sample) 
    (h1 : pop.size = 5000) (h2 : sam.size = 250) (h3 : sam.population = pop) : 
    sam.size = 250 ∧ sam.population = pop := by
  sorry

#check sample_size_theorem

end sample_size_theorem_l3742_374239


namespace pastry_production_theorem_l3742_374206

/-- Represents a baker's production --/
structure BakerProduction where
  mini_cupcakes : ℕ
  pop_tarts : ℕ
  blueberry_pies : ℕ
  chocolate_eclairs : ℕ
  macarons : ℕ

/-- Calculates the total number of pastries for a baker --/
def total_pastries (bp : BakerProduction) : ℕ :=
  bp.mini_cupcakes + bp.pop_tarts + bp.blueberry_pies + bp.chocolate_eclairs + bp.macarons

/-- Calculates the total cost of pastries for a baker --/
def total_cost (bp : BakerProduction) : ℚ :=
  bp.mini_cupcakes * (1/2) + bp.pop_tarts * 1 + bp.blueberry_pies * 3 + bp.chocolate_eclairs * 2 + bp.macarons * (3/2)

theorem pastry_production_theorem (lola lulu lila luka : BakerProduction) : 
  lola = { mini_cupcakes := 13, pop_tarts := 10, blueberry_pies := 8, chocolate_eclairs := 6, macarons := 0 } →
  lulu = { mini_cupcakes := 16, pop_tarts := 12, blueberry_pies := 14, chocolate_eclairs := 9, macarons := 0 } →
  lila = { mini_cupcakes := 22, pop_tarts := 15, blueberry_pies := 10, chocolate_eclairs := 12, macarons := 0 } →
  luka = { mini_cupcakes := 18, pop_tarts := 20, blueberry_pies := 7, chocolate_eclairs := 14, macarons := 25 } →
  (total_pastries lola + total_pastries lulu + total_pastries lila + total_pastries luka = 231) ∧
  (total_cost lola + total_cost lulu + total_cost lila + total_cost luka = 328) := by
  sorry

end pastry_production_theorem_l3742_374206


namespace max_prime_factor_of_arithmetic_sequence_number_l3742_374202

/-- A 3-digit decimal number with digits forming an arithmetic sequence -/
def ArithmeticSequenceNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    b = a + d ∧
    c = a + 2 * d

theorem max_prime_factor_of_arithmetic_sequence_number :
  ∀ n : ℕ, ArithmeticSequenceNumber n →
    (∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 317) ∧
    (∃ m : ℕ, ArithmeticSequenceNumber m ∧ ∃ p : ℕ, Nat.Prime p ∧ p ∣ m ∧ p = 317) :=
by sorry

end max_prime_factor_of_arithmetic_sequence_number_l3742_374202


namespace smallest_natural_satisfying_congruences_l3742_374297

theorem smallest_natural_satisfying_congruences : 
  ∃ N : ℕ, (∀ m : ℕ, m > N → 
    (m % 9 ≠ 8 ∨ m % 8 ≠ 7 ∨ m % 7 ≠ 6 ∨ m % 6 ≠ 5 ∨ 
     m % 5 ≠ 4 ∨ m % 4 ≠ 3 ∨ m % 3 ≠ 2 ∨ m % 2 ≠ 1)) ∧
  N % 9 = 8 ∧ N % 8 = 7 ∧ N % 7 = 6 ∧ N % 6 = 5 ∧ 
  N % 5 = 4 ∧ N % 4 = 3 ∧ N % 3 = 2 ∧ N % 2 = 1 ∧ 
  N = 2519 :=
sorry

end smallest_natural_satisfying_congruences_l3742_374297


namespace tv_show_length_specific_l3742_374265

/-- The length of a TV show, given the total airtime and duration of commercials and breaks -/
def tv_show_length (total_airtime : ℕ) (commercial_durations : List ℕ) (break_durations : List ℕ) : ℚ :=
  let total_minutes : ℕ := total_airtime
  let commercial_time : ℕ := commercial_durations.sum
  let break_time : ℕ := break_durations.sum
  let show_time : ℕ := total_minutes - commercial_time - break_time
  (show_time : ℚ) / 60

/-- Theorem stating the length of the TV show given specific conditions -/
theorem tv_show_length_specific : 
  let total_airtime : ℕ := 150  -- 2 hours and 30 minutes
  let commercial_durations : List ℕ := [7, 7, 13, 5, 9, 9]
  let break_durations : List ℕ := [4, 2, 8]
  abs (tv_show_length total_airtime commercial_durations break_durations - 1.4333) < 0.0001 := by
  sorry

#eval tv_show_length 150 [7, 7, 13, 5, 9, 9] [4, 2, 8]

end tv_show_length_specific_l3742_374265


namespace black_area_after_changes_l3742_374241

/-- Represents the fraction of black area remaining after a single change --/
def remaining_black_fraction : ℚ := 2 / 3

/-- Represents the number of changes --/
def num_changes : ℕ := 3

/-- Theorem stating that after three changes, 8/27 of the original area remains black --/
theorem black_area_after_changes :
  remaining_black_fraction ^ num_changes = 8 / 27 := by
  sorry

end black_area_after_changes_l3742_374241


namespace investment_result_l3742_374253

/-- Given a total investment split between two interest rates, calculates the total investment with interest after one year. -/
def total_investment_with_interest (total_investment : ℝ) (amount_at_low_rate : ℝ) (low_rate : ℝ) (high_rate : ℝ) : ℝ :=
  let amount_at_high_rate := total_investment - amount_at_low_rate
  let interest_low := amount_at_low_rate * low_rate
  let interest_high := amount_at_high_rate * high_rate
  total_investment + interest_low + interest_high

/-- Theorem stating that given the specific investment conditions, the total investment with interest is $1,046.00 -/
theorem investment_result : 
  let total_investment := 1000
  let amount_at_low_rate := 699.99
  let low_rate := 0.04
  let high_rate := 0.06
  (total_investment_with_interest total_investment amount_at_low_rate low_rate high_rate) = 1046 := by
sorry

end investment_result_l3742_374253


namespace tire_promotion_price_l3742_374269

/-- The regular price of a tire under the given promotion -/
def regular_price : ℝ := 105

/-- The total cost of five tires under the promotion -/
def total_cost : ℝ := 421

/-- The promotion: buy four tires at regular price, get the fifth for $1 -/
theorem tire_promotion_price : 
  4 * regular_price + 1 = total_cost := by sorry

end tire_promotion_price_l3742_374269


namespace triangle_side_length_l3742_374246

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  a : Real
  b : Real

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.A = π / 3)  -- 60 degrees in radians
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.B = π / 6)  -- 30 degrees in radians
  : t.b = 1 := by
  sorry

end triangle_side_length_l3742_374246


namespace hyperbola_conjugate_axis_length_l3742_374207

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    with eccentricity e = √7/2, and a point P on the right branch of the hyperbola
    such that PF₂ ⊥ F₁F₂ and PF₂ = 9/2, prove that the length of the conjugate axis
    is 6√3. -/
theorem hyperbola_conjugate_axis_length
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (he : Real.sqrt 7 / 2 = Real.sqrt (1 + b^2 / a^2))
  (hP : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0)
  (hPF2 : b^2 / a = 9 / 2) :
  2 * b = 6 * Real.sqrt 3 := by
  sorry

end hyperbola_conjugate_axis_length_l3742_374207


namespace cubic_factorization_sum_of_squares_l3742_374243

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x, 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 := by
  sorry

end cubic_factorization_sum_of_squares_l3742_374243


namespace chips_calories_is_310_l3742_374266

/-- Represents the calorie content of various food items and daily calorie limits --/
structure CalorieData where
  cake : ℕ
  coke : ℕ
  breakfast : ℕ
  lunch : ℕ
  daily_limit : ℕ
  remaining : ℕ

/-- Calculates the calorie content of the pack of chips --/
def calculate_chips_calories (data : CalorieData) : ℕ :=
  data.daily_limit - data.remaining - (data.cake + data.coke + data.breakfast + data.lunch)

/-- Theorem stating that the calorie content of the pack of chips is 310 --/
theorem chips_calories_is_310 (data : CalorieData) 
    (h1 : data.cake = 110)
    (h2 : data.coke = 215)
    (h3 : data.breakfast = 560)
    (h4 : data.lunch = 780)
    (h5 : data.daily_limit = 2500)
    (h6 : data.remaining = 525) :
  calculate_chips_calories data = 310 := by
  sorry

end chips_calories_is_310_l3742_374266


namespace star_three_four_l3742_374208

-- Define the star operation
def star (a b : ℝ) : ℝ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_three_four : star 3 4 = 8 := by
  sorry

end star_three_four_l3742_374208


namespace butter_cheese_ratio_l3742_374252

/-- Represents the prices of items bought by Ursula -/
structure Prices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ

/-- The conditions of Ursula's shopping trip -/
def shopping_conditions (p : Prices) : Prop :=
  p.tea = 10 ∧
  p.tea = 2 * p.cheese ∧
  p.bread = p.butter / 2 ∧
  p.butter + p.bread + p.cheese + p.tea = 21

/-- The theorem stating that under the given conditions, 
    the price of butter is 80% of the price of cheese -/
theorem butter_cheese_ratio (p : Prices) 
  (h : shopping_conditions p) : p.butter / p.cheese = 0.8 := by
  sorry

end butter_cheese_ratio_l3742_374252


namespace solve_exponential_equation_l3742_374237

theorem solve_exponential_equation :
  ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^3 ∧ x = 2 := by
  sorry

end solve_exponential_equation_l3742_374237


namespace hamburger_combinations_l3742_374205

theorem hamburger_combinations : 
  let num_condiments : ℕ := 10
  let num_bun_types : ℕ := 2
  let num_patty_choices : ℕ := 3
  (2 ^ num_condiments) * num_bun_types * num_patty_choices = 6144 :=
by
  sorry

end hamburger_combinations_l3742_374205


namespace potassium_bromate_weight_l3742_374264

/-- The atomic weight of Potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Potassium atoms in Potassium Bromate -/
def num_K : ℕ := 1

/-- The number of Bromine atoms in Potassium Bromate -/
def num_Br : ℕ := 1

/-- The number of Oxygen atoms in Potassium Bromate -/
def num_O : ℕ := 3

/-- The molecular weight of Potassium Bromate in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  num_K * atomic_weight_K + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem stating that the molecular weight of Potassium Bromate is 167.00 g/mol -/
theorem potassium_bromate_weight : molecular_weight_KBrO3 = 167.00 := by
  sorry

end potassium_bromate_weight_l3742_374264


namespace min_value_fractional_sum_l3742_374203

theorem min_value_fractional_sum (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2)) + (y^3 / (x - 2)) ≥ 96 ∧
  ((x^3 / (y - 2)) + (y^3 / (x - 2)) = 96 ↔ x = 4 ∧ y = 4) := by
  sorry

end min_value_fractional_sum_l3742_374203


namespace probability_at_most_one_first_class_l3742_374218

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of products -/
def total_products : ℕ := 5

/-- The number of first-class products -/
def first_class_products : ℕ := 3

/-- The number of second-class products -/
def second_class_products : ℕ := 2

/-- The number of products to be selected -/
def selected_products : ℕ := 2

theorem probability_at_most_one_first_class :
  (choose first_class_products 1 * choose second_class_products 1 + choose second_class_products 2) /
  choose total_products selected_products = 7 / 10 := by sorry

end probability_at_most_one_first_class_l3742_374218


namespace arrasta_um_solvable_l3742_374261

/-- Represents a move in the Arrasta Um game -/
inductive Move
| Up : Move
| Down : Move
| Left : Move
| Right : Move

/-- Represents the state of the Arrasta Um game -/
structure ArrastaUmState where
  n : Nat  -- size of the board
  blackPos : Nat × Nat  -- position of the black piece
  emptyPos : Nat × Nat  -- position of the empty cell

/-- Checks if a position is valid on the board -/
def isValidPosition (n : Nat) (pos : Nat × Nat) : Prop :=
  pos.1 < n ∧ pos.2 < n

/-- Checks if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Nat × Nat) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2 + 1 = pos2.2 ∨ pos2.2 + 1 = pos1.2)) ∨
  (pos1.2 = pos2.2 ∧ (pos1.1 + 1 = pos2.1 ∨ pos2.1 + 1 = pos1.1))

/-- Applies a move to the game state -/
def applyMove (state : ArrastaUmState) (move : Move) : ArrastaUmState :=
  match move with
  | Move.Up => { state with blackPos := (state.blackPos.1 - 1, state.blackPos.2), emptyPos := state.blackPos }
  | Move.Down => { state with blackPos := (state.blackPos.1 + 1, state.blackPos.2), emptyPos := state.blackPos }
  | Move.Left => { state with blackPos := (state.blackPos.1, state.blackPos.2 - 1), emptyPos := state.blackPos }
  | Move.Right => { state with blackPos := (state.blackPos.1, state.blackPos.2 + 1), emptyPos := state.blackPos }

/-- Checks if a move is valid -/
def isValidMove (state : ArrastaUmState) (move : Move) : Prop :=
  isAdjacent state.blackPos state.emptyPos ∧
  isValidPosition state.n (applyMove state move).blackPos

/-- Theorem: It's possible to finish Arrasta Um in 6n-8 moves on an n × n board -/
theorem arrasta_um_solvable (n : Nat) (h : n ≥ 2) :
  ∃ (moves : List Move), moves.length = 6 * n - 8 ∧
    (moves.foldl applyMove { n := n, blackPos := (n - 1, 0), emptyPos := (n - 1, 1) }).blackPos = (0, n - 1) :=
  sorry


end arrasta_um_solvable_l3742_374261


namespace points_three_units_from_negative_two_l3742_374201

theorem points_three_units_from_negative_two : 
  ∀ x : ℝ, (abs (x - (-2)) = 3) ↔ (x = -5 ∨ x = 1) := by
  sorry

end points_three_units_from_negative_two_l3742_374201


namespace tenth_term_geometric_sequence_l3742_374209

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5  -- First term
  let r : ℚ := 4/3  -- Common ratio
  let n : ℕ := 10  -- Term number we're looking for
  let a_n : ℚ := a * r^(n - 1)  -- Formula for nth term of geometric sequence
  a_n = 1310720/19683 := by
  sorry

end tenth_term_geometric_sequence_l3742_374209


namespace solution_equation1_solution_equation2_l3742_374216

-- Define the equations
def equation1 (x : ℝ) : Prop := x - 2 * Real.sqrt x + 1 = 0
def equation2 (x : ℝ) : Prop := x + 2 + Real.sqrt (x + 2) = 0

-- Theorem for the first equation
theorem solution_equation1 : ∃ (x : ℝ), equation1 x ∧ x = 1 :=
  sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ (x : ℝ), equation2 x ∧ x = -2 :=
  sorry

end solution_equation1_solution_equation2_l3742_374216


namespace profit_share_difference_l3742_374262

/-- Given the investments of A, B, and C, and B's profit share,
    prove that the difference between A's and C's profit shares is 1600. -/
theorem profit_share_difference
  (investment_A investment_B investment_C : ℕ)
  (profit_share_B : ℕ)
  (h1 : investment_A = 8000)
  (h2 : investment_B = 10000)
  (h3 : investment_C = 12000)
  (h4 : profit_share_B = 4000) :
  let total_investment := investment_A + investment_B + investment_C
  let total_profit := (total_investment * profit_share_B) / investment_B
  let profit_share_A := (investment_A * total_profit) / total_investment
  let profit_share_C := (investment_C * total_profit) / total_investment
  profit_share_C - profit_share_A = 1600 := by
sorry


end profit_share_difference_l3742_374262


namespace appliance_price_ratio_l3742_374268

theorem appliance_price_ratio : 
  ∀ (c p q : ℝ), 
  p = 0.8 * c →  -- 20% loss
  q = 1.25 * c → -- 25% profit
  q / p = 25 / 16 := by
sorry

end appliance_price_ratio_l3742_374268


namespace total_pens_count_l3742_374240

theorem total_pens_count (red_pens : ℕ) (black_pens : ℕ) (blue_pens : ℕ) 
  (h1 : red_pens = 8)
  (h2 : black_pens = red_pens + 10)
  (h3 : blue_pens = red_pens + 7) :
  red_pens + black_pens + blue_pens = 41 := by
  sorry

end total_pens_count_l3742_374240


namespace no_geometric_progression_with_1_2_5_l3742_374229

theorem no_geometric_progression_with_1_2_5 :
  ¬ ∃ (a q : ℝ) (m n p : ℕ), 
    m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    a * q^m = 1 ∧ a * q^n = 2 ∧ a * q^p = 5 :=
by sorry

end no_geometric_progression_with_1_2_5_l3742_374229


namespace candy_probability_l3742_374250

/-- Represents the number of red candies in the jar -/
def red_candies : ℕ := 15

/-- Represents the number of blue candies in the jar -/
def blue_candies : ℕ := 20

/-- Represents the total number of candies in the jar -/
def total_candies : ℕ := red_candies + blue_candies

/-- Represents the number of candies each person picks -/
def picks : ℕ := 3

/-- The probability of Terry and Mary getting the same color combination -/
def same_color_probability : ℚ := 243 / 6825

theorem candy_probability : 
  let terry_red_prob := (red_candies.choose picks : ℚ) / (total_candies.choose picks)
  let terry_blue_prob := (blue_candies.choose picks : ℚ) / (total_candies.choose picks)
  let mary_red_prob := ((red_candies - picks).choose picks : ℚ) / ((total_candies - picks).choose picks)
  let mary_blue_prob := ((blue_candies - picks).choose picks : ℚ) / ((total_candies - picks).choose picks)
  terry_red_prob * mary_red_prob + terry_blue_prob * mary_blue_prob = same_color_probability :=
sorry

end candy_probability_l3742_374250


namespace valid_course_combinations_l3742_374254

def total_courses : ℕ := 7
def required_courses : ℕ := 4
def math_courses : ℕ := 3
def other_courses : ℕ := 4

def valid_combinations : ℕ := (total_courses - 1).choose (required_courses - 1) - other_courses.choose (required_courses - 1)

theorem valid_course_combinations :
  valid_combinations = 16 :=
sorry

end valid_course_combinations_l3742_374254


namespace f_decreasing_implies_a_nonnegative_l3742_374298

/-- A function that represents f(x) = x^2 + |x - a| + b -/
def f (a b x : ℝ) : ℝ := x^2 + |x - a| + b

/-- Theorem: If f(x) is decreasing on (-∞, 0], then a ≥ 0 -/
theorem f_decreasing_implies_a_nonnegative (a b : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f a b x ≥ f a b y) → a ≥ 0 := by
  sorry

end f_decreasing_implies_a_nonnegative_l3742_374298


namespace onions_count_prove_onions_count_l3742_374286

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onion_difference : ℕ := 5200

theorem onions_count : ℕ :=
  (tomatoes + corn) - onion_difference

theorem prove_onions_count : onions_count = 985 := by
  sorry

end onions_count_prove_onions_count_l3742_374286


namespace calculation_proof_l3742_374238

theorem calculation_proof :
  (let a := 3 + 4/5
   let b := (1 - 9/10) / (1/100)
   a * b = 38) ∧
  (let c := 5/6 + 20
   let d := 5/4
   c / d = 50/3) ∧
  (3/7 * 5/9 * 28 * 45 = 300) := by
  sorry

end calculation_proof_l3742_374238


namespace sum_of_x_and_y_l3742_374257

theorem sum_of_x_and_y (x y : ℝ) (hx : 3 + x = 5) (hy : -3 + y = 5) : x + y = 10 := by
  sorry

end sum_of_x_and_y_l3742_374257


namespace platform_length_platform_length_is_150_l3742_374273

/-- Given a train passing a platform and a man, calculate the platform length -/
theorem platform_length (train_speed : Real) (platform_time : Real) (man_time : Real) : Real :=
  let train_speed_ms := train_speed * 1000 / 3600
  let train_length := train_speed_ms * man_time
  let platform_length := train_speed_ms * platform_time - train_length
  platform_length

/-- Prove that the platform length is 150 meters given the specified conditions -/
theorem platform_length_is_150 :
  platform_length 54 30 20 = 150 := by
  sorry

end platform_length_platform_length_is_150_l3742_374273


namespace find_divisor_l3742_374289

theorem find_divisor (N : ℕ) (D : ℕ) (h1 : N = 44 * 432) 
  (h2 : ∃ Q : ℕ, N = D * Q + 3) (h3 : D > 0) : D = 43 := by
  sorry

end find_divisor_l3742_374289


namespace solve_for_t_l3742_374259

theorem solve_for_t (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 236)
  (eq2 : t = 2 * s + 1) : 
  t = 487 / 29 := by
sorry

end solve_for_t_l3742_374259


namespace significant_figures_220_and_0_101_l3742_374288

/-- Represents an approximate number with its value and precision -/
structure ApproximateNumber where
  value : ℝ
  precision : ℕ

/-- Returns the number of significant figures in an approximate number -/
def significantFigures (n : ApproximateNumber) : ℕ :=
  sorry

theorem significant_figures_220_and_0_101 :
  ∃ (a b : ApproximateNumber),
    a.value = 220 ∧
    b.value = 0.101 ∧
    significantFigures a = 3 ∧
    significantFigures b = 3 :=
  sorry

end significant_figures_220_and_0_101_l3742_374288


namespace waiter_net_earnings_waiter_earnings_result_l3742_374279

/-- Calculates the waiter's net earnings from tips after commission --/
theorem waiter_net_earnings (customers : Nat) 
  (tipping_customers : Nat)
  (bill1 bill2 bill3 bill4 : ℝ)
  (tip_percent1 tip_percent2 tip_percent3 tip_percent4 : ℝ)
  (commission_rate : ℝ) : ℝ :=
  let total_tips := 
    bill1 * tip_percent1 + 
    bill2 * tip_percent2 + 
    bill3 * tip_percent3 + 
    bill4 * tip_percent4
  let commission := total_tips * commission_rate
  let net_earnings := total_tips - commission
  net_earnings

/-- The waiter's net earnings are approximately $16.82 --/
theorem waiter_earnings_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |waiter_net_earnings 9 4 25 22 35 30 0.15 0.18 0.20 0.10 0.05 - 16.82| < ε :=
sorry

end waiter_net_earnings_waiter_earnings_result_l3742_374279


namespace quadratic_from_means_l3742_374222

theorem quadratic_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 10) :
  ∀ x : ℝ, x^2 - 12*x + 100 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_from_means_l3742_374222


namespace polynomial_multiplication_l3742_374247

theorem polynomial_multiplication (a b : ℝ) : (2*a + 3*b) * (2*a - b) = 4*a^2 + 4*a*b - 3*b^2 := by
  sorry

end polynomial_multiplication_l3742_374247


namespace expected_value_is_correct_l3742_374281

def number_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_colored (n : ℕ) : Bool := sorry

def is_red (n : ℕ) : Bool := sorry

def is_blue (n : ℕ) : Bool := sorry

def probability_red : ℚ := 1/2

def probability_blue : ℚ := 1/2

def is_sum_of_red_and_blue (n : ℕ) : Bool := sorry

def expected_value : ℚ := sorry

theorem expected_value_is_correct : expected_value = 423/32 := by sorry

end expected_value_is_correct_l3742_374281


namespace cos_420_degrees_l3742_374244

theorem cos_420_degrees : Real.cos (420 * π / 180) = 1 / 2 := by
  sorry

end cos_420_degrees_l3742_374244
