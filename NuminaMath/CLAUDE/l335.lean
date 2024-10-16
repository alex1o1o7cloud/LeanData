import Mathlib

namespace NUMINAMATH_CALUDE_gcd_1729_867_l335_33526

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l335_33526


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_subset_l335_33581

def total_elements : ℕ := 4
def selected_elements : ℕ := 2
def subset_size : ℕ := 2

theorem probability_at_least_one_from_subset :
  (1 : ℚ) - (Nat.choose (total_elements - subset_size) selected_elements : ℚ) / 
  (Nat.choose total_elements selected_elements : ℚ) = 5/6 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_subset_l335_33581


namespace NUMINAMATH_CALUDE_simplify_square_roots_l335_33500

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 49 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l335_33500


namespace NUMINAMATH_CALUDE_base12_remainder_is_4_l335_33533

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of the number --/
def base12Number : List Nat := [1, 5, 3, 4]

/-- The theorem stating that the remainder of the base-12 number divided by 9 is 4 --/
theorem base12_remainder_is_4 : 
  (base12ToBase10 base12Number) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_is_4_l335_33533


namespace NUMINAMATH_CALUDE_price_reduction_profit_l335_33563

/-- Represents the daily sales and profit scenario of a product in a shopping mall -/
structure MallSales where
  initialSales : ℕ  -- Initial daily sales in units
  initialProfit : ℕ  -- Initial profit per unit in yuan
  salesIncrease : ℕ  -- Increase in sales units per yuan of price reduction
  priceReduction : ℕ  -- Price reduction per unit in yuan

/-- Calculates the daily profit based on the given sales scenario -/
def dailyProfit (m : MallSales) : ℕ :=
  (m.initialSales + m.salesIncrease * m.priceReduction) * (m.initialProfit - m.priceReduction)

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_profit (m : MallSales) 
  (h1 : m.initialSales = 30)
  (h2 : m.initialProfit = 50)
  (h3 : m.salesIncrease = 2)
  (h4 : m.priceReduction = 20) :
  dailyProfit m = 2100 := by
  sorry

#eval dailyProfit { initialSales := 30, initialProfit := 50, salesIncrease := 2, priceReduction := 20 }

end NUMINAMATH_CALUDE_price_reduction_profit_l335_33563


namespace NUMINAMATH_CALUDE_smallest_number_of_weights_l335_33547

/-- A function that determines if a given number of weights can measure all masses -/
def can_measure_all (n : ℕ) : Prop :=
  ∃ (weights : Fin n → ℝ), 
    (∀ i, weights i ≥ 0.01) ∧ 
    (∀ m : ℝ, 0 ≤ m ∧ m ≤ 20.2 → 
      ∃ (subset : Fin n → Bool), 
        abs (m - (Finset.sum (Finset.filter (λ i => subset i = true) Finset.univ) weights)) ≤ 0.01)

theorem smallest_number_of_weights : 
  (∀ k < 2020, ¬ can_measure_all k) ∧ can_measure_all 2020 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_weights_l335_33547


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l335_33508

-- Problem 1
theorem problem_1 (α : Real) (h : Real.tan (π/4 + α) = 2) :
  Real.sin (2*α) + Real.cos α ^ 2 = 3/2 := by sorry

-- Problem 2
theorem problem_2 (x₁ y₁ x₂ y₂ α : Real) 
  (h1 : x₁^2 + y₁^2 = 1) 
  (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : Real.sin α + Real.cos α = 7/17) 
  (h4 : 0 < α) (h5 : α < π) :
  x₁*x₂ + y₁*y₂ = -8/17 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l335_33508


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l335_33521

theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) 
  (quadruplets_in_lineup : ℕ) :
  total_players = 17 →
  quadruplets = 4 →
  starters = 6 →
  quadruplets_in_lineup = 2 →
  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 4290 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l335_33521


namespace NUMINAMATH_CALUDE_solution_set_inequality_l335_33532

theorem solution_set_inequality (x : ℝ) :
  x^2 - |x| - 2 ≤ 0 ↔ x ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l335_33532


namespace NUMINAMATH_CALUDE_triangle_inradius_l335_33553

/-- The inradius of a triangle with side lengths 7, 11, and 14 is 3√10 / 4 -/
theorem triangle_inradius (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 14) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  A / s = (3 * Real.sqrt 10) / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_inradius_l335_33553


namespace NUMINAMATH_CALUDE_prob_diamond_or_ace_and_heart_l335_33562

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of cards that are neither diamonds nor aces -/
def NeitherDiamondNorAce : ℕ := 36

/-- Number of hearts in a standard deck -/
def Hearts : ℕ := 13

/-- Probability of drawing a card that is neither a diamond nor an ace -/
def probNeitherDiamondNorAce : ℚ := NeitherDiamondNorAce / StandardDeck

/-- Probability of drawing a heart -/
def probHeart : ℚ := Hearts / StandardDeck

theorem prob_diamond_or_ace_and_heart :
  let probAtLeastOneDiamondOrAce := 1 - probNeitherDiamondNorAce ^ 2
  probAtLeastOneDiamondOrAce * probHeart = 88 / 676 := by
  sorry

end NUMINAMATH_CALUDE_prob_diamond_or_ace_and_heart_l335_33562


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cuboid_l335_33585

theorem sphere_surface_area_from_cuboid (a : ℝ) (h : a > 0) :
  let cuboid_dimensions := (2*a, a, a)
  let sphere_radius := Real.sqrt (3/2 * a^2)
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 6 * Real.pi * a^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cuboid_l335_33585


namespace NUMINAMATH_CALUDE_g_stable_point_fixed_points_subset_stable_points_l335_33514

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define fixed point
def is_fixed_point (f : RealFunction) (x : ℝ) : Prop := f x = x

-- Define stable point
def is_stable_point (f : RealFunction) (x : ℝ) : Prop := f (f x) = x

-- Define the set of fixed points
def fixed_points (f : RealFunction) : Set ℝ := {x | is_fixed_point f x}

-- Define the set of stable points
def stable_points (f : RealFunction) : Set ℝ := {x | is_stable_point f x}

-- Define the function g(x) = 3x - 8
def g : RealFunction := λ x ↦ 3 * x - 8

-- Theorem: The stable point of g(x) = 3x - 8 is x = 4
theorem g_stable_point : is_stable_point g 4 := by sorry

-- Theorem: For any function, the set of fixed points is a subset of the set of stable points
theorem fixed_points_subset_stable_points (f : RealFunction) : 
  fixed_points f ⊆ stable_points f := by sorry

end NUMINAMATH_CALUDE_g_stable_point_fixed_points_subset_stable_points_l335_33514


namespace NUMINAMATH_CALUDE_compute_m_3v_minus_2w_l335_33507

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Fin 2 → ℝ)

def Mv : Fin 2 → ℝ := ![3, -1]
def Mw : Fin 2 → ℝ := ![4, 3]

axiom mv_eq : M.mulVec v = Mv
axiom mw_eq : M.mulVec w = Mw

theorem compute_m_3v_minus_2w : M.mulVec (3 • v - 2 • w) = ![1, -9] := by sorry

end NUMINAMATH_CALUDE_compute_m_3v_minus_2w_l335_33507


namespace NUMINAMATH_CALUDE_black_white_area_ratio_l335_33583

/-- The ratio of black to white area in concentric circles -/
theorem black_white_area_ratio :
  let radii : Fin 5 → ℝ := ![2, 4, 6, 8, 10]
  let circle_area (r : ℝ) := π * r^2
  let ring_area (i : Fin 4) := circle_area (radii (i + 1)) - circle_area (radii i)
  let black_area := circle_area (radii 0) + ring_area 1 + ring_area 3
  let white_area := ring_area 0 + ring_area 2
  black_area / white_area = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_black_white_area_ratio_l335_33583


namespace NUMINAMATH_CALUDE_cherry_strawberry_cost_ratio_l335_33528

/-- The cost of a pound of strawberries in dollars -/
def strawberry_cost : ℚ := 2.20

/-- The cost of 5 pounds of strawberries and 5 pounds of cherries in dollars -/
def total_cost : ℚ := 77

/-- The ratio of the cost of cherries to strawberries -/
def cherry_strawberry_ratio : ℚ := 6

theorem cherry_strawberry_cost_ratio :
  ∃ (cherry_cost : ℚ),
    cherry_cost > 0 ∧
    5 * strawberry_cost + 5 * cherry_cost = total_cost ∧
    cherry_cost / strawberry_cost = cherry_strawberry_ratio :=
by sorry

end NUMINAMATH_CALUDE_cherry_strawberry_cost_ratio_l335_33528


namespace NUMINAMATH_CALUDE_fifth_term_ratio_l335_33574

/-- Two arithmetic sequences and their sum ratios -/
structure ArithmeticSequences where
  a : ℕ → ℝ  -- First arithmetic sequence
  b : ℕ → ℝ  -- Second arithmetic sequence
  S : ℕ → ℝ  -- Sum of first n terms of sequence a
  T : ℕ → ℝ  -- Sum of first n terms of sequence b
  sum_ratio : ∀ n : ℕ, S n / T n = (2 * n - 3) / (3 * n - 2)

/-- The ratio of the 5th terms of the sequences is 3/5 -/
theorem fifth_term_ratio (seq : ArithmeticSequences) : seq.a 5 / seq.b 5 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_ratio_l335_33574


namespace NUMINAMATH_CALUDE_tail_cut_divisibility_by_7_l335_33580

def tail_cut (n : ℕ) : ℕ :=
  (n / 10) - 2 * (n % 10)

def is_divisible_by_7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem tail_cut_divisibility_by_7 (A : ℕ) :
  (A > 0) →
  (is_divisible_by_7 A ↔ 
    ∃ (k : ℕ), is_divisible_by_7 (Nat.iterate tail_cut k A)) :=
by sorry

end NUMINAMATH_CALUDE_tail_cut_divisibility_by_7_l335_33580


namespace NUMINAMATH_CALUDE_number_property_l335_33549

theorem number_property (y : ℝ) : y = (1 / y) * (-y) + 3 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_property_l335_33549


namespace NUMINAMATH_CALUDE_expression_value_proof_l335_33595

theorem expression_value_proof (a b c k : ℤ) 
  (ha : a = 30) (hb : b = 25) (hc : c = 4) (hk : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_proof_l335_33595


namespace NUMINAMATH_CALUDE_complex_equation_implies_ratio_l335_33542

theorem complex_equation_implies_ratio (m n : ℝ) :
  (2 + m * Complex.I) * (n - 2 * Complex.I) = -4 - 3 * Complex.I →
  m / n = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_implies_ratio_l335_33542


namespace NUMINAMATH_CALUDE_target_walmart_tool_difference_l335_33561

/-- Represents a multitool with its components -/
structure Multitool where
  screwdrivers : Nat
  knives : Nat
  files : Nat
  scissors : Nat
  other_tools : Nat

/-- The Walmart multitool -/
def walmart_multitool : Multitool :=
  { screwdrivers := 1
    knives := 3
    files := 0
    scissors := 0
    other_tools := 2 }

/-- The Target multitool -/
def target_multitool : Multitool :=
  { screwdrivers := 1
    knives := 2 * walmart_multitool.knives
    files := 3
    scissors := 1
    other_tools := 0 }

/-- Total number of tools in a multitool -/
def total_tools (m : Multitool) : Nat :=
  m.screwdrivers + m.knives + m.files + m.scissors + m.other_tools

/-- Theorem stating the difference in the number of tools between Target and Walmart multitools -/
theorem target_walmart_tool_difference :
  total_tools target_multitool - total_tools walmart_multitool = 5 := by
  sorry


end NUMINAMATH_CALUDE_target_walmart_tool_difference_l335_33561


namespace NUMINAMATH_CALUDE_missing_number_proof_l335_33502

theorem missing_number_proof (x : ℝ) (h : 0.00375 * x = 153.75) : x = 41000 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l335_33502


namespace NUMINAMATH_CALUDE_partner_q_invest_time_l335_33534

/-- Represents the investment and profit data for three partners -/
structure PartnerData where
  investment_ratio : Fin 3 → ℚ
  profit_ratio : Fin 3 → ℚ
  p_invest_time : ℚ
  r_invest_time : ℚ

/-- Calculates the investment time for partner q given the partner data -/
def calculate_q_invest_time (data : PartnerData) : ℚ :=
  (data.investment_ratio 0 * data.p_invest_time * data.profit_ratio 1) /
  (data.investment_ratio 1 * data.profit_ratio 0)

/-- Theorem stating that partner q's investment time is 14 months -/
theorem partner_q_invest_time (data : PartnerData)
  (h1 : data.investment_ratio 0 = 7)
  (h2 : data.investment_ratio 1 = 5)
  (h3 : data.investment_ratio 2 = 3)
  (h4 : data.profit_ratio 0 = 7)
  (h5 : data.profit_ratio 1 = 14)
  (h6 : data.profit_ratio 2 = 9)
  (h7 : data.p_invest_time = 5)
  (h8 : data.r_invest_time = 9) :
  calculate_q_invest_time data = 14 := by
  sorry

end NUMINAMATH_CALUDE_partner_q_invest_time_l335_33534


namespace NUMINAMATH_CALUDE_water_boiling_point_l335_33537

/-- The temperature in Fahrenheit at which water boils -/
def boiling_point_f : ℝ := 212

/-- The temperature in Fahrenheit at which water melts -/
def melting_point_f : ℝ := 32

/-- The temperature in Celsius at which water melts -/
def melting_point_c : ℝ := 0

/-- A function to convert Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := sorry

/-- A function to convert Fahrenheit to Celsius -/
def fahrenheit_to_celsius (f : ℝ) : ℝ := sorry

/-- The boiling point of water in Celsius -/
def boiling_point_c : ℝ := 100

theorem water_boiling_point :
  ∃ (temp_c temp_f : ℝ),
    celsius_to_fahrenheit temp_c = temp_f ∧
    temp_c = 35 ∧
    temp_f = 95 →
  fahrenheit_to_celsius boiling_point_f = boiling_point_c :=
sorry

end NUMINAMATH_CALUDE_water_boiling_point_l335_33537


namespace NUMINAMATH_CALUDE_carmen_earnings_l335_33523

-- Define the sales for each house
def green_house_sales : ℕ := 3
def green_house_price : ℚ := 4

def yellow_house_thin_mints : ℕ := 2
def yellow_house_thin_mints_price : ℚ := 3.5
def yellow_house_fudge_delights : ℕ := 1
def yellow_house_fudge_delights_price : ℚ := 5

def brown_house_sales : ℕ := 9
def brown_house_price : ℚ := 2

-- Define the total earnings
def total_earnings : ℚ := 
  green_house_sales * green_house_price +
  yellow_house_thin_mints * yellow_house_thin_mints_price +
  yellow_house_fudge_delights * yellow_house_fudge_delights_price +
  brown_house_sales * brown_house_price

-- Theorem statement
theorem carmen_earnings : total_earnings = 42 := by
  sorry

end NUMINAMATH_CALUDE_carmen_earnings_l335_33523


namespace NUMINAMATH_CALUDE_only_A_is_impossible_l335_33505

-- Define the set of possible ball colors in the bag
inductive BallColor
| Red
| White

-- Define the set of possible outcomes for a dice roll
inductive DiceOutcome
| One | Two | Three | Four | Five | Six

-- Define the set of possible last digits for a license plate
inductive LicensePlateLastDigit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

-- Define the events
def event_A : Prop := ∃ (ball : BallColor), ball = BallColor.Red ∨ ball = BallColor.White

def event_B : Prop := True  -- We can't model weather prediction precisely, so we assume it's always possible

def event_C : Prop := ∃ (outcome : DiceOutcome), outcome = DiceOutcome.Six

def event_D : Prop := ∃ (digit : LicensePlateLastDigit), 
  digit = LicensePlateLastDigit.Zero ∨ 
  digit = LicensePlateLastDigit.Two ∨ 
  digit = LicensePlateLastDigit.Four ∨ 
  digit = LicensePlateLastDigit.Six ∨ 
  digit = LicensePlateLastDigit.Eight

-- Theorem stating that only event A is impossible
theorem only_A_is_impossible :
  (¬ event_A) ∧ event_B ∧ event_C ∧ event_D :=
sorry

end NUMINAMATH_CALUDE_only_A_is_impossible_l335_33505


namespace NUMINAMATH_CALUDE_jessica_attended_one_game_l335_33504

/-- The number of soccer games Jessica actually attended -/
def jessica_attended (total games : ℕ) (planned : ℕ) (skipped : ℕ) (rescheduled : ℕ) (additional_missed : ℕ) : ℕ :=
  planned - skipped - additional_missed

/-- Theorem stating that Jessica attended 1 game given the problem conditions -/
theorem jessica_attended_one_game :
  jessica_attended 12 8 3 2 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jessica_attended_one_game_l335_33504


namespace NUMINAMATH_CALUDE_initial_carrots_count_l335_33551

/-- Proves that the initial number of carrots is 300 given the problem conditions --/
theorem initial_carrots_count : ℕ :=
  let initial_carrots : ℕ := 300
  let before_lunch_fraction : ℚ := 2/5
  let after_lunch_fraction : ℚ := 3/5
  let unused_carrots : ℕ := 72

  have h1 : (1 - before_lunch_fraction) * (1 - after_lunch_fraction) * initial_carrots = unused_carrots := by sorry

  initial_carrots


end NUMINAMATH_CALUDE_initial_carrots_count_l335_33551


namespace NUMINAMATH_CALUDE_f_properties_l335_33579

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x + 1

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x ∈ Set.Icc (2 - Real.exp 1) 1) ∧
  (∀ a ≤ 0, ∃! x, f a x = 0) ∧
  (∀ a > 0, ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z, f a z = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l335_33579


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l335_33560

theorem imaginary_part_of_complex_number :
  (Complex.im ((2 : ℂ) - Complex.I * Complex.I)) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l335_33560


namespace NUMINAMATH_CALUDE_not_all_triangles_divisible_to_square_l335_33572

/-- A triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- A square with side length -/
structure Square where
  side : ℝ

/-- Represents a division of a shape into parts -/
structure Division where
  parts : ℕ

/-- Represents the ability to form a shape from parts -/
def can_form (d : Division) (s : Square) : Prop := sorry

/-- The theorem stating that not all triangles can be divided into 1000 parts to form a square -/
theorem not_all_triangles_divisible_to_square :
  ∃ t : Triangle, ¬ ∃ (d : Division) (s : Square), d.parts = 1000 ∧ can_form d s := by sorry

end NUMINAMATH_CALUDE_not_all_triangles_divisible_to_square_l335_33572


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element6_l335_33589

theorem pascal_triangle_row20_element6 : Nat.choose 20 5 = 7752 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element6_l335_33589


namespace NUMINAMATH_CALUDE_five_thirteenths_repeating_decimal_sum_l335_33557

theorem five_thirteenths_repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d + 
    (0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d) / 999999 →
  c + d = 11 :=
by sorry

end NUMINAMATH_CALUDE_five_thirteenths_repeating_decimal_sum_l335_33557


namespace NUMINAMATH_CALUDE_tamara_brownie_earnings_l335_33520

/-- Calculates the earnings from selling brownies --/
def brownie_earnings (pans : ℕ) (pieces_per_pan : ℕ) (price_per_piece : ℕ) : ℕ :=
  pans * pieces_per_pan * price_per_piece

/-- Proves that Tamara's earnings from brownies equal $32 --/
theorem tamara_brownie_earnings :
  brownie_earnings 2 8 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_tamara_brownie_earnings_l335_33520


namespace NUMINAMATH_CALUDE_final_quantities_correct_l335_33578

/-- Represents the inventory and transactions of a stationery shop -/
structure StationeryShop where
  x : ℝ  -- initial number of pencils
  y : ℝ  -- initial number of pens
  z : ℝ  -- initial number of rulers

/-- Calculates the final quantities after transactions -/
def finalQuantities (shop : StationeryShop) : ℝ × ℝ × ℝ :=
  let remainingPencils := shop.x * 0.75
  let remainingPens := shop.y * 0.60
  let remainingRulers := shop.z * 0.80
  let finalPencils := remainingPencils + remainingPencils * 2.50
  let finalPens := remainingPens + 100
  let finalRulers := remainingRulers + remainingRulers * 5
  (finalPencils, finalPens, finalRulers)

/-- Theorem stating the correctness of the final quantities calculation -/
theorem final_quantities_correct (shop : StationeryShop) :
  finalQuantities shop = (2.625 * shop.x, 0.60 * shop.y + 100, 4.80 * shop.z) := by
  sorry

end NUMINAMATH_CALUDE_final_quantities_correct_l335_33578


namespace NUMINAMATH_CALUDE_motorcycles_sold_is_eight_l335_33524

/-- Represents the monthly production and sales data for a vehicle factory -/
structure VehicleProduction where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycle_price : ℕ
  profit_increase : ℕ

/-- Calculates the number of motorcycles sold per month -/
def motorcycles_sold (data : VehicleProduction) : ℕ :=
  sorry

/-- Theorem stating that the number of motorcycles sold is 8 -/
theorem motorcycles_sold_is_eight (data : VehicleProduction) 
  (h1 : data.car_material_cost = 100)
  (h2 : data.cars_produced = 4)
  (h3 : data.car_price = 50)
  (h4 : data.motorcycle_material_cost = 250)
  (h5 : data.motorcycle_price = 50)
  (h6 : data.profit_increase = 50) :
  motorcycles_sold data = 8 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_sold_is_eight_l335_33524


namespace NUMINAMATH_CALUDE_number_problem_l335_33522

theorem number_problem (x : ℝ) : 0.5 * x = 0.1667 * x + 10 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l335_33522


namespace NUMINAMATH_CALUDE_probability_second_red_given_first_red_l335_33543

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4

theorem probability_second_red_given_first_red :
  let p_first_red := red_balls / total_balls
  let p_both_red := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))
  let p_second_red_given_first_red := p_both_red / p_first_red
  p_second_red_given_first_red = 5 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_second_red_given_first_red_l335_33543


namespace NUMINAMATH_CALUDE_first_chapter_pages_l335_33529

theorem first_chapter_pages (total_chapters : Nat) (second_chapter_pages : Nat) (third_chapter_pages : Nat) (total_pages : Nat)
  (h1 : total_chapters = 3)
  (h2 : second_chapter_pages = 35)
  (h3 : third_chapter_pages = 24)
  (h4 : total_pages = 125) :
  total_pages - (second_chapter_pages + third_chapter_pages) = 66 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l335_33529


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_radius_l335_33538

/-- Given a sphere that transforms into a hemisphere, this theorem relates the radius of the 
    hemisphere to the radius of the original sphere. -/
theorem sphere_to_hemisphere_radius (r : ℝ) (h : r = 5 * Real.rpow 2 (1/3)) : 
  ∃ R : ℝ, R = 5 ∧ (4/3) * Real.pi * R^3 = (2/3) * Real.pi * r^3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_radius_l335_33538


namespace NUMINAMATH_CALUDE_geometric_series_relation_l335_33555

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 5/7 -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (∑' n, c / d^n) = 5) :
    (∑' n, c / (c + 2*d)^n) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l335_33555


namespace NUMINAMATH_CALUDE_rectangle_area_is_144_l335_33599

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the property that circles touch the sides of the rectangle
def circles_touch_sides (r : Rectangle) : Prop :=
  r.length = 4 * circle_radius ∧ r.width = 4 * circle_radius

-- Define the area of the rectangle
def rectangle_area (r : Rectangle) : ℝ :=
  r.length * r.width

-- Theorem statement
theorem rectangle_area_is_144 (r : Rectangle) 
  (h : circles_touch_sides r) : rectangle_area r = 144 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_144_l335_33599


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l335_33550

/-- A square garden with an area of 9 square meters has a perimeter of 12 meters. -/
theorem square_garden_perimeter : 
  ∀ (side : ℝ), side > 0 → side^2 = 9 → 4 * side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l335_33550


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l335_33509

/-- Proves that the loss percentage on the first book is 15% given the problem conditions --/
theorem book_sale_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h1 : total_cost = 360)
  (h2 : cost_book1 = 210)
  (h3 : gain_percentage = 19)
  (h4 : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - (loss_percentage / 100)) ∧
    selling_price = (total_cost - cost_book1) * (1 + (gain_percentage / 100))) :
  ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
sorry


end NUMINAMATH_CALUDE_book_sale_loss_percentage_l335_33509


namespace NUMINAMATH_CALUDE_divisibility_of_n_l335_33539

theorem divisibility_of_n : ∀ (n : ℕ),
  n = (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1) →
  5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_n_l335_33539


namespace NUMINAMATH_CALUDE_routes_to_n_2_l335_33531

/-- The number of possible routes from (0, 0) to (x, y) moving only right or up -/
def f (x y : ℕ) : ℕ := sorry

/-- Theorem: The number of routes from (0, 0) to (n, 2) is (n^2 + 3n + 2) / 2 -/
theorem routes_to_n_2 (n : ℕ) : f n 2 = (n^2 + 3*n + 2) / 2 := by sorry

end NUMINAMATH_CALUDE_routes_to_n_2_l335_33531


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l335_33566

theorem quadratic_roots_distinct (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a*x^2 + b*x + c = 0 ∧ discriminant > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l335_33566


namespace NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l335_33515

/-- The number of ways to arrange n distinguishable people around a circular table,
    where rotations are considered the same arrangement -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem senate_committee_seating_arrangements :
  circularArrangements 10 = 362880 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l335_33515


namespace NUMINAMATH_CALUDE_points_form_circle_l335_33597

theorem points_form_circle :
  ∀ (t : ℝ), ∃ (x y : ℝ), x = Real.cos t ∧ y = Real.sin t → x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_points_form_circle_l335_33597


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l335_33570

/-- Given two planar vectors m and n, where m is parallel to n, 
    prove that the magnitude of n is 2√5. -/
theorem parallel_vectors_magnitude (m n : ℝ × ℝ) : 
  m = (-1, 2) → 
  n.1 = 2 → 
  (∃ k : ℝ, n = k • m) → 
  ‖n‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l335_33570


namespace NUMINAMATH_CALUDE_line_passes_through_point_with_slope_l335_33546

/-- The slope of the line -/
def m : ℝ := 2

/-- The x-coordinate of the point P -/
def x₀ : ℝ := 3

/-- The y-coordinate of the point P -/
def y₀ : ℝ := 4

/-- The equation of the line passing through (x₀, y₀) with slope m -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 2 = 0

theorem line_passes_through_point_with_slope :
  line_equation x₀ y₀ ∧ 
  ∀ x y : ℝ, line_equation x y → (y - y₀) = m * (x - x₀) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_with_slope_l335_33546


namespace NUMINAMATH_CALUDE_quadratic_roots_l335_33513

theorem quadratic_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l335_33513


namespace NUMINAMATH_CALUDE_ef_fraction_of_gh_l335_33519

/-- Given a line segment GH with points E and F on it, prove that EF is 5/11 of GH -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  (E ≤ F) → -- E is before or at F on the line
  (F ≤ H) → -- F is before or at H on the line
  (G ≤ E) → -- G is before or at E on the line
  (G - E = 5 * (H - E)) → -- GE = 5 * EH
  (G - F = 10 * (H - F)) → -- GF = 10 * FH
  F - E = 5 / 11 * (H - G) := by
sorry

end NUMINAMATH_CALUDE_ef_fraction_of_gh_l335_33519


namespace NUMINAMATH_CALUDE_pizza_theorem_l335_33510

def pizza_problem (total_served : ℕ) (successfully_served : ℕ) : Prop :=
  total_served - successfully_served = 6

theorem pizza_theorem : pizza_problem 9 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l335_33510


namespace NUMINAMATH_CALUDE_quadrilateral_area_implies_k_l335_33535

/-- A quadrilateral with vertices A(0,3), B(0,k), C(5,10), and D(5,0) -/
structure Quadrilateral (k : ℝ) :=
  (A : ℝ × ℝ := (0, 3))
  (B : ℝ × ℝ := (0, k))
  (C : ℝ × ℝ := (5, 10))
  (D : ℝ × ℝ := (5, 0))

/-- The area of a quadrilateral -/
def area (q : Quadrilateral k) : ℝ :=
  sorry

/-- Theorem stating that if k > 3 and the area of the quadrilateral is 50, then k = 13 -/
theorem quadrilateral_area_implies_k (k : ℝ) (q : Quadrilateral k)
    (h1 : k > 3)
    (h2 : area q = 50) :
  k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_implies_k_l335_33535


namespace NUMINAMATH_CALUDE_max_value_of_g_l335_33590

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l335_33590


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l335_33567

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 689 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l335_33567


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l335_33527

/-- A rectangle with perimeter 60 meters and area 221 square meters has a shorter side of 13 meters. -/
theorem rectangle_shorter_side (a b : ℝ) : 
  a > 0 → b > 0 →  -- positive sides
  2 * (a + b) = 60 →  -- perimeter condition
  a * b = 221 →  -- area condition
  min a b = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l335_33527


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l335_33588

-- Define the two lines
def line1 (x y : ℝ) : Prop := 6 * x - 8 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define what it means for a point to be between two lines
def between (x y : ℝ) : Prop :=
  ∃ (y1 y2 : ℝ), line1 x y1 ∧ line2 x y2 ∧ ((y1 < y ∧ y < y2) ∨ (y2 < y ∧ y < y1))

-- Theorem statement
theorem point_between_parallel_lines :
  ∀ b : ℤ, between 5 (b : ℝ) → b = 4 := by sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l335_33588


namespace NUMINAMATH_CALUDE_sin_330_degrees_l335_33548

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l335_33548


namespace NUMINAMATH_CALUDE_log_27_3_l335_33582

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  -- Define 27 as 3³
  have h : 27 = 3^3 := by norm_num
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_log_27_3_l335_33582


namespace NUMINAMATH_CALUDE_only_ball_draw_is_classical_l335_33584

/-- Represents a probability experiment -/
inductive Experiment
| ballDraw
| busWait
| coinToss
| waterTest

/-- Checks if an experiment has a finite number of outcomes -/
def isFinite (e : Experiment) : Prop :=
  match e with
  | Experiment.ballDraw => true
  | Experiment.busWait => false
  | Experiment.coinToss => true
  | Experiment.waterTest => false

/-- Checks if an experiment has equally likely outcomes -/
def isEquallyLikely (e : Experiment) : Prop :=
  match e with
  | Experiment.ballDraw => true
  | Experiment.busWait => false
  | Experiment.coinToss => false
  | Experiment.waterTest => false

/-- Defines a classical probability model -/
def isClassicalProbabilityModel (e : Experiment) : Prop :=
  isFinite e ∧ isEquallyLikely e

/-- Theorem stating that only the ball draw experiment is a classical probability model -/
theorem only_ball_draw_is_classical : 
  ∀ e : Experiment, isClassicalProbabilityModel e ↔ e = Experiment.ballDraw :=
by sorry

end NUMINAMATH_CALUDE_only_ball_draw_is_classical_l335_33584


namespace NUMINAMATH_CALUDE_parabola_y_comparison_l335_33541

/-- Given a parabola y = -x² + 4x + c, prove that the y-coordinate of the point (-1, y₁) 
    is less than the y-coordinate of the point (1, y₂) on this parabola. -/
theorem parabola_y_comparison (c : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = -(-1)^2 + 4*(-1) + c) 
  (h₂ : y₂ = -(1)^2 + 4*(1) + c) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_comparison_l335_33541


namespace NUMINAMATH_CALUDE_order_of_exponentials_l335_33592

theorem order_of_exponentials :
  let a : ℝ := (2 : ℝ) ^ (4/5)
  let b : ℝ := (4 : ℝ) ^ (2/7)
  let c : ℝ := (25 : ℝ) ^ (1/5)
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_order_of_exponentials_l335_33592


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l335_33545

/-- The number of ways to arrange 8 athletes on 8 tracks with 3 specified athletes in consecutive tracks -/
def arrangement_count : ℕ := 4320

/-- The number of tracks in the stadium -/
def num_tracks : ℕ := 8

/-- The total number of athletes -/
def num_athletes : ℕ := 8

/-- The number of specified athletes that must be in consecutive tracks -/
def num_specified : ℕ := 3

/-- The number of ways to arrange the specified athletes in consecutive tracks -/
def consecutive_arrangements : ℕ := num_tracks - num_specified + 1

/-- The number of permutations of the specified athletes -/
def specified_permutations : ℕ := Nat.factorial num_specified

/-- The number of permutations of the remaining athletes -/
def remaining_permutations : ℕ := Nat.factorial (num_athletes - num_specified)

theorem arrangement_count_proof : 
  arrangement_count = consecutive_arrangements * specified_permutations * remaining_permutations :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l335_33545


namespace NUMINAMATH_CALUDE_sheridan_fish_problem_l335_33569

/-- The number of fish Mrs. Sheridan gave to her sister -/
def fish_given (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem sheridan_fish_problem :
  fish_given 47.0 25 = 22 :=
by sorry

end NUMINAMATH_CALUDE_sheridan_fish_problem_l335_33569


namespace NUMINAMATH_CALUDE_sufficient_condition_range_exclusive_or_range_l335_33586

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

/-- Definition of proposition q -/
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

/-- Theorem for part (1) -/
theorem sufficient_condition_range (m : ℝ) :
  (∀ x : ℝ, p x → q m x) → m ∈ Set.Ici 4 :=
sorry

/-- Theorem for part (2) -/
theorem exclusive_or_range (x : ℝ) :
  (p x ∨ q 5 x) ∧ ¬(p x ∧ q 5 x) → x ∈ Set.Icc (-3) (-2) ∪ Set.Ioo 6 7 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_exclusive_or_range_l335_33586


namespace NUMINAMATH_CALUDE_cheryl_skittles_l335_33565

/-- 
Given that Cheryl starts with a certain number of Skittles and receives additional Skittles,
this theorem proves the total number of Skittles Cheryl ends up with.
-/
theorem cheryl_skittles (initial : ℕ) (additional : ℕ) :
  initial = 8 → additional = 89 → initial + additional = 97 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_skittles_l335_33565


namespace NUMINAMATH_CALUDE_total_flowers_l335_33556

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) 
  (h1 : num_pots = 544) (h2 : flowers_per_pot = 32) : 
  num_pots * flowers_per_pot = 17408 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l335_33556


namespace NUMINAMATH_CALUDE_division_problem_l335_33575

theorem division_problem : (((120 / 5) / 2) / 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l335_33575


namespace NUMINAMATH_CALUDE_karthik_weight_upper_bound_l335_33598

-- Define the lower and upper bounds for Karthik's weight according to different opinions
def karthik_lower_bound : ℝ := 55
def brother_lower_bound : ℝ := 50
def brother_upper_bound : ℝ := 60
def father_upper_bound : ℝ := 58

-- Define the average weight
def average_weight : ℝ := 56.5

-- Define Karthik's upper bound as a variable
def karthik_upper_bound : ℝ := sorry

-- Theorem statement
theorem karthik_weight_upper_bound :
  karthik_lower_bound < karthik_upper_bound ∧
  brother_lower_bound < karthik_upper_bound ∧
  karthik_upper_bound ≤ brother_upper_bound ∧
  karthik_upper_bound ≤ father_upper_bound ∧
  average_weight = (karthik_lower_bound + karthik_upper_bound) / 2 →
  karthik_upper_bound = 58 := by sorry

end NUMINAMATH_CALUDE_karthik_weight_upper_bound_l335_33598


namespace NUMINAMATH_CALUDE_gcd_63_84_l335_33540

theorem gcd_63_84 : Nat.gcd 63 84 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_63_84_l335_33540


namespace NUMINAMATH_CALUDE_product_of_digits_5432_base8_l335_33577

/-- Convert a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_digits_5432_base8 :
  productOfList (toBase8 5432) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_5432_base8_l335_33577


namespace NUMINAMATH_CALUDE_infinite_series_sum_l335_33559

/-- The sum of the infinite series ∑(k=1 to ∞) [12^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1)))] is equal to 3. -/
theorem infinite_series_sum : 
  (∑' k, (12 : ℝ)^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))) = 3 :=
sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l335_33559


namespace NUMINAMATH_CALUDE_share_a_correct_l335_33530

/-- Calculates the share of profit for partner A given the investment details and total profit -/
def calculate_share_a (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * change_month + (initial_a - withdraw_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + advance_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

theorem share_a_correct (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) :
  initial_a = 3000 →
  initial_b = 4000 →
  withdraw_a = 1000 →
  advance_b = 1000 →
  total_months = 12 →
  change_month = 8 →
  total_profit = 840 →
  calculate_share_a initial_a initial_b withdraw_a advance_b total_months change_month total_profit = 320 :=
by sorry

end NUMINAMATH_CALUDE_share_a_correct_l335_33530


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_l335_33576

theorem integral_one_plus_sin : ∫ x in -π..π, (1 + Real.sin x) = 2 * π := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_l335_33576


namespace NUMINAMATH_CALUDE_m_less_than_two_l335_33506

open Real

/-- Proposition p -/
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + 1 > 0

/-- Proposition q -/
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 ≤ 0

/-- The main theorem -/
theorem m_less_than_two (m : ℝ) (h : ¬(p m) ∨ ¬(q m)) : m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_two_l335_33506


namespace NUMINAMATH_CALUDE_zeros_in_fraction_l335_33554

-- Define the fraction
def fraction : ℚ := 18 / 50000

-- Define the function to count zeros after the decimal point
def count_zeros_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem zeros_in_fraction :
  count_zeros_after_decimal fraction = 3 := by sorry

end NUMINAMATH_CALUDE_zeros_in_fraction_l335_33554


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l335_33573

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), y - 5 > 4*y - 1 → y ≤ x) ∧ (x - 5 > 4*x - 1) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l335_33573


namespace NUMINAMATH_CALUDE_problem_solution_l335_33558

theorem problem_solution (a z : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * z) : z = 2205 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l335_33558


namespace NUMINAMATH_CALUDE_hyperbola_equation_l335_33594

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0) with focal length 2√5,
    and a parabola y = (1/4)x² + 1/4 tangent to its asymptote,
    prove that the equation of the hyperbola C is x²/4 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_focal : 5 = a^2 + b^2)
  (h_tangent : ∃ (x : ℝ), (1/4) * x^2 + (1/4) = (b/a) * x) :
  a = 2 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l335_33594


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l335_33511

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ∈ Set.Ici (-2) → x + 3 ≥ 1)) ↔ 
  (∃ x : ℝ, x ∈ Set.Ici (-2) ∧ x + 3 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l335_33511


namespace NUMINAMATH_CALUDE_pet_insurance_coverage_percentage_l335_33518

theorem pet_insurance_coverage_percentage
  (insurance_duration : ℕ)
  (insurance_monthly_cost : ℚ)
  (procedure_cost : ℚ)
  (amount_saved : ℚ)
  (h1 : insurance_duration = 24)
  (h2 : insurance_monthly_cost = 20)
  (h3 : procedure_cost = 5000)
  (h4 : amount_saved = 3520)
  : (1 - (amount_saved / procedure_cost)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_coverage_percentage_l335_33518


namespace NUMINAMATH_CALUDE_unique_number_property_l335_33596

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l335_33596


namespace NUMINAMATH_CALUDE_seven_classes_matches_l335_33544

/-- 
Given a number of classes, calculates the total number of matches 
when each class plays against every other class exactly once.
-/
def totalMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- 
Theorem: When 7 classes play against each other once, 
the total number of matches is 21.
-/
theorem seven_classes_matches : totalMatches 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_classes_matches_l335_33544


namespace NUMINAMATH_CALUDE_shaded_area_is_73_l335_33568

/-- The total area of two overlapping rectangles minus their common area -/
def total_shaded_area (length1 width1 length2 width2 overlap_area : ℕ) : ℕ :=
  length1 * width1 + length2 * width2 - overlap_area

/-- Theorem stating that the total shaded area is 73 for the given dimensions -/
theorem shaded_area_is_73 :
  total_shaded_area 8 5 4 9 3 = 73 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_73_l335_33568


namespace NUMINAMATH_CALUDE_team_selections_count_l335_33501

/-- The number of ways to select a team leader and a secretary from a team of 5 members -/
def team_selections : ℕ := 5 * 4

/-- Theorem stating that the number of ways to select a team leader and a secretary 
    from a team of 5 members is 20 -/
theorem team_selections_count : team_selections = 20 := by
  sorry

end NUMINAMATH_CALUDE_team_selections_count_l335_33501


namespace NUMINAMATH_CALUDE_asaf_age_l335_33564

theorem asaf_age :
  ∀ (asaf_age alexander_age asaf_pencils : ℕ),
    -- Sum of ages is 140
    asaf_age + alexander_age = 140 →
    -- Age difference is half of Asaf's pencils
    alexander_age - asaf_age = asaf_pencils / 2 →
    -- Total pencils is 220
    asaf_pencils + (asaf_pencils + 60) = 220 →
    -- Asaf's age is 90
    asaf_age = 90 := by
  sorry

end NUMINAMATH_CALUDE_asaf_age_l335_33564


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l335_33517

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + (3 * x) + 15 + (3 * x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l335_33517


namespace NUMINAMATH_CALUDE_xy_value_l335_33591

theorem xy_value (x y : ℝ) 
  (h1 : (4:ℝ)^x / (2:ℝ)^(x+y) = 16)
  (h2 : (9:ℝ)^(x+y) / (3:ℝ)^(5*y) = 81) : 
  x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l335_33591


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l335_33552

/-- Hyperbola C with equation x²/16 - y²/4 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 16 - y^2 / 4 = 1

/-- Point P with coordinates (0, 3) -/
def point_P : ℝ × ℝ := (0, 3)

/-- Line l passing through point P -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- Condition for point A to be on the hyperbola C and line l -/
def point_A_condition (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola_C x y ∧ y = line_l k x ∧ x > 0

/-- Condition for point B to be on the hyperbola C and line l -/
def point_B_condition (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola_C x y ∧ y = line_l k x ∧ x > 0

/-- Condition for point D to be on line l -/
def point_D_condition (k : ℝ) (x y : ℝ) : Prop :=
  y = line_l k x ∧ (x, y) ≠ point_P

/-- Condition for the cross ratio equality |PA| * |DB| = |PB| * |DA| -/
def cross_ratio_condition (xa ya xb yb xd yd : ℝ) : Prop :=
  (xa - 0) * (xd - xb) = (xb - 0) * (xd - xa)

theorem hyperbola_intersection_theorem (k : ℝ) 
  (xa ya xb yb xd yd : ℝ) :
  point_A_condition k xa ya →
  point_B_condition k xb yb →
  point_D_condition k xd yd →
  (xa ≠ xb) →
  cross_ratio_condition xa ya xb yb xd yd →
  yd = -4/3 := by sorry


end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l335_33552


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l335_33593

theorem smallest_number_divisible (n : ℕ) : n = 746 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m - 18 = 8 * k₁ ∧ 
    m - 18 = 14 * k₂ ∧ 
    m - 18 = 26 * k₃ ∧ 
    m - 18 = 28 * k₄)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n - 18 = 8 * k₁ ∧ 
    n - 18 = 14 * k₂ ∧ 
    n - 18 = 26 * k₃ ∧ 
    n - 18 = 28 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l335_33593


namespace NUMINAMATH_CALUDE_toy_store_problem_l335_33587

/-- Toy store problem -/
theorem toy_store_problem 
  (cost_sum : ℝ) 
  (budget_A budget_B : ℝ) 
  (total_toys : ℕ) 
  (max_A : ℕ) 
  (total_budget : ℝ) 
  (sell_price_A sell_price_B : ℝ) :
  cost_sum = 40 →
  budget_A = 90 →
  budget_B = 150 →
  total_toys = 48 →
  max_A = 23 →
  total_budget = 1000 →
  sell_price_A = 30 →
  sell_price_B = 45 →
  ∃ (cost_A cost_B : ℝ) (num_plans : ℕ) (profit_function : ℕ → ℝ) (max_profit : ℝ),
    cost_A + cost_B = cost_sum ∧
    budget_A / cost_A = budget_B / cost_B ∧
    cost_A = 15 ∧
    cost_B = 25 ∧
    num_plans = 4 ∧
    (∀ m : ℕ, profit_function m = -5 * m + 960) ∧
    max_profit = 860 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_problem_l335_33587


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal_l335_33571

theorem x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  let y := x + 1/x
  (x^2 + 1/x^2 = y^2 - 2) ∧ (x^3 + 1/x^3 = y^3 - 3*y) := by
  sorry

#check x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal_l335_33571


namespace NUMINAMATH_CALUDE_specific_triangle_l335_33503

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- The main theorem about the specific acute triangle -/
theorem specific_triangle (t : AcuteTriangle) 
  (h1 : t.a = 2 * t.b * Real.sin t.A)
  (h2 : t.a = 3 * Real.sqrt 3)
  (h3 : t.c = 5) :
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_specific_triangle_l335_33503


namespace NUMINAMATH_CALUDE_jed_card_collection_l335_33536

def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (cards_given_biweekly : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + weeks * cards_per_week - (weeks / 2) * cards_given_biweekly

theorem jed_card_collection (target_cards : ℕ) : 
  cards_after_weeks 20 6 2 4 = target_cards ∧ target_cards = 40 :=
by sorry

end NUMINAMATH_CALUDE_jed_card_collection_l335_33536


namespace NUMINAMATH_CALUDE_first_project_depth_l335_33525

-- Define the parameters for the first digging project
def length1 : ℝ := 25
def breadth1 : ℝ := 30
def days1 : ℝ := 12

-- Define the parameters for the second digging project
def length2 : ℝ := 20
def breadth2 : ℝ := 50
def depth2 : ℝ := 75
def days2 : ℝ := 12

-- Define the function to calculate volume
def volume (length : ℝ) (breadth : ℝ) (depth : ℝ) : ℝ :=
  length * breadth * depth

-- Theorem statement
theorem first_project_depth :
  ∃ (depth1 : ℝ),
    volume length1 breadth1 depth1 = volume length2 breadth2 depth2 ∧
    depth1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_project_depth_l335_33525


namespace NUMINAMATH_CALUDE_coefficient_of_expression_l335_33516

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (expression : ℚ) : ℚ := sorry

/-- The expression -2ab/3 -/
def expression : ℚ := -2 / 3

theorem coefficient_of_expression :
  coefficient expression = -2 / 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_expression_l335_33516


namespace NUMINAMATH_CALUDE_subcommittees_with_coach_count_l335_33512

def total_members : ℕ := 12
def coach_members : ℕ := 5
def subcommittee_size : ℕ := 5

def subcommittees_with_coach : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - coach_members) subcommittee_size

theorem subcommittees_with_coach_count : subcommittees_with_coach = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittees_with_coach_count_l335_33512
