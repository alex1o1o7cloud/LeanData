import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_multiplication_sum_l4052_405212

theorem two_digit_multiplication_sum (a b : ℕ) : 
  a ≥ 10 ∧ a < 100 ∧ b ≥ 10 ∧ b < 100 →
  a * (b + 40) = 2496 →
  a * b = 936 →
  a + b = 63 := by
sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_sum_l4052_405212


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l4052_405247

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic (b c x : ℂ) : ℂ := x^2 + b*x + c

theorem quadratic_root_implies_coefficients :
  ∀ (b c : ℝ), quadratic b c (2 - i) = 0 → b = -4 ∧ c = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l4052_405247


namespace NUMINAMATH_CALUDE_cricketer_average_score_l4052_405273

/-- Represents a cricketer's scoring statistics -/
structure CricketerStats where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverageScore (stats : CricketerStats) : ℕ :=
  sorry

theorem cricketer_average_score 
  (stats : CricketerStats) 
  (h1 : stats.innings = 19) 
  (h2 : stats.lastInningScore = 98) 
  (h3 : stats.averageIncrease = 4) : 
  newAverageScore stats = 26 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l4052_405273


namespace NUMINAMATH_CALUDE_projectile_trajectory_area_l4052_405246

theorem projectile_trajectory_area 
  (u : ℝ) 
  (k : ℝ) 
  (φ : ℝ) 
  (h_φ_range : 30 * π / 180 ≤ φ ∧ φ ≤ 150 * π / 180) 
  (h_u_pos : u > 0) 
  (h_k_pos : k > 0) : 
  ∃ d : ℝ, d = π / 8 ∧ 
    (∀ x y : ℝ, (x^2 / (u^2 / (2 * k))^2 + (y - u^2 / (4 * k))^2 / (u^2 / (4 * k))^2 = 1) → 
      π * (u^2 / (2 * k)) * (u^2 / (4 * k)) = d * u^4 / k^2) := by
sorry

end NUMINAMATH_CALUDE_projectile_trajectory_area_l4052_405246


namespace NUMINAMATH_CALUDE_rs_length_l4052_405299

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let XY := dist t.X t.Y
  let YZ := dist t.Y t.Z
  let ZX := dist t.Z t.X
  XY = 13 ∧ YZ = 14 ∧ ZX = 15

-- Define the median XM
def isMedian (t : Triangle) (M : ℝ × ℝ) : Prop :=
  dist t.X M = dist M ((t.Y.1 + t.Z.1, t.Y.2 + t.Z.2))

-- Define points G and F
def isOnSide (A B P : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧ P = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

-- Define angle bisectors
def isAngleBisector (t : Triangle) (P : ℝ × ℝ) (V : ℝ × ℝ) : Prop :=
  ∃ G F : ℝ × ℝ, 
    isOnSide t.Z t.X G ∧ 
    isOnSide t.X t.Y F ∧
    dist t.Y G * dist t.Z V = dist t.Z G * dist t.Y V ∧
    dist t.Z F * dist t.X V = dist t.X F * dist t.Z V

-- Define the theorem
theorem rs_length (t : Triangle) (M R S : ℝ × ℝ) :
  isValidTriangle t →
  isMedian t M →
  isAngleBisector t R t.Y →
  isAngleBisector t S t.Z →
  dist R S = 129 / 203 :=
sorry


end NUMINAMATH_CALUDE_rs_length_l4052_405299


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4052_405297

/-- Given that y^4 varies inversely with ⁴√z, prove that z = 1/4096 when y = 6, given that y = 3 when z = 16 -/
theorem inverse_variation_problem (y z : ℝ) (h1 : ∃ k : ℝ, ∀ y z, y^4 * z^(1/4) = k) 
  (h2 : 3^4 * 16^(1/4) = 6^4 * z^(1/4)) : 
  y = 6 → z = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4052_405297


namespace NUMINAMATH_CALUDE_tank_filling_time_l4052_405244

/-- Represents the time (in hours) it takes to fill the tank without the hole -/
def T : ℝ := 15

/-- Represents the time (in hours) it takes to fill the tank with the hole -/
def fill_time_with_hole : ℝ := 20

/-- Represents the time (in hours) it takes for the hole to empty the full tank -/
def empty_time : ℝ := 60

theorem tank_filling_time :
  (1 / T - 1 / empty_time = 1 / fill_time_with_hole) ∧
  (T > 0) ∧ (fill_time_with_hole > 0) ∧ (empty_time > 0) :=
sorry

end NUMINAMATH_CALUDE_tank_filling_time_l4052_405244


namespace NUMINAMATH_CALUDE_possible_sets_for_B_l4052_405230

def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {1, 2, 3, 4}

theorem possible_sets_for_B (B : Set ℕ) 
  (h1 : set_A ⊆ B) (h2 : B ⊆ set_B) :
  B = set_A ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end NUMINAMATH_CALUDE_possible_sets_for_B_l4052_405230


namespace NUMINAMATH_CALUDE_shopping_lottery_results_l4052_405240

/-- Represents the lottery event with 10 coupons -/
structure LotteryEvent where
  total_coupons : Nat
  first_prize_coupons : Nat
  second_prize_coupons : Nat
  non_prize_coupons : Nat
  first_prize_value : Nat
  second_prize_value : Nat
  drawn_coupons : Nat

/-- The specific lottery event described in the problem -/
def shopping_lottery : LotteryEvent :=
  { total_coupons := 10
  , first_prize_coupons := 1
  , second_prize_coupons := 3
  , non_prize_coupons := 6
  , first_prize_value := 50
  , second_prize_value := 10
  , drawn_coupons := 2
  }

/-- The probability of winning a prize in the shopping lottery -/
def win_probability (l : LotteryEvent) : Rat :=
  1 - (Nat.choose l.non_prize_coupons l.drawn_coupons) / (Nat.choose l.total_coupons l.drawn_coupons)

/-- The mathematical expectation of the total prize value in the shopping lottery -/
def prize_expectation (l : LotteryEvent) : Rat :=
  let p0 := (Nat.choose l.non_prize_coupons l.drawn_coupons) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p10 := (Nat.choose l.second_prize_coupons 1 * Nat.choose l.non_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p20 := (Nat.choose l.second_prize_coupons 2) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p50 := (Nat.choose l.first_prize_coupons 1 * Nat.choose l.non_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p60 := (Nat.choose l.first_prize_coupons 1 * Nat.choose l.second_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  0 * p0 + 10 * p10 + 20 * p20 + 50 * p50 + 60 * p60

theorem shopping_lottery_results :
  win_probability shopping_lottery = 2/3 ∧
  prize_expectation shopping_lottery = 16 := by
  sorry

end NUMINAMATH_CALUDE_shopping_lottery_results_l4052_405240


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l4052_405248

theorem lawn_mowing_time (mary_rate tom_rate : ℚ) (mary_time : ℚ) : 
  mary_rate = 1 / 3 →
  tom_rate = 1 / 6 →
  mary_time = 2 →
  (1 - mary_rate * mary_time) / tom_rate = 2 :=
by sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l4052_405248


namespace NUMINAMATH_CALUDE_range_of_a_l4052_405283

theorem range_of_a (a : ℝ) : 
  ((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ 
   (∃ x : ℝ, x^2 - x + a = 0)) ∧ 
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
    (∃ x : ℝ, x^2 - x + a = 0)) ↔ 
  a < 0 ∨ (1/4 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4052_405283


namespace NUMINAMATH_CALUDE_calculation_proof_l4052_405255

theorem calculation_proof : 2 * (-1/4) - |1 - Real.sqrt 3| + (-2023)^0 = 3/2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4052_405255


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l4052_405267

/-- Given two points A(a, 3) and B(-4, b) that are symmetric with respect to the origin,
    prove that a - b = 7 -/
theorem symmetric_points_difference (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (a, 3) ∧ B = (-4, b) ∧ A = (-B.1, -B.2)) →
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l4052_405267


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_299_l4052_405278

theorem greatest_prime_factor_of_299 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 299 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 299 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_299_l4052_405278


namespace NUMINAMATH_CALUDE_james_out_of_pocket_cost_l4052_405238

/-- Calculates the out-of-pocket cost for a given service -/
def outOfPocketCost (cost : ℝ) (coveragePercent : ℝ) : ℝ :=
  cost - (cost * coveragePercent)

/-- Theorem: James's total out-of-pocket cost is $262.70 -/
theorem james_out_of_pocket_cost : 
  let consultation_cost : ℝ := 300
  let consultation_coverage : ℝ := 0.83
  let xray_cost : ℝ := 150
  let xray_coverage : ℝ := 0.74
  let medication_cost : ℝ := 75
  let medication_coverage : ℝ := 0.55
  let therapy_cost : ℝ := 120
  let therapy_coverage : ℝ := 0.62
  let equipment_cost : ℝ := 85
  let equipment_coverage : ℝ := 0.49
  let followup_cost : ℝ := 200
  let followup_coverage : ℝ := 0.75
  
  (outOfPocketCost consultation_cost consultation_coverage +
   outOfPocketCost xray_cost xray_coverage +
   outOfPocketCost medication_cost medication_coverage +
   outOfPocketCost therapy_cost therapy_coverage +
   outOfPocketCost equipment_cost equipment_coverage +
   outOfPocketCost followup_cost followup_coverage) = 262.70 := by
  sorry


end NUMINAMATH_CALUDE_james_out_of_pocket_cost_l4052_405238


namespace NUMINAMATH_CALUDE_sixteen_percent_of_forty_percent_of_93_75_l4052_405217

theorem sixteen_percent_of_forty_percent_of_93_75 : 
  (0.16 * (0.4 * 93.75)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_percent_of_forty_percent_of_93_75_l4052_405217


namespace NUMINAMATH_CALUDE_ball_max_height_l4052_405202

/-- The height function of a ball thrown upwards -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 130 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l4052_405202


namespace NUMINAMATH_CALUDE_meal_combinations_l4052_405292

theorem meal_combinations (n : ℕ) (h : n = 15) : n * (n - 1) = 210 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l4052_405292


namespace NUMINAMATH_CALUDE_wall_height_proof_l4052_405291

/-- Proves that the height of each wall is 2 meters given the painting conditions --/
theorem wall_height_proof (num_walls : ℕ) (wall_width : ℝ) (paint_rate : ℝ) 
  (total_time : ℝ) (spare_time : ℝ) :
  num_walls = 5 →
  wall_width = 3 →
  paint_rate = 1 / 10 →
  total_time = 10 →
  spare_time = 5 →
  ∃ (wall_height : ℝ), 
    wall_height = 2 ∧ 
    (total_time - spare_time) * 60 * paint_rate = num_walls * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_wall_height_proof_l4052_405291


namespace NUMINAMATH_CALUDE_next_simultaneous_event_is_180_lcm_9_60_is_180_l4052_405269

/-- Represents the interval in minutes between lighting up events -/
def light_interval : ℕ := 9

/-- Represents the interval in minutes between chiming events -/
def chime_interval : ℕ := 60

/-- Calculates the next time both events occur simultaneously -/
def next_simultaneous_event : ℕ := Nat.lcm light_interval chime_interval

/-- Theorem stating that the next simultaneous event occurs after 180 minutes -/
theorem next_simultaneous_event_is_180 : next_simultaneous_event = 180 := by
  sorry

/-- Theorem stating that 180 minutes is the least common multiple of 9 and 60 -/
theorem lcm_9_60_is_180 : Nat.lcm 9 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_event_is_180_lcm_9_60_is_180_l4052_405269


namespace NUMINAMATH_CALUDE_triangle_area_approx_l4052_405261

-- Define the triangle DEF and point Q
structure Triangle :=
  (D E F Q : ℝ × ℝ)

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  let d_to_q := Real.sqrt ((t.D.1 - t.Q.1)^2 + (t.D.2 - t.Q.2)^2)
  let e_to_q := Real.sqrt ((t.E.1 - t.Q.1)^2 + (t.E.2 - t.Q.2)^2)
  let f_to_q := Real.sqrt ((t.F.1 - t.Q.1)^2 + (t.F.2 - t.Q.2)^2)
  d_to_q = 5 ∧ e_to_q = 13 ∧ f_to_q = 12

def is_equilateral (t : Triangle) : Prop :=
  let d_to_e := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let e_to_f := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let f_to_d := Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)
  d_to_e = e_to_f ∧ e_to_f = f_to_d

-- Define the theorem
theorem triangle_area_approx (t : Triangle) 
  (h1 : is_valid_triangle t) (h2 : is_equilateral t) : 
  ∃ (area : ℝ), abs (area - 132) < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l4052_405261


namespace NUMINAMATH_CALUDE_cycle_cut_orthogonality_l4052_405295

-- Define a graph
structure Graph where
  V : Type
  E : Type
  incident : E → V → Prop

-- Define cycle space and cut space
def CycleSpace (G : Graph) : Type := sorry
def CutSpace (G : Graph) : Type := sorry

-- Define orthogonal complement
def OrthogonalComplement (S : Type) : Type := sorry

-- State the theorem
theorem cycle_cut_orthogonality (G : Graph) :
  (CycleSpace G = OrthogonalComplement (CutSpace G)) ∧
  (CutSpace G = OrthogonalComplement (CycleSpace G)) := by
  sorry

end NUMINAMATH_CALUDE_cycle_cut_orthogonality_l4052_405295


namespace NUMINAMATH_CALUDE_intersection_perpendicular_bisector_l4052_405289

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem intersection_perpendicular_bisector :
  ∀ A B : ℝ × ℝ,
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B →
  ∀ x y : ℝ,
  perp_bisector x y ↔
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_bisector_l4052_405289


namespace NUMINAMATH_CALUDE_product_equivalence_l4052_405249

theorem product_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 
  5^128 - 4^128 := by sorry

end NUMINAMATH_CALUDE_product_equivalence_l4052_405249


namespace NUMINAMATH_CALUDE_square_sum_of_linear_equations_l4052_405203

theorem square_sum_of_linear_equations (x y : ℝ) 
  (eq1 : 3 * x + y = 20) 
  (eq2 : 4 * x + y = 25) : 
  x^2 + y^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_linear_equations_l4052_405203


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l4052_405254

theorem largest_n_binomial_sum : 
  (∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
    (∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m)) → 
  (∃ n : ℕ, n = 7 ∧ (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
    (∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l4052_405254


namespace NUMINAMATH_CALUDE_difference_of_squares_ratio_l4052_405252

theorem difference_of_squares_ratio : 
  (1732^2 - 1725^2) / (1739^2 - 1718^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_ratio_l4052_405252


namespace NUMINAMATH_CALUDE_age_difference_32_12_l4052_405201

/-- The difference in ages between two people given their present ages -/
def age_difference (elder_age younger_age : ℕ) : ℕ :=
  elder_age - younger_age

/-- Theorem stating the age difference between two people with given ages -/
theorem age_difference_32_12 :
  age_difference 32 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_32_12_l4052_405201


namespace NUMINAMATH_CALUDE_coin_collection_problem_l4052_405225

/-- Represents the state of a coin collection --/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- Calculates the ratio of gold to silver coins --/
def goldSilverRatio (c : CoinCollection) : ℚ :=
  c.gold / c.silver

/-- Represents the coin collection problem --/
theorem coin_collection_problem 
  (initial : CoinCollection)
  (final : CoinCollection)
  (added_gold : ℕ) :
  goldSilverRatio initial = 1 / 3 →
  goldSilverRatio final = 1 / 2 →
  final.gold + final.silver = 135 →
  final.gold = initial.gold + added_gold →
  final.silver = initial.silver →
  added_gold = 15 := by
  sorry


end NUMINAMATH_CALUDE_coin_collection_problem_l4052_405225


namespace NUMINAMATH_CALUDE_neznaika_expression_problem_l4052_405220

theorem neznaika_expression_problem :
  ∃ (f : ℝ → ℝ → ℝ → ℝ),
    (∀ x y z, f x y z = x / (y - Real.sqrt z)) →
    f 20 2 2 > 30 := by
  sorry

end NUMINAMATH_CALUDE_neznaika_expression_problem_l4052_405220


namespace NUMINAMATH_CALUDE_vasims_share_l4052_405209

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℕ)

/-- Represents the distribution of money among three people -/
structure Distribution :=
  (faruk : Share)
  (vasim : Share)
  (ranjith : Share)

/-- The ratio of the distribution -/
def distribution_ratio (d : Distribution) : Prop :=
  5 * d.faruk.amount = 3 * d.vasim.amount ∧
  6 * d.faruk.amount = 3 * d.ranjith.amount

/-- The difference between the largest and smallest share is 900 -/
def share_difference (d : Distribution) : Prop :=
  d.ranjith.amount - d.faruk.amount = 900

theorem vasims_share (d : Distribution) 
  (h1 : distribution_ratio d) 
  (h2 : share_difference d) : 
  d.vasim.amount = 1500 :=
sorry

end NUMINAMATH_CALUDE_vasims_share_l4052_405209


namespace NUMINAMATH_CALUDE_table_price_is_300_l4052_405286

def table_selling_price (num_trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) 
                        (labor_cost : ℕ) (profit : ℕ) : ℕ :=
  let total_planks := num_trees * planks_per_tree
  let num_tables := total_planks / planks_per_table
  let total_revenue := labor_cost + profit
  total_revenue / num_tables

theorem table_price_is_300 :
  table_selling_price 30 25 15 3000 12000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_table_price_is_300_l4052_405286


namespace NUMINAMATH_CALUDE_complex_number_pure_imaginary_l4052_405250

/-- Given a complex number z = (m-1) + (m+1)i where m is a real number and z is pure imaginary, prove that m = 1 -/
theorem complex_number_pure_imaginary (m : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (m - 1) (m + 1))
  (h2 : z.re = 0) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_pure_imaginary_l4052_405250


namespace NUMINAMATH_CALUDE_inequality_proof_l4052_405275

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0)
  (h5 : x * y * z = 1) (h6 : y + z + t = 2) :
  x^2 + y^2 + z^2 + t^2 ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4052_405275


namespace NUMINAMATH_CALUDE_min_value_expression_l4052_405241

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  ((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) ≥ 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4052_405241


namespace NUMINAMATH_CALUDE_investment_rate_proof_l4052_405271

theorem investment_rate_proof (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) :
  initial_investment = 8000 →
  initial_rate = 0.05 →
  additional_investment = 4000 →
  additional_rate = 0.08 →
  let total_interest := initial_investment * initial_rate + additional_investment * additional_rate
  let total_investment := initial_investment + additional_investment
  (total_interest / total_investment) = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l4052_405271


namespace NUMINAMATH_CALUDE_power_six_mod_eleven_l4052_405272

theorem power_six_mod_eleven : 6^2045 % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_six_mod_eleven_l4052_405272


namespace NUMINAMATH_CALUDE_marilyns_bottle_caps_l4052_405223

/-- The problem of Marilyn's bottle caps -/
theorem marilyns_bottle_caps :
  ∀ (initial : ℕ), 
    (initial - 36 = 15) → 
    initial = 51 := by
  sorry

end NUMINAMATH_CALUDE_marilyns_bottle_caps_l4052_405223


namespace NUMINAMATH_CALUDE_complete_square_factorization_l4052_405205

theorem complete_square_factorization :
  ∀ (x : ℝ), x^2 + 2*x + 1 = (x + 1)^2 := by
  sorry

#check complete_square_factorization

end NUMINAMATH_CALUDE_complete_square_factorization_l4052_405205


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l4052_405221

/-- Given four points on a Cartesian plane where segment AB is parallel to segment XY, 
    prove that k = -8. -/
theorem parallel_segments_k_value 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-6, 0))
  (hB : B = (0, -6))
  (hX : X = (0, 10))
  (hY : Y = (18, k))
  (h_parallel : (B.2 - A.2) * (Y.1 - X.1) = (Y.2 - X.2) * (B.1 - A.1)) :
  k = -8 :=
by sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l4052_405221


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l4052_405222

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 126 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l4052_405222


namespace NUMINAMATH_CALUDE_time_to_cross_signal_pole_l4052_405231

-- Define the train and platform parameters
def train_length : ℝ := 300
def platform_length : ℝ := 400
def time_cross_platform : ℝ := 42

-- Define the theorem
theorem time_to_cross_signal_pole :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_cross_platform
  let time_cross_pole := train_length / train_speed
  time_cross_pole = 18 := by
  sorry


end NUMINAMATH_CALUDE_time_to_cross_signal_pole_l4052_405231


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l4052_405215

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, x^2 - k*x + 1 = 0) ↔ k = 2 ∨ k = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l4052_405215


namespace NUMINAMATH_CALUDE_sisters_and_brothers_in_family_l4052_405204

/-- Represents a family with boys and girls -/
structure Family where
  boys : Nat
  girls : Nat

/-- Calculates the number of sisters a girl has in the family (excluding herself) -/
def sisters_of_girl (f : Family) : Nat :=
  f.girls - 1

/-- Calculates the number of brothers a girl has in the family -/
def brothers_of_girl (f : Family) : Nat :=
  f.boys

theorem sisters_and_brothers_in_family (harry_sisters : Nat) (harry_brothers : Nat) :
  harry_sisters = 4 → harry_brothers = 3 →
  ∃ (f : Family),
    f.girls = harry_sisters + 1 ∧
    f.boys = harry_brothers + 1 ∧
    sisters_of_girl f = 3 ∧
    brothers_of_girl f = 3 :=
by sorry

end NUMINAMATH_CALUDE_sisters_and_brothers_in_family_l4052_405204


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4052_405268

/-- Given three mutually externally tangent circles with radii a, b, and c,
    the radius r of the inscribed circle satisfies the given equation. -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 9 / 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4052_405268


namespace NUMINAMATH_CALUDE_counterfeit_identification_possible_l4052_405294

/-- Represents the result of weighing two coins on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftLighter : WeighResult
  | RightLighter : WeighResult

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real : Coin
  | Counterfeit : Coin

/-- A function that simulates weighing two coins on a balance scale -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.Real, Coin.Real => WeighResult.Equal
  | Coin.Counterfeit, Coin.Real => WeighResult.LeftLighter
  | Coin.Real, Coin.Counterfeit => WeighResult.RightLighter
  | Coin.Counterfeit, Coin.Counterfeit => WeighResult.Equal

/-- A function that identifies the counterfeit coin based on one weighing -/
def identifyCounterfeit (coins : Fin 3 → Coin) : Fin 3 :=
  match weigh (coins 0) (coins 1) with
  | WeighResult.Equal => 2
  | WeighResult.LeftLighter => 0
  | WeighResult.RightLighter => 1

theorem counterfeit_identification_possible :
  ∀ (coins : Fin 3 → Coin),
  (∃! i, coins i = Coin.Counterfeit) →
  coins (identifyCounterfeit coins) = Coin.Counterfeit :=
sorry


end NUMINAMATH_CALUDE_counterfeit_identification_possible_l4052_405294


namespace NUMINAMATH_CALUDE_steves_pencils_l4052_405285

/-- Steve's pencil distribution problem -/
theorem steves_pencils (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra : ℕ) :
  boxes = 2 →
  pencils_per_box = 12 →
  lauren_pencils = 6 →
  matt_extra = 3 →
  boxes * pencils_per_box - lauren_pencils - (lauren_pencils + matt_extra) = 9 :=
by sorry

end NUMINAMATH_CALUDE_steves_pencils_l4052_405285


namespace NUMINAMATH_CALUDE_orange_count_l4052_405265

/-- Represents the count of fruits in a box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ
  oranges : ℕ

/-- The properties of the fruit box as described in the problem -/
def is_valid_fruit_box (box : FruitBox) : Prop :=
  box.apples + box.pears + box.oranges = 60 ∧
  box.apples = 3 * (box.pears + box.oranges) ∧
  box.pears * 5 = box.apples + box.oranges

/-- Theorem stating that a valid fruit box has 5 oranges -/
theorem orange_count (box : FruitBox) (h : is_valid_fruit_box box) : box.oranges = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l4052_405265


namespace NUMINAMATH_CALUDE_smallest_subtraction_for_divisibility_l4052_405282

theorem smallest_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 100 ∧ (427751 - x) % 101 = 0 ∧ ∀ y : ℕ, y < x → (427751 - y) % 101 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_subtraction_for_divisibility_l4052_405282


namespace NUMINAMATH_CALUDE_min_value_sum_l4052_405277

theorem min_value_sum (a b : ℝ) (h : a^2 + 2*b^2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y) ∧ (∃ (u v : ℝ), u^2 + 2*v^2 = 6 ∧ m = u + v) ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l4052_405277


namespace NUMINAMATH_CALUDE_probability_after_removal_l4052_405210

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (each : ℕ)
  (h1 : total = numbers * each)

/-- Calculates the number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the probability of selecting a pair from the deck after removing two pairs -/
def probability_of_pair (d : Deck) : ℚ :=
  let remaining := d.total - 4
  let total_choices := choose_two remaining
  let pair_choices := (d.numbers - 1) * choose_two d.each
  pair_choices / total_choices

theorem probability_after_removal (d : Deck) 
  (h2 : d.total = 60) 
  (h3 : d.numbers = 12) 
  (h4 : d.each = 5) : 
  probability_of_pair d = 11 / 154 := by
  sorry

end NUMINAMATH_CALUDE_probability_after_removal_l4052_405210


namespace NUMINAMATH_CALUDE_line_equation_45_degree_slope_2_intercept_l4052_405235

/-- The equation of a line with a slope angle of 45° and a y-intercept of 2 is y = x + 2 -/
theorem line_equation_45_degree_slope_2_intercept :
  let slope_angle : Real := 45 * (π / 180)  -- Convert 45° to radians
  let y_intercept : Real := 2
  let slope : Real := Real.tan slope_angle
  ∀ x y : Real, y = slope * x + y_intercept ↔ y = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_45_degree_slope_2_intercept_l4052_405235


namespace NUMINAMATH_CALUDE_spinner_probability_l4052_405298

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_A + p_B + p_C + p_D = 1 →
  p_C = 3/16 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l4052_405298


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l4052_405237

theorem geometric_progression_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 9)
  (h2 : sum_first_two = 7) :
  ∃ (a : ℝ), (a = 3 * (3 - Real.sqrt 2) ∨ a = 3 * (3 + Real.sqrt 2)) ∧
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l4052_405237


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l4052_405257

theorem solve_quadratic_equation (B : ℝ) :
  5 * B^2 + 5 = 30 → B = Real.sqrt 5 ∨ B = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l4052_405257


namespace NUMINAMATH_CALUDE_weekend_finances_correct_l4052_405214

/-- Represents Tom's financial situation over the weekend -/
structure WeekendFinances where
  initial : ℝ  -- Initial amount
  car_wash : ℝ  -- Amount earned from washing cars
  lawn_mow : ℝ  -- Amount earned from mowing lawns
  painting : ℝ  -- Amount earned from painting
  expenses : ℝ  -- Amount spent on gas and food
  final : ℝ  -- Final amount

/-- Theorem stating that Tom's final amount is correctly calculated -/
theorem weekend_finances_correct (tom : WeekendFinances) 
  (h1 : tom.initial = 74)
  (h2 : tom.final = 86) :
  tom.initial + tom.car_wash + tom.lawn_mow + tom.painting - tom.expenses = tom.final := by
  sorry

end NUMINAMATH_CALUDE_weekend_finances_correct_l4052_405214


namespace NUMINAMATH_CALUDE_complex_calculation_l4052_405279

theorem complex_calculation (z : ℂ) (h : z = 1 + I) : z^2 + 2/z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l4052_405279


namespace NUMINAMATH_CALUDE_least_factorial_divisible_by_7875_l4052_405262

theorem least_factorial_divisible_by_7875 :
  ∃ (n : ℕ), n > 0 ∧ 7875 ∣ n.factorial ∧ ∀ (m : ℕ), m > 0 → 7875 ∣ m.factorial → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_factorial_divisible_by_7875_l4052_405262


namespace NUMINAMATH_CALUDE_g_of_3_l4052_405245

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem g_of_3 : g 3 = 126 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l4052_405245


namespace NUMINAMATH_CALUDE_adolfo_blocks_l4052_405263

theorem adolfo_blocks (initial_blocks added_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : added_blocks = 30) :
  initial_blocks + added_blocks = 65 := by
  sorry

end NUMINAMATH_CALUDE_adolfo_blocks_l4052_405263


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l4052_405280

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ -- Angles are positive
  a + b = 90 ∧ -- Sum of acute angles in a right triangle
  b = 4 * a -- Ratio of angles is 4:1
  → a = 18 ∧ b = 72 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l4052_405280


namespace NUMINAMATH_CALUDE_rational_function_sum_l4052_405253

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  h_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  h_horiz_asymp : ∀ ε > 0, ∃ M, ∀ x, |x| > M → |p x / q x| < ε
  h_vert_asymp : ContinuousAt q (-2) ∧ q (-2) = 0
  h_p3 : p 3 = 1
  h_q3 : q 3 = 4

/-- The main theorem -/
theorem rational_function_sum (f : RationalFunction) : 
  ∀ x, f.p x + f.q x = (4 * x^2 + 7 * x - 9) / 10 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_sum_l4052_405253


namespace NUMINAMATH_CALUDE_evaluate_expression_l4052_405281

theorem evaluate_expression : (1 / ((5^2)^4)) * 5^11 * 2 = 250 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4052_405281


namespace NUMINAMATH_CALUDE_or_false_necessary_not_sufficient_for_and_false_l4052_405233

theorem or_false_necessary_not_sufficient_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬(p ∨ q)) :=
by sorry

end NUMINAMATH_CALUDE_or_false_necessary_not_sufficient_for_and_false_l4052_405233


namespace NUMINAMATH_CALUDE_tangency_values_l4052_405251

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 5

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 1

/-- The tangency condition -/
def tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' → hyperbola m x' y' → (x = x' ∧ y = y')

/-- The theorem stating the values of m for which the parabola and hyperbola are tangent -/
theorem tangency_values :
  ∀ m : ℝ, tangent m ↔ (m = 10 + 4 * Real.sqrt 6 ∨ m = 10 - 4 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangency_values_l4052_405251


namespace NUMINAMATH_CALUDE_range_of_a_l4052_405216

-- Define the conditions
def p (x : ℝ) : Prop := x^2 + 2*x > 3
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a (h : ∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ ∃ x a, ¬(p x) ∧ (q x a)) :
  ∀ a : ℝ, (∃ x : ℝ, q x a) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4052_405216


namespace NUMINAMATH_CALUDE_characterize_solution_set_l4052_405256

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ), n ≥ 2 ∧ ∀ (x y : ℝ), f (x + y^n) = f x + (f y)^n

/-- The set of functions that satisfy the functional equation -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | SatisfiesFunctionalEquation f}

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := fun _ ↦ 0

/-- The identity function -/
def IdentityFunction : ℝ → ℝ := fun x ↦ x

/-- The negation function -/
def NegationFunction : ℝ → ℝ := fun x ↦ -x

/-- The main theorem characterizing the solution set -/
theorem characterize_solution_set :
  SolutionSet = {ZeroFunction, IdentityFunction, NegationFunction} := by sorry

end NUMINAMATH_CALUDE_characterize_solution_set_l4052_405256


namespace NUMINAMATH_CALUDE_pear_price_is_correct_l4052_405266

/-- The price of a pear in won -/
def pear_price : ℕ := 6300

/-- The price of an apple in won -/
def apple_price : ℕ := pear_price + 2400

/-- The sum of the prices of an apple and a pear in won -/
def total_price : ℕ := 15000

theorem pear_price_is_correct : pear_price = 6300 := by
  have h1 : apple_price + pear_price = total_price := by sorry
  have h2 : apple_price = pear_price + 2400 := by sorry
  sorry

end NUMINAMATH_CALUDE_pear_price_is_correct_l4052_405266


namespace NUMINAMATH_CALUDE_simplify_expression_l4052_405284

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  |a - 2| - Real.sqrt ((a - 3)^2) = 2*a - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4052_405284


namespace NUMINAMATH_CALUDE_specificPolygonArea_l4052_405287

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A polygon defined by a list of grid points -/
def Polygon := List GridPoint

/-- The polygon formed by connecting specific points on a 4x4 grid -/
def specificPolygon : Polygon :=
  [⟨0,0⟩, ⟨1,0⟩, ⟨1,1⟩, ⟨0,1⟩, ⟨1,2⟩, ⟨0,2⟩, ⟨1,3⟩, ⟨0,3⟩, 
   ⟨3,3⟩, ⟨2,3⟩, ⟨3,2⟩, ⟨2,2⟩, ⟨2,1⟩, ⟨3,1⟩, ⟨3,0⟩, ⟨2,0⟩]

/-- Function to calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℕ := sorry

/-- Theorem stating that the area of the specific polygon is 16 square units -/
theorem specificPolygonArea : calculateArea specificPolygon = 16 := by sorry

end NUMINAMATH_CALUDE_specificPolygonArea_l4052_405287


namespace NUMINAMATH_CALUDE_limit_of_expression_l4052_405243

theorem limit_of_expression (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    |((4 * (n : ℝ)^2 + 4 * n - 1) / (4 * (n : ℝ)^2 + 2 * n + 3))^(1 - 2 * n) - Real.exp (-1)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_expression_l4052_405243


namespace NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l4052_405296

theorem twenty_is_eighty_percent_of_twentyfive : 
  ∃ x : ℝ, (20 : ℝ) / x = (80 : ℝ) / 100 ∧ x = 25 := by sorry

end NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l4052_405296


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l4052_405229

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1305 is 13% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1500
  let selling_price : ℚ := 1305
  loss_percentage cost_price selling_price = 13 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l4052_405229


namespace NUMINAMATH_CALUDE_work_completion_time_l4052_405258

theorem work_completion_time (original_laborers : ℕ) (absent_laborers : ℕ) (actual_days : ℕ) : 
  original_laborers = 20 → 
  absent_laborers = 5 → 
  actual_days = 20 → 
  ∃ (original_days : ℕ), 
    original_days * original_laborers = actual_days * (original_laborers - absent_laborers) ∧ 
    original_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4052_405258


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l4052_405232

theorem logarithm_expression_equals_zero :
  Real.log 14 - 2 * Real.log (7/3) + Real.log 7 - Real.log 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l4052_405232


namespace NUMINAMATH_CALUDE_profit_increase_1995_to_1997_l4052_405211

/-- Represents the financial data of a company over three years -/
structure CompanyFinances where
  R1 : ℝ  -- Revenue in 1995
  E1 : ℝ  -- Expenses in 1995
  P1 : ℝ  -- Profit in 1995
  R2 : ℝ  -- Revenue in 1996
  E2 : ℝ  -- Expenses in 1996
  P2 : ℝ  -- Profit in 1996
  R3 : ℝ  -- Revenue in 1997
  E3 : ℝ  -- Expenses in 1997
  P3 : ℝ  -- Profit in 1997

/-- The profit increase from 1995 to 1997 is 55.25% -/
theorem profit_increase_1995_to_1997 (cf : CompanyFinances)
  (h1 : cf.P1 = cf.R1 - cf.E1)
  (h2 : cf.R2 = 1.20 * cf.R1)
  (h3 : cf.E2 = 1.10 * cf.E1)
  (h4 : cf.P2 = 1.15 * cf.P1)
  (h5 : cf.R3 = 1.25 * cf.R2)
  (h6 : cf.E3 = 1.20 * cf.E2)
  (h7 : cf.P3 = 1.35 * cf.P2) :
  cf.P3 = 1.5525 * cf.P1 := by
  sorry

#check profit_increase_1995_to_1997

end NUMINAMATH_CALUDE_profit_increase_1995_to_1997_l4052_405211


namespace NUMINAMATH_CALUDE_books_loaned_out_l4052_405242

/-- Proves that the number of books loaned out is 50 given the initial and final book counts and return rate -/
theorem books_loaned_out (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 → final_books = 65 → return_rate = 4/5 → 
  (initial_books - final_books) / (1 - return_rate) = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_l4052_405242


namespace NUMINAMATH_CALUDE_circular_track_catchup_l4052_405260

/-- The time (in minutes) for Person A to catch up with Person B on a circular track -/
def catchUpTime (trackCircumference : ℝ) (speedA speedB : ℝ) (restInterval : ℝ) (restDuration : ℝ) : ℝ :=
  sorry

theorem circular_track_catchup :
  let trackCircumference : ℝ := 400
  let speedA : ℝ := 52
  let speedB : ℝ := 46
  let restInterval : ℝ := 100
  let restDuration : ℝ := 1
  catchUpTime trackCircumference speedA speedB restInterval restDuration = 147 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_catchup_l4052_405260


namespace NUMINAMATH_CALUDE_charity_race_total_l4052_405239

/-- Represents the total amount raised by students in a charity race -/
def total_raised (
  total_students : ℕ
  ) (
  group_a_students : ℕ
  ) (
  group_b_students : ℕ
  ) (
  group_c_students : ℕ
  ) (
  group_a_race_amount : ℕ
  ) (
  group_a_extra_amount : ℕ
  ) (
  group_b_race_amount : ℕ
  ) (
  group_b_extra_amount : ℕ
  ) (
  group_c_race_amount : ℕ
  ) (
  group_c_extra_total : ℕ
  ) : ℕ :=
  (group_a_students * (group_a_race_amount + group_a_extra_amount)) +
  (group_b_students * (group_b_race_amount + group_b_extra_amount)) +
  (group_c_students * group_c_race_amount + group_c_extra_total)

/-- Theorem stating that the total amount raised is $1080 -/
theorem charity_race_total :
  total_raised 30 10 12 8 20 5 30 10 25 150 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_total_l4052_405239


namespace NUMINAMATH_CALUDE_book_page_ratio_l4052_405208

/-- Given a set of books with specific page counts, prove the ratio of pages between middle and shortest books --/
theorem book_page_ratio (longest middle shortest : ℕ) : 
  longest = 396 → 
  shortest = longest / 4 → 
  middle = 297 → 
  middle / shortest = 3 := by
sorry

end NUMINAMATH_CALUDE_book_page_ratio_l4052_405208


namespace NUMINAMATH_CALUDE_mike_picked_12_pears_l4052_405234

/-- The number of pears Mike picked -/
def mike_pears : ℕ := sorry

/-- The number of pears Keith picked initially -/
def keith_initial_pears : ℕ := 47

/-- The number of pears Keith gave away -/
def keith_gave_away : ℕ := 46

/-- The total number of pears Keith and Mike have left -/
def total_pears_left : ℕ := 13

theorem mike_picked_12_pears : mike_pears = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_12_pears_l4052_405234


namespace NUMINAMATH_CALUDE_prop_one_prop_two_l4052_405236

-- Proposition 1
theorem prop_one (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a > b) :
  a - 1 / a > b - 1 / b :=
sorry

-- Proposition 2
theorem prop_two (a b : ℝ) (hb : b ≠ 0) :
  b * (b - a) ≤ 0 ↔ a / b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_prop_one_prop_two_l4052_405236


namespace NUMINAMATH_CALUDE_village_population_equality_l4052_405213

/-- The initial population of Village 1 -/
def initial_population_village1 : ℕ := 68000

/-- The yearly decrease in population of Village 1 -/
def yearly_decrease_village1 : ℕ := 1200

/-- The initial population of Village 2 -/
def initial_population_village2 : ℕ := 42000

/-- The yearly increase in population of Village 2 -/
def yearly_increase_village2 : ℕ := 800

/-- The number of years after which the populations are equal -/
def years_until_equal : ℕ := 13

theorem village_population_equality :
  initial_population_village1 - yearly_decrease_village1 * years_until_equal =
  initial_population_village2 + yearly_increase_village2 * years_until_equal :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_l4052_405213


namespace NUMINAMATH_CALUDE_min_intersection_cardinality_l4052_405290

-- Define the cardinality function
def card (S : Set α) : ℕ := sorry

-- Define the power set function
def powerset (S : Set α) : Set (Set α) := sorry

theorem min_intersection_cardinality 
  (A B C D : Set α) 
  (h1 : card A = 150) 
  (h2 : card B = 150) 
  (h3 : card D = 102) 
  (h4 : card (powerset A) + card (powerset B) + card (powerset C) + card (powerset D) = 
        card (powerset (A ∪ B ∪ C ∪ D)))
  (h5 : card (powerset A) + card (powerset B) + card (powerset C) + card (powerset D) = 2^152) :
  card (A ∩ B ∩ C ∩ D) ≥ 99 ∧ ∃ (A' B' C' D' : Set α), 
    card A' = 150 ∧ 
    card B' = 150 ∧ 
    card D' = 102 ∧ 
    card (powerset A') + card (powerset B') + card (powerset C') + card (powerset D') = 
      card (powerset (A' ∪ B' ∪ C' ∪ D')) ∧
    card (powerset A') + card (powerset B') + card (powerset C') + card (powerset D') = 2^152 ∧
    card (A' ∩ B' ∩ C' ∩ D') = 99 :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_cardinality_l4052_405290


namespace NUMINAMATH_CALUDE_ship_length_proof_l4052_405293

/-- The length of the ship in meters -/
def ship_length : ℝ := 72

/-- The speed of the ship in meters per second -/
def ship_speed : ℝ := 4

/-- Emily's walking speed in meters per second -/
def emily_speed : ℝ := 6

/-- The number of steps Emily takes from back to front of the ship -/
def steps_back_to_front : ℕ := 300

/-- The number of steps Emily takes from front to back of the ship -/
def steps_front_to_back : ℕ := 60

/-- The length of each of Emily's steps in meters -/
def step_length : ℝ := 2

theorem ship_length_proof :
  let relative_speed_forward := emily_speed - ship_speed
  let relative_speed_backward := emily_speed + ship_speed
  let distance_forward := steps_back_to_front * step_length
  let distance_backward := steps_front_to_back * step_length
  let time_forward := distance_forward / relative_speed_forward
  let time_backward := distance_backward / relative_speed_backward
  ship_length = distance_forward - ship_speed * time_forward ∧
  ship_length = distance_backward + ship_speed * time_backward :=
by sorry

end NUMINAMATH_CALUDE_ship_length_proof_l4052_405293


namespace NUMINAMATH_CALUDE_hypergeom_expected_and_variance_l4052_405207

/-- Hypergeometric distribution parameters -/
structure HyperGeomParams where
  N : ℕ  -- Population size
  K : ℕ  -- Number of success states in the population
  n : ℕ  -- Number of draws
  h1 : K ≤ N
  h2 : n ≤ N

/-- Expected value of a hypergeometric distribution -/
def expected_value (p : HyperGeomParams) : ℚ :=
  (p.n * p.K : ℚ) / p.N

/-- Variance of a hypergeometric distribution -/
def variance (p : HyperGeomParams) : ℚ :=
  (p.n * p.K * (p.N - p.K) * (p.N - p.n) : ℚ) / (p.N^2 * (p.N - 1))

/-- Theorem: Expected value and variance for the given problem -/
theorem hypergeom_expected_and_variance :
  ∃ (p : HyperGeomParams),
    p.N = 100 ∧ p.K = 10 ∧ p.n = 3 ∧
    expected_value p = 3/10 ∧
    variance p = 51/200 := by
  sorry

end NUMINAMATH_CALUDE_hypergeom_expected_and_variance_l4052_405207


namespace NUMINAMATH_CALUDE_probability_of_drawing_balls_l4052_405274

theorem probability_of_drawing_balls (prob_A prob_B : ℝ) 
  (h_prob_A : prob_A = 1/3) (h_prob_B : prob_B = 1/2) :
  let prob_both_red := prob_A * prob_B
  let prob_exactly_one_red := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
  let prob_both_not_red := (1 - prob_A) * (1 - prob_B)
  let prob_at_least_one_red := 1 - prob_both_not_red
  (prob_both_red = 1/6) ∧
  (prob_exactly_one_red = 1/2) ∧
  (prob_both_not_red = 5/6) ∧
  (prob_at_least_one_red = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_balls_l4052_405274


namespace NUMINAMATH_CALUDE_multiplication_simplification_l4052_405270

theorem multiplication_simplification : 11 * (1 / 17) * 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l4052_405270


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l4052_405276

theorem sqrt_sum_equals_eleven_sqrt_two_over_six :
  Real.sqrt (9 / 2) + Real.sqrt (2 / 9) = 11 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l4052_405276


namespace NUMINAMATH_CALUDE_football_field_length_proof_l4052_405264

/-- The length of a football field in yards -/
def football_field_length : ℝ := 200

/-- The number of football fields a potato is launched across -/
def fields_crossed : ℕ := 6

/-- The speed of the dog in feet per minute -/
def dog_speed : ℝ := 400

/-- The time taken by the dog to fetch the potato in minutes -/
def fetch_time : ℝ := 9

/-- The number of feet in a yard -/
def feet_per_yard : ℝ := 3

theorem football_field_length_proof :
  football_field_length = 
    (dog_speed / feet_per_yard * fetch_time) / fields_crossed := by
  sorry

end NUMINAMATH_CALUDE_football_field_length_proof_l4052_405264


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l4052_405227

/-- The minimum distance between a point on the ellipse x²/3 + y² = 1 and 
    a point on the line x + y = 4, along with the coordinates of the point 
    on the ellipse at this minimum distance. -/
theorem min_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}
  let line := {q : ℝ × ℝ | q.1 + q.2 = 4}
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧ 
    (∀ (p' : ℝ × ℝ) (q : ℝ × ℝ), p' ∈ ellipse → q ∈ line → 
      Real.sqrt 2 ≤ Real.sqrt ((p'.1 - q.1)^2 + (p'.2 - q.2)^2)) ∧
    p = (3/2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l4052_405227


namespace NUMINAMATH_CALUDE_det_A_eq_neg94_l4052_405226

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 4, -2; 3, -1, 5; -1, 3, 2]

theorem det_A_eq_neg94 : Matrix.det A = -94 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_neg94_l4052_405226


namespace NUMINAMATH_CALUDE_bacon_percentage_of_total_l4052_405228

def total_sandwich_calories : ℕ := 1250
def bacon_strips : ℕ := 2
def calories_per_bacon_strip : ℕ := 125

def bacon_calories : ℕ := bacon_strips * calories_per_bacon_strip

theorem bacon_percentage_of_total (h : bacon_calories = bacon_strips * calories_per_bacon_strip) :
  (bacon_calories : ℚ) / total_sandwich_calories * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bacon_percentage_of_total_l4052_405228


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4052_405288

/-- The function f(x) = a^(x-1) + 7 always passes through the point (1, 8) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x-1) + 7
  f 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4052_405288


namespace NUMINAMATH_CALUDE_units_digit_of_product_l4052_405224

theorem units_digit_of_product : ((30 * 31 * 32 * 33 * 34 * 35) / 1000) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l4052_405224


namespace NUMINAMATH_CALUDE_lucy_crayons_count_l4052_405259

/-- The number of crayons Willy has -/
def willys_crayons : ℕ := 5092

/-- The difference between Willy's and Lucy's crayons -/
def difference : ℕ := 1121

/-- The number of crayons Lucy has -/
def lucys_crayons : ℕ := willys_crayons - difference

theorem lucy_crayons_count : lucys_crayons = 3971 := by
  sorry

end NUMINAMATH_CALUDE_lucy_crayons_count_l4052_405259


namespace NUMINAMATH_CALUDE_ratio_b_to_a_is_one_l4052_405206

/-- An arithmetic sequence with first four terms a, b, x, and 2x - 1/2 -/
structure ArithmeticSequence (a b x : ℝ) : Prop where
  term1 : a = a
  term2 : b = b
  term3 : x = x
  term4 : 2 * x - 1/2 = 2 * x - 1/2
  is_arithmetic : ∃ (d : ℝ), b - a = d ∧ x - b = d ∧ (2 * x - 1/2) - x = d

/-- The ratio of b to a in the arithmetic sequence is 1 -/
theorem ratio_b_to_a_is_one {a b x : ℝ} (h : ArithmeticSequence a b x) : b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_b_to_a_is_one_l4052_405206


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l4052_405200

def is_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def is_digit (n : ℕ) : Prop := n < 10

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cryptarithmetic_solution :
  ∃! (X Y B M C : ℕ),
    is_distinct X Y B M C ∧
    is_nonzero_digit X ∧
    is_digit Y ∧
    is_nonzero_digit B ∧
    is_digit M ∧
    is_digit C ∧
    X * 1000 + Y * 100 + 70 + B * 100 + M * 10 + C =
    B * 1000 + M * 100 + C * 10 + 0 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l4052_405200


namespace NUMINAMATH_CALUDE_alice_and_bob_savings_l4052_405218

theorem alice_and_bob_savings (alice_money : ℚ) (bob_money : ℚ) :
  alice_money = 2 / 5 →
  bob_money = 1 / 4 →
  2 * (alice_money + bob_money) = 13 / 10 := by
sorry

end NUMINAMATH_CALUDE_alice_and_bob_savings_l4052_405218


namespace NUMINAMATH_CALUDE_f_even_implies_a_zero_min_value_when_a_greater_than_two_l4052_405219

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |2*x - a|

-- Theorem 1: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

-- Theorem 2: If a > 2, then the minimum value of f(x) is a - 1
theorem min_value_when_a_greater_than_two (a : ℝ) :
  a > 2 → ∃ m : ℝ, (∀ x : ℝ, f a x ≥ m) ∧ (∃ x : ℝ, f a x = m) ∧ m = a - 1 := by sorry

end NUMINAMATH_CALUDE_f_even_implies_a_zero_min_value_when_a_greater_than_two_l4052_405219
