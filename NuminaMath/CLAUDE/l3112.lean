import Mathlib

namespace candy_difference_example_l3112_311257

/-- Given a total number of candies and the number of strawberry candies,
    calculate the difference between grape and strawberry candies. -/
def candy_difference (total : ℕ) (strawberry : ℕ) : ℕ :=
  (total - strawberry) - strawberry

/-- Theorem stating that given 821 total candies and 267 strawberry candies,
    the difference between grape and strawberry candies is 287. -/
theorem candy_difference_example : candy_difference 821 267 = 287 := by
  sorry

end candy_difference_example_l3112_311257


namespace consecutive_square_roots_l3112_311264

theorem consecutive_square_roots (x : ℝ) (n : ℕ) :
  (∃ m : ℕ, n = m ∧ x^2 = m) →
  Real.sqrt ((n + 1 : ℝ)) = Real.sqrt (x^2 + 1) :=
sorry

end consecutive_square_roots_l3112_311264


namespace area_of_triangle_l3112_311259

/-- The hyperbola with equation x^2 - y^2/12 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2/12 = 1}

/-- The foci of the hyperbola -/
def Foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The distance ratio condition -/
axiom distance_ratio : 
  let (f1x, f1y, f2x, f2y) := Foci
  let (px, py) := P
  3 * ((f2x - px)^2 + (f2y - py)^2) = 2 * ((f1x - px)^2 + (f1y - py)^2)

/-- P is on the hyperbola -/
axiom P_on_hyperbola : P ∈ Hyperbola

/-- The theorem to be proved -/
theorem area_of_triangle : 
  let (f1x, f1y, f2x, f2y) := Foci
  let (px, py) := P
  (1/2) * |f1x - f2x| * |f1y - f2y| = 12 := by sorry

end area_of_triangle_l3112_311259


namespace harry_age_l3112_311247

/-- Given the ages of Kiarra, Bea, Job, Figaro, and Harry, prove that Harry is 26 years old. -/
theorem harry_age (kiarra bea job figaro harry : ℕ) 
  (h1 : kiarra = 2 * bea)
  (h2 : job = 3 * bea)
  (h3 : figaro = job + 7)
  (h4 : harry * 2 = figaro)
  (h5 : kiarra = 30) : 
  harry = 26 := by
  sorry

end harry_age_l3112_311247


namespace double_frosted_cubes_count_l3112_311253

/-- Represents a cube with dimensions n × n × n -/
structure Cube (n : ℕ) where
  size : ℕ := n

/-- Represents a cake with frosting on top and sides, but not on bottom -/
structure FrostedCake (n : ℕ) extends Cube n where
  frosted_top : Bool := true
  frosted_sides : Bool := true
  frosted_bottom : Bool := false

/-- Counts the number of 1×1×1 cubes with exactly two frosted faces in a FrostedCake -/
def count_double_frosted_cubes (cake : FrostedCake 4) : ℕ :=
  sorry

theorem double_frosted_cubes_count :
  ∀ (cake : FrostedCake 4), count_double_frosted_cubes cake = 20 :=
by sorry

end double_frosted_cubes_count_l3112_311253


namespace student_age_problem_l3112_311207

theorem student_age_problem (n : ℕ) : 
  n < 10 →
  (8 : ℝ) * n = (10 : ℝ) * (n + 1) - 28 →
  n = 9 := by
sorry

end student_age_problem_l3112_311207


namespace z_share_per_x_rupee_l3112_311208

/-- Given a total amount divided among three parties x, y, and z, 
    this theorem proves the ratio of z's share to x's share. -/
theorem z_share_per_x_rupee 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 78) 
  (h2 : y_share = 18) 
  (h3 : y_share = 0.45 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  z_share / x_share = 0.5 := by
sorry

end z_share_per_x_rupee_l3112_311208


namespace prob_two_gray_rabbits_l3112_311221

/-- The probability of selecting 2 gray rabbits out of a group of 5 rabbits, 
    where 3 are gray and 2 are white, given that each rabbit has an equal 
    chance of being selected. -/
theorem prob_two_gray_rabbits (total : Nat) (gray : Nat) (white : Nat) 
    (h1 : total = gray + white) 
    (h2 : total = 5) 
    (h3 : gray = 3) 
    (h4 : white = 2) : 
  (Nat.choose gray 2 : ℚ) / (Nat.choose total 2) = 3 / 10 := by
  sorry

end prob_two_gray_rabbits_l3112_311221


namespace park_warden_citations_l3112_311213

theorem park_warden_citations :
  ∀ (littering off_leash parking : ℕ),
    littering = off_leash →
    parking = 2 * (littering + off_leash) →
    littering + off_leash + parking = 24 →
    littering = 4 := by
  sorry

end park_warden_citations_l3112_311213


namespace expected_red_balls_l3112_311276

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- The probability of drawing a red ball in a single draw -/
def p_red : ℚ := red_balls / total_balls

/-- The number of draws -/
def num_draws : ℕ := 6

/-- The random variable representing the number of red balls drawn -/
def ξ : ℕ → ℚ := sorry

/-- The expected value of ξ -/
def E_ξ : ℚ := num_draws * p_red

theorem expected_red_balls : E_ξ = 4 := by sorry

end expected_red_balls_l3112_311276


namespace intersection_of_A_and_B_l3112_311241

def A : Set ℝ := {x | x^2 - 4 < 0}

def B : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end intersection_of_A_and_B_l3112_311241


namespace train_journey_time_l3112_311246

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 3 / 4) = usual_speed * usual_time → 
  usual_time = 3 := by
  sorry

end train_journey_time_l3112_311246


namespace chlorine_treatment_capacity_l3112_311267

/-- Proves that given a rectangular pool with specified dimensions and chlorine costs,
    one quart of chlorine treats 120 cubic feet of water. -/
theorem chlorine_treatment_capacity
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (chlorine_cost : ℝ) (total_spent : ℝ)
  (h1 : length = 10)
  (h2 : width = 8)
  (h3 : depth = 6)
  (h4 : chlorine_cost = 3)
  (h5 : total_spent = 12) :
  (length * width * depth) / (total_spent / chlorine_cost) = 120 := by
  sorry


end chlorine_treatment_capacity_l3112_311267


namespace film_rewinding_time_l3112_311283

/-- Time required to rewind a film onto a reel -/
theorem film_rewinding_time
  (a L S ω : ℝ)
  (ha : a > 0)
  (hL : L > 0)
  (hS : S > 0)
  (hω : ω > 0) :
  ∃ T : ℝ,
    T > 0 ∧
    T = (π / (S * ω)) * (Real.sqrt (a^2 + (4 * S * L / π)) - a) :=
by sorry

end film_rewinding_time_l3112_311283


namespace solution_to_polynomial_equation_l3112_311292

theorem solution_to_polynomial_equation : ∃ x : ℤ, x^5 - 101*x^3 - 999*x^2 + 100900 = 0 :=
by
  use 10
  -- Proof goes here
  sorry

end solution_to_polynomial_equation_l3112_311292


namespace inequality_proof_l3112_311237

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l3112_311237


namespace johns_hair_growth_l3112_311243

/-- Represents John's hair growth and haircut information -/
structure HairGrowthInfo where
  cutFrom : ℝ  -- Length of hair before cut
  cutTo : ℝ    -- Length of hair after cut
  baseCost : ℝ -- Base cost of a haircut
  tipPercent : ℝ -- Tip percentage
  yearlySpend : ℝ -- Total spent on haircuts per year

/-- Calculates the monthly hair growth rate -/
def monthlyGrowthRate (info : HairGrowthInfo) : ℝ :=
  -- Definition of the function
  sorry

/-- Theorem stating that John's hair grows 1.5 inches per month -/
theorem johns_hair_growth (info : HairGrowthInfo) 
  (h1 : info.cutFrom = 9)
  (h2 : info.cutTo = 6)
  (h3 : info.baseCost = 45)
  (h4 : info.tipPercent = 0.2)
  (h5 : info.yearlySpend = 324) :
  monthlyGrowthRate info = 1.5 := by
  sorry

end johns_hair_growth_l3112_311243


namespace equal_interval_line_segments_l3112_311260

/-- Given two line segments with equal interval spacing between points,
    where one segment has 10 points over length a and the other has 100 points over length b,
    prove that b = 11a. -/
theorem equal_interval_line_segments (a b : ℝ) : 
  (∃ (interval : ℝ), 
    a = 9 * interval ∧ 
    b = 99 * interval) → 
  b = 11 * a := by sorry

end equal_interval_line_segments_l3112_311260


namespace positive_xy_l3112_311228

theorem positive_xy (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 := by
  sorry

end positive_xy_l3112_311228


namespace intersection_condition_l3112_311284

/-- The set A defined by the equation y = x^2 + mx + 2 -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + m * p.1 + 2}

/-- The set B defined by the equation y = x + 1 -/
def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1}

/-- The theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 ∨ m ≥ 3 := by
  sorry

end intersection_condition_l3112_311284


namespace symmetric_line_theorem_l3112_311282

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the x-axis -/
def symmetricLineEquation (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, c)

/-- Proves that the equation of the line symmetric to 2x-y+4=0 
    with respect to the x-axis is 2x+y+4=0 -/
theorem symmetric_line_theorem :
  let original := (2, -1, 4)
  let symmetric := symmetricLineEquation 2 (-1) 4
  symmetric = (2, 1, 4) := by sorry

end symmetric_line_theorem_l3112_311282


namespace cheerful_not_green_l3112_311219

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (green : Snake → Prop)
variable (cheerful : Snake → Prop)
variable (can_sing : Snake → Prop)
variable (can_multiply : Snake → Prop)

-- Define the conditions
axiom all_cheerful_can_sing : ∀ s : Snake, cheerful s → can_sing s
axiom no_green_can_multiply : ∀ s : Snake, green s → ¬can_multiply s
axiom cannot_multiply_cannot_sing : ∀ s : Snake, ¬can_multiply s → ¬can_sing s

-- Theorem to prove
theorem cheerful_not_green : ∀ s : Snake, cheerful s → ¬green s := by
  sorry

end cheerful_not_green_l3112_311219


namespace sin_product_equality_l3112_311205

theorem sin_product_equality : (1 - Real.sin (π / 6)) * (1 - Real.sin (5 * π / 6)) = 1 / 4 := by
  sorry

end sin_product_equality_l3112_311205


namespace arccos_equation_solution_l3112_311214

theorem arccos_equation_solution :
  ∃ x : ℝ, x = Real.sqrt (1 / (64 - 36 * Real.sqrt 3)) ∧ 
    Real.arccos (3 * x) - Real.arccos x = π / 6 := by
  sorry

end arccos_equation_solution_l3112_311214


namespace not_necessarily_square_lt_of_lt_l3112_311290

theorem not_necessarily_square_lt_of_lt {a b : ℝ} (h : a < b) : 
  ¬(∀ a b : ℝ, a < b → a^2 < b^2) :=
by sorry

end not_necessarily_square_lt_of_lt_l3112_311290


namespace max_generatable_number_l3112_311297

def powers_of_three : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def can_generate (n : ℤ) : Prop :=
  ∃ (coeffs : List ℤ), 
    coeffs.length = powers_of_three.length ∧ 
    (∀ c ∈ coeffs, c = 1 ∨ c = 0 ∨ c = -1) ∧
    n = List.sum (List.zipWith (· * ·) coeffs (powers_of_three.map Int.ofNat))

theorem max_generatable_number :
  (∀ n : ℕ, n ≤ 1093 → can_generate n) ∧
  ¬(can_generate 1094) :=
sorry

end max_generatable_number_l3112_311297


namespace point_A_in_fourth_quadrant_l3112_311261

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point A -/
def point_A : Point :=
  { x := 5, y := -4 }

/-- Theorem: Point A is in the fourth quadrant -/
theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A := by
  sorry


end point_A_in_fourth_quadrant_l3112_311261


namespace other_root_of_quadratic_l3112_311289

theorem other_root_of_quadratic (c : ℝ) : 
  (∃ x : ℝ, 6 * x^2 + c * x = -3) → 
  (-1/2 : ℝ) ∈ {x : ℝ | 6 * x^2 + c * x = -3} →
  (-1 : ℝ) ∈ {x : ℝ | 6 * x^2 + c * x = -3} := by
sorry

end other_root_of_quadratic_l3112_311289


namespace negation_of_forall_exp_geq_x_plus_one_l3112_311270

theorem negation_of_forall_exp_geq_x_plus_one :
  (¬ ∀ x : ℝ, Real.exp x ≥ x + 1) ↔ (∃ x₀ : ℝ, Real.exp x₀ < x₀ + 1) := by
  sorry

end negation_of_forall_exp_geq_x_plus_one_l3112_311270


namespace cos_arcsin_eight_seventeenths_l3112_311242

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end cos_arcsin_eight_seventeenths_l3112_311242


namespace annual_grass_cutting_cost_l3112_311298

/-- The annual cost of grass cutting given specific conditions -/
theorem annual_grass_cutting_cost
  (initial_height : ℝ)
  (growth_rate : ℝ)
  (cutting_threshold : ℝ)
  (cost_per_cut : ℝ)
  (h1 : initial_height = 2)
  (h2 : growth_rate = 0.5)
  (h3 : cutting_threshold = 4)
  (h4 : cost_per_cut = 100)
  : ℝ :=
by
  -- Prove that the annual cost of grass cutting is $300
  sorry

#check annual_grass_cutting_cost

end annual_grass_cutting_cost_l3112_311298


namespace lava_lamp_probability_is_one_seventh_l3112_311279

/-- The probability of a specific arrangement of lava lamps -/
def lava_lamp_probability : ℚ :=
  let total_lamps : ℕ := 4 + 4  -- 4 red + 4 blue lamps
  let lamps_on : ℕ := 4  -- 4 lamps are turned on
  let remaining_lamps : ℕ := total_lamps - 2  -- excluding leftmost and rightmost
  let remaining_on : ℕ := lamps_on - 1  -- excluding the rightmost lamp which is on
  let favorable_arrangements : ℕ := Nat.choose remaining_lamps (total_lamps / 2 - 1)  -- arranging remaining red lamps
  let favorable_on_choices : ℕ := Nat.choose (total_lamps - 1) (lamps_on - 1)  -- choosing remaining on lamps
  let total_arrangements : ℕ := Nat.choose total_lamps (total_lamps / 2)  -- total ways to arrange red and blue lamps
  let total_on_choices : ℕ := Nat.choose total_lamps lamps_on  -- total ways to choose on lamps
  (favorable_arrangements * favorable_on_choices : ℚ) / (total_arrangements * total_on_choices)

/-- The probability of the specific lava lamp arrangement is 1/7 -/
theorem lava_lamp_probability_is_one_seventh : lava_lamp_probability = 1 / 7 := by
  sorry

end lava_lamp_probability_is_one_seventh_l3112_311279


namespace least_k_factorial_divisible_by_315_l3112_311209

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem least_k_factorial_divisible_by_315 :
  ∀ k : ℕ, k > 1 → (factorial k) % 315 = 0 → k ≥ 7 :=
by sorry

end least_k_factorial_divisible_by_315_l3112_311209


namespace sum_of_numbers_in_ratio_l3112_311244

theorem sum_of_numbers_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 4725 →
  a + b + c = 105 := by
sorry

end sum_of_numbers_in_ratio_l3112_311244


namespace factorization_of_16x_squared_minus_4_l3112_311251

theorem factorization_of_16x_squared_minus_4 (x : ℝ) :
  16 * x^2 - 4 = 4 * (2*x + 1) * (2*x - 1) := by
  sorry

end factorization_of_16x_squared_minus_4_l3112_311251


namespace min_score_is_45_l3112_311269

/-- Represents the test scores and conditions -/
structure TestScores where
  num_tests : ℕ
  max_score : ℕ
  first_three : Fin 3 → ℕ
  target_average : ℕ

/-- Calculates the minimum score needed on one of the last two tests -/
def min_score (ts : TestScores) : ℕ :=
  let total_needed := ts.target_average * ts.num_tests
  let first_three_sum := (ts.first_three 0) + (ts.first_three 1) + (ts.first_three 2)
  let remaining_sum := total_needed - first_three_sum
  remaining_sum - ts.max_score

/-- Theorem stating the minimum score needed is 45 -/
theorem min_score_is_45 (ts : TestScores) 
  (h1 : ts.num_tests = 5)
  (h2 : ts.max_score = 120)
  (h3 : ts.first_three 0 = 86 ∧ ts.first_three 1 = 102 ∧ ts.first_three 2 = 97)
  (h4 : ts.target_average = 90) :
  min_score ts = 45 := by
  sorry

#eval min_score { num_tests := 5, max_score := 120, first_three := ![86, 102, 97], target_average := 90 }

end min_score_is_45_l3112_311269


namespace largest_solution_quadratic_l3112_311254

theorem largest_solution_quadratic (x : ℝ) : 
  (3 * (8 * x^2 + 10 * x + 8) = x * (8 * x - 34)) →
  x ≤ (-4 + Real.sqrt 10) / 2 :=
by sorry

end largest_solution_quadratic_l3112_311254


namespace tanner_video_game_cost_l3112_311230

/-- The cost of Tanner's video game purchase -/
def video_game_cost (september_savings october_savings november_savings remaining_amount : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - remaining_amount

/-- Theorem stating the cost of Tanner's video game -/
theorem tanner_video_game_cost :
  video_game_cost 17 48 25 41 = 49 := by
  sorry

end tanner_video_game_cost_l3112_311230


namespace inequality_proof_l3112_311262

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^4 + b^4 + c^4 = 3) :
  1 / (4 - a*b) + 1 / (4 - b*c) + 1 / (4 - c*a) ≤ 1 := by
  sorry

end inequality_proof_l3112_311262


namespace fruits_left_l3112_311255

def fruits_problem (plums guavas apples given_away : ℕ) : ℕ :=
  (plums + guavas + apples) - given_away

theorem fruits_left (plums guavas apples given_away : ℕ) 
  (h : given_away ≤ plums + guavas + apples) : 
  fruits_problem plums guavas apples given_away = 
  (plums + guavas + apples) - given_away :=
by
  sorry

end fruits_left_l3112_311255


namespace bus_trip_speed_l3112_311258

/-- Proves that for a trip of 880 miles, if increasing the speed by 10 mph
    reduces the trip time by 2 hours, then the original speed was 61.5 mph. -/
theorem bus_trip_speed (v : ℝ) (h : v > 0) : 
  (880 / v) - (880 / (v + 10)) = 2 → v = 61.5 := by
  sorry

end bus_trip_speed_l3112_311258


namespace surface_area_bound_l3112_311288

/-- A convex broken line -/
structure ConvexBrokenLine where
  points : List (ℝ × ℝ)
  is_convex : Bool
  length : ℝ

/-- The surface area of revolution of a convex broken line -/
def surface_area_of_revolution (line : ConvexBrokenLine) : ℝ := sorry

/-- Theorem: The surface area of revolution of a convex broken line
    is less than or equal to π * d² / 2, where d is the length of the line -/
theorem surface_area_bound (line : ConvexBrokenLine) :
  surface_area_of_revolution line ≤ Real.pi * line.length^2 / 2 := by
  sorry

end surface_area_bound_l3112_311288


namespace A_intersect_complement_B_eq_a_l3112_311278

-- Define the universal set U
def U : Set Char := {'{', 'a', 'b', 'c', 'd', 'e', '}'}

-- Define set A
def A : Set Char := {'{', 'a', 'b', '}'}

-- Define set B
def B : Set Char := {'{', 'b', 'c', 'd', '}'}

-- Theorem to prove
theorem A_intersect_complement_B_eq_a : A ∩ (U \ B) = {'{', 'a', '}'} := by
  sorry

end A_intersect_complement_B_eq_a_l3112_311278


namespace star_calculation_l3112_311286

def star (x y : ℤ) : ℤ := x * y - 1

theorem star_calculation : (star (star 2 3) 4) = 19 := by sorry

end star_calculation_l3112_311286


namespace sets_are_equal_l3112_311225

-- Define the sets A and B
def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, |-(Real.sqrt 3)|}

-- State the theorem
theorem sets_are_equal : A = B := by sorry

end sets_are_equal_l3112_311225


namespace base_r_is_seven_l3112_311287

/-- Represents a number in base r --/
def BaseR (n : ℕ) (r : ℕ) : ℕ → ℕ
| 0 => 0
| (k+1) => (n % r) * r^k + BaseR (n / r) r k

/-- The equation representing the transaction in base r --/
def TransactionEquation (r : ℕ) : Prop :=
  BaseR 210 r 2 + BaseR 260 r 2 = BaseR 500 r 2

theorem base_r_is_seven :
  ∃ r : ℕ, r > 1 ∧ TransactionEquation r ∧ r = 7 := by
  sorry

end base_r_is_seven_l3112_311287


namespace intersection_y_intercept_sum_l3112_311240

/-- Given two lines that intersect at a point, prove the sum of their y-intercepts. -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∃ x y : ℝ, x = (1/3)*y + a ∧ y = (1/3)*x + b ∧ x = 3 ∧ y = -1) →
  a + b = 4/3 := by
  sorry

end intersection_y_intercept_sum_l3112_311240


namespace inequality_relationship_l3112_311295

theorem inequality_relationship (x : ℝ) : 
  (∀ x, x - 2 > 0 → (x - 2) * (x - 1) > 0) ∧ 
  (∃ x, (x - 2) * (x - 1) > 0 ∧ ¬(x - 2 > 0)) :=
by sorry

end inequality_relationship_l3112_311295


namespace prob_at_least_one_consonant_l3112_311218

def word : String := "barkhint"

def is_consonant (c : Char) : Bool :=
  c ∈ ['b', 'r', 'k', 'h', 'n', 't']

def num_letters : Nat := word.length

def num_vowels : Nat := word.toList.filter (fun c => !is_consonant c) |>.length

def num_ways_to_select_two : Nat := num_letters * (num_letters - 1) / 2

def num_ways_to_select_two_vowels : Nat := num_vowels * (num_vowels - 1) / 2

theorem prob_at_least_one_consonant :
  (1 : ℚ) - (num_ways_to_select_two_vowels : ℚ) / num_ways_to_select_two = 27 / 28 := by
  sorry

end prob_at_least_one_consonant_l3112_311218


namespace ap_sum_possible_n_values_l3112_311273

theorem ap_sum_possible_n_values :
  let S (n : ℕ) (a : ℤ) := (n : ℤ) * (2 * a + (n - 1) * 3) / 2
  (∃! k : ℕ, k > 1 ∧ (∃ a : ℤ, S k a = 180) ∧
    ∀ m : ℕ, m > 1 → (∃ b : ℤ, S m b = 180) → m ∈ Finset.range k) :=
by sorry

end ap_sum_possible_n_values_l3112_311273


namespace hiking_distance_sum_l3112_311268

theorem hiking_distance_sum : 
  let leg1 : ℝ := 3.8
  let leg2 : ℝ := 1.75
  let leg3 : ℝ := 2.3
  let leg4 : ℝ := 0.45
  let leg5 : ℝ := 1.92
  leg1 + leg2 + leg3 + leg4 + leg5 = 10.22 := by
  sorry

end hiking_distance_sum_l3112_311268


namespace sophie_donuts_l3112_311275

/-- The number of donuts left for Sophie after giving some away -/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) : ℕ :=
  (total_boxes - boxes_given) * donuts_per_box - donuts_given

/-- Theorem stating that Sophie is left with 30 donuts -/
theorem sophie_donuts :
  donuts_left 4 12 1 6 = 30 := by
  sorry

end sophie_donuts_l3112_311275


namespace min_shipping_cost_l3112_311224

/-- Represents the shipping problem with given stock, demand, and costs. -/
structure ShippingProblem where
  shanghai_stock : ℕ
  nanjing_stock : ℕ
  suzhou_demand : ℕ
  changsha_demand : ℕ
  cost_shanghai_suzhou : ℕ
  cost_shanghai_changsha : ℕ
  cost_nanjing_suzhou : ℕ
  cost_nanjing_changsha : ℕ

/-- Calculates the total shipping cost given the number of units shipped from Shanghai to Suzhou. -/
def total_cost (problem : ShippingProblem) (x : ℕ) : ℕ :=
  problem.cost_shanghai_suzhou * x +
  problem.cost_shanghai_changsha * (problem.shanghai_stock - x) +
  problem.cost_nanjing_suzhou * (problem.suzhou_demand - x) +
  problem.cost_nanjing_changsha * (x - (problem.suzhou_demand - problem.nanjing_stock))

/-- Theorem stating that the minimum shipping cost is 8600 yuan for the given problem. -/
theorem min_shipping_cost (problem : ShippingProblem) 
  (h1 : problem.shanghai_stock = 12)
  (h2 : problem.nanjing_stock = 6)
  (h3 : problem.suzhou_demand = 10)
  (h4 : problem.changsha_demand = 8)
  (h5 : problem.cost_shanghai_suzhou = 400)
  (h6 : problem.cost_shanghai_changsha = 800)
  (h7 : problem.cost_nanjing_suzhou = 300)
  (h8 : problem.cost_nanjing_changsha = 500) :
  ∃ x : ℕ, x ≥ 4 ∧ x ≤ 10 ∧ total_cost problem x = 8600 ∧ 
  ∀ y : ℕ, y ≥ 4 → y ≤ 10 → total_cost problem y ≥ total_cost problem x :=
sorry

end min_shipping_cost_l3112_311224


namespace geometric_sequence_sum_l3112_311236

/-- A geometric sequence with sum of first n terms S_n -/
structure GeometricSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given conditions for the geometric sequence -/
def given_sequence : GeometricSequence where
  S := fun n => 
    if n = 2 then 6
    else if n = 4 then 18
    else 0  -- We only know S_2 and S_4, other values are placeholders

theorem geometric_sequence_sum (seq : GeometricSequence) :
  seq.S 2 = 6 → seq.S 4 = 18 → seq.S 6 = 42 := by
  sorry

end geometric_sequence_sum_l3112_311236


namespace greatest_integer_with_gcf_nine_l3112_311280

theorem greatest_integer_with_gcf_nine : ∃ n : ℕ, n < 200 ∧ 
  Nat.gcd n 45 = 9 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 45 = 9 → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_integer_with_gcf_nine_l3112_311280


namespace gift_cost_theorem_l3112_311201

def polo_price : ℚ := 26
def necklace_price : ℚ := 83
def game_price : ℚ := 90
def sock_price : ℚ := 7
def book_price : ℚ := 15
def scarf_price : ℚ := 22

def polo_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def game_quantity : ℕ := 1
def sock_quantity : ℕ := 4
def book_quantity : ℕ := 3
def scarf_quantity : ℕ := 2

def sales_tax_rate : ℚ := 13 / 200  -- 6.5%
def book_discount_rate : ℚ := 1 / 10  -- 10%
def rebate : ℚ := 12

def total_cost : ℚ :=
  polo_price * polo_quantity +
  necklace_price * necklace_quantity +
  game_price * game_quantity +
  sock_price * sock_quantity +
  book_price * book_quantity +
  scarf_price * scarf_quantity

def discounted_book_cost : ℚ := book_price * book_quantity * (1 - book_discount_rate)

def total_cost_after_book_discount : ℚ :=
  total_cost - (book_price * book_quantity) + discounted_book_cost

def total_cost_with_tax : ℚ :=
  total_cost_after_book_discount * (1 + sales_tax_rate)

def final_cost : ℚ := total_cost_with_tax - rebate

theorem gift_cost_theorem :
  final_cost = 46352 / 100 := by sorry

end gift_cost_theorem_l3112_311201


namespace triangle_area_l3112_311285

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  b^2 - b*c - 2*c^2 = 0 →
  a = Real.sqrt 6 →
  Real.cos A = 7/8 →
  (1/2) * b * c * Real.sin A = Real.sqrt 15 / 2 := by
sorry

end triangle_area_l3112_311285


namespace trig_identity_l3112_311239

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin ((5*π)/6 - x) + (Real.cos ((π/3) - x))^2 = 5/16 := by
  sorry

end trig_identity_l3112_311239


namespace binomial_coefficient_x4_in_x_plus_1_to_10_l3112_311233

theorem binomial_coefficient_x4_in_x_plus_1_to_10 :
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (1^(10 - k)) * (1^k)) = 210 := by
  sorry

end binomial_coefficient_x4_in_x_plus_1_to_10_l3112_311233


namespace algebraic_expression_value_l3112_311245

theorem algebraic_expression_value (x : ℝ) : 2 * x^2 + 2 * x + 5 = 9 → 3 * x^2 + 3 * x - 7 = -1 := by
  sorry

end algebraic_expression_value_l3112_311245


namespace fathers_age_is_32_l3112_311204

/-- The present age of the father -/
def father_age : ℕ := 32

/-- The present age of the older son -/
def older_son_age : ℕ := 22

/-- The present age of the younger son -/
def younger_son_age : ℕ := 18

/-- The average age of the father and his two sons is 24 years -/
axiom average_age : (father_age + older_son_age + younger_son_age) / 3 = 24

/-- 5 years ago, the average age of the two sons was 15 years -/
axiom sons_average_age_5_years_ago : (older_son_age - 5 + younger_son_age - 5) / 2 = 15

/-- The difference between the ages of the two sons is 4 years -/
axiom sons_age_difference : older_son_age - younger_son_age = 4

/-- Theorem: Given the conditions, the father's present age is 32 years -/
theorem fathers_age_is_32 : father_age = 32 := by
  sorry

end fathers_age_is_32_l3112_311204


namespace always_odd_l3112_311212

theorem always_odd (a b c : ℕ+) (ha : a.val % 2 = 1) (hb : b.val % 2 = 1) :
  (3^a.val + (b.val - 1)^2 * c.val) % 2 = 1 := by
  sorry

end always_odd_l3112_311212


namespace vitamin_pack_size_l3112_311296

theorem vitamin_pack_size (vitamin_a_pack_size : ℕ) 
  (vitamin_a_packs : ℕ) (vitamin_d_packs : ℕ) : 
  (vitamin_a_pack_size * vitamin_a_packs = 17 * vitamin_d_packs) →  -- Equal quantities condition
  (vitamin_a_pack_size * vitamin_a_packs = 119) →                   -- Smallest number condition
  (∀ x y : ℕ, x * y = 119 → x ≤ vitamin_a_pack_size ∨ y ≤ vitamin_a_packs) →  -- Smallest positive integer values
  vitamin_a_pack_size = 7 :=
by sorry

end vitamin_pack_size_l3112_311296


namespace stacy_paper_pages_per_day_l3112_311210

/-- Calculates the number of pages to write per day given total pages and number of days -/
def pagesPerDay (totalPages : ℕ) (numDays : ℕ) : ℚ :=
  totalPages / numDays

theorem stacy_paper_pages_per_day :
  let totalPages : ℕ := 33
  let numDays : ℕ := 3
  pagesPerDay totalPages numDays = 11 := by
  sorry

end stacy_paper_pages_per_day_l3112_311210


namespace log_square_ratio_l3112_311227

theorem log_square_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (h2 : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
sorry

end log_square_ratio_l3112_311227


namespace paper_flowers_per_hour_l3112_311226

/-- The number of paper flowers Person B makes per hour -/
def flowers_per_hour_B : ℕ := 80

/-- The number of paper flowers Person A makes per hour -/
def flowers_per_hour_A : ℕ := flowers_per_hour_B - 20

/-- The time it takes Person A to make 120 flowers -/
def time_A : ℚ := 120 / flowers_per_hour_A

/-- The time it takes Person B to make 160 flowers -/
def time_B : ℚ := 160 / flowers_per_hour_B

theorem paper_flowers_per_hour :
  (flowers_per_hour_A = flowers_per_hour_B - 20) ∧
  (time_A = time_B) →
  flowers_per_hour_B = 80 := by
  sorry

end paper_flowers_per_hour_l3112_311226


namespace total_cats_l3112_311220

/-- The number of cats owned by Mr. Thompson -/
def thompson_cats : ℕ := 15

/-- The number of cats owned by Mrs. Sheridan -/
def sheridan_cats : ℕ := 11

/-- The number of cats owned by Mrs. Garrett -/
def garrett_cats : ℕ := 24

/-- The number of cats owned by Mr. Ravi -/
def ravi_cats : ℕ := 18

/-- The theorem stating that the total number of cats is 68 -/
theorem total_cats : thompson_cats + sheridan_cats + garrett_cats + ravi_cats = 68 := by
  sorry

end total_cats_l3112_311220


namespace money_ratio_proof_l3112_311266

theorem money_ratio_proof (alison brittany brooke kent : ℕ) : 
  alison = brittany / 2 →
  brittany = 4 * brooke →
  kent = 1000 →
  alison = 4000 →
  brooke / kent = 2 := by
sorry

end money_ratio_proof_l3112_311266


namespace one_carton_per_case_l3112_311272

/-- Given a case containing cartons, each carton containing b boxes,
    each box containing 400 paper clips, and 800 paper clips in 2 cases,
    prove that there is 1 carton in a case. -/
theorem one_carton_per_case (b : ℕ) (h1 : b ≥ 1) :
  ∃ (c : ℕ), c = 1 ∧ 2 * c * b * 400 = 800 := by
  sorry

#check one_carton_per_case

end one_carton_per_case_l3112_311272


namespace sqrt_five_irrational_and_greater_than_two_l3112_311211

theorem sqrt_five_irrational_and_greater_than_two :
  ∃ x : ℝ, Irrational x ∧ x > 2 ∧ x = Real.sqrt 5 := by sorry

end sqrt_five_irrational_and_greater_than_two_l3112_311211


namespace profit_percentage_is_twenty_l3112_311248

/-- Calculates the percentage profit on wholesale price given wholesale price, retail price, and discount percentage. -/
def percentage_profit (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that given the specific values in the problem, the percentage profit is 20%. -/
theorem profit_percentage_is_twenty :
  percentage_profit 108 144 10 = 20 := by
  sorry

end profit_percentage_is_twenty_l3112_311248


namespace triangle_sides_perfect_square_l3112_311215

theorem triangle_sides_perfect_square
  (a b c : ℤ)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_condition : Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs = 1)
  (int_quotient_1 : ∃ k : ℤ, k * (a + b - c) = a^2 + b^2 - c^2)
  (int_quotient_2 : ∃ k : ℤ, k * (b + c - a) = b^2 + c^2 - a^2)
  (int_quotient_3 : ∃ k : ℤ, k * (c + a - b) = c^2 + a^2 - b^2) :
  ∃ n : ℤ, (a + b - c) * (b + c - a) * (c + a - b) = n^2 ∨
           2 * (a + b - c) * (b + c - a) * (c + a - b) = n^2 :=
by sorry

end triangle_sides_perfect_square_l3112_311215


namespace expand_expression_l3112_311203

theorem expand_expression (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 := by
  sorry

end expand_expression_l3112_311203


namespace min_area_special_square_l3112_311223

/-- A square with one side on y = 2x - 17 and two vertices on y = x^2 -/
structure SpecialSquare where
  /-- Side length of the square -/
  a : ℝ
  /-- Parameter for the line y = 2x + b passing through two vertices on the parabola -/
  b : ℝ
  /-- The square has one side on y = 2x - 17 -/
  side_on_line : a = (17 + b) / Real.sqrt 5
  /-- Two vertices of the square are on y = x^2 -/
  vertices_on_parabola : a^2 = 20 * (1 + b)

/-- The minimum area of a SpecialSquare is 80 -/
theorem min_area_special_square :
  ∀ s : SpecialSquare, s.a^2 ≥ 80 := by
  sorry

#check min_area_special_square

end min_area_special_square_l3112_311223


namespace fish_upstream_speed_l3112_311229

/-- The upstream speed of a fish given its downstream speed and speed in still water -/
theorem fish_upstream_speed (downstream_speed still_water_speed : ℝ) :
  downstream_speed = 55 →
  still_water_speed = 45 →
  still_water_speed - (downstream_speed - still_water_speed) = 35 := by
  sorry

#check fish_upstream_speed

end fish_upstream_speed_l3112_311229


namespace fraction_inequality_l3112_311271

theorem fraction_inequality (x : ℝ) : (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end fraction_inequality_l3112_311271


namespace angle_measure_proof_l3112_311265

theorem angle_measure_proof (x : ℝ) : 
  (180 - x) = 3 * (90 - x) + 10 → x = 50 := by
  sorry

end angle_measure_proof_l3112_311265


namespace ink_bottle_arrangement_l3112_311235

-- Define the type for a row of bottles
def Row := Fin 7 → Bool

-- Define the type for the arrangement of bottles
def Arrangement := Fin 130 → Row

-- Theorem statement
theorem ink_bottle_arrangement (arr : Arrangement) :
  (∃ i j k : Fin 130, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ arr i = arr j ∧ arr j = arr k) ∨
  (∃ i₁ j₁ i₂ j₂ : Fin 130, i₁ ≠ j₁ ∧ i₂ ≠ j₂ ∧ i₁ ≠ i₂ ∧ i₁ ≠ j₂ ∧ j₁ ≠ i₂ ∧ j₁ ≠ j₂ ∧
    arr i₁ = arr j₁ ∧ arr i₂ = arr j₂) :=
by
  sorry

end ink_bottle_arrangement_l3112_311235


namespace root_sum_reciprocal_l3112_311222

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 1 = 0) → 
  (b^3 - 2*b + 1 = 0) → 
  (c^3 - 2*c + 1 = 0) → 
  (a ≠ b) → (b ≠ c) → (c ≠ a) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 10 / 3) := by
sorry

end root_sum_reciprocal_l3112_311222


namespace child_tickets_sold_l3112_311231

/-- Proves the number of child tickets sold in a theater --/
theorem child_tickets_sold (total_tickets : ℕ) (adult_price child_price total_revenue : ℚ) :
  total_tickets = 80 →
  adult_price = 12 →
  child_price = 5 →
  total_revenue = 519 →
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 63 := by
  sorry

end child_tickets_sold_l3112_311231


namespace hall_area_l3112_311274

/-- Proves that the area of a rectangular hall is 500 square meters, given that its length is 25 meters and 5 meters more than its breadth. -/
theorem hall_area : 
  ∀ (length breadth : ℝ),
  length = 25 →
  length = breadth + 5 →
  length * breadth = 500 := by
sorry

end hall_area_l3112_311274


namespace largest_among_expressions_l3112_311232

theorem largest_among_expressions : 
  let a := -|(-3)|^3
  let b := -(-3)^3
  let c := (-3)^3
  let d := -(3^3)
  (b ≥ a) ∧ (b ≥ c) ∧ (b ≥ d) :=
by sorry

end largest_among_expressions_l3112_311232


namespace grid_routes_3x2_l3112_311281

theorem grid_routes_3x2 :
  let total_moves : ℕ := 3 + 2
  let right_moves : ℕ := 3
  let down_moves : ℕ := 2
  let num_routes : ℕ := Nat.choose total_moves down_moves
  num_routes = 10 := by sorry

end grid_routes_3x2_l3112_311281


namespace ellipse_foci_distance_l3112_311249

/-- The distance between the foci of the ellipse 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  let a : ℝ := 4
  let b : ℝ := 3
  ∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 →
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 7 :=
by sorry

end ellipse_foci_distance_l3112_311249


namespace muffin_profit_l3112_311234

/-- Bob's muffin business profit calculation -/
theorem muffin_profit : 
  ∀ (muffins_per_day : ℕ) 
    (cost_price selling_price : ℚ) 
    (days_in_week : ℕ),
  muffins_per_day = 12 →
  cost_price = 3/4 →
  selling_price = 3/2 →
  days_in_week = 7 →
  (selling_price - cost_price) * muffins_per_day * days_in_week = 63 := by
  sorry


end muffin_profit_l3112_311234


namespace wrapping_paper_area_formula_l3112_311263

/-- Represents a rectangular box with length, width, and height. -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- Calculates the area of wrapping paper needed to wrap a box. -/
def wrappingPaperArea (box : Box) : ℝ :=
  6 * box.length * box.height + 2 * box.width * box.height

/-- Theorem stating that the wrapping paper area for a box is 6lh + 2wh. -/
theorem wrapping_paper_area_formula (box : Box) :
  wrappingPaperArea box = 6 * box.length * box.height + 2 * box.width * box.height :=
by sorry

end wrapping_paper_area_formula_l3112_311263


namespace stream_speed_l3112_311252

theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 14 →
  distance = 4864 →
  total_time = 700 →
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time) →
  stream_speed = 1.2 := by
sorry

end stream_speed_l3112_311252


namespace no_prime_solution_l3112_311256

-- Define a function to convert a number from base p to base 10
def to_base_10 (digits : List Nat) (p : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * p ^ i) 0

-- Define the left-hand side of the equation
def lhs (p : Nat) : Nat :=
  to_base_10 [6, 0, 0, 2] p +
  to_base_10 [4, 0, 4] p +
  to_base_10 [5, 1, 2] p +
  to_base_10 [2, 2, 2] p +
  to_base_10 [9] p

-- Define the right-hand side of the equation
def rhs (p : Nat) : Nat :=
  to_base_10 [3, 3, 4] p +
  to_base_10 [2, 7, 5] p +
  to_base_10 [1, 2, 3] p

-- State the theorem
theorem no_prime_solution :
  ¬ ∃ p : Nat, Nat.Prime p ∧ lhs p = rhs p :=
sorry

end no_prime_solution_l3112_311256


namespace order_of_trig_values_l3112_311202

theorem order_of_trig_values :
  let a := Real.tan (70 * π / 180)
  let b := Real.sin (25 * π / 180)
  let c := Real.cos (25 * π / 180)
  b < c ∧ c < a := by sorry

end order_of_trig_values_l3112_311202


namespace perpendicular_lines_sum_l3112_311238

/-- Two perpendicular lines with a given perpendicular foot -/
structure PerpendicularLines where
  a : ℝ
  b : ℝ
  c : ℝ
  line1 : ∀ x y : ℝ, a * x + 4 * y - 2 = 0
  line2 : ∀ x y : ℝ, 2 * x - 5 * y + b = 0
  perpendicular : (a / 4) * (2 / 5) = -1
  foot_on_line1 : a * 1 + 4 * c - 2 = 0
  foot_on_line2 : 2 * 1 - 5 * c + b = 0

/-- The sum of a, b, and c for perpendicular lines with given conditions is -4 -/
theorem perpendicular_lines_sum (l : PerpendicularLines) : l.a + l.b + l.c = -4 := by
  sorry

end perpendicular_lines_sum_l3112_311238


namespace cubic_polynomials_common_root_l3112_311250

theorem cubic_polynomials_common_root :
  ∃ (c d : ℝ), c = -3 ∧ d = -4 ∧
  ∃ (x : ℝ), x^3 + c*x^2 + 15*x + 10 = 0 ∧ x^3 + d*x^2 + 17*x + 12 = 0 := by
  sorry

end cubic_polynomials_common_root_l3112_311250


namespace sufficient_condition_for_p_l3112_311293

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define what it means for a condition to be sufficient but not necessary
def sufficient_but_not_necessary (condition : ℝ → Prop) (proposition : ℝ → Prop) : Prop :=
  (∃ a : ℝ, condition a ∧ proposition a) ∧
  (∃ a : ℝ, ¬condition a ∧ proposition a)

-- Theorem statement
theorem sufficient_condition_for_p :
  sufficient_but_not_necessary (λ a : ℝ => a = 2) p :=
sorry

end sufficient_condition_for_p_l3112_311293


namespace triangle_with_ap_angles_and_altitudes_is_equilateral_l3112_311217

/-- A triangle with angles and altitudes in arithmetic progression is equilateral -/
theorem triangle_with_ap_angles_and_altitudes_is_equilateral 
  (A B C : ℝ) (a b c : ℝ) (ha hb hc : ℝ) : 
  (∃ (d : ℝ), A = B - d ∧ C = B + d) →  -- Angles in arithmetic progression
  (A + B + C = 180) →                   -- Sum of angles in a triangle
  (ha + hc = 2 * hb) →                  -- Altitudes in arithmetic progression
  (ha = 2 * area / a) →                 -- Relation between altitude and side
  (hb = 2 * area / b) → 
  (hc = 2 * area / c) → 
  (b^2 = a^2 + c^2 - a*c) →             -- Law of cosines for 60° angle
  (a = b ∧ b = c) :=                    -- Triangle is equilateral
by sorry

end triangle_with_ap_angles_and_altitudes_is_equilateral_l3112_311217


namespace fourth_group_frequency_count_l3112_311206

theorem fourth_group_frequency_count 
  (f₁ f₂ f₃ : ℝ) 
  (n₁ : ℕ) 
  (h₁ : f₁ = 0.1) 
  (h₂ : f₂ = 0.3) 
  (h₃ : f₃ = 0.4) 
  (h₄ : n₁ = 5) 
  (h₅ : f₁ + f₂ + f₃ < 1) : 
  ∃ (N : ℕ) (n₄ : ℕ), 
    N > 0 ∧ 
    f₁ = n₁ / N ∧ 
    n₄ = N * (1 - (f₁ + f₂ + f₃)) ∧ 
    n₄ = 10 := by
  sorry

end fourth_group_frequency_count_l3112_311206


namespace playground_width_l3112_311291

/-- The number of playgrounds -/
def num_playgrounds : ℕ := 8

/-- The length of each playground in meters -/
def playground_length : ℝ := 300

/-- The total area of all playgrounds in square kilometers -/
def total_area_km2 : ℝ := 0.6

/-- Conversion factor from square kilometers to square meters -/
def km2_to_m2 : ℝ := 1000000

theorem playground_width :
  ∀ (width : ℝ),
  (width * playground_length * num_playgrounds = total_area_km2 * km2_to_m2) →
  width = 250 := by
sorry

end playground_width_l3112_311291


namespace distance_between_foci_l3112_311277

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 5)^2) = 26

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 5)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 26 :=
by sorry

end distance_between_foci_l3112_311277


namespace root_sum_reciprocal_l3112_311294

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ r ≠ p) →
  (∀ (x : ℝ), x^3 - 20*x^2 + 99*x - 154 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ (t : ℝ), t ≠ p ∧ t ≠ q ∧ t ≠ r → 
    1 / (t^3 - 20*t^2 + 99*t - 154) = A / (t - p) + B / (t - q) + C / (t - r)) →
  1 / A + 1 / B + 1 / C = 245 :=
by sorry

end root_sum_reciprocal_l3112_311294


namespace largest_expression_l3112_311299

def y : ℝ := 0.0002

theorem largest_expression (a b c d e : ℝ) 
  (ha : a = 5 + y)
  (hb : b = 5 - y)
  (hc : c = 5 * y)
  (hd : d = 5 / y)
  (he : e = y / 5) :
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_expression_l3112_311299


namespace divisibility_implies_equation_existence_l3112_311200

theorem divisibility_implies_equation_existence (p x y : ℕ) (hp : Prime p) 
  (hp_form : ∃ k : ℕ, p = 4 * k + 3) (hx : x > 0) (hy : y > 0)
  (hdiv : p ∣ (x^2 - x*y + ((p+1)/4) * y^2)) :
  ∃ u v : ℤ, x^2 - x*y + ((p+1)/4) * y^2 = p * (u^2 - u*v + ((p+1)/4) * v^2) := by
sorry

end divisibility_implies_equation_existence_l3112_311200


namespace ellipse_hyperbola_theorem_l3112_311216

-- Define the ellipse T
def ellipse_T (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola S
def hyperbola_S (x y m n : ℝ) : Prop :=
  x^2 / m^2 - y^2 / n^2 = 1 ∧ m > 0 ∧ n > 0

-- Define the common focus
def common_focus (a b m n : ℝ) : Prop :=
  a^2 - b^2 = m^2 + n^2 ∧ a^2 - b^2 = 4

-- Define the asymptotic line l
def asymptotic_line (x y m n : ℝ) : Prop :=
  y = (n / m) * x

-- Define the symmetry condition
def symmetry_condition (a b m n : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola_S x y m n ∧
  ((x = m^2 - 2 ∧ y = m * n) ∨ (x = 4*b/5 ∧ y = 3*b/5))

-- Main theorem
theorem ellipse_hyperbola_theorem (a b m n : ℝ) :
  ellipse_T 0 b a b ∧
  hyperbola_S 2 0 m n ∧
  common_focus a b m n ∧
  (∃ (x y : ℝ), hyperbola_S x y m n ∧ asymptotic_line x y m n) ∧
  symmetry_condition a b m n →
  a^2 = 5 ∧ b^2 = 4 ∧ m^2 = 4/5 ∧ n^2 = 16/5 :=
by
  sorry

end ellipse_hyperbola_theorem_l3112_311216
