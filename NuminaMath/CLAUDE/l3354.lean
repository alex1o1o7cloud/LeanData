import Mathlib

namespace train_length_l3354_335426

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 108 →
  platform_length = 300.06 →
  crossing_time = 25 →
  (train_speed * 1000 / 3600 * crossing_time) - platform_length = 449.94 := by
  sorry

#check train_length

end train_length_l3354_335426


namespace omega_squared_plus_7omega_plus_40_abs_l3354_335486

def ω : ℂ := 4 + 3 * Complex.I

theorem omega_squared_plus_7omega_plus_40_abs : 
  Complex.abs (ω^2 + 7*ω + 40) = 15 * Real.sqrt 34 := by
  sorry

end omega_squared_plus_7omega_plus_40_abs_l3354_335486


namespace range_of_f_l3354_335419

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.Icc (-5 : ℝ) 13, ∃ x ∈ Set.Icc 2 5, f x = y ∧
  ∀ x ∈ Set.Icc 2 5, f x ∈ Set.Icc (-5 : ℝ) 13 :=
sorry

end range_of_f_l3354_335419


namespace min_sum_floor_l3354_335407

theorem min_sum_floor (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (⌊(x^2 + y^2) / z⌋ + ⌊(y^2 + z^2) / x⌋ + ⌊(z^2 + x^2) / y⌋ = 4) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ ≥ 4 :=
by sorry

end min_sum_floor_l3354_335407


namespace f_equals_negative_two_iff_b_equals_negative_one_l3354_335442

def f (x : ℝ) : ℝ := 5 * x + 3

theorem f_equals_negative_two_iff_b_equals_negative_one :
  ∀ b : ℝ, f b = -2 ↔ b = -1 := by sorry

end f_equals_negative_two_iff_b_equals_negative_one_l3354_335442


namespace inverse_variation_problem_l3354_335415

theorem inverse_variation_problem (k : ℝ) (h1 : k > 0) :
  (∀ x y : ℝ, x ≠ 0 → y * x^2 = k) →
  (2 * 3^2 = k) →
  (∃ x : ℝ, x > 0 ∧ 8 * x^2 = k) →
  (∃ x : ℝ, x > 0 ∧ 8 * x^2 = k ∧ x = 3/2) :=
by sorry

end inverse_variation_problem_l3354_335415


namespace pet_shop_kittens_l3354_335414

/-- Calculates the number of kittens in a pet shop given the following conditions:
  * The pet shop has 2 puppies
  * A puppy costs $20
  * A kitten costs $15
  * The total stock is worth $100
-/
theorem pet_shop_kittens (num_puppies : ℕ) (puppy_cost kitten_cost total_stock : ℚ) : 
  num_puppies = 2 → 
  puppy_cost = 20 → 
  kitten_cost = 15 → 
  total_stock = 100 → 
  (total_stock - num_puppies * puppy_cost) / kitten_cost = 4 := by
  sorry

#check pet_shop_kittens

end pet_shop_kittens_l3354_335414


namespace lcm_hcf_problem_l3354_335463

theorem lcm_hcf_problem (a b : ℕ+) :
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 47 →
  a = 210 →
  b = 517 := by
sorry

end lcm_hcf_problem_l3354_335463


namespace tangent_line_problem_l3354_335406

theorem tangent_line_problem (k a : ℝ) : 
  (∃ b : ℝ, (3 = 4 + a / 2 + 1) ∧ 
             (3 = 2 * k + b) ∧ 
             (k = 2 * 2 - a / 4)) → 
  (∃ b : ℝ, (3 = 4 + a / 2 + 1) ∧ 
             (3 = 2 * k + b) ∧ 
             (k = 2 * 2 - a / 4) ∧ 
             b = -7) :=
by sorry

end tangent_line_problem_l3354_335406


namespace profit_and_max_profit_l3354_335498

/-- Initial profit per visitor in yuan -/
def initial_profit_per_visitor : ℝ := 10

/-- Initial daily visitor count -/
def initial_visitor_count : ℝ := 500

/-- Visitor loss per yuan of price increase -/
def visitor_loss_per_yuan : ℝ := 20

/-- Calculate profit based on price increase -/
def profit (price_increase : ℝ) : ℝ :=
  (initial_profit_per_visitor + price_increase) * (initial_visitor_count - visitor_loss_per_yuan * price_increase)

/-- Ticket price increase for 6000 yuan daily profit -/
def price_increase_for_target_profit : ℝ := 10

/-- Ticket price increase for maximum profit -/
def price_increase_for_max_profit : ℝ := 7.5

theorem profit_and_max_profit :
  (profit price_increase_for_target_profit = 6000) ∧
  (∀ x : ℝ, profit x ≤ profit price_increase_for_max_profit) := by
  sorry

end profit_and_max_profit_l3354_335498


namespace inverse_proportional_cube_root_l3354_335404

theorem inverse_proportional_cube_root (x y : ℝ) (k : ℝ) : 
  (x ^ 2 * y ^ (1/3) = k) →  -- x² and ³√y are inversely proportional
  (3 ^ 2 * 216 ^ (1/3) = k) →  -- x = 3 when y = 216
  (x * y = 54) →  -- xy = 54
  y = 18 * 4 ^ (1/3) :=  -- y = 18 ³√4
by sorry

end inverse_proportional_cube_root_l3354_335404


namespace bus_overlap_count_l3354_335476

-- Define the bus schedules
def busA_interval : ℕ := 6
def busB_interval : ℕ := 10
def busC_interval : ℕ := 14

-- Define the time range in minutes (5:00 PM to 10:00 PM)
def start_time : ℕ := 240  -- 4 hours after 1:00 PM
def end_time : ℕ := 540    -- 9 hours after 1:00 PM

-- Function to calculate the number of overlaps between two buses
def count_overlaps (interval1 interval2 start_time end_time : ℕ) : ℕ :=
  (end_time - start_time) / Nat.lcm interval1 interval2 + 1

-- Function to calculate the total number of distinct overlaps
def total_distinct_overlaps (start_time end_time : ℕ) : ℕ :=
  let ab_overlaps := count_overlaps busA_interval busB_interval start_time end_time
  let bc_overlaps := count_overlaps busB_interval busC_interval start_time end_time
  let ac_overlaps := count_overlaps busA_interval busC_interval start_time end_time
  ab_overlaps + bc_overlaps + ac_overlaps - 2  -- Subtracting 2 for common overlaps

-- The main theorem
theorem bus_overlap_count : 
  total_distinct_overlaps start_time end_time = 18 := by
  sorry

end bus_overlap_count_l3354_335476


namespace union_of_A_and_B_complement_of_intersection_A_and_B_l3354_335481

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ -1}
def B : Set ℝ := {x | x + 4 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -5} := by sorry

-- Theorem for ∁ᵤ(A ∩ B)
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x : ℝ | x < -4 ∨ x > -1} := by sorry

end union_of_A_and_B_complement_of_intersection_A_and_B_l3354_335481


namespace corner_sum_is_164_l3354_335434

/-- Represents a 9x9 checkerboard filled with numbers 1 through 81 -/
def Checkerboard := Fin 9 → Fin 9 → Nat

/-- The number at position (i, j) on the checkerboard -/
def number_at (board : Checkerboard) (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The sum of numbers in the four corners of the checkerboard -/
def corner_sum (board : Checkerboard) : Nat :=
  number_at board 0 0 + number_at board 0 8 + 
  number_at board 8 0 + number_at board 8 8

/-- Theorem stating that the sum of numbers in the four corners is 164 -/
theorem corner_sum_is_164 (board : Checkerboard) : corner_sum board = 164 := by
  sorry

end corner_sum_is_164_l3354_335434


namespace parallelogram_base_length_l3354_335402

/-- Given a parallelogram with area 288 square centimeters and height 16 cm, 
    prove that its base length is 18 cm. -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 288) 
  (h2 : height = 16) : 
  area / height = 18 := by
  sorry

end parallelogram_base_length_l3354_335402


namespace inequality_and_equality_condition_l3354_335469

theorem inequality_and_equality_condition (a b c : ℝ) :
  (5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c) ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by sorry

end inequality_and_equality_condition_l3354_335469


namespace same_color_probability_l3354_335497

/-- The probability of drawing two balls of the same color from a bag with green and white balls -/
theorem same_color_probability (green white : ℕ) (h : green = 5 ∧ white = 9) :
  let total := green + white
  let p_green := green / total
  let p_white := white / total
  let p_same_color := p_green * ((green - 1) / (total - 1)) + p_white * ((white - 1) / (total - 1))
  p_same_color = 46 / 91 := by
  sorry

end same_color_probability_l3354_335497


namespace center_trajectory_is_parabola_l3354_335449

/-- A circle passing through a point and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passesThrough : center.1^2 + (center.2 - 3)^2 = radius^2
  tangentToLine : center.2 + radius = 3

/-- The trajectory of the center of a moving circle -/
def centerTrajectory (c : TangentCircle) : Prop :=
  c.center.1^2 = 12 * c.center.2

/-- Theorem: The trajectory of the center of a circle passing through (0, 3) 
    and tangent to y + 3 = 0 is described by x^2 = 12y -/
theorem center_trajectory_is_parabola :
  ∀ c : TangentCircle, centerTrajectory c :=
sorry

end center_trajectory_is_parabola_l3354_335449


namespace absolute_value_expression_l3354_335473

theorem absolute_value_expression (x : ℤ) (h : x = 1999) :
  |4*x^2 - 5*x + 1| - 4*|x^2 + 2*x + 2| + 3*x + 7 = -19990 := by
  sorry

end absolute_value_expression_l3354_335473


namespace x_minus_y_equals_three_l3354_335462

theorem x_minus_y_equals_three (x y : ℝ) 
  (eq1 : 3 * x - 5 * y = 5)
  (eq2 : x / (x + y) = 5 / 7) :
  x - y = 3 := by sorry

end x_minus_y_equals_three_l3354_335462


namespace possible_m_values_l3354_335440

-- Define set A
def A : Set ℤ := {-1, 1}

-- Define set B
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

-- Theorem statement
theorem possible_m_values :
  ∀ m : ℤ, B m ⊆ A → (m = 0 ∨ m = 1 ∨ m = -1) :=
by
  sorry

end possible_m_values_l3354_335440


namespace spade_calculation_l3354_335422

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : spade 3 (spade 5 (spade 7 10)) = 1 := by
  sorry

end spade_calculation_l3354_335422


namespace sum_of_solutions_squared_equation_l3354_335483

theorem sum_of_solutions_squared_equation (x₁ x₂ : ℝ) :
  (x₁ - 8)^2 = 49 → (x₂ - 8)^2 = 49 → x₁ + x₂ = 16 := by
  sorry

end sum_of_solutions_squared_equation_l3354_335483


namespace line_through_points_l3354_335456

/-- Given a line y = ax + b passing through points (3, 2) and (7, 14), prove that 2a - b = 13 -/
theorem line_through_points (a b : ℝ) : 
  (2 : ℝ) = a * 3 + b → 
  (14 : ℝ) = a * 7 + b → 
  2 * a - b = 13 := by
  sorry

end line_through_points_l3354_335456


namespace power_mod_seventeen_l3354_335491

theorem power_mod_seventeen : 5^2021 ≡ 11 [ZMOD 17] := by
  sorry

end power_mod_seventeen_l3354_335491


namespace coat_price_and_tax_l3354_335450

/-- Represents the price of a coat -/
structure CoatPrice where
  original : ℝ
  discounted : ℝ
  taxRate : ℝ

/-- Calculates the tax amount based on the original price and tax rate -/
def calculateTax (price : CoatPrice) : ℝ :=
  price.original * price.taxRate

theorem coat_price_and_tax (price : CoatPrice) 
  (h1 : price.discounted = 72)
  (h2 : price.discounted = (2/5) * price.original)
  (h3 : price.taxRate = 0.05) :
  price.original = 180 ∧ calculateTax price = 9 := by
  sorry

end coat_price_and_tax_l3354_335450


namespace problem_solution_l3354_335496

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1/p + 1/q = 1) 
  (h4 : p*q = 12) : 
  q = 6 := by
sorry

end problem_solution_l3354_335496


namespace incorrect_games_proportion_l3354_335448

/-- Represents a chess tournament -/
structure ChessTournament where
  N : ℕ  -- number of players
  incorrect_games : ℕ  -- number of incorrect games

/-- Definition of a round-robin tournament -/
def is_round_robin (t : ChessTournament) : Prop :=
  t.incorrect_games ≤ t.N * (t.N - 1) / 2

/-- The main theorem: incorrect games are less than 75% of total games -/
theorem incorrect_games_proportion (t : ChessTournament) 
  (h : is_round_robin t) : 
  (4 * t.incorrect_games : ℚ) < (3 * t.N * (t.N - 1) : ℚ) := by
  sorry


end incorrect_games_proportion_l3354_335448


namespace quadratic_equation_roots_l3354_335480

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 + 2*m*x1 + m^2 + m = 0 ∧ 
    x2^2 + 2*m*x2 + m^2 + m = 0 ∧ 
    x1^2 + x2^2 = 12) → 
  m = -2 :=
by sorry

end quadratic_equation_roots_l3354_335480


namespace cone_base_circumference_l3354_335464

/-- For a right circular cone with volume 16π cubic centimeters and height 6 cm,
    the circumference of its base is 4√2π cm. -/
theorem cone_base_circumference :
  ∀ (r : ℝ), 
    (1 / 3 * π * r^2 * 6 = 16 * π) →
    (2 * π * r = 4 * Real.sqrt 2 * π) :=
by sorry

end cone_base_circumference_l3354_335464


namespace yanna_apples_l3354_335412

theorem yanna_apples (apples_to_zenny apples_to_andrea apples_kept : ℕ) : 
  apples_to_zenny = 18 → 
  apples_to_andrea = 6 → 
  apples_kept = 36 → 
  apples_to_zenny + apples_to_andrea + apples_kept = 60 :=
by
  sorry

end yanna_apples_l3354_335412


namespace max_value_cos_theta_l3354_335482

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sin x

theorem max_value_cos_theta (θ : ℝ) 
  (h : ∀ x, f x ≤ f θ) : 
  Real.cos θ = 3 * Real.sqrt 10 / 10 := by
  sorry

end max_value_cos_theta_l3354_335482


namespace overlap_area_is_one_l3354_335420

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Calculate the area of overlap between two triangles on a 3x3 grid -/
def triangleOverlapArea (t1 t2 : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the area of overlap between the given triangles is 1 square unit -/
theorem overlap_area_is_one :
  let t1 : Triangle := { v1 := {x := 0, y := 2}, v2 := {x := 2, y := 1}, v3 := {x := 0, y := 0} }
  let t2 : Triangle := { v1 := {x := 2, y := 2}, v2 := {x := 0, y := 1}, v3 := {x := 2, y := 0} }
  triangleOverlapArea t1 t2 = 1 := by
  sorry

end overlap_area_is_one_l3354_335420


namespace combination_problem_l3354_335432

theorem combination_problem (m : ℕ) : 
  (1 : ℚ) / Nat.choose 5 m - (1 : ℚ) / Nat.choose 6 m = (7 : ℚ) / (10 * Nat.choose 7 m) → 
  Nat.choose 8 m = 28 := by
  sorry

end combination_problem_l3354_335432


namespace max_integer_k_l3354_335408

theorem max_integer_k (x y k : ℝ) : 
  x - 4*y = k - 1 →
  2*x + y = k →
  x - y ≤ 0 →
  ∀ m : ℤ, m ≤ k → m ≤ 0 :=
by sorry

end max_integer_k_l3354_335408


namespace min_teams_for_employees_l3354_335441

theorem min_teams_for_employees (total_employees : ℕ) (max_team_size : ℕ) (h1 : total_employees = 36) (h2 : max_team_size = 12) : 
  (total_employees + max_team_size - 1) / max_team_size = 3 :=
by sorry

end min_teams_for_employees_l3354_335441


namespace arccos_sin_five_equals_five_minus_pi_over_two_l3354_335468

theorem arccos_sin_five_equals_five_minus_pi_over_two :
  Real.arccos (Real.sin 5) = (5 - Real.pi) / 2 := by
  sorry

end arccos_sin_five_equals_five_minus_pi_over_two_l3354_335468


namespace pattern_cost_is_15_l3354_335493

/-- The cost of a sewing pattern given the total spent, fabric cost per yard, yards of fabric bought,
    thread cost per spool, and number of thread spools bought. -/
def pattern_cost (total_spent fabric_cost_per_yard yards_fabric thread_cost_per_spool num_thread_spools : ℕ) : ℕ :=
  total_spent - (fabric_cost_per_yard * yards_fabric + thread_cost_per_spool * num_thread_spools)

/-- Theorem stating that the pattern cost is $15 given the specific conditions. -/
theorem pattern_cost_is_15 :
  pattern_cost 141 24 5 3 2 = 15 := by
  sorry

end pattern_cost_is_15_l3354_335493


namespace weight_of_replaced_person_l3354_335495

theorem weight_of_replaced_person
  (n : ℕ) 
  (avg_increase : ℝ)
  (new_person_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  new_person_weight = 95 →
  new_person_weight - n * avg_increase = 75 :=
by sorry

end weight_of_replaced_person_l3354_335495


namespace Φ_is_connected_l3354_335457

-- Define the set Φ
def Φ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt (y^2 - 8*x^2 - 6*y + 9) ≤ 3*y - 1 ∧
               x^2 + y^2 ≤ 9}

-- Theorem statement
theorem Φ_is_connected : IsConnected Φ := by
  sorry

end Φ_is_connected_l3354_335457


namespace cows_count_l3354_335453

/-- Represents the number of animals in the farm -/
structure FarmAnimals where
  ducks : ℕ
  cows : ℕ
  spiders : ℕ

/-- Checks if the given farm animals satisfy all the conditions -/
def satisfiesConditions (animals : FarmAnimals) : Prop :=
  let totalLegs := 2 * animals.ducks + 4 * animals.cows + 8 * animals.spiders
  let totalHeads := animals.ducks + animals.cows + animals.spiders
  totalLegs = 2 * totalHeads + 72 ∧
  animals.spiders = 2 * animals.ducks ∧
  totalHeads ≤ 40

/-- Theorem stating that the number of cows is 30 given the conditions -/
theorem cows_count (animals : FarmAnimals) :
  satisfiesConditions animals → animals.cows = 30 := by
  sorry

end cows_count_l3354_335453


namespace probability_twelve_no_consecutive_ones_l3354_335478

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- Probability of a valid sequence of length n -/
def probability (n : ℕ) : ℚ :=
  (validSequences n : ℚ) / (totalSequences n : ℚ)

theorem probability_twelve_no_consecutive_ones :
  probability 12 = 377 / 4096 :=
sorry

end probability_twelve_no_consecutive_ones_l3354_335478


namespace no_roots_in_interval_l3354_335485

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 10*x^2

-- State the theorem
theorem no_roots_in_interval :
  ∀ x ∈ Set.Icc 1 2, f x ≠ 0 := by sorry

end no_roots_in_interval_l3354_335485


namespace cube_root_sum_of_threes_l3354_335484

theorem cube_root_sum_of_threes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := by
  sorry

end cube_root_sum_of_threes_l3354_335484


namespace min_perimeter_triangle_ABC_l3354_335487

/-- Triangle ABC with integer side lengths, BD angle bisector of ∠ABC, AD = 4, DC = 6, D on AC -/
structure TriangleABC where
  AB : ℕ
  BC : ℕ
  AC : ℕ
  AD : ℕ
  DC : ℕ
  hAD : AD = 4
  hDC : DC = 6
  hAC : AC = AD + DC
  hAngleBisector : AB * DC = BC * AD

/-- The minimum possible perimeter of triangle ABC is 25 -/
theorem min_perimeter_triangle_ABC (t : TriangleABC) : 
  (∀ t' : TriangleABC, t'.AB + t'.BC + t'.AC ≥ t.AB + t.BC + t.AC) → 
  t.AB + t.BC + t.AC = 25 := by
  sorry

end min_perimeter_triangle_ABC_l3354_335487


namespace derivative_sin_plus_exp_cos_l3354_335425

theorem derivative_sin_plus_exp_cos (x : ℝ) :
  let y : ℝ → ℝ := λ x => Real.sin x + Real.exp x * Real.cos x
  deriv y x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x :=
by
  sorry

end derivative_sin_plus_exp_cos_l3354_335425


namespace complex_equation_real_solution_l3354_335400

theorem complex_equation_real_solution (a : ℝ) : 
  (((2 * a) / (1 + Complex.I) + 1 + Complex.I).im = 0) → a = 1 := by
  sorry

end complex_equation_real_solution_l3354_335400


namespace f_composition_value_l3354_335444

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 0
  else if x = 0 then Real.pi
  else Real.pi^2 + 1

theorem f_composition_value : f (f (f 1)) = 0 := by
  sorry

end f_composition_value_l3354_335444


namespace smallest_angle_in_3_4_5_ratio_triangle_l3354_335439

theorem smallest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    (b = (4/3) * a) → (c = (5/3) * a) →
    (a + b + c = 180) →
    a = 45 := by
sorry

end smallest_angle_in_3_4_5_ratio_triangle_l3354_335439


namespace no_valid_distribution_of_skittles_l3354_335431

theorem no_valid_distribution_of_skittles : ¬ ∃ (F : ℕ+), 
  (14 - 3 * F.val ≥ 3) ∧ (14 - 3 * F.val) % 3 = 0 := by
  sorry

end no_valid_distribution_of_skittles_l3354_335431


namespace range_of_sum_l3354_335472

theorem range_of_sum (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) :
  2 < x + y ∧ x + y < 6 := by
  sorry

end range_of_sum_l3354_335472


namespace anna_toy_production_l3354_335409

/-- Anna's toy production problem -/
theorem anna_toy_production (t : ℕ) : 
  let w : ℕ := 3 * t
  let monday_production : ℕ := w * t
  let tuesday_production : ℕ := (w + 5) * (t - 3)
  monday_production - tuesday_production = 4 * t + 15 := by
sorry

end anna_toy_production_l3354_335409


namespace books_sold_l3354_335499

theorem books_sold (initial_books : ℕ) (remaining_books : ℕ) : initial_books = 136 → remaining_books = 27 → initial_books - remaining_books = 109 := by
  sorry

end books_sold_l3354_335499


namespace imaginary_part_of_z_l3354_335435

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l3354_335435


namespace midpoint_coordinate_relation_l3354_335454

/-- Given two points D and E in the plane, if F is their midpoint,
    then 3 times the x-coordinate of F minus 5 times the y-coordinate of F equals 9. -/
theorem midpoint_coordinate_relation :
  let D : ℝ × ℝ := (30, 10)
  let E : ℝ × ℝ := (6, 8)
  let F : ℝ × ℝ := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  3 * F.1 - 5 * F.2 = 9 := by
  sorry

end midpoint_coordinate_relation_l3354_335454


namespace min_value_expression_l3354_335460

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sqrt : y^2 = x) :
  (x^2 + y^4) / (x * y^2) = 2 := by
  sorry

end min_value_expression_l3354_335460


namespace expand_expression_l3354_335416

theorem expand_expression (x : ℝ) : 20 * (3 * x + 4) - 10 = 60 * x + 70 := by
  sorry

end expand_expression_l3354_335416


namespace prime_dates_in_leap_year_l3354_335445

def isPrimeMonth (m : Nat) : Bool :=
  m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 7 ∨ m = 11

def isPrimeDay (d : Nat) : Bool :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 17 ∨ d = 19 ∨ d = 23 ∨ d = 29 ∨ d = 31

def daysInMonth (m : Nat) : Nat :=
  if m = 2 then 29
  else if m = 4 ∨ m = 11 then 30
  else 31

def countPrimeDates : Nat :=
  (List.range 12).filter isPrimeMonth
    |>.map (fun m => (List.range (daysInMonth m)).filter isPrimeDay |>.length)
    |>.sum

theorem prime_dates_in_leap_year :
  countPrimeDates = 63 := by
  sorry

end prime_dates_in_leap_year_l3354_335445


namespace simplify_expression_l3354_335403

theorem simplify_expression (b : ℝ) : (2 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 720 * b^10 := by
  sorry

end simplify_expression_l3354_335403


namespace guitar_center_shipping_fee_l3354_335421

/-- The shipping fee of Guitar Center given the conditions of the guitar purchase --/
theorem guitar_center_shipping_fee :
  let suggested_price : ℚ := 1000
  let guitar_center_discount : ℚ := 15 / 100
  let sweetwater_discount : ℚ := 10 / 100
  let savings : ℚ := 50
  let guitar_center_price := suggested_price * (1 - guitar_center_discount)
  let sweetwater_price := suggested_price * (1 - sweetwater_discount)
  guitar_center_price + (sweetwater_price - guitar_center_price - savings) = guitar_center_price :=
by sorry

end guitar_center_shipping_fee_l3354_335421


namespace percentage_of_red_non_honda_cars_l3354_335475

theorem percentage_of_red_non_honda_cars 
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (honda_red_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 900) 
  (h2 : honda_cars = 500) 
  (h3 : honda_red_ratio = 90 / 100) 
  (h4 : total_red_ratio = 60 / 100) :
  (((total_red_ratio * total_cars) - (honda_red_ratio * honda_cars)) / (total_cars - honda_cars)) = 225 / 1000 := by
  sorry

end percentage_of_red_non_honda_cars_l3354_335475


namespace triangle_area_l3354_335458

theorem triangle_area (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → 
  b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 →
  abs (1/2 * a * b * cos_theta) = 4.5 :=
sorry

end triangle_area_l3354_335458


namespace exists_expr_2023_l3354_335411

/-- An arithmetic expression without parentheses -/
inductive ArithExpr
  | Const : ℤ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr

/-- Evaluation function for ArithExpr -/
def eval : ArithExpr → ℤ
  | ArithExpr.Const n => n
  | ArithExpr.Add a b => eval a + eval b
  | ArithExpr.Sub a b => eval a - eval b
  | ArithExpr.Mul a b => eval a * eval b
  | ArithExpr.Div a b => eval a / eval b

/-- Theorem stating the existence of an arithmetic expression evaluating to 2023 -/
theorem exists_expr_2023 : ∃ e : ArithExpr, eval e = 2023 := by
  sorry


end exists_expr_2023_l3354_335411


namespace football_players_l3354_335470

theorem football_players (total : ℕ) (cricket : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 420)
  (h2 : cricket = 175)
  (h3 : both = 130)
  (h4 : neither = 50) :
  total - neither - (cricket - both) = 325 :=
sorry

end football_players_l3354_335470


namespace walking_speed_calculation_l3354_335488

/-- Proves that the walking speed is 4 km/hr given the conditions of the problem -/
theorem walking_speed_calculation (run_speed : ℝ) (total_distance : ℝ) (total_time : ℝ)
  (h1 : run_speed = 8)
  (h2 : total_distance = 20)
  (h3 : total_time = 3.75) :
  ∃ (walk_speed : ℝ),
    walk_speed = 4 ∧
    (total_distance / 2) / walk_speed + (total_distance / 2) / run_speed = total_time :=
by sorry

end walking_speed_calculation_l3354_335488


namespace units_digit_problem_l3354_335494

theorem units_digit_problem : ∃ n : ℕ, (3 * 19 * 1933 - 3^4) % 10 = 0 :=
by sorry

end units_digit_problem_l3354_335494


namespace warehouse_space_theorem_l3354_335447

/-- Represents the warehouse with two floors and some occupied space -/
structure Warehouse :=
  (second_floor : ℝ)
  (first_floor : ℝ)
  (occupied_space : ℝ)

/-- The remaining available space in the warehouse -/
def remaining_space (w : Warehouse) : ℝ :=
  w.first_floor + w.second_floor - w.occupied_space

/-- The theorem stating the remaining available space in the warehouse -/
theorem warehouse_space_theorem (w : Warehouse) 
  (h1 : w.first_floor = 2 * w.second_floor)
  (h2 : w.occupied_space = w.second_floor / 4)
  (h3 : w.occupied_space = 5000) : 
  remaining_space w = 55000 := by
  sorry

#check warehouse_space_theorem

end warehouse_space_theorem_l3354_335447


namespace max_value_abc_l3354_335455

theorem max_value_abc (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/4) ≤ 3 ∧
  ∃ a' b' c' : ℝ, a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 3 ∧
    a' + Real.sqrt (a' * b') + (a' * b' * c') ^ (1/4) = 3 :=
sorry

end max_value_abc_l3354_335455


namespace final_state_is_green_l3354_335417

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Represents a color change event between two chameleons -/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Red => Color.Green
  | Color.Yellow, Color.Green => Color.Red
  | Color.Red, Color.Yellow => Color.Green
  | Color.Red, Color.Green => Color.Yellow
  | Color.Green, Color.Yellow => Color.Red
  | Color.Green, Color.Red => Color.Yellow
  | _, _ => c1  -- No change if colors are the same

/-- Theorem: The only possible final state is all chameleons being green -/
theorem final_state_is_green (finalState : ChameleonState) :
  (finalState.yellow + finalState.red + finalState.green = totalChameleons) →
  (∀ (c1 c2 : Color), colorChange c1 c2 = colorChange c2 c1) →
  (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
by sorry

#check final_state_is_green

end final_state_is_green_l3354_335417


namespace sum_of_fraction_parts_is_correct_l3354_335452

/-- The repeating decimal 3.71717171... -/
def repeating_decimal : ℚ := 3 + 71/99

/-- The sum of the numerator and denominator of the fraction representing
    the repeating decimal 3.71717171... in its lowest terms -/
def sum_of_fraction_parts : ℕ := 467

/-- Theorem stating that the sum of the numerator and denominator of the fraction
    representing 3.71717171... in its lowest terms is 467 -/
theorem sum_of_fraction_parts_is_correct :
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n + d = sum_of_fraction_parts := by
  sorry

end sum_of_fraction_parts_is_correct_l3354_335452


namespace ninety_sixth_digit_of_5_div_37_l3354_335489

/-- The decimal representation of 5/37 has a repeating pattern of length 3 -/
def decimal_repeat_length : ℕ := 3

/-- The repeating pattern in the decimal representation of 5/37 -/
def decimal_pattern : Fin 3 → ℕ
| 0 => 1
| 1 => 3
| 2 => 5

/-- The 96th digit after the decimal point in the decimal representation of 5/37 is 5 -/
theorem ninety_sixth_digit_of_5_div_37 : 
  decimal_pattern ((96 : ℕ) % decimal_repeat_length) = 5 := by
  sorry

end ninety_sixth_digit_of_5_div_37_l3354_335489


namespace sum_of_reciprocals_l3354_335474

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 4 * x * y) : 1 / x + 1 / y = 4 := by
  sorry

end sum_of_reciprocals_l3354_335474


namespace combination_problem_l3354_335477

/-- Given that 1/C_5^m - 1/C_6^m = 7/(10C_7^m), prove that C_{21}^m = 210 -/
theorem combination_problem (m : ℕ) : 
  (1 / (Nat.choose 5 m : ℚ) - 1 / (Nat.choose 6 m : ℚ) = 7 / (10 * (Nat.choose 7 m : ℚ))) → 
  Nat.choose 21 m = 210 := by
sorry

end combination_problem_l3354_335477


namespace layla_earnings_correct_l3354_335437

/-- Calculates the babysitting earnings for a given family -/
def family_earnings (base_rate : ℚ) (hours : ℚ) (bonus_threshold : ℚ) (bonus_amount : ℚ) 
  (discount_rate : ℚ) (flat_rate : ℚ) (is_weekend : Bool) (past_midnight : Bool) : ℚ :=
  sorry

/-- Calculates Layla's total babysitting earnings -/
def layla_total_earnings : ℚ :=
  let donaldsons := family_earnings 15 7 5 5 0 0 false false
  let merck := family_earnings 18 6 3 0 0.1 0 false false
  let hille := family_earnings 20 3 0 10 0 0 false true
  let johnson := family_earnings 22 4 4 0 0 80 false false
  let ramos := family_earnings 25 2 0 20 0 0 true false
  donaldsons + merck + hille + johnson + ramos

theorem layla_earnings_correct : layla_total_earnings = 435.2 := by
  sorry

end layla_earnings_correct_l3354_335437


namespace unique_integer_with_conditions_l3354_335443

theorem unique_integer_with_conditions : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 100 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 84 := by
  sorry

end unique_integer_with_conditions_l3354_335443


namespace work_completion_proof_l3354_335490

/-- A's work rate in days -/
def a_rate : ℚ := 1 / 15

/-- B's work rate in days -/
def b_rate : ℚ := 1 / 20

/-- The fraction of work left after A and B work together -/
def work_left : ℚ := 65 / 100

/-- The number of days A and B worked together -/
def days_worked : ℚ := 3

theorem work_completion_proof :
  (a_rate + b_rate) * days_worked = 1 - work_left :=
sorry

end work_completion_proof_l3354_335490


namespace melanie_brownies_batches_l3354_335436

/-- Represents the number of brownies in each batch -/
def brownies_per_batch : ℕ := 20

/-- Represents the fraction of brownies set aside for the bake sale -/
def bake_sale_fraction : ℚ := 3/4

/-- Represents the fraction of remaining brownies put in a container -/
def container_fraction : ℚ := 3/5

/-- Represents the number of brownies given out -/
def brownies_given_out : ℕ := 20

/-- Proves that Melanie baked 10 batches of brownies -/
theorem melanie_brownies_batches :
  ∃ (batches : ℕ),
    batches = 10 ∧
    (brownies_per_batch * batches : ℚ) * (1 - bake_sale_fraction) * (1 - container_fraction) =
      brownies_given_out :=
by sorry

end melanie_brownies_batches_l3354_335436


namespace gift_packaging_combinations_l3354_335479

/-- The number of varieties of wrapping paper. -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of colors of ribbon. -/
def ribbon_colors : ℕ := 5

/-- The number of types of gift tags. -/
def gift_tag_types : ℕ := 6

/-- The total number of possible gift packaging combinations. -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_tag_types

/-- Theorem stating that the total number of gift packaging combinations is 300. -/
theorem gift_packaging_combinations :
  total_combinations = 300 :=
by sorry

end gift_packaging_combinations_l3354_335479


namespace inequality_proof_l3354_335413

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b :=
by sorry

end inequality_proof_l3354_335413


namespace arithmetic_calculation_l3354_335492

theorem arithmetic_calculation : 4 * (8 - 3) + 6 / 2 = 23 := by
  sorry

end arithmetic_calculation_l3354_335492


namespace sin_alpha_value_l3354_335433

-- Define the angle α
def α : Real := sorry

-- Define the point P on the terminal side of α
def P : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem sin_alpha_value :
  (α.sin = -2 * Real.sqrt 5 / 5) ∧
  (α.cos ≥ 0) ∧
  (α.sin * 2 + α.cos * (-2) = 0) := by
  sorry

end sin_alpha_value_l3354_335433


namespace cone_height_equals_six_l3354_335429

/-- Proves that given a cylinder M with base radius 2 and height 6, and a cone N whose base diameter equals its slant height, if their volumes are equal, then the height of cone N is 6. -/
theorem cone_height_equals_six (r : ℝ) (h : ℝ) :
  (2 : ℝ) ^ 2 * 6 = (1 / 3) * r ^ 2 * h ∧ 
  h = r * Real.sqrt 3 →
  h = 6 := by sorry

end cone_height_equals_six_l3354_335429


namespace vector_combination_equality_l3354_335446

/-- Given vectors a, b, and c in ℝ³, prove that 2a - 3b + 4c equals (16, 0, -19) -/
theorem vector_combination_equality (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (3, 5, 1)) 
  (hb : b = (2, 2, 3)) 
  (hc : c = (4, -1, -3)) : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -19) := by
  sorry

end vector_combination_equality_l3354_335446


namespace polynomial_arrangement_l3354_335471

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := 3*x^2 - x + x^3 - 1

-- Define the arranged polynomial
def arranged_polynomial (x : ℝ) : ℝ := -1 - x + 3*x^2 + x^3

-- Theorem stating that the original polynomial is equal to the arranged polynomial
theorem polynomial_arrangement :
  ∀ x : ℝ, original_polynomial x = arranged_polynomial x :=
by
  sorry

end polynomial_arrangement_l3354_335471


namespace rectangular_garden_length_l3354_335405

/-- Theorem: For a rectangular garden with a perimeter of 600 m and a breadth of 95 m, the length is 205 m. -/
theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 600) 
  (h2 : breadth = 95) :
  2 * (breadth + 205) = perimeter := by
  sorry

end rectangular_garden_length_l3354_335405


namespace total_dress_designs_l3354_335428

/-- Represents the number of fabric color choices --/
def num_colors : Nat := 5

/-- Represents the number of pattern choices --/
def num_patterns : Nat := 4

/-- Represents the number of sleeve length choices --/
def num_sleeve_lengths : Nat := 2

/-- Theorem stating the total number of possible dress designs --/
theorem total_dress_designs : num_colors * num_patterns * num_sleeve_lengths = 40 := by
  sorry

end total_dress_designs_l3354_335428


namespace line_length_difference_l3354_335461

-- Define the lengths of the lines in inches
def white_line_inches : ℝ := 7.666666666666667
def blue_line_inches : ℝ := 3.3333333333333335

-- Define conversion rates
def inches_to_cm : ℝ := 2.54
def cm_to_mm : ℝ := 10

-- Theorem statement
theorem line_length_difference : 
  (white_line_inches * inches_to_cm - blue_line_inches * inches_to_cm) * cm_to_mm = 110.05555555555553 := by
  sorry

end line_length_difference_l3354_335461


namespace parabola_transformation_l3354_335438

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = -2x^2 + 1 -/
def original_parabola : Parabola := { a := -2, b := 0, c := 1 }

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h, c := p.a * h^2 + p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The resulting parabola after transformations -/
def transformed_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) (-1)

theorem parabola_transformation :
  transformed_parabola = { a := -2, b := 12, c := -18 } :=
sorry

end parabola_transformation_l3354_335438


namespace arithmetic_geometric_progression_l3354_335451

theorem arithmetic_geometric_progression (k : ℝ) :
  ∃ (x y z : ℝ),
    x + y + z = k ∧
    y - x = z - y ∧
    y^2 = x * (z + k/6) ∧
    ((x = k/6 ∧ y = k/3 ∧ z = k/2) ∨ (x = 2*k/3 ∧ y = k/3 ∧ z = 0)) :=
by sorry

end arithmetic_geometric_progression_l3354_335451


namespace a_b_parallel_opposite_l3354_335418

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -4)

/-- Predicate to check if two vectors are parallel and in opposite directions -/
def parallel_opposite (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

/-- Theorem stating that vectors a and b are parallel and in opposite directions -/
theorem a_b_parallel_opposite : parallel_opposite a b := by
  sorry

end a_b_parallel_opposite_l3354_335418


namespace sum_of_fifth_and_eighth_l3354_335423

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fifth_and_eighth (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) - 5 = 0 →
  (a 10)^2 - 3*(a 10) - 5 = 0 →
  a 5 + a 8 = 3 :=
by sorry

end sum_of_fifth_and_eighth_l3354_335423


namespace part_one_part_two_l3354_335465

/-- Definition of arithmetic sequence sum -/
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem for part (I) -/
theorem part_one :
  ∃! k : ℕ+, arithmetic_sum (3/2) 1 (k^2) = (arithmetic_sum (3/2) 1 k)^2 :=
sorry

/-- Definition of arithmetic sequence -/
def arithmetic_seq (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

/-- Theorem for part (II) -/
theorem part_two :
  ∀ a₁ d : ℚ, (∀ k : ℕ+, arithmetic_sum a₁ d (k^2) = (arithmetic_sum a₁ d k)^2) ↔
    ((∀ n, arithmetic_seq a₁ d n = 0) ∨
     (∀ n, arithmetic_seq a₁ d n = 1) ∨
     (∀ n, arithmetic_seq a₁ d n = 2 * n - 1)) :=
sorry

end part_one_part_two_l3354_335465


namespace tens_digit_of_19_power_2021_l3354_335430

theorem tens_digit_of_19_power_2021 : ∃ n : ℕ, 19^2021 ≡ 10*n + 1 [ZMOD 100] := by
  sorry

end tens_digit_of_19_power_2021_l3354_335430


namespace xy_value_given_condition_l3354_335427

theorem xy_value_given_condition (x y : ℝ) : 
  |x - 2| + Real.sqrt (y + 3) = 0 → x * y = -6 := by
  sorry

end xy_value_given_condition_l3354_335427


namespace fixed_point_theorem_l3354_335459

-- Define the curve E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through two points
def Line (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define a line with a given slope passing through a point
def LineWithSlope (x0 y0 m : ℝ) (x y : ℝ) : Prop :=
  y - y0 = m * (x - x0)

theorem fixed_point_theorem (xA yA xB yB xC yC : ℝ) :
  E xA yA →
  E xB yB →
  E xC yC →
  Line (-3) 2 xA yA xB yB →
  LineWithSlope xA yA 1 xC yC →
  Line xB yB xC yC 5 2 := by
  sorry

end fixed_point_theorem_l3354_335459


namespace simplify_expression_l3354_335401

theorem simplify_expression (a b c : ℝ) (h : b^2 = c^2) :
  -|b| - |a-b| + |a-c| - |b+c| = - |a-b| + |a-c| - |b+c| := by
  sorry

end simplify_expression_l3354_335401


namespace young_photographer_club_l3354_335424

theorem young_photographer_club (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos :=
by sorry

end young_photographer_club_l3354_335424


namespace smallest_with_specific_divisor_counts_l3354_335467

/-- Count of positive odd integer divisors of n -/
def oddDivisorsCount (n : ℕ+) : ℕ := sorry

/-- Count of positive even integer divisors of n -/
def evenDivisorsCount (n : ℕ+) : ℕ := sorry

/-- Predicate to check if a number satisfies the divisor count conditions -/
def satisfiesDivisorCounts (n : ℕ+) : Prop :=
  oddDivisorsCount n = 7 ∧ evenDivisorsCount n = 14

theorem smallest_with_specific_divisor_counts :
  satisfiesDivisorCounts 1458 ∧
  ∀ m : ℕ+, m < 1458 → ¬satisfiesDivisorCounts m := by
  sorry

end smallest_with_specific_divisor_counts_l3354_335467


namespace angle_B_is_60_l3354_335410

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : Real
  B : Real
  C : Real
  scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A
  sum_180 : A + B + C = 180

-- Define the specific triangle with given angle relationships
def SpecificTriangle (t : ScaleneTriangle) : Prop :=
  t.C = 3 * t.A ∧ t.B = 2 * t.A

-- Theorem statement
theorem angle_B_is_60 (t : ScaleneTriangle) (h : SpecificTriangle t) : t.B = 60 := by
  sorry

end angle_B_is_60_l3354_335410


namespace twenty_percent_equals_fiftyfour_l3354_335466

theorem twenty_percent_equals_fiftyfour (x : ℝ) : (20 / 100) * x = 54 → x = 270 := by
  sorry

end twenty_percent_equals_fiftyfour_l3354_335466
