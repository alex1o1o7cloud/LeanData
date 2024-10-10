import Mathlib

namespace triangle_inequality_l3241_324127

theorem triangle_inequality (a b c S r R : ℝ) (ha hb hc : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < S ∧ 0 < r ∧ 0 < R →
  9 * r ≤ ha + hb + hc →
  ha + hb + hc ≤ 9 * R / 2 →
  1 / a + 1 / b + 1 / c = (ha + hb + hc) / (2 * S) →
  9 * r / (2 * S) ≤ 1 / a + 1 / b + 1 / c ∧ 1 / a + 1 / b + 1 / c ≤ 9 * R / (4 * S) :=
by sorry

end triangle_inequality_l3241_324127


namespace christinas_earnings_l3241_324178

/-- The amount Christina earns for planting flowers and mowing the lawn -/
theorem christinas_earnings (flower_rate : ℚ) (mow_rate : ℚ) (flowers_planted : ℚ) (area_mowed : ℚ) :
  flower_rate = 8/3 →
  mow_rate = 5/2 →
  flowers_planted = 9/4 →
  area_mowed = 7/3 →
  flower_rate * flowers_planted + mow_rate * area_mowed = 71/6 := by
sorry

end christinas_earnings_l3241_324178


namespace lottery_probability_l3241_324168

theorem lottery_probability : 
  let powerball_count : ℕ := 30
  let luckyball_count : ℕ := 49
  let luckyball_picks : ℕ := 6
  let powerball_prob : ℚ := 1 / powerball_count
  let luckyball_prob : ℚ := 1 / (Nat.choose luckyball_count luckyball_picks)
  powerball_prob * luckyball_prob = 1 / 419512480 :=
by sorry

end lottery_probability_l3241_324168


namespace horner_rule_v4_l3241_324173

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_rule_v4 :
  horner_v4 2 = 80 :=
by sorry

end horner_rule_v4_l3241_324173


namespace student_group_assignments_non_empty_coin_subsets_l3241_324150

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of coins --/
def num_coins : ℕ := 7

/-- Theorem for the number of ways to assign students to groups --/
theorem student_group_assignments :
  (num_groups : ℕ) ^ num_students = 32 := by sorry

/-- Theorem for the number of non-empty subsets of coins --/
theorem non_empty_coin_subsets :
  2 ^ num_coins - 1 = 127 := by sorry

end student_group_assignments_non_empty_coin_subsets_l3241_324150


namespace intersection_range_l3241_324195

def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

theorem intersection_range (k : ℝ) (m : ℝ) :
  (∀ x y, line k x = y ∧ ellipse m x y → 
    ∃ x' y', x ≠ x' ∧ line k x' = y' ∧ ellipse m x' y') →
  m ∈ Set.Ioo 1 5 ∪ Set.Ioi 5 :=
sorry

end intersection_range_l3241_324195


namespace cars_meeting_time_l3241_324175

theorem cars_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance = 60) 
  (h2 : speed1 = 13) (h3 : speed2 = 17) : 
  distance / (speed1 + speed2) = 2 := by
  sorry

end cars_meeting_time_l3241_324175


namespace gcf_72_108_l3241_324138

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by sorry

end gcf_72_108_l3241_324138


namespace debt_installments_l3241_324103

theorem debt_installments (x : ℝ) : 
  (8 * x + 44 * (x + 65)) / 52 = 465 → x = 410 := by
  sorry

end debt_installments_l3241_324103


namespace value_of_a_l3241_324152

theorem value_of_a (a : ℝ) : -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end value_of_a_l3241_324152


namespace completing_square_l3241_324180

theorem completing_square (x : ℝ) : x^2 + 2*x - 1 = 0 ↔ (x + 1)^2 = 2 := by
  sorry

end completing_square_l3241_324180


namespace imaginary_part_of_z_l3241_324108

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) : 
  Complex.im z = -1/5 := by
  sorry

end imaginary_part_of_z_l3241_324108


namespace bridgette_guest_count_l3241_324149

/-- The number of guests Bridgette is inviting -/
def bridgette_guests : ℕ := 84

/-- The number of guests Alex is inviting -/
def alex_guests : ℕ := (2 * bridgette_guests) / 3

/-- The number of extra plates the caterer makes -/
def extra_plates : ℕ := 10

/-- The number of asparagus spears per plate -/
def spears_per_plate : ℕ := 8

/-- The total number of asparagus spears needed -/
def total_spears : ℕ := 1200

theorem bridgette_guest_count : 
  spears_per_plate * (bridgette_guests + alex_guests + extra_plates) = total_spears :=
by sorry

end bridgette_guest_count_l3241_324149


namespace parabola_vertex_l3241_324161

/-- Given a quadratic function f(x) = -x^2 + cx + d whose inequality f(x) ≤ 0
    has the solution set (-∞, -4] ∪ [6, ∞), prove that its vertex is (5, 1) -/
theorem parabola_vertex (c d : ℝ) : 
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) → 
  ∃ x y, x = 5 ∧ y = 1 ∧ ∀ t, -t^2 + c*t + d ≤ -(-t + x)^2 + y :=
sorry

end parabola_vertex_l3241_324161


namespace nine_sided_polygon_diagonals_l3241_324111

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n.choose 2) - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l3241_324111


namespace decreasing_f_implies_a_greater_than_five_l3241_324109

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 5) / Real.log (Real.sin 1)

theorem decreasing_f_implies_a_greater_than_five (a : ℝ) :
  (∀ x y, a < x ∧ x < y → f y < f x) →
  a > 5 :=
by sorry

end decreasing_f_implies_a_greater_than_five_l3241_324109


namespace average_daily_temp_range_l3241_324121

def high_temps : List ℝ := [49, 62, 58, 57, 46, 60, 55]
def low_temps : List ℝ := [40, 47, 45, 41, 39, 42, 44]

def daily_range (high low : List ℝ) : List ℝ :=
  List.zipWith (·-·) high low

theorem average_daily_temp_range :
  let ranges := daily_range high_temps low_temps
  (ranges.sum / ranges.length : ℝ) = 89 / 7 := by
  sorry

end average_daily_temp_range_l3241_324121


namespace unused_edge_exists_l3241_324123

/-- Represents a token on a vertex of the 2n-gon -/
structure Token (n : ℕ) where
  position : Fin (2 * n)

/-- Represents a move (swapping tokens on an edge) -/
structure Move (n : ℕ) where
  edge : Fin (2 * n) × Fin (2 * n)

/-- Represents the state of the 2n-gon after some moves -/
structure GameState (n : ℕ) where
  tokens : Fin (2 * n) → Token n
  moves : List (Move n)

/-- Predicate to check if two tokens have been swapped -/
def haveBeenSwapped (n : ℕ) (t1 t2 : Token n) (moves : List (Move n)) : Prop :=
  sorry

/-- Predicate to check if an edge has been used for swapping -/
def edgeUsed (n : ℕ) (edge : Fin (2 * n) × Fin (2 * n)) (moves : List (Move n)) : Prop :=
  sorry

/-- The main theorem -/
theorem unused_edge_exists (n : ℕ) (finalState : GameState n) :
  (∀ t1 t2 : Token n, t1 ≠ t2 → haveBeenSwapped n t1 t2 finalState.moves) →
  ∃ edge : Fin (2 * n) × Fin (2 * n), ¬edgeUsed n edge finalState.moves :=
sorry

end unused_edge_exists_l3241_324123


namespace taehyungs_mother_age_l3241_324134

/-- Given the age differences and the younger brother's age, prove Taehyung's mother's age --/
theorem taehyungs_mother_age :
  ∀ (taehyung_age mother_age brother_age : ℕ),
    mother_age - taehyung_age = 31 →
    taehyung_age - brother_age = 5 →
    brother_age = 7 →
    mother_age = 43 := by
  sorry

end taehyungs_mother_age_l3241_324134


namespace inscribed_circle_radius_rhombus_l3241_324105

/-- The radius of the inscribed circle in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let area := d1 * d2 / 2
  area / (4 * a) = 30 / Real.sqrt 241 :=
by sorry

end inscribed_circle_radius_rhombus_l3241_324105


namespace fourth_term_of_geometric_sequence_l3241_324179

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 := by
  sorry

end fourth_term_of_geometric_sequence_l3241_324179


namespace book_page_numbering_l3241_324128

def total_digits (n : ℕ) : ℕ :=
  let d1 := min n 9
  let d2 := min (n - 9) 90
  let d3 := min (n - 99) 900
  let d4 := max (n - 999) 0
  d1 + 2 * d2 + 3 * d3 + 4 * d4

theorem book_page_numbering :
  total_digits 5000 = 18893 := by
sorry

end book_page_numbering_l3241_324128


namespace quadratic_root_existence_l3241_324174

theorem quadratic_root_existence (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (h1 : a * x₁^2 + b * x₁ + c = 0)
  (h2 : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end quadratic_root_existence_l3241_324174


namespace jeans_and_shirts_cost_l3241_324191

/-- The cost of one pair of jeans -/
def jean_cost : ℝ := 11

/-- The cost of one shirt -/
def shirt_cost : ℝ := 18

/-- The cost of 2 pairs of jeans and 3 shirts -/
def cost_2j_3s : ℝ := 76

/-- The cost of 3 pairs of jeans and 2 shirts -/
def cost_3j_2s : ℝ := 3 * jean_cost + 2 * shirt_cost

theorem jeans_and_shirts_cost : cost_3j_2s = 69 := by
  sorry

end jeans_and_shirts_cost_l3241_324191


namespace unique_c_for_quadratic_equation_l3241_324196

theorem unique_c_for_quadratic_equation :
  ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b + 1/b) * x + c = 0)) ∧
  c = 1 := by
sorry

end unique_c_for_quadratic_equation_l3241_324196


namespace certain_number_problem_l3241_324145

theorem certain_number_problem (x : ℕ) (n : ℕ) : x = 4 → 3 * x + n = 48 → n = 36 := by
  sorry

end certain_number_problem_l3241_324145


namespace georgesBirthdayMoneyIs12_l3241_324120

/-- Calculates the amount George will receive on his 25th birthday --/
def georgesBirthdayMoney (currentAge : ℕ) (startAge : ℕ) (spendPercentage : ℚ) (exchangeRate : ℚ) : ℚ :=
  let totalBills : ℕ := currentAge - startAge
  let remainingBills : ℚ := (1 - spendPercentage) * totalBills
  exchangeRate * remainingBills

/-- Theorem stating the amount George will receive --/
theorem georgesBirthdayMoneyIs12 : 
  georgesBirthdayMoney 25 15 (1/5) (3/2) = 12 := by
  sorry


end georgesBirthdayMoneyIs12_l3241_324120


namespace scientific_notation_of_80_million_l3241_324190

theorem scientific_notation_of_80_million :
  ∃ (n : ℕ), 80000000 = 8 * (10 ^ n) ∧ n = 7 := by
  sorry

end scientific_notation_of_80_million_l3241_324190


namespace integer_terms_count_l3241_324153

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a₃ : ℝ  -- 3rd term
  a₁₈ : ℝ -- 18th term
  h₃ : a₃ = 14
  h₁₈ : a₁₈ = 23

/-- The number of integer terms in the first 2010 terms of the sequence -/
def integerTermCount (seq : ArithmeticSequence) : ℕ :=
  402

/-- Theorem stating the number of integer terms in the first 2010 terms -/
theorem integer_terms_count (seq : ArithmeticSequence) :
  integerTermCount seq = 402 := by
  sorry

end integer_terms_count_l3241_324153


namespace shaded_area_proof_l3241_324156

def circle_radius : ℝ := 3

def pi_value : ℝ := 3

theorem shaded_area_proof :
  let circle_area := pi_value * circle_radius^2
  let square_side := circle_radius * Real.sqrt 2
  let square_area := square_side^2
  let total_square_area := 2 * square_area
  circle_area - total_square_area = 9 := by sorry

end shaded_area_proof_l3241_324156


namespace lines_are_parallel_l3241_324163

-- Define the slope and y-intercept of a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the two lines
def l1 : Line := { slope := 2, intercept := 1 }
def l2 : Line := { slope := 2, intercept := 5 }

-- Define parallel lines
def parallel (a b : Line) : Prop := a.slope = b.slope

-- Theorem statement
theorem lines_are_parallel : parallel l1 l2 := by
  sorry

end lines_are_parallel_l3241_324163


namespace journey_time_ratio_l3241_324110

/-- Represents a two-part journey with given speeds and times -/
structure Journey where
  v : ℝ  -- Initial speed
  t : ℝ  -- Initial time
  total_distance : ℝ  -- Total distance traveled

/-- The theorem statement -/
theorem journey_time_ratio 
  (j : Journey) 
  (h1 : j.v = 30)  -- Initial speed is 30 mph
  (h2 : j.v * j.t + (2 * j.v) * (2 * j.t) = j.total_distance)  -- Total distance equation
  (h3 : j.total_distance = 75)  -- Total distance is 75 miles
  : j.t / (2 * j.t) = 1 / 2 := by
  sorry

#check journey_time_ratio

end journey_time_ratio_l3241_324110


namespace bridge_length_l3241_324139

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 205 ∧
    bridge_length = train_speed_kmh * (1000 / 3600) * crossing_time - train_length :=
by sorry

end bridge_length_l3241_324139


namespace integral_cube_root_problem_l3241_324114

open Real MeasureTheory

theorem integral_cube_root_problem :
  ∫ x in (1 : ℝ)..64, (2 + x^(1/3)) / ((x^(1/6) + 2*x^(1/3) + x^(1/2)) * x^(1/2)) = 6 := by
  sorry

end integral_cube_root_problem_l3241_324114


namespace reciprocal_of_negative_three_l3241_324165

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end reciprocal_of_negative_three_l3241_324165


namespace inequality_proof_l3241_324130

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = a^2 + b^2 + c^2) : 
  a^2 / (a^2 + b*c) + b^2 / (b^2 + c*a) + c^2 / (c^2 + a*b) ≥ (a + b + c) / 2 := by
sorry

end inequality_proof_l3241_324130


namespace factorization_f_max_value_g_l3241_324137

-- Define the polynomials
def f (x : ℝ) : ℝ := x^2 - 4*x - 5
def g (x : ℝ) : ℝ := -2*x^2 - 4*x + 3

-- Theorem for factorization of f
theorem factorization_f : ∀ x : ℝ, f x = (x + 1) * (x - 5) := by sorry

-- Theorem for maximum value of g
theorem max_value_g : 
  (∀ x : ℝ, g x ≤ 5) ∧ g (-1) = 5 := by sorry

end factorization_f_max_value_g_l3241_324137


namespace article_selling_price_l3241_324113

theorem article_selling_price (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 10 →
  gain_percent = 150 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 25 := by
sorry

end article_selling_price_l3241_324113


namespace vasya_petya_number_ambiguity_l3241_324170

theorem vasya_petya_number_ambiguity (a : ℝ) (ha : a ≠ 0) :
  ∃ b : ℝ, b ≠ a ∧ a^4 + a^2 = b^4 + b^2 := by
  sorry

end vasya_petya_number_ambiguity_l3241_324170


namespace waiter_tables_l3241_324141

theorem waiter_tables (total_customers : ℕ) (people_per_table : ℕ) (h1 : total_customers = 90) (h2 : people_per_table = 10) :
  total_customers / people_per_table = 9 := by
  sorry

end waiter_tables_l3241_324141


namespace x_squared_less_than_abs_x_plus_two_l3241_324146

theorem x_squared_less_than_abs_x_plus_two (x : ℝ) :
  x^2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by sorry

end x_squared_less_than_abs_x_plus_two_l3241_324146


namespace partition_naturals_l3241_324171

theorem partition_naturals (c : ℚ) (hc : c > 0) (hc_ne_one : c ≠ 1) :
  ∃ (A B : Set ℕ), (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) ∧
  (∀ (a₁ a₂ : ℕ), a₁ ∈ A → a₂ ∈ A → a₁ ≠ 0 → a₂ ≠ 0 → (a₁ : ℚ) / a₂ ≠ c) ∧
  (∀ (b₁ b₂ : ℕ), b₁ ∈ B → b₂ ∈ B → b₁ ≠ 0 → b₂ ≠ 0 → (b₁ : ℚ) / b₂ ≠ c) :=
by sorry

end partition_naturals_l3241_324171


namespace projected_revenue_increase_l3241_324106

theorem projected_revenue_increase (actual_decrease : Real) (actual_to_projected_ratio : Real) 
  (h1 : actual_decrease = 0.3)
  (h2 : actual_to_projected_ratio = 0.5) :
  ∃ (projected_increase : Real), 
    (1 - actual_decrease) = actual_to_projected_ratio * (1 + projected_increase) ∧ 
    projected_increase = 0.4 := by
  sorry

end projected_revenue_increase_l3241_324106


namespace linear_increase_l3241_324112

theorem linear_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 6) :
  ∀ x, f (x + 12) - f x = 18 := by
  sorry

end linear_increase_l3241_324112


namespace chord_intersection_triangle_area_l3241_324184

/-- Given two chords of a circle intersecting at a point, this theorem
    calculates the area of one triangle formed by the chords, given the
    area of the other triangle and the lengths of two segments. -/
theorem chord_intersection_triangle_area
  (PO SO : ℝ) (area_POR : ℝ) (h1 : PO = 3) (h2 : SO = 4) (h3 : area_POR = 7) :
  let area_QOS := (16 * area_POR) / (9 : ℝ)
  area_QOS = 112 / 9 := by sorry

end chord_intersection_triangle_area_l3241_324184


namespace zero_subset_X_l3241_324154

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by
  sorry

end zero_subset_X_l3241_324154


namespace real_part_of_complex_product_l3241_324119

theorem real_part_of_complex_product : 
  let z : ℂ := (1 + Complex.I) * (1 + 2 * Complex.I)
  Complex.re z = -1 := by sorry

end real_part_of_complex_product_l3241_324119


namespace time_for_A_alone_l3241_324158

-- Define the work rates for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
def condition1 : Prop := 3 * (rA + rB) = 1
def condition2 : Prop := 6 * (rB + rC) = 1
def condition3 : Prop := (15/4) * (rA + rC) = 1

-- Theorem statement
theorem time_for_A_alone 
  (h1 : condition1 rA rB)
  (h2 : condition2 rB rC)
  (h3 : condition3 rA rC) :
  1 / rA = 60 / 13 :=
by sorry

end time_for_A_alone_l3241_324158


namespace arithmetic_sequence_sum_l3241_324159

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end arithmetic_sequence_sum_l3241_324159


namespace mistaken_divisor_l3241_324102

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = 32 * correct_divisor →
  dividend = 56 * mistaken_divisor →
  mistaken_divisor = 12 := by
sorry

end mistaken_divisor_l3241_324102


namespace right_triangle_perimeter_l3241_324116

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end right_triangle_perimeter_l3241_324116


namespace sarah_bottle_caps_l3241_324155

/-- The total number of bottle caps Sarah has after buying more -/
def total_bottle_caps (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Sarah has 29 bottle caps in total -/
theorem sarah_bottle_caps : total_bottle_caps 26 3 = 29 := by
  sorry

end sarah_bottle_caps_l3241_324155


namespace acute_triangle_median_inequality_l3241_324122

/-- For an acute triangle ABC with side lengths a, b, c and corresponding median lengths m_a, m_b, m_c,
    the sum of the squared medians divided by the sum of two squared sides minus the third squared side
    is greater than or equal to 9/4. -/
theorem acute_triangle_median_inequality (a b c m_a m_b m_c : ℝ) 
  (h_acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (h_medians : m_a^2 = (2*b^2 + 2*c^2 - a^2)/4 ∧ 
               m_b^2 = (2*a^2 + 2*c^2 - b^2)/4 ∧ 
               m_c^2 = (2*a^2 + 2*b^2 - c^2)/4) :
  m_a^2 / (-a^2 + b^2 + c^2) + m_b^2 / (-b^2 + a^2 + c^2) + m_c^2 / (-c^2 + a^2 + b^2) ≥ 9/4 :=
by sorry

end acute_triangle_median_inequality_l3241_324122


namespace banana_price_is_five_l3241_324131

/-- Represents the market problem with Peter's purchases --/
def market_problem (banana_price : ℝ) : Prop :=
  let initial_money : ℝ := 500
  let potato_kilos : ℝ := 6
  let potato_price : ℝ := 2
  let tomato_kilos : ℝ := 9
  let tomato_price : ℝ := 3
  let cucumber_kilos : ℝ := 5
  let cucumber_price : ℝ := 4
  let banana_kilos : ℝ := 3
  let remaining_money : ℝ := 426
  initial_money - (potato_kilos * potato_price + tomato_kilos * tomato_price + 
    cucumber_kilos * cucumber_price + banana_kilos * banana_price) = remaining_money

/-- Theorem stating that the price per kilo of bananas is $5 --/
theorem banana_price_is_five : 
  ∃ (banana_price : ℝ), market_problem banana_price ∧ banana_price = 5 :=
sorry

end banana_price_is_five_l3241_324131


namespace jess_walks_to_store_l3241_324107

/-- The number of blocks Jess walks to the store -/
def blocks_to_store : ℕ := sorry

/-- The total number of blocks Jess walks -/
def total_blocks : ℕ := 25

/-- Theorem stating that Jess walks 11 blocks to the store -/
theorem jess_walks_to_store : 
  blocks_to_store = 11 :=
by
  have h1 : blocks_to_store + 6 + 8 = total_blocks := sorry
  sorry


end jess_walks_to_store_l3241_324107


namespace hyperbola_equation_l3241_324117

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := ∃ (a b : ℝ), x^2/a^2 - y^2/b^2 = 1

-- Define the condition of shared foci
def shared_foci (e h : (ℝ → ℝ → Prop)) : Prop := 
  ∃ (c : ℝ), c^2 = 5 ∧ 
    (∀ x y, e x y ↔ x^2/(c^2+4) + y^2/4 = 1) ∧
    (∀ x y, h x y ↔ ∃ (a b : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ a^2 - b^2 = c^2)

-- Define the asymptote condition
def asymptote_condition (h : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), h x y ∧ x - 2*y = 0

-- The theorem to prove
theorem hyperbola_equation 
  (h : shared_foci ellipse hyperbola_C)
  (a : asymptote_condition hyperbola_C) :
  ∀ x y, hyperbola_C x y ↔ x^2/4 - y^2 = 1 :=
sorry

end hyperbola_equation_l3241_324117


namespace union_complement_problem_l3241_324126

theorem union_complement_problem (U A B : Set Nat) :
  U = {1, 2, 3, 4} →
  A = {1, 2} →
  B = {2, 3} →
  A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end union_complement_problem_l3241_324126


namespace zeros_of_f_l3241_324125

def f (x : ℝ) : ℝ := (x^2 - 3*x) * (x + 4)

theorem zeros_of_f : {x : ℝ | f x = 0} = {0, 3, -4} := by sorry

end zeros_of_f_l3241_324125


namespace subtracted_number_l3241_324176

theorem subtracted_number (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 := by
  sorry

end subtracted_number_l3241_324176


namespace total_pure_acid_l3241_324177

theorem total_pure_acid (solution1_volume : Real) (solution1_concentration : Real)
                        (solution2_volume : Real) (solution2_concentration : Real)
                        (solution3_volume : Real) (solution3_concentration : Real) :
  solution1_volume = 6 →
  solution1_concentration = 0.40 →
  solution2_volume = 4 →
  solution2_concentration = 0.35 →
  solution3_volume = 3 →
  solution3_concentration = 0.55 →
  solution1_volume * solution1_concentration +
  solution2_volume * solution2_concentration +
  solution3_volume * solution3_concentration = 5.45 := by
sorry

end total_pure_acid_l3241_324177


namespace infinite_inequality_occurrences_l3241_324143

theorem infinite_inequality_occurrences (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 1 + a n > a (n - 1) * (2 : ℝ)^(1/5) :=
sorry

end infinite_inequality_occurrences_l3241_324143


namespace max_value_a_l3241_324169

theorem max_value_a (a b c d : ℝ) 
  (h1 : b + c + d = 3 - a) 
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) : 
  a ≤ 2 ∧ ∃ (b c d : ℝ), b + c + d = 3 - 2 ∧ 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - 2^2 :=
sorry

end max_value_a_l3241_324169


namespace journey_distance_is_70_l3241_324183

-- Define the journey parameters
def journey_time_at_40 : Real := 1.75
def journey_time_at_35 : Real := 2

-- Theorem statement
theorem journey_distance_is_70 :
  ∃ (distance : Real),
    distance = 40 * journey_time_at_40 ∧
    distance = 35 * journey_time_at_35 ∧
    distance = 70 := by
  sorry

end journey_distance_is_70_l3241_324183


namespace human_habitable_area_l3241_324129

/-- The fraction of Earth's surface that is not covered by water -/
def land_fraction : ℚ := 1/3

/-- The fraction of land that is inhabitable for humans -/
def inhabitable_land_fraction : ℚ := 2/3

/-- The fraction of Earth's surface that humans can live on -/
def human_habitable_fraction : ℚ := land_fraction * inhabitable_land_fraction

theorem human_habitable_area :
  human_habitable_fraction = 2/9 := by sorry

end human_habitable_area_l3241_324129


namespace probability_of_rolling_six_l3241_324132

/-- The probability of rolling a total of 6 with two fair dice -/
theorem probability_of_rolling_six (dice : ℕ) (faces : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  dice = 2 →
  faces = 6 →
  total_outcomes = faces * faces →
  favorable_outcomes = 5 →
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 :=
by sorry

end probability_of_rolling_six_l3241_324132


namespace projected_revenue_increase_l3241_324136

theorem projected_revenue_increase (last_year_revenue : ℝ) :
  let actual_revenue := 0.9 * last_year_revenue
  let projected_revenue := last_year_revenue * (1 + 0.2)
  actual_revenue = 0.75 * projected_revenue :=
by sorry

end projected_revenue_increase_l3241_324136


namespace derivative_of_power_function_l3241_324115

open Real

/-- Given differentiable functions u and v, where u is positive,
    f(x) = u(x)^(v(x)) is differentiable and its derivative is as stated. -/
theorem derivative_of_power_function (u v : ℝ → ℝ) (hu : Differentiable ℝ u)
    (hv : Differentiable ℝ v) (hup : ∀ x, u x > 0) :
  let f := λ x => (u x) ^ (v x)
  Differentiable ℝ f ∧ 
  ∀ x, deriv f x = (u x)^(v x) * (deriv v x * log (u x) + v x * deriv u x / u x) :=
by sorry

end derivative_of_power_function_l3241_324115


namespace solution_set_f_range_of_m_l3241_324199

-- Define the functions f and g
def f (x : ℝ) := |x + 3| + |x - 1|
def g (m : ℝ) (x : ℝ) := -x^2 + 2*m*x

-- Statement for the solution set of f(x) > 4
theorem solution_set_f (x : ℝ) : f x > 4 ↔ x < -3 ∨ x > 1 := by sorry

-- Statement for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, f x₁ ≥ g m x₂) → -2 < m ∧ m < 2 := by sorry

end solution_set_f_range_of_m_l3241_324199


namespace right_rectangular_prism_volume_specific_prism_volume_l3241_324187

/-- Given a right rectangular prism with face areas a₁, a₂, and a₃,
    prove that its volume is the square root of the product of these areas. -/
theorem right_rectangular_prism_volume 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) : 
  ∃ (l w h : ℝ), l * w = a₁ ∧ w * h = a₂ ∧ l * h = a₃ ∧ 
  l * w * h = Real.sqrt (a₁ * a₂ * a₃) := by
  sorry

/-- The volume of a right rectangular prism with face areas 56, 63, and 72 
    square units is 504 cubic units. -/
theorem specific_prism_volume : 
  ∃ (l w h : ℝ), l * w = 56 ∧ w * h = 63 ∧ l * h = 72 ∧ 
  l * w * h = 504 := by
  sorry

end right_rectangular_prism_volume_specific_prism_volume_l3241_324187


namespace two_from_five_permutation_l3241_324162

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem two_from_five_permutation : 
  permutations 5 2 = 20 := by sorry

end two_from_five_permutation_l3241_324162


namespace tetrahedron_sum_is_15_l3241_324185

/-- Represents a tetrahedron -/
structure Tetrahedron where
  edges : ℕ
  vertices : ℕ
  faces : ℕ

/-- The properties of a tetrahedron -/
def is_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges = 6 ∧ t.vertices = 4 ∧ t.faces = 4

/-- The sum calculation with one vertex counted twice -/
def sum_with_extra_vertex (t : Tetrahedron) : ℕ :=
  t.edges + (t.vertices + 1) + t.faces

/-- Theorem: The sum of edges, faces, and vertices (with one counted twice) of a tetrahedron is 15 -/
theorem tetrahedron_sum_is_15 (t : Tetrahedron) (h : is_tetrahedron t) :
  sum_with_extra_vertex t = 15 := by
  sorry

end tetrahedron_sum_is_15_l3241_324185


namespace part_one_part_two_l3241_324194

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem part_one (m : ℝ) (h_m : m > 0) 
  (h_set : Set.Icc (-3/2) (1/2) = {x | f (x + 1) ≤ 2 * m}) : 
  m = 1 := by sorry

-- Part 2
theorem part_two : 
  (∃ n : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n / 2018^y + |2*x - 1|) ∧ 
  (∀ n' : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n' / 2018^y + |2*x - 1|) → n ≤ n')) ∧
  (∀ n : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n / 2018^y + |2*x - 1|) → n ≥ 1) := by sorry

end part_one_part_two_l3241_324194


namespace percentage_equality_l3241_324193

theorem percentage_equality : (0.2 * 4 : ℝ) = (0.8 * 1 : ℝ) := by sorry

end percentage_equality_l3241_324193


namespace matrix_power_100_l3241_324197

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_100 : A^100 = !![1, 0; 200, 1] := by sorry

end matrix_power_100_l3241_324197


namespace rectangle_length_l3241_324142

/-- Given a rectangle with width 5 feet and perimeter 22 feet, prove that its length is 6 feet. -/
theorem rectangle_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 5 → perimeter = 22 → 2 * (length + width) = perimeter → length = 6 := by
  sorry

end rectangle_length_l3241_324142


namespace line_representation_slope_nonexistence_x_intercept_angle_of_inclination_l3241_324133

/-- Represents the equation ((m^2 - 2m - 3)x + (2m^2 + m - 1)y + 6 - 2m = 0) -/
def equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y + 6 - 2*m = 0

/-- The equation represents a line if and only if m ≠ -1 -/
theorem line_representation (m : ℝ) : 
  (∃ x y, equation m x y) ↔ m ≠ -1 := by sorry

/-- The slope of the line does not exist when m = 1/2 -/
theorem slope_nonexistence (m : ℝ) : 
  (∀ x y, equation m x y → x = 4/3) ↔ m = 1/2 := by sorry

/-- When the x-intercept is -3, m = -5/3 -/
theorem x_intercept (m : ℝ) : 
  (∃ y, equation m (-3) y) ↔ m = -5/3 := by sorry

/-- When the angle of inclination is 45°, m = 4/3 -/
theorem angle_of_inclination (m : ℝ) : 
  (∀ x₁ y₁ x₂ y₂, equation m x₁ y₁ ∧ equation m x₂ y₂ ∧ x₁ ≠ x₂ → 
    (y₂ - y₁) / (x₂ - x₁) = 1) ↔ m = 4/3 := by sorry

end line_representation_slope_nonexistence_x_intercept_angle_of_inclination_l3241_324133


namespace max_sum_reciprocals_l3241_324157

theorem max_sum_reciprocals (k l m : ℕ+) (h : (k : ℝ)⁻¹ + (l : ℝ)⁻¹ + (m : ℝ)⁻¹ < 1) :
  ∃ (a b c : ℕ+), (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ = 41/42 ∧
    ∀ (x y z : ℕ+), (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ < 1 →
      (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ ≤ 41/42 := by
  sorry

end max_sum_reciprocals_l3241_324157


namespace largest_x_sqrt_3x_eq_5x_l3241_324198

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℚ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) ∧ 
  (∀ (y : ℚ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) :=
by sorry

end largest_x_sqrt_3x_eq_5x_l3241_324198


namespace uncertain_sum_l3241_324164

theorem uncertain_sum (a b c : ℤ) (h : |a - b|^19 + |c - a|^95 = 1) :
  ∃ (x : ℤ), (x = 1 ∨ x = 2) ∧ |c - a| + |a - b| + |b - a| = x :=
sorry

end uncertain_sum_l3241_324164


namespace oliver_stickers_l3241_324166

theorem oliver_stickers (initial_stickers : ℕ) (used_fraction : ℚ) (kept_stickers : ℕ) 
  (h1 : initial_stickers = 135)
  (h2 : used_fraction = 1/3)
  (h3 : kept_stickers = 54) :
  let remaining_stickers := initial_stickers - (used_fraction * initial_stickers).num
  let given_stickers := remaining_stickers - kept_stickers
  (given_stickers : ℚ) / remaining_stickers = 2/5 := by
sorry

end oliver_stickers_l3241_324166


namespace four_of_a_kind_count_l3241_324151

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Represents a "Four of a Kind" combination -/
structure FourOfAKind :=
  (number : ℕ)
  (fifth_card : ℕ)
  (h_number_valid : number ≤ 13)
  (h_fifth_card_valid : fifth_card ≤ 52)
  (h_fifth_card_diff : fifth_card ≠ number)

/-- The number of different "Four of a Kind" combinations in a standard deck -/
def count_four_of_a_kind (d : Deck) : ℕ :=
  13 * (d.total_cards - d.num_suits)

/-- Theorem stating that the number of "Four of a Kind" combinations is 624 -/
theorem four_of_a_kind_count (d : Deck) 
  (h_standard : d.total_cards = 52 ∧ d.num_suits = 4 ∧ d.cards_per_suit = 13) : 
  count_four_of_a_kind d = 624 := by
  sorry

end four_of_a_kind_count_l3241_324151


namespace complex_absolute_value_l3241_324101

theorem complex_absolute_value (x y : ℝ) : 
  (Complex.I : ℂ) * (x + 3 * Complex.I) = y - Complex.I → 
  Complex.abs (x - y * Complex.I) = Real.sqrt 10 := by
  sorry

end complex_absolute_value_l3241_324101


namespace units_digit_of_sqrt_product_sum_last_three_digits_of_2012_cubed_l3241_324104

-- Define the function to calculate the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function to calculate the sum of the last 3 digits
def sumLastThreeDigits (n : ℕ) : ℕ := (n % 1000) / 100 + ((n % 100) / 10) + (n % 10)

-- Theorem 1
theorem units_digit_of_sqrt_product (Q : ℤ) :
  ∃ X : ℕ, X^2 = (100 * 102 * 103 * 105 + (Q - 3)) ∧ unitsDigit X = 3 := by sorry

-- Theorem 2
theorem sum_last_three_digits_of_2012_cubed :
  sumLastThreeDigits (2012^3) = 17 := by sorry

end units_digit_of_sqrt_product_sum_last_three_digits_of_2012_cubed_l3241_324104


namespace find_n_l3241_324147

theorem find_n : ∃ n : ℝ, n + (n + 1) + (n + 2) + (n + 3) = 20 ∧ n = 3.5 := by
  sorry

end find_n_l3241_324147


namespace messages_in_week_after_removal_l3241_324188

/-- Calculates the total number of messages sent in a week by remaining members of a group after some members were removed. -/
def total_messages_in_week (initial_members : ℕ) (removed_members : ℕ) (messages_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  (initial_members - removed_members) * messages_per_day * days_in_week

/-- Proves that the total number of messages sent in a week by remaining members is 45500, given the specified conditions. -/
theorem messages_in_week_after_removal :
  total_messages_in_week 150 20 50 7 = 45500 := by
  sorry

end messages_in_week_after_removal_l3241_324188


namespace closest_integer_to_cube_root_l3241_324172

theorem closest_integer_to_cube_root (n : ℕ) : 
  ∃ (m : ℤ), ∀ (k : ℤ), |k - (5^3 + 9^3 : ℝ)^(1/3)| ≥ |m - (5^3 + 9^3 : ℝ)^(1/3)| ∧ m = 9 := by
  sorry

end closest_integer_to_cube_root_l3241_324172


namespace sandy_carrots_l3241_324100

theorem sandy_carrots (initial_carrots : ℕ) (taken_carrots : ℕ) :
  initial_carrots = 6 →
  taken_carrots = 3 →
  initial_carrots - taken_carrots = 3 := by
  sorry

end sandy_carrots_l3241_324100


namespace arithmetic_sequence_solution_l3241_324160

theorem arithmetic_sequence_solution :
  ∀ (y : ℚ),
  let a₁ : ℚ := 3/4
  let a₂ : ℚ := y - 2
  let a₃ : ℚ := 4*y
  (a₂ - a₁ = a₃ - a₂) →  -- arithmetic sequence condition
  y = -19/8 :=
by
  sorry

end arithmetic_sequence_solution_l3241_324160


namespace trigonometric_system_solution_l3241_324186

theorem trigonometric_system_solution (x y z : ℝ) : 
  Real.sin x + Real.sin y + Real.sin (x + y + z) = 0 ∧
  Real.sin x + 2 * Real.sin z = 0 ∧
  Real.sin y + 3 * Real.sin z = 0 →
  ∃ (k m n : ℤ), x = π * k ∧ y = π * m ∧ z = π * n := by
sorry

end trigonometric_system_solution_l3241_324186


namespace sqrt_sum_simplification_l3241_324144

theorem sqrt_sum_simplification : Real.sqrt 3600 + Real.sqrt 1600 = 100 := by
  sorry

end sqrt_sum_simplification_l3241_324144


namespace shift_direct_proportion_l3241_324181

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shift a linear function horizontally -/
def shift_right (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.slope * units + f.intercept }

/-- The original direct proportion function y = -2x -/
def original_function : LinearFunction :=
  { slope := -2, intercept := 0 }

theorem shift_direct_proportion :
  shift_right original_function 3 = { slope := -2, intercept := 6 } := by
  sorry

end shift_direct_proportion_l3241_324181


namespace f_always_positive_implies_m_greater_than_e_range_of_y_when_f_has_two_zeros_l3241_324148

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x - 2

-- Theorem 1: If f(x) > 0 for all x in ℝ, then m > e
theorem f_always_positive_implies_m_greater_than_e (m : ℝ) :
  (∀ x : ℝ, f m x > 0) → m > Real.exp 1 := by sorry

-- Theorem 2: Range of y when f has two zeros
theorem range_of_y_when_f_has_two_zeros (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  f m x₁ = 0 →
  f m x₂ = 0 →
  let y := (Real.exp x₂ - Real.exp x₁) * (1 / (Real.exp x₂ + Real.exp x₁) - m)
  ∀ z : ℝ, z < 0 ∧ ∃ (t : ℝ), y = t := by sorry

end

end f_always_positive_implies_m_greater_than_e_range_of_y_when_f_has_two_zeros_l3241_324148


namespace fisherman_catch_l3241_324140

theorem fisherman_catch (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) : 
  bass = 32 → 
  trout = bass / 4 → 
  blue_gill = 2 * bass → 
  bass + trout + blue_gill = 104 := by
  sorry

end fisherman_catch_l3241_324140


namespace min_teams_for_athletes_l3241_324135

theorem min_teams_for_athletes (total_athletes : ℕ) (max_per_team : ℕ) (h1 : total_athletes = 30) (h2 : max_per_team = 9) :
  ∃ (num_teams : ℕ) (athletes_per_team : ℕ),
    num_teams * athletes_per_team = total_athletes ∧
    athletes_per_team ≤ max_per_team ∧
    num_teams = 5 ∧
    ∀ (other_num_teams : ℕ) (other_athletes_per_team : ℕ),
      other_num_teams * other_athletes_per_team = total_athletes →
      other_athletes_per_team ≤ max_per_team →
      other_num_teams ≥ num_teams :=
by sorry

end min_teams_for_athletes_l3241_324135


namespace x_less_than_one_implications_l3241_324182

theorem x_less_than_one_implications (x : ℝ) (h : x < 1) : x^3 < 1 ∧ |x| < 1 := by
  sorry

end x_less_than_one_implications_l3241_324182


namespace fraction_product_equality_l3241_324167

theorem fraction_product_equality : (3 / 4) * (5 / 9) * (8 / 13) * (3 / 7) = 10 / 91 := by
  sorry

end fraction_product_equality_l3241_324167


namespace permutations_with_non_adjacency_l3241_324192

theorem permutations_with_non_adjacency (n : ℕ) (h : n ≥ 4) :
  let total_permutations := n.factorial
  let adjacent_a1_a2 := 2 * (n - 1).factorial
  let adjacent_a3_a4 := 2 * (n - 1).factorial
  let both_adjacent := 4 * (n - 2).factorial
  total_permutations - adjacent_a1_a2 - adjacent_a3_a4 + both_adjacent = (n^2 - 5*n + 8) * (n - 2).factorial :=
by sorry

#check permutations_with_non_adjacency

end permutations_with_non_adjacency_l3241_324192


namespace largest_certain_divisor_l3241_324124

def is_valid_roll (roll : Finset Nat) : Prop :=
  roll.card = 7 ∧ roll ⊆ Finset.range 9 \ {0}

def product_of_roll (roll : Finset Nat) : Nat :=
  roll.prod id

theorem largest_certain_divisor :
  ∃ (n : Nat), n = 192 ∧
  (∀ (roll : Finset Nat), is_valid_roll roll → n ∣ product_of_roll roll) ∧
  (∀ (m : Nat), m > n →
    ∃ (roll : Finset Nat), is_valid_roll roll ∧ ¬(m ∣ product_of_roll roll)) :=
sorry

end largest_certain_divisor_l3241_324124


namespace product_congruence_l3241_324189

theorem product_congruence : 56 * 89 * 94 ≡ 21 [ZMOD 25] := by
  sorry

end product_congruence_l3241_324189


namespace abc_product_l3241_324118

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 165)
  (h2 : b * (c + a) = 156)
  (h3 : c * (a + b) = 180) :
  a * b * c = 100 * Real.sqrt 39 := by
  sorry

end abc_product_l3241_324118
