import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_inequality_l411_41172

theorem absolute_value_inequality (x y : ℝ) : 
  |y - 3*x| < 2*x ↔ x > 0 ∧ x < y ∧ y < 5*x :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l411_41172


namespace NUMINAMATH_CALUDE_afternoon_bags_count_l411_41176

def morning_bags : ℕ := 29
def bag_weight : ℕ := 7
def total_weight : ℕ := 322

def afternoon_bags : ℕ := (total_weight - morning_bags * bag_weight) / bag_weight

theorem afternoon_bags_count : afternoon_bags = 17 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_bags_count_l411_41176


namespace NUMINAMATH_CALUDE_subset_A_l411_41101

def A : Set ℝ := {x | x > -1}

theorem subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_A_l411_41101


namespace NUMINAMATH_CALUDE_right_angled_triangle_l411_41178

theorem right_angled_triangle (A B C : Real) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ B ∧ B ≤ π) (h3 : 0 ≤ C ∧ C ≤ π) 
  (h4 : A + B + C = π) (h5 : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2) : 
  A = π/2 ∨ B = π/2 ∨ C = π/2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l411_41178


namespace NUMINAMATH_CALUDE_chocolate_division_l411_41113

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) :
  total_chocolate = 64/7 →
  num_piles = 6 →
  piles_to_shaina = 2 →
  piles_to_shaina * (total_chocolate / num_piles) = 64/21 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l411_41113


namespace NUMINAMATH_CALUDE_min_moves_to_no_moves_l411_41126

/-- Represents a chessboard configuration -/
structure ChessBoard (n : ℕ) where
  pieces : Fin n → Fin n → Bool

/-- A move on the chessboard -/
inductive Move (n : ℕ)
  | jump : Fin n → Fin n → Fin n → Fin n → Move n

/-- Predicate to check if a move is valid -/
def is_valid_move (n : ℕ) (board : ChessBoard n) (move : Move n) : Prop :=
  match move with
  | Move.jump from_x from_y to_x to_y =>
    -- Implement the logic for a valid move
    sorry

/-- Predicate to check if no further moves are possible -/
def no_moves_possible (n : ℕ) (board : ChessBoard n) : Prop :=
  ∀ (move : Move n), ¬(is_valid_move n board move)

/-- The main theorem -/
theorem min_moves_to_no_moves (n : ℕ) :
  ∀ (move_sequence : List (Move n)),
    (∃ (final_board : ChessBoard n),
      no_moves_possible n final_board ∧
      -- final_board is the result of applying move_sequence to the initial board
      sorry) →
    move_sequence.length ≥ ⌈(n^2 : ℚ) / 3⌉ :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_no_moves_l411_41126


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l411_41158

theorem restaurant_bill_proof : 
  ∀ (total_bill : ℝ),
  (∃ (individual_share : ℝ),
    -- 9 friends initially splitting the bill equally
    individual_share = total_bill / 9 ∧ 
    -- 8 friends each paying an extra $3.00
    8 * (individual_share + 3) = total_bill) →
  total_bill = 216 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l411_41158


namespace NUMINAMATH_CALUDE_final_paycheck_amount_l411_41195

def biweekly_gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

theorem final_paycheck_amount :
  biweekly_gross_pay * (1 - retirement_rate) - tax_deduction = 740 := by
  sorry

end NUMINAMATH_CALUDE_final_paycheck_amount_l411_41195


namespace NUMINAMATH_CALUDE_peters_to_amandas_flower_ratio_l411_41183

theorem peters_to_amandas_flower_ratio : 
  ∀ (amanda_flowers peter_flowers peter_flowers_after : ℕ),
    amanda_flowers = 20 →
    peter_flowers = peter_flowers_after + 15 →
    peter_flowers_after = 45 →
    peter_flowers = 3 * amanda_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_peters_to_amandas_flower_ratio_l411_41183


namespace NUMINAMATH_CALUDE_jane_work_days_l411_41145

theorem jane_work_days (john_rate : ℚ) (total_days : ℕ) (jane_stop_days : ℕ) :
  john_rate = 1/20 →
  total_days = 10 →
  jane_stop_days = 5 →
  ∃ jane_rate : ℚ,
    (5 * (john_rate + jane_rate) + 5 * john_rate = 1) ∧
    (jane_rate = 1/10) :=
by sorry

end NUMINAMATH_CALUDE_jane_work_days_l411_41145


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_proof_l411_41104

/-- The original cost of one chocolate bar before discount -/
def chocolate_bar_cost : ℝ := 4.82

theorem chocolate_bar_cost_proof (
  gummy_bear_cost : ℝ)
  (chocolate_chip_cost : ℝ)
  (total_cost : ℝ)
  (gummy_bear_discount : ℝ)
  (chocolate_chip_discount : ℝ)
  (chocolate_bar_discount : ℝ)
  (h1 : gummy_bear_cost = 2)
  (h2 : chocolate_chip_cost = 5)
  (h3 : total_cost = 150)
  (h4 : gummy_bear_discount = 0.05)
  (h5 : chocolate_chip_discount = 0.10)
  (h6 : chocolate_bar_discount = 0.15)
  : chocolate_bar_cost = 4.82 := by
  sorry

#check chocolate_bar_cost_proof

end NUMINAMATH_CALUDE_chocolate_bar_cost_proof_l411_41104


namespace NUMINAMATH_CALUDE_find_divisor_l411_41106

theorem find_divisor (divisor : ℕ) : 
  (144 / divisor = 13) ∧ (144 % divisor = 1) → divisor = 11 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l411_41106


namespace NUMINAMATH_CALUDE_unique_triple_solution_l411_41161

theorem unique_triple_solution (a b c : ℝ) : 
  a > 5 ∧ b > 5 ∧ c > 5 ∧
  ((a + 3)^2 / (b + c - 5) + (b + 5)^2 / (c + a - 7) + (c + 7)^2 / (a + b - 9) = 49) →
  a = 13 ∧ b = 9 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l411_41161


namespace NUMINAMATH_CALUDE_sqrt_three_minus_fraction_bound_l411_41117

theorem sqrt_three_minus_fraction_bound (n m : ℕ) (h : Real.sqrt 3 - (m : ℝ) / n > 0) :
  Real.sqrt 3 - (m : ℝ) / n > 1 / (2 * (m : ℝ) * n) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_fraction_bound_l411_41117


namespace NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l411_41136

theorem no_positive_a_satisfies_inequality :
  ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l411_41136


namespace NUMINAMATH_CALUDE_paint_for_large_cube_l411_41114

-- Define the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge ^ 2

-- Define the paint required for a cube with edge 2 cm
def paint_for_2cm : ℝ := 1

-- Define the edge length of the larger cube
def large_cube_edge : ℝ := 6

-- Theorem to prove
theorem paint_for_large_cube : 
  (surface_area large_cube_edge / surface_area 2) * paint_for_2cm = 9 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_large_cube_l411_41114


namespace NUMINAMATH_CALUDE_business_value_l411_41150

/-- Proves the value of a business given partial ownership and sale information -/
theorem business_value (
  total_shares : ℚ)
  (owner_share : ℚ)
  (sold_fraction : ℚ)
  (sale_price : ℚ)
  (h1 : owner_share = 1 / 3)
  (h2 : sold_fraction = 3 / 5)
  (h3 : sale_price = 15000) :
  total_shares = 75000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_l411_41150


namespace NUMINAMATH_CALUDE_wrench_force_problem_l411_41156

/-- The force required to loosen a nut varies inversely with the length of the wrench handle -/
def inverse_variation (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem wrench_force_problem (force₁ : ℝ) (length₁ : ℝ) (force₂ : ℝ) (length₂ : ℝ) :
  inverse_variation force₁ length₁ →
  inverse_variation force₂ length₂ →
  force₁ = 300 →
  length₁ = 12 →
  length₂ = 18 →
  force₂ = 200 := by
  sorry

end NUMINAMATH_CALUDE_wrench_force_problem_l411_41156


namespace NUMINAMATH_CALUDE_gcd_12547_23791_l411_41162

theorem gcd_12547_23791 : Nat.gcd 12547 23791 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12547_23791_l411_41162


namespace NUMINAMATH_CALUDE_impossible_coverage_l411_41159

/-- Represents a rectangular paper strip -/
structure PaperStrip where
  width : ℕ
  length : ℕ

/-- Represents a cube -/
structure Cube where
  sideLength : ℕ

/-- Represents the configuration of paper strips on cube faces -/
def CubeConfiguration := Cube → List PaperStrip

/-- Checks if a configuration covers exactly three faces sharing a vertex -/
def coversThreeFaces (config : CubeConfiguration) (cube : Cube) : Prop :=
  sorry

/-- Checks if strips in a configuration overlap -/
def hasOverlap (config : CubeConfiguration) : Prop :=
  sorry

/-- Checks if a configuration leaves any gaps -/
def hasGaps (config : CubeConfiguration) (cube : Cube) : Prop :=
  sorry

/-- Main theorem: It's impossible to cover three faces of a 4x4x4 cube with 16 1x3 strips -/
theorem impossible_coverage : 
  ∀ (config : CubeConfiguration),
    let cube := Cube.mk 4
    let strips := List.replicate 16 (PaperStrip.mk 1 3)
    (coversThreeFaces config cube) → 
    (¬ hasOverlap config) → 
    (¬ hasGaps config cube) → 
    False :=
  sorry

end NUMINAMATH_CALUDE_impossible_coverage_l411_41159


namespace NUMINAMATH_CALUDE_acute_angles_equal_positive_angles_less_than_90_l411_41139

-- Define the sets A and D
def A : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def D : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}

-- Theorem statement
theorem acute_angles_equal_positive_angles_less_than_90 : A = D := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_equal_positive_angles_less_than_90_l411_41139


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l411_41110

/-- A rectangle with perimeter 72 meters and length-to-width ratio of 5:2 has a diagonal of 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l411_41110


namespace NUMINAMATH_CALUDE_complex_number_modulus_l411_41164

theorem complex_number_modulus (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 1) :
  Complex.abs z = 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l411_41164


namespace NUMINAMATH_CALUDE_exists_vertex_reach_all_l411_41142

/-- A directed graph where every pair of vertices is connected by a directed edge. -/
structure CompleteDigraph (V : Type) where
  edge : V → V → Prop
  complete : ∀ (u v : V), u ≠ v → edge u v ∨ edge v u

/-- A path of length at most 2 exists between two vertices. -/
def PathLengthAtMostTwo {V : Type} (G : CompleteDigraph V) (u v : V) : Prop :=
  G.edge u v ∨ ∃ w : V, G.edge u w ∧ G.edge w v

/-- There exists a vertex from which every other vertex can be reached by a path of length at most 2. -/
theorem exists_vertex_reach_all {V : Type} (G : CompleteDigraph V) [Finite V] [Nonempty V] :
  ∃ u : V, ∀ v : V, u ≠ v → PathLengthAtMostTwo G u v := by sorry

end NUMINAMATH_CALUDE_exists_vertex_reach_all_l411_41142


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l411_41180

-- Define the quadratic function
def f (x : ℝ) := x^2 + x - 2

-- Define the solution set
def solution_set := {x : ℝ | x ≤ -2 ∨ x ≥ 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l411_41180


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l411_41160

/-- Given Melissa's game scoring information, calculate her points per game without bonus -/
theorem melissa_points_per_game (bonus_per_game : ℕ) (total_points : ℕ) (num_games : ℕ) 
  (h1 : bonus_per_game = 82)
  (h2 : total_points = 15089)
  (h3 : num_games = 79) :
  (total_points - bonus_per_game * num_games) / num_games = 109 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l411_41160


namespace NUMINAMATH_CALUDE_marble_jar_count_l411_41144

theorem marble_jar_count :
  ∀ (total blue red green yellow : ℕ),
    2 * blue = total →
    4 * red = total →
    green = 27 →
    yellow = 14 →
    blue + red + green + yellow = total →
    total = 164 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_count_l411_41144


namespace NUMINAMATH_CALUDE_quadratic_sum_of_solutions_l411_41189

theorem quadratic_sum_of_solutions : ∃ a b : ℝ, 
  (∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = a ∨ x = b)) ∧ 
  a ≥ b ∧ 
  3*a + 2*b = 15 + Real.sqrt 92 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_solutions_l411_41189


namespace NUMINAMATH_CALUDE_wire_necklace_length_l411_41137

def wire_problem (num_spools : ℕ) (spool_length : ℕ) (total_necklaces : ℕ) : ℕ :=
  (num_spools * spool_length) / total_necklaces

theorem wire_necklace_length :
  wire_problem 3 20 15 = 4 :=
by sorry

end NUMINAMATH_CALUDE_wire_necklace_length_l411_41137


namespace NUMINAMATH_CALUDE_cards_distribution_l411_41165

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person : ℕ := total_cards / num_people
  let remaining_cards : ℕ := total_cards % num_people
  let people_with_extra : ℕ := remaining_cards
  (num_people - people_with_extra) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cards_distribution_l411_41165


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l411_41182

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (loss_margin : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 4400 →
  loss_margin = 1760 →
  candidate_percentage = total_votes.cast⁻¹ * (total_votes - loss_margin) / 2 →
  candidate_percentage = 30 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l411_41182


namespace NUMINAMATH_CALUDE_derivative_zero_implies_x_equals_plus_minus_a_l411_41134

theorem derivative_zero_implies_x_equals_plus_minus_a (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x ↦ (x^2 + a^2) / x
  let f' : ℝ → ℝ := fun x ↦ (x^2 - a^2) / x^2
  ∀ x₀ : ℝ, x₀ ≠ 0 → f' x₀ = 0 → x₀ = a ∨ x₀ = -a := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_implies_x_equals_plus_minus_a_l411_41134


namespace NUMINAMATH_CALUDE_octavia_photos_count_l411_41152

/-- Represents the number of photographs in a photography exhibition --/
structure PhotoExhibition where
  total : ℕ
  octavia_photos : ℕ
  jack_framed : ℕ
  jack_framed_octavia : ℕ
  jack_framed_others : ℕ

/-- The photography exhibition satisfies the given conditions --/
def exhibition_conditions (e : PhotoExhibition) : Prop :=
  e.jack_framed_octavia = 24 ∧
  e.jack_framed_others = 12 ∧
  e.jack_framed = e.jack_framed_octavia + e.jack_framed_others ∧
  e.total = 48 ∧
  e.total = e.octavia_photos + e.jack_framed - e.jack_framed_octavia

/-- Theorem stating that under the given conditions, Octavia took 36 photographs --/
theorem octavia_photos_count (e : PhotoExhibition) 
  (h : exhibition_conditions e) : e.octavia_photos = 36 := by
  sorry


end NUMINAMATH_CALUDE_octavia_photos_count_l411_41152


namespace NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l411_41122

theorem sum_and_ratio_implies_difference (x y : ℝ) 
  (sum_eq : x + y = 540)
  (ratio_eq : x / y = 4 / 5) :
  y - x = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l411_41122


namespace NUMINAMATH_CALUDE_debt_doubling_time_l411_41174

theorem debt_doubling_time (interest_rate : ℝ) (doubling_factor : ℝ) : 
  interest_rate = 0.06 → doubling_factor = 2 → 
  (∀ t : ℕ, t < 12 → (1 + interest_rate)^t ≤ doubling_factor) ∧ 
  (1 + interest_rate)^12 > doubling_factor := by
  sorry

end NUMINAMATH_CALUDE_debt_doubling_time_l411_41174


namespace NUMINAMATH_CALUDE_total_cans_eq_319_l411_41105

/-- The number of cans collected by five people given certain relationships between their collections. -/
def total_cans (solomon : ℕ) : ℕ :=
  let juwan := solomon / 3
  let levi := juwan / 2
  let gaby := (5 * solomon) / 2
  let michelle := gaby / 3
  solomon + juwan + levi + gaby + michelle

/-- Theorem stating that when Solomon collects 66 cans, the total number of cans collected by all five people is 319. -/
theorem total_cans_eq_319 : total_cans 66 = 319 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_eq_319_l411_41105


namespace NUMINAMATH_CALUDE_f_at_5_l411_41146

/-- A function satisfying the given functional equation -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation that f satisfies for all x -/
axiom f_eq (x : ℝ) : 3 * f x + f (2 - x) = 4 * x^2 + 1

/-- The theorem to be proved -/
theorem f_at_5 : f 5 = 133 / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l411_41146


namespace NUMINAMATH_CALUDE_china_mobile_charges_l411_41116

/-- Represents a mobile plan with a base fee and an excess charge per minute -/
structure MobilePlan where
  base_fee : ℝ
  excess_charge : ℝ

/-- Calculates the total call charges for a given mobile plan and excess minutes -/
def total_charges (plan : MobilePlan) (excess_minutes : ℝ) : ℝ :=
  plan.base_fee + plan.excess_charge * excess_minutes

/-- Theorem stating the relationship between total charges and excess minutes for the specific plan -/
theorem china_mobile_charges (x : ℝ) :
  let plan := MobilePlan.mk 39 0.19
  total_charges plan x = 0.19 * x + 39 := by
  sorry


end NUMINAMATH_CALUDE_china_mobile_charges_l411_41116


namespace NUMINAMATH_CALUDE_candy_bar_weight_reduction_l411_41175

/-- Represents the change in weight and price of a candy bar -/
structure CandyBar where
  original_weight : ℝ
  new_weight : ℝ
  price : ℝ
  price_per_ounce_increase : ℝ

/-- The theorem stating the relationship between weight reduction and price per ounce increase -/
theorem candy_bar_weight_reduction (c : CandyBar) 
  (h1 : c.price_per_ounce_increase = 2/3)
  (h2 : c.price > 0)
  (h3 : c.original_weight > 0)
  (h4 : c.new_weight > 0)
  (h5 : c.new_weight < c.original_weight) :
  (c.original_weight - c.new_weight) / c.original_weight = 0.4 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_weight_reduction_l411_41175


namespace NUMINAMATH_CALUDE_three_digit_sum_not_2021_l411_41115

theorem three_digit_sum_not_2021 (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a ≠ b → b ≠ c → a ≠ c → 
  a < 10 → b < 10 → c < 10 → 
  222 * (a + b + c) ≠ 2021 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_not_2021_l411_41115


namespace NUMINAMATH_CALUDE_m_range_l411_41185

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 4*x + 3)}

-- Define set B
def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ (Set.univ \ A), y = x + m/x ∧ m > 0}

-- Theorem statement
theorem m_range (m : ℝ) : (2 * Real.sqrt m ∈ B m) ↔ (1 < m ∧ m < 9) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l411_41185


namespace NUMINAMATH_CALUDE_deposit_percentage_l411_41155

def deposit : ℝ := 120
def remaining : ℝ := 1080

theorem deposit_percentage :
  (deposit / (deposit + remaining)) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_deposit_percentage_l411_41155


namespace NUMINAMATH_CALUDE_joao_salary_height_l411_41124

/-- Conversion rate from real to cruzado -/
def real_to_cruzado : ℝ := 2750000000

/-- João's monthly salary in reais -/
def joao_salary : ℝ := 640

/-- Height of 100 cruzado notes in centimeters -/
def stack_height : ℝ := 1.5

/-- Number of cruzado notes in a stack -/
def notes_per_stack : ℝ := 100

/-- Conversion factor from centimeters to kilometers -/
def cm_to_km : ℝ := 100000

theorem joao_salary_height : 
  (joao_salary * real_to_cruzado / notes_per_stack * stack_height) / cm_to_km = 264000 := by
  sorry

end NUMINAMATH_CALUDE_joao_salary_height_l411_41124


namespace NUMINAMATH_CALUDE_simplify_negative_a_minus_a_l411_41103

theorem simplify_negative_a_minus_a (a : ℝ) : -a - a = -2 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_a_minus_a_l411_41103


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l411_41171

theorem largest_multiple_of_15_under_500 : ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l411_41171


namespace NUMINAMATH_CALUDE_expression_value_l411_41148

def numerator : ℤ := 20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1

def denominator : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20

theorem expression_value : (numerator : ℚ) / denominator = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l411_41148


namespace NUMINAMATH_CALUDE_shopping_spree_theorem_l411_41140

def shopping_spree (initial_amount : ℝ) (book_price : ℝ) (num_books : ℕ) 
  (game_price : ℝ) (water_bottle_price : ℝ) (snack_price : ℝ) (num_snacks : ℕ)
  (bundle_price : ℝ) (book_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let book_total := book_price * num_books
  let discounted_book_total := book_total * (1 - book_discount)
  let subtotal := discounted_book_total + game_price + water_bottle_price + 
                  (snack_price * num_snacks) + bundle_price
  let total_with_tax := subtotal * (1 + tax_rate)
  initial_amount - total_with_tax

theorem shopping_spree_theorem :
  shopping_spree 200 12 5 45 10 3 3 20 0.1 0.12 = 45.44 := by sorry

end NUMINAMATH_CALUDE_shopping_spree_theorem_l411_41140


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l411_41102

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 24 → 
  difference = 6 → 
  girls + boys = total → 
  girls = boys + difference → 
  (girls : ℚ) / (boys : ℚ) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l411_41102


namespace NUMINAMATH_CALUDE_handshakes_count_l411_41130

/-- Represents a social gathering with specific group interactions -/
structure SocialGathering where
  total_people : Nat
  group1_size : Nat
  subgroup_size : Nat
  group2_size : Nat
  outsiders : Nat

/-- Calculates the number of handshakes in a social gathering -/
def handshakes (sg : SocialGathering) : Nat :=
  sg.subgroup_size * (sg.group2_size + sg.outsiders) +
  (sg.group1_size - sg.subgroup_size) * sg.outsiders +
  sg.group2_size * sg.outsiders

/-- Theorem stating the number of handshakes in the specific social gathering -/
theorem handshakes_count :
  let sg : SocialGathering := {
    total_people := 36,
    group1_size := 25,
    subgroup_size := 15,
    group2_size := 6,
    outsiders := 5
  }
  handshakes sg = 245 := by sorry

end NUMINAMATH_CALUDE_handshakes_count_l411_41130


namespace NUMINAMATH_CALUDE_inverse_of_singular_matrix_l411_41179

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; 8, -4]

theorem inverse_of_singular_matrix :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_singular_matrix_l411_41179


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l411_41196

/-- A function that returns the number of positive integer divisors of a given positive integer -/
def numberOfDivisors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a given positive integer has exactly 12 positive integer divisors -/
def hasTwelveDivisors (n : ℕ+) : Prop :=
  numberOfDivisors n = 12

/-- Theorem stating that 108 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_twelve_divisors :
  (∀ m : ℕ+, m < 108 → ¬(hasTwelveDivisors m)) ∧ hasTwelveDivisors 108 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l411_41196


namespace NUMINAMATH_CALUDE_rational_equation_solution_l411_41118

theorem rational_equation_solution (x : ℝ) : 
  -5 < x ∧ x < 3 → ((x^2 - 4*x + 5) / (2*x - 2) = 2 ↔ x = 4 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l411_41118


namespace NUMINAMATH_CALUDE_find_other_number_l411_41198

theorem find_other_number (a b : ℤ) (h1 : a - b = 8) (h2 : a = 16) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l411_41198


namespace NUMINAMATH_CALUDE_parabola_sum_l411_41119

/-- A parabola with coefficients p, q, and r. -/
structure Parabola where
  p : ℚ
  q : ℚ
  r : ℚ

/-- The y-coordinate of a point on the parabola given its x-coordinate. -/
def Parabola.y_coord (para : Parabola) (x : ℚ) : ℚ :=
  para.p * x^2 + para.q * x + para.r

theorem parabola_sum (para : Parabola) 
    (vertex : para.y_coord 3 = -2)
    (point : para.y_coord 6 = 5) :
    para.p + para.q + para.r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l411_41119


namespace NUMINAMATH_CALUDE_larry_stickers_l411_41108

theorem larry_stickers (initial : ℕ) (lost : ℕ) (gained : ℕ) 
  (h1 : initial = 193) 
  (h2 : lost = 6) 
  (h3 : gained = 12) : 
  initial - lost + gained = 199 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l411_41108


namespace NUMINAMATH_CALUDE_incircle_radius_given_tangent_circles_l411_41125

-- Define the triangle and circles
structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the property of being tangent
def is_tangent (c1 c2 : Circle) : Prop := sorry

-- Define the property of being inside a triangle
def is_inside (c : Circle) (t : Triangle) : Prop := sorry

-- Define the incircle of a triangle
def incircle (t : Triangle) : Circle := sorry

-- Main theorem
theorem incircle_radius_given_tangent_circles 
  (t : Triangle) (k : Circle) (k1 k2 k3 : Circle) :
  k = incircle t →
  is_inside k1 t ∧ is_inside k2 t ∧ is_inside k3 t →
  is_tangent k k1 ∧ is_tangent k k2 ∧ is_tangent k k3 →
  k1.radius = 1 ∧ k2.radius = 4 ∧ k3.radius = 9 →
  k.radius = 11 := by
  sorry

end NUMINAMATH_CALUDE_incircle_radius_given_tangent_circles_l411_41125


namespace NUMINAMATH_CALUDE_rug_area_is_24_l411_41193

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
def rugArea (floorLength floorWidth stripWidth : ℝ) : ℝ :=
  (floorLength - 2 * stripWidth) * (floorWidth - 2 * stripWidth)

/-- Theorem stating that the area of the rug is 24 square meters given the specific dimensions -/
theorem rug_area_is_24 :
  rugArea 10 8 2 = 24 := by
  sorry

#eval rugArea 10 8 2

end NUMINAMATH_CALUDE_rug_area_is_24_l411_41193


namespace NUMINAMATH_CALUDE_dog_grooming_time_l411_41194

theorem dog_grooming_time :
  let short_hair_time : ℕ := 10 -- Time to dry a short-haired dog in minutes
  let full_hair_time : ℕ := 2 * short_hair_time -- Time to dry a full-haired dog
  let short_hair_count : ℕ := 6 -- Number of short-haired dogs
  let full_hair_count : ℕ := 9 -- Number of full-haired dogs
  let total_time : ℕ := short_hair_time * short_hair_count + full_hair_time * full_hair_count
  total_time / 60 = 4 -- Total time in hours
  := by sorry

end NUMINAMATH_CALUDE_dog_grooming_time_l411_41194


namespace NUMINAMATH_CALUDE_symmetric_curve_l411_41153

/-- The equation of a curve symmetric to y^2 = 4x with respect to the line x = 2 -/
theorem symmetric_curve (x y : ℝ) : 
  (∀ x₀ y₀ : ℝ, y₀^2 = 4*x₀ → (4 - x₀)^2 = 4*(2 - (4 - x₀))) → 
  y^2 = 16 - 4*x :=
sorry

end NUMINAMATH_CALUDE_symmetric_curve_l411_41153


namespace NUMINAMATH_CALUDE_sin_graph_translation_l411_41190

open Real

theorem sin_graph_translation (a : ℝ) (h1 : 0 < a) (h2 : a < π) :
  (∀ x, sin (2 * (x - a) + π / 3) = sin (2 * x)) → a = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_translation_l411_41190


namespace NUMINAMATH_CALUDE_min_workers_for_profit_is_16_l411_41181

/-- Represents the minimum number of workers required for a manufacturing plant to make a profit -/
def min_workers_for_profit (
  maintenance_cost : ℕ)  -- Daily maintenance cost in dollars
  (hourly_wage : ℕ)      -- Hourly wage per worker in dollars
  (widgets_per_hour : ℕ) -- Number of widgets produced per worker per hour
  (widget_price : ℕ)     -- Selling price of each widget in dollars
  (work_hours : ℕ)       -- Number of work hours per day
  : ℕ :=
  16

/-- Theorem stating that given the specific conditions, the minimum number of workers for profit is 16 -/
theorem min_workers_for_profit_is_16 :
  min_workers_for_profit 600 20 4 4 10 = 16 := by
  sorry

#eval min_workers_for_profit 600 20 4 4 10

end NUMINAMATH_CALUDE_min_workers_for_profit_is_16_l411_41181


namespace NUMINAMATH_CALUDE_sports_club_problem_l411_41129

theorem sports_club_problem (total_members badminton_players tennis_players both : ℕ) 
  (h1 : total_members = 30)
  (h2 : badminton_players = 17)
  (h3 : tennis_players = 17)
  (h4 : both = 6) :
  total_members - (badminton_players + tennis_players - both) = 2 := by
sorry

end NUMINAMATH_CALUDE_sports_club_problem_l411_41129


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l411_41191

/-- The x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_points (x : ℝ) :
  (3 * x^2 - 4 * x + 7 = 6 * x^2 + x + 3) ↔ 
  (x = (5 + Real.sqrt 73) / -6 ∨ x = (5 - Real.sqrt 73) / -6) := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l411_41191


namespace NUMINAMATH_CALUDE_stamp_book_gcd_l411_41187

theorem stamp_book_gcd : Nat.gcd (Nat.gcd 1260 1470) 1890 = 210 := by
  sorry

end NUMINAMATH_CALUDE_stamp_book_gcd_l411_41187


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l411_41121

/-- The complex number z = (3+i)/(1-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l411_41121


namespace NUMINAMATH_CALUDE_percentage_calculation_l411_41154

theorem percentage_calculation (whole : ℝ) (part : ℝ) :
  whole = 475.25 →
  part = 129.89 →
  (part / whole) * 100 = 27.33 :=
by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l411_41154


namespace NUMINAMATH_CALUDE_quadratic_properties_l411_41133

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the equation
def are_roots (a b c x₁ x₂ : ℝ) : Prop :=
  quadratic_equation a b c x₁ ∧ quadratic_equation a b c x₂

theorem quadratic_properties
  (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) (h_roots : are_roots a b c x₁ x₂) :
  (¬ (∃ z : ℂ, x₁ = z ∧ x₂ = z ∧ z.im ≠ 0)) ∧
  (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧
  (x₁^2 * x₂ + x₁ * x₂^2 = -b * c / a^2) ∧
  (b^2 - 4*a*c < 0 → ∃ y : ℝ, x₁ - x₂ = Complex.I * y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l411_41133


namespace NUMINAMATH_CALUDE_vertical_line_no_slope_l411_41192

/-- A line parallel to the y-axis has no defined slope -/
theorem vertical_line_no_slope (a : ℝ) : 
  ¬ ∃ (m : ℝ), ∀ (x y : ℝ), x = a → (∀ ε > 0, ∃ δ > 0, ∀ x' y', |x' - x| < δ → |y' - y| < ε * |x' - x|) :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_line_no_slope_l411_41192


namespace NUMINAMATH_CALUDE_triangle_parallel_ratio_bounds_l411_41111

/-- Given a triangle ABC with sides a, b, c and an interior point O, 
    the ratios formed by lines through O parallel to the sides satisfy
    the given inequalities. -/
theorem triangle_parallel_ratio_bounds 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ)
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (h_sum : a' / a + b' / b + c' / c = 3) :
  (max (a' / a) (max (b' / b) (c' / c)) ≥ 2 / 3) ∧ 
  (min (a' / a) (min (b' / b) (c' / c)) ≤ 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_parallel_ratio_bounds_l411_41111


namespace NUMINAMATH_CALUDE_min_sum_of_product_2550_l411_41112

theorem min_sum_of_product_2550 (a b c : ℕ+) (h : a * b * c = 2550) :
  ∃ (x y z : ℕ+), x * y * z = 2550 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2550_l411_41112


namespace NUMINAMATH_CALUDE_intersection_implies_m_values_l411_41170

theorem intersection_implies_m_values (m : ℝ) : 
  let M : Set ℝ := {4, 5, -3*m}
  let N : Set ℝ := {-9, 3}
  (M ∩ N).Nonempty → m = 3 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_values_l411_41170


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l411_41188

/-- The speed of the boat in still water in kmph -/
def boat_speed : ℝ := 57

/-- The speed of the stream in kmph -/
def stream_speed : ℝ := 19

/-- The time taken to row upstream -/
def time_upstream : ℝ := sorry

/-- The time taken to row downstream -/
def time_downstream : ℝ := sorry

/-- The distance traveled (assumed to be the same for both upstream and downstream) -/
def distance : ℝ := sorry

theorem upstream_downstream_time_ratio :
  time_upstream / time_downstream = 2 := by sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l411_41188


namespace NUMINAMATH_CALUDE_last_digit_theorem_l411_41132

-- Define the property for the last digit of powers
def last_digit_property (a n k : ℕ) : Prop :=
  a^(4*n + k) % 10 = a^k % 10

-- Define the sum of specific powers
def sum_of_powers : ℕ :=
  (2^1997 + 3^1997 + 7^1997 + 9^1997) % 10

-- Theorem statement
theorem last_digit_theorem :
  (∀ (a n k : ℕ), last_digit_property a n k) ∧
  sum_of_powers = 1 := by
sorry

end NUMINAMATH_CALUDE_last_digit_theorem_l411_41132


namespace NUMINAMATH_CALUDE_pierre_cake_consumption_l411_41131

theorem pierre_cake_consumption (total_weight : ℝ) (parts : ℕ) 
  (h1 : total_weight = 546)
  (h2 : parts = 12)
  (h3 : parts > 0) :
  let nathalie_portion := total_weight / parts
  let pierre_portion := 2.5 * nathalie_portion
  pierre_portion = 113.75 := by sorry

end NUMINAMATH_CALUDE_pierre_cake_consumption_l411_41131


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l411_41128

theorem polygon_interior_angle_sum (n : ℕ) (h : n * 40 = 360) : 
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l411_41128


namespace NUMINAMATH_CALUDE_sum_of_alpha_beta_l411_41197

/-- Given constants α and β satisfying the rational equation, prove their sum is 176 -/
theorem sum_of_alpha_beta (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102*x + 2021) / (x^2 + 89*x - 3960)) : 
  α + β = 176 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_alpha_beta_l411_41197


namespace NUMINAMATH_CALUDE_tangent_and_perpendicular_l411_41138

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x + y + 2 = 0

theorem tangent_and_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve f
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is f'(x₀)
    (3 : ℝ) = -f' x₀ ∧
    -- The given line and tangent line are perpendicular
    (2 : ℝ) * (3 : ℝ) = -(6 : ℝ) * (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_perpendicular_l411_41138


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l411_41199

theorem unique_four_digit_number : ∃! (abcd : ℕ), 
  (1000 ≤ abcd ∧ abcd < 10000) ∧  -- 4-digit number
  (abcd % 11 = 0) ∧  -- multiple of 11
  (((abcd / 1000) * 10 + ((abcd / 100) % 10)) % 7 = 0) ∧  -- ac is multiple of 7
  ((abcd / 1000) + ((abcd / 100) % 10) + ((abcd / 10) % 10) + (abcd % 10) = (abcd % 10)^2) ∧  -- sum of digits equals square of last digit
  abcd = 3454 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l411_41199


namespace NUMINAMATH_CALUDE_vector_angle_problem_l411_41107

/-- The angle between two 2D vectors -/
def angle_between (v w : ℝ × ℝ) : ℝ := sorry

/-- Converts degrees to radians -/
def deg_to_rad (deg : ℝ) : ℝ := sorry

theorem vector_angle_problem (a b : ℝ × ℝ) 
  (sum_eq : a.1 + b.1 = 2 ∧ a.2 + b.2 = -1)
  (a_eq : a = (1, 2)) :
  angle_between a b = deg_to_rad 135 := by sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l411_41107


namespace NUMINAMATH_CALUDE_equation_solution_l411_41127

theorem equation_solution (a x : ℚ) : 
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ 
  ((x + a) / 9 - (1 - 3 * x) / 12 = 1) →
  a = 65 / 11 ∧ x = 13 / 11 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l411_41127


namespace NUMINAMATH_CALUDE_min_tiles_needed_is_260_l411_41123

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of the tile -/
def tileDimensions : Dimensions := ⟨2, 5⟩

/-- The dimensions of the floor in feet -/
def floorDimensionsFeet : Dimensions := ⟨3, 6⟩

/-- The dimensions of the floor in inches -/
def floorDimensionsInches : Dimensions :=
  ⟨feetToInches floorDimensionsFeet.length, feetToInches floorDimensionsFeet.width⟩

/-- Calculates the minimum number of tiles needed to cover the floor -/
def minTilesNeeded : ℕ :=
  (area floorDimensionsInches + area tileDimensions - 1) / area tileDimensions

theorem min_tiles_needed_is_260 : minTilesNeeded = 260 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_needed_is_260_l411_41123


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l411_41173

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth fencing_cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * fencing_cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_calculation :
  let length : ℝ := 75
  let breadth : ℝ := 25
  let fencing_cost_per_meter : ℝ := 26.50
  (length = breadth + 50) →
  total_fencing_cost length breadth fencing_cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 75 25 26.50

end NUMINAMATH_CALUDE_fencing_cost_calculation_l411_41173


namespace NUMINAMATH_CALUDE_negation_equivalence_l411_41109

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l411_41109


namespace NUMINAMATH_CALUDE_smallest_nine_digit_divisible_by_11_l411_41147

theorem smallest_nine_digit_divisible_by_11 : ℕ :=
  let n := 100000010
  have h1 : n ≥ 100000000 ∧ n < 1000000000 := by sorry
  have h2 : n % 11 = 0 := by sorry
  have h3 : ∀ m : ℕ, m ≥ 100000000 ∧ m < n → m % 11 ≠ 0 := by sorry
  n

end NUMINAMATH_CALUDE_smallest_nine_digit_divisible_by_11_l411_41147


namespace NUMINAMATH_CALUDE_combined_shoe_size_l411_41149

-- Define Jasmine's shoe size
def jasmine_size : ℕ := 7

-- Define the relationship between Alexa's and Jasmine's shoe sizes
def alexa_size : ℕ := 2 * jasmine_size

-- Define the combined shoe size
def combined_size : ℕ := jasmine_size + alexa_size

-- Theorem to prove
theorem combined_shoe_size : combined_size = 21 := by
  sorry

end NUMINAMATH_CALUDE_combined_shoe_size_l411_41149


namespace NUMINAMATH_CALUDE_divide_multiply_result_l411_41135

theorem divide_multiply_result (x : ℝ) (h : x = 4.5) : (x / 6) * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_result_l411_41135


namespace NUMINAMATH_CALUDE_length_of_BC_l411_41151

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) (b c : ℝ) : Prop :=
  t.A = (0, 0) ∧
  t.B = (-b, parabola (-b)) ∧
  t.C = (c, parabola c) ∧
  b > 0 ∧
  c > 0 ∧
  t.B.2 = t.C.2 ∧  -- BC is parallel to x-axis
  (1/2 * (c + b) * (parabola (-b))) = 96  -- Area of the triangle is 96

-- Theorem to prove
theorem length_of_BC (t : Triangle) (b c : ℝ) 
  (h : triangle_conditions t b c) : 
  (t.C.1 - t.B.1) = 59/9 := by sorry

end NUMINAMATH_CALUDE_length_of_BC_l411_41151


namespace NUMINAMATH_CALUDE_rulers_added_l411_41169

theorem rulers_added (initial_rulers : ℕ) (final_rulers : ℕ) (added_rulers : ℕ) : 
  initial_rulers = 46 → final_rulers = 71 → added_rulers = final_rulers - initial_rulers → 
  added_rulers = 25 := by
  sorry

end NUMINAMATH_CALUDE_rulers_added_l411_41169


namespace NUMINAMATH_CALUDE_flea_misses_point_l411_41177

/-- Represents the number of points on the circle. -/
def n : ℕ := 101

/-- Represents the position of the flea after k jumps. -/
def flea_position (k : ℕ) : ℕ := (k * (k + 1) / 2) % n

/-- States that there exists a point that the flea never lands on. -/
theorem flea_misses_point : ∃ p : Fin n, ∀ k : ℕ, flea_position k ≠ p.val :=
sorry

end NUMINAMATH_CALUDE_flea_misses_point_l411_41177


namespace NUMINAMATH_CALUDE_sum_longest_altitudes_eq_21_l411_41100

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 9
  side_b : b = 12
  side_c : c = 15

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ :=
  t.a + t.b

theorem sum_longest_altitudes_eq_21 (t : RightTriangle) :
  sum_longest_altitudes t = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_longest_altitudes_eq_21_l411_41100


namespace NUMINAMATH_CALUDE_permutations_6_3_l411_41186

/-- The number of permutations of k elements chosen from a set of n elements -/
def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- Theorem: The number of permutations of 3 elements chosen from a set of 6 elements is 120 -/
theorem permutations_6_3 : permutations 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_6_3_l411_41186


namespace NUMINAMATH_CALUDE_cubic_root_cubes_l411_41141

/-- Given a cubic equation x^3 + ax^2 + bx + c = 0 with roots α, β, and γ,
    the cubic equation with roots α^3, β^3, and γ^3 is
    x^3 + (a^3 - 3ab + 3c)x^2 + (b^3 + 3c^2 - 3abc)x + c^3 -/
theorem cubic_root_cubes (a b c : ℝ) (α β γ : ℝ) :
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∀ x : ℝ, x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3 = 0
           ↔ x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_cubes_l411_41141


namespace NUMINAMATH_CALUDE_expression_simplification_l411_41163

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l411_41163


namespace NUMINAMATH_CALUDE_bumper_car_line_after_three_rounds_l411_41166

def bumper_car_line (initial_people : ℕ) (capacity : ℕ) (leave_once : ℕ) (priority_join : ℕ) (rounds : ℕ) : ℕ :=
  let first_round := initial_people - capacity - leave_once + priority_join
  let subsequent_rounds := first_round - (rounds - 1) * capacity + (rounds - 1) * priority_join
  subsequent_rounds

theorem bumper_car_line_after_three_rounds :
  bumper_car_line 30 5 10 5 3 = 20 := by sorry

end NUMINAMATH_CALUDE_bumper_car_line_after_three_rounds_l411_41166


namespace NUMINAMATH_CALUDE_black_ball_from_red_bag_impossible_l411_41143

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black

/-- Represents the contents of a bag -/
structure Bag where
  balls : List BallColor

/-- Defines an impossible event -/
def impossibleEvent (p : ℝ) : Prop := p = 0

/-- Theorem: Drawing a black ball from a bag with only red balls is an impossible event -/
theorem black_ball_from_red_bag_impossible (bag : Bag) 
    (h : ∀ b ∈ bag.balls, b = BallColor.Red) : 
  impossibleEvent (Nat.card {i | bag.balls.get? i = some BallColor.Black} / bag.balls.length) := by
  sorry

end NUMINAMATH_CALUDE_black_ball_from_red_bag_impossible_l411_41143


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l411_41184

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l411_41184


namespace NUMINAMATH_CALUDE_discount_problem_l411_41157

/-- Given a purchase with a 25% discount where the discount amount is $40, 
    prove that the total amount paid is $120. -/
theorem discount_problem (original_price : ℝ) (discount_rate : ℝ) (discount_amount : ℝ) (total_paid : ℝ) : 
  discount_rate = 0.25 →
  discount_amount = 40 →
  discount_amount = discount_rate * original_price →
  total_paid = original_price - discount_amount →
  total_paid = 120 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l411_41157


namespace NUMINAMATH_CALUDE_overall_profit_calculation_john_profit_is_50_l411_41167

/-- Calculates the overall profit from selling two items with given costs and profit/loss percentages -/
theorem overall_profit_calculation 
  (grinder_cost mobile_cost : ℕ) 
  (grinder_loss_percent mobile_profit_percent : ℚ) : ℕ :=
  let grinder_selling_price := grinder_cost - (grinder_cost * grinder_loss_percent).floor
  let mobile_selling_price := mobile_cost + (mobile_cost * mobile_profit_percent).ceil
  let total_selling_price := grinder_selling_price + mobile_selling_price
  let total_cost := grinder_cost + mobile_cost
  (total_selling_price - total_cost).toNat

/-- Proves that given the specific costs and percentages, the overall profit is 50 -/
theorem john_profit_is_50 : 
  overall_profit_calculation 15000 8000 (5/100) (10/100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_calculation_john_profit_is_50_l411_41167


namespace NUMINAMATH_CALUDE_tan_315_degrees_l411_41168

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l411_41168


namespace NUMINAMATH_CALUDE_pokemon_card_ratio_l411_41120

theorem pokemon_card_ratio (mark_cards lloyd_cards michael_cards : ℕ) : 
  mark_cards = lloyd_cards →
  mark_cards = michael_cards - 10 →
  michael_cards = 100 →
  mark_cards + lloyd_cards + michael_cards + 80 = 300 →
  mark_cards = lloyd_cards :=
by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_ratio_l411_41120
