import Mathlib

namespace NUMINAMATH_GPT_n_is_one_sixth_sum_of_list_l59_5952

-- Define the condition that n is 4 times the average of the other 20 numbers
def satisfies_condition (n : ℝ) (l : List ℝ) : Prop :=
  l.length = 21 ∧
  n ∈ l ∧
  n = 4 * (l.erase n).sum / 20

-- State the main theorem
theorem n_is_one_sixth_sum_of_list {n : ℝ} {l : List ℝ} (h : satisfies_condition n l) :
  n = (1 / 6) * l.sum :=
by
  sorry

end NUMINAMATH_GPT_n_is_one_sixth_sum_of_list_l59_5952


namespace NUMINAMATH_GPT_median_of_first_ten_positive_integers_l59_5927

def first_ten_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem median_of_first_ten_positive_integers : 
  ∃ median : ℝ, median = 5.5 := by
  sorry

end NUMINAMATH_GPT_median_of_first_ten_positive_integers_l59_5927


namespace NUMINAMATH_GPT_multiply_binomials_l59_5956

theorem multiply_binomials :
  ∀ (x : ℝ), 
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 :=
by
  sorry

end NUMINAMATH_GPT_multiply_binomials_l59_5956


namespace NUMINAMATH_GPT_x_less_than_y_by_35_percent_l59_5967

noncomputable def percentage_difference (x y : ℝ) : ℝ :=
  ((y / x) - 1) * 100

theorem x_less_than_y_by_35_percent (x y : ℝ) (h : y = 1.5384615384615385 * x) :
  percentage_difference x y = 53.846153846153854 :=
by
  sorry

end NUMINAMATH_GPT_x_less_than_y_by_35_percent_l59_5967


namespace NUMINAMATH_GPT_calc_3a2008_minus_5b2008_l59_5974

theorem calc_3a2008_minus_5b2008 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : 3 * a ^ 2008 - 5 * b ^ 2008 = -5 :=
by
  sorry

end NUMINAMATH_GPT_calc_3a2008_minus_5b2008_l59_5974


namespace NUMINAMATH_GPT_impossible_relationships_l59_5902

theorem impossible_relationships (a b : ℝ) (h : (1 / a) = (1 / b)) :
  (¬ (0 < a ∧ a < b)) ∧ (¬ (b < a ∧ a < 0)) :=
by
  sorry

end NUMINAMATH_GPT_impossible_relationships_l59_5902


namespace NUMINAMATH_GPT_correct_operation_l59_5915

variable (a b : ℝ)

theorem correct_operation :
  ¬ (a^2 + a^3 = a^5) ∧
  ¬ ((a^2)^3 = a^5) ∧
  ¬ (a^2 * a^3 = a^6) ∧
  ((-a * b)^5 / (-a * b)^3 = a^2 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l59_5915


namespace NUMINAMATH_GPT_value_of_clothing_piece_eq_l59_5942

def annual_remuneration := 10
def work_months := 7
def received_silver_coins := 2

theorem value_of_clothing_piece_eq : 
  ∃ x : ℝ, (x + received_silver_coins) * 12 = (x + annual_remuneration) * work_months → x = 9.2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_clothing_piece_eq_l59_5942


namespace NUMINAMATH_GPT_ellipse_a_value_l59_5986

theorem ellipse_a_value
  (a : ℝ)
  (h1 : 0 < a)
  (h2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1)
  (e : ℝ)
  (h3 : e = 2 / 3)
  : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_a_value_l59_5986


namespace NUMINAMATH_GPT_revision_cost_per_page_is_4_l59_5957

-- Definitions based on conditions
def initial_cost_per_page := 6
def total_pages := 100
def revised_once_pages := 35
def revised_twice_pages := 15
def no_revision_pages := total_pages - revised_once_pages - revised_twice_pages
def total_cost := 860

-- Theorem to be proved
theorem revision_cost_per_page_is_4 : 
  ∃ x : ℝ, 
    ((initial_cost_per_page * total_pages) + 
     (revised_once_pages * x) + 
     (revised_twice_pages * (2 * x)) = total_cost) ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_revision_cost_per_page_is_4_l59_5957


namespace NUMINAMATH_GPT_inradius_one_third_height_l59_5979

-- The problem explicitly states this triangle's sides form an arithmetic progression.
-- We need to define conditions and then prove the question is equivalent to the answer given those conditions.
theorem inradius_one_third_height (a b c r h_b : ℝ) (h : a ≤ b ∧ b ≤ c) (h_arith : 2 * b = a + c) :
  r = h_b / 3 :=
sorry

end NUMINAMATH_GPT_inradius_one_third_height_l59_5979


namespace NUMINAMATH_GPT_geometric_to_arithmetic_common_ratio_greater_than_1_9_l59_5951

theorem geometric_to_arithmetic (q : ℝ) (h : q = (1 + Real.sqrt 5) / 2) :
  ∃ (a b c : ℝ), b - a = c - b ∧ a / b = b / c := 
sorry

theorem common_ratio_greater_than_1_9 (q : ℝ) (h_pos : q > 1.9 ∧ q < 2) :
  ∃ (n : ℕ), q^(n+1) - 2 * q^n + 1 = 0 :=
sorry

end NUMINAMATH_GPT_geometric_to_arithmetic_common_ratio_greater_than_1_9_l59_5951


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l59_5971

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℝ) (d : ℝ) (h_nonzero : d ≠ 0) 
  (h_a3 : a 3 = 7)
  (h_geo_seq : (a 2 - 1)^2 = (a 1 - 1) * (a 4 - 1)) : 
  a 10 = 21 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l59_5971


namespace NUMINAMATH_GPT_find_a_l59_5948

-- Definitions matching the conditions
def seq (a b c d : ℤ) := [a, b, c, d, 0, 1, 1, 2, 3, 5, 8]

-- Conditions provided in the problem
def fib_property (a b c d : ℤ) : Prop :=
    d + 0 = 1 ∧ 
    c + 1 = 0 ∧ 
    b + (-1) = 1 ∧ 
    a + 2 = -1

-- Theorem statement to prove
theorem find_a (a b c d : ℤ) (h : fib_property a b c d) : a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l59_5948


namespace NUMINAMATH_GPT_select_from_companyA_l59_5947

noncomputable def companyA_representatives : ℕ := 40
noncomputable def companyB_representatives : ℕ := 60
noncomputable def total_representatives : ℕ := companyA_representatives + companyB_representatives
noncomputable def sample_size : ℕ := 10
noncomputable def sampling_ratio : ℚ := sample_size / total_representatives
noncomputable def selected_from_companyA : ℚ := companyA_representatives * sampling_ratio

theorem select_from_companyA : selected_from_companyA = 4 := by
  sorry


end NUMINAMATH_GPT_select_from_companyA_l59_5947


namespace NUMINAMATH_GPT_probability_of_head_l59_5920

def events : Type := {e // e = "H" ∨ e = "T"}

def equallyLikely (e : events) : Prop :=
  e = ⟨"H", Or.inl rfl⟩ ∨ e = ⟨"T", Or.inr rfl⟩

def totalOutcomes := 2

def probOfHead : ℚ := 1 / totalOutcomes

theorem probability_of_head : probOfHead = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_head_l59_5920


namespace NUMINAMATH_GPT_jane_received_change_l59_5931

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end NUMINAMATH_GPT_jane_received_change_l59_5931


namespace NUMINAMATH_GPT_smallest_number_collected_l59_5965

-- Define the numbers collected by each person according to the conditions
def jungkook : ℕ := 6 * 3
def yoongi : ℕ := 4
def yuna : ℕ := 5

-- The statement to prove
theorem smallest_number_collected : yoongi = min (min jungkook yoongi) yuna :=
by sorry

end NUMINAMATH_GPT_smallest_number_collected_l59_5965


namespace NUMINAMATH_GPT_ratio_black_white_extended_pattern_l59_5919

def originalBlackTiles : ℕ := 8
def originalWhiteTiles : ℕ := 17
def originalSquareSide : ℕ := 5
def extendedSquareSide : ℕ := 7
def newBlackTiles : ℕ := (extendedSquareSide * extendedSquareSide) - (originalSquareSide * originalSquareSide)
def totalBlackTiles : ℕ := originalBlackTiles + newBlackTiles
def totalWhiteTiles : ℕ := originalWhiteTiles

theorem ratio_black_white_extended_pattern : totalBlackTiles / totalWhiteTiles = 32 / 17 := sorry

end NUMINAMATH_GPT_ratio_black_white_extended_pattern_l59_5919


namespace NUMINAMATH_GPT_common_area_of_rectangle_and_circle_eqn_l59_5906

theorem common_area_of_rectangle_and_circle_eqn :
  let rect_length := 8
  let rect_width := 4
  let circle_radius := 3
  let common_area := (3^2 * 2 * Real.pi / 4) - 2 * Real.sqrt 5  
  common_area = (9 * Real.pi / 2) - 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_common_area_of_rectangle_and_circle_eqn_l59_5906


namespace NUMINAMATH_GPT_onewaynia_road_closure_l59_5926

variable {V : Type} -- Denoting the type of cities
variable (G : V → V → Prop) -- G represents the directed graph

-- Conditions
variables (outdegree : V → Nat) (indegree : V → Nat)
variables (two_ways : ∀ (u v : V), u ≠ v → ¬(G u v ∧ G v u))
variables (two_out : ∀ v : V, outdegree v = 2)
variables (two_in : ∀ v : V, indegree v = 2)

theorem onewaynia_road_closure:
  ∃ n : Nat, n ≥ 1 ∧ (number_of_closures : Nat) = 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_onewaynia_road_closure_l59_5926


namespace NUMINAMATH_GPT_simplify_expression_l59_5941

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l59_5941


namespace NUMINAMATH_GPT_max_pies_without_ingredients_l59_5914

theorem max_pies_without_ingredients :
  let total_pies := 48
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 2
  let cayenne_pies := 3 * total_pies / 8
  let soy_nut_pies := total_pies / 8
  total_pies - max chocolate_pies (max marshmallow_pies (max cayenne_pies soy_nut_pies)) = 24 := by
{
  sorry
}

end NUMINAMATH_GPT_max_pies_without_ingredients_l59_5914


namespace NUMINAMATH_GPT_triangle_shortest_side_condition_l59_5987

theorem triangle_shortest_side_condition
  (A B C : Type) 
  (r : ℝ) (AF FB : ℝ)
  (P : ℝ)
  (h_AF : AF = 7)
  (h_FB : FB = 9)
  (h_r : r = 5)
  (h_P : P = 46) 
  : (min (min (7 + 9) (2 * 14)) ((7 + 9) - 14)) = 2 := 
by sorry

end NUMINAMATH_GPT_triangle_shortest_side_condition_l59_5987


namespace NUMINAMATH_GPT_sonnets_not_read_l59_5937

-- Define the conditions in the original problem
def sonnet_lines := 14
def unheard_lines := 70

-- Define a statement that needs to be proven
-- Prove that the number of sonnets not read is 5
theorem sonnets_not_read : unheard_lines / sonnet_lines = 5 := by
  sorry

end NUMINAMATH_GPT_sonnets_not_read_l59_5937


namespace NUMINAMATH_GPT_sum_of_n_for_perfect_square_l59_5970

theorem sum_of_n_for_perfect_square (n : ℕ) (Sn : ℕ) 
  (hSn : Sn = n^2 + 20 * n + 12) 
  (hn : n > 0) :
  ∃ k : ℕ, k^2 = Sn → (sum_of_possible_n = 16) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_n_for_perfect_square_l59_5970


namespace NUMINAMATH_GPT_base_s_computation_l59_5908

theorem base_s_computation (s : ℕ) (h : 550 * s + 420 * s = 1100 * s) : s = 7 := by
  sorry

end NUMINAMATH_GPT_base_s_computation_l59_5908


namespace NUMINAMATH_GPT_range_of_m_l59_5943

noncomputable def inequality_solutions (x m : ℝ) := |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) : (∃ x : ℝ, inequality_solutions x m) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l59_5943


namespace NUMINAMATH_GPT_first_ring_time_l59_5928

-- Define the properties of the clock
def rings_every_three_hours : Prop := ∀ n : ℕ, 3 * n < 24
def rings_eight_times_a_day : Prop := ∀ n : ℕ, n = 8 → 3 * n = 24

-- The theorem statement
theorem first_ring_time : rings_every_three_hours → rings_eight_times_a_day → (∀ n : ℕ, n = 1 → 3 * n = 3) := 
    sorry

end NUMINAMATH_GPT_first_ring_time_l59_5928


namespace NUMINAMATH_GPT_intersection_A_B_l59_5975

open Set

def f (x : ℕ) : ℕ := x^2 - 12 * x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a, a ∈ A ∧ b = f a}

theorem intersection_A_B : A ∩ B = {1, 4, 9} :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_intersection_A_B_l59_5975


namespace NUMINAMATH_GPT_range_of_m_l59_5980

noncomputable def f (x m : ℝ) : ℝ := -x^2 - 4 * m * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x1 x2 : ℝ, 2 ≤ x1 → x1 ≤ x2 → f x1 m ≥ f x2 m) ↔ m ≥ -1 := 
sorry

end NUMINAMATH_GPT_range_of_m_l59_5980


namespace NUMINAMATH_GPT_part1_part2_l59_5950

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x + a - 1) + abs (x - 2 * a)

-- Part (1) of the proof problem
theorem part1 (a : ℝ) : f 1 a < 3 → - (2 : ℝ)/3 < a ∧ a < 4 / 3 := sorry

-- Part (2) of the proof problem
theorem part2 (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := sorry

end NUMINAMATH_GPT_part1_part2_l59_5950


namespace NUMINAMATH_GPT_function_characterization_l59_5973

theorem function_characterization (f : ℤ → ℤ)
  (h : ∀ a b : ℤ, ∃ k : ℤ, f (f a - b) + b * f (2 * a) = k ^ 2) :
  (∀ n : ℤ, (n % 2 = 0 → f n = 0) ∧ (n % 2 ≠ 0 → ∃ k: ℤ, f n = k ^ 2))
  ∨ (∀ n : ℤ, ∃ k: ℤ, f n = k ^ 2 ∧ k = n) :=
sorry

end NUMINAMATH_GPT_function_characterization_l59_5973


namespace NUMINAMATH_GPT_roots_polynomial_pq_sum_l59_5925

theorem roots_polynomial_pq_sum :
  ∀ p q : ℝ, 
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) * (x - 4) = x^4 - 10 * x^3 + p * x^2 - q * x + 24) 
  → p + q = 85 :=
by 
  sorry

end NUMINAMATH_GPT_roots_polynomial_pq_sum_l59_5925


namespace NUMINAMATH_GPT_markup_percentage_l59_5988

variable (W R : ℝ)

-- Condition: When sold at a 40% discount, a sweater nets the merchant a 30% profit on the wholesale cost.
def discount_condition : Prop := 0.6 * R = 1.3 * W

-- Theorem: The percentage markup of the sweater from wholesale to normal retail price is 116.67%
theorem markup_percentage (h : discount_condition W R) : (R - W) / W * 100 = 116.67 :=
by sorry

end NUMINAMATH_GPT_markup_percentage_l59_5988


namespace NUMINAMATH_GPT_pyramid_bottom_right_value_l59_5997

theorem pyramid_bottom_right_value (a x y z b : ℕ) (h1 : 18 = (21 + x) / 2)
  (h2 : 14 = (21 + y) / 2) (h3 : 16 = (15 + z) / 2) (h4 : b = (21 + y) / 2) :
  a = 6 := 
sorry

end NUMINAMATH_GPT_pyramid_bottom_right_value_l59_5997


namespace NUMINAMATH_GPT_ellipse_standard_and_trajectory_l59_5994

theorem ellipse_standard_and_trajectory :
  ∀ a b x y : ℝ, 
  a > b ∧ 0 < b ∧ 
  (b^2 = a^2 - 1) ∧ 
  (9/4 + 6/(8) = 1) →
  (∃ x y : ℝ, (x / 2)^2 / 9 + (y)^2 / 8 = 1) ∧ 
  (x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3) := 
  sorry

end NUMINAMATH_GPT_ellipse_standard_and_trajectory_l59_5994


namespace NUMINAMATH_GPT_sixth_power_sum_l59_5916

theorem sixth_power_sum (a b c d e f : ℤ) :
  a^6 + b^6 + c^6 + d^6 + e^6 + f^6 = 6 * a * b * c * d * e * f + 1 → 
  (a = 1 ∨ a = -1 ∨ b = 1 ∨ b = -1 ∨ c = 1 ∨ c = -1 ∨ 
   d = 1 ∨ d = -1 ∨ e = 1 ∨ e = -1 ∨ f = 1 ∨ f = -1) ∧
  ((a = 1 ∨ a = -1 ∨ a = 0) ∧ 
   (b = 1 ∨ b = -1 ∨ b = 0) ∧ 
   (c = 1 ∨ c = -1 ∨ c = 0) ∧ 
   (d = 1 ∨ d = -1 ∨ d = 0) ∧ 
   (e = 1 ∨ e = -1 ∨ e = 0) ∧ 
   (f = 1 ∨ f = -1 ∨ f = 0)) ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0 ∨ f ≠ 0) ∧
  (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0 ∨ f = 0) := 
sorry

end NUMINAMATH_GPT_sixth_power_sum_l59_5916


namespace NUMINAMATH_GPT_supplement_of_complement_of_30_degrees_l59_5921

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α
def α : ℝ := 30

theorem supplement_of_complement_of_30_degrees : supplement (complement α) = 120 := 
by
  sorry

end NUMINAMATH_GPT_supplement_of_complement_of_30_degrees_l59_5921


namespace NUMINAMATH_GPT_lorelai_jellybeans_correct_l59_5949

-- Define the number of jellybeans Gigi has
def gigi_jellybeans : Nat := 15

-- Define the number of additional jellybeans Rory has compared to Gigi
def rory_additional_jellybeans : Nat := 30

-- Define the number of jellybeans both girls together have
def total_jellybeans : Nat := gigi_jellybeans + (gigi_jellybeans + rory_additional_jellybeans)

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : Nat := 3 * total_jellybeans

-- The theorem to prove the number of jellybeans Lorelai has eaten is 180
theorem lorelai_jellybeans_correct : lorelai_jellybeans = 180 := by
  sorry

end NUMINAMATH_GPT_lorelai_jellybeans_correct_l59_5949


namespace NUMINAMATH_GPT_parallel_vectors_implies_scalar_l59_5991

-- Defining the vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Stating the condition and required proof
theorem parallel_vectors_implies_scalar (m : ℝ) (h : (vector_a.snd / vector_a.fst) = (vector_b m).snd / (vector_b m).fst) : m = -4 :=
by sorry

end NUMINAMATH_GPT_parallel_vectors_implies_scalar_l59_5991


namespace NUMINAMATH_GPT_weights_problem_l59_5938

theorem weights_problem (n : ℕ) (x : ℝ) (h_avg : ∀ (i : ℕ), i < n → ∃ (w : ℝ), w = x) 
  (h_heaviest : ∃ (w_max : ℝ), w_max = 5 * x) : n > 5 :=
by
  sorry

end NUMINAMATH_GPT_weights_problem_l59_5938


namespace NUMINAMATH_GPT_four_thirds_of_nine_halves_l59_5929

theorem four_thirds_of_nine_halves :
  (4 / 3) * (9 / 2) = 6 := 
sorry

end NUMINAMATH_GPT_four_thirds_of_nine_halves_l59_5929


namespace NUMINAMATH_GPT_typing_time_in_hours_l59_5912

def words_per_minute := 32
def word_count := 7125
def break_interval := 25
def break_time := 5
def mistake_interval := 100
def correction_time_per_mistake := 1

theorem typing_time_in_hours :
  let typing_time := (word_count + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let total_break_time := breaks * break_time
  let mistakes := (word_count + mistake_interval - 1) / mistake_interval
  let total_correction_time := mistakes * correction_time_per_mistake
  let total_time := typing_time + total_break_time + total_correction_time
  let total_hours := (total_time + 60 - 1) / 60
  total_hours = 6 :=
by
  sorry

end NUMINAMATH_GPT_typing_time_in_hours_l59_5912


namespace NUMINAMATH_GPT_range_of_values_includes_one_integer_l59_5977

theorem range_of_values_includes_one_integer (x : ℝ) (h : -1 < 2 * x + 3 ∧ 2 * x + 3 < 1) :
  ∃! n : ℤ, -7 < (2 * x - 3) ∧ (2 * x - 3) < -5 ∧ n = -6 :=
sorry

end NUMINAMATH_GPT_range_of_values_includes_one_integer_l59_5977


namespace NUMINAMATH_GPT_length_of_plot_l59_5932

theorem length_of_plot (W P C r : ℝ) (hW : W = 65) (hP : P = 2.5) (hC : C = 340) (hr : r = 0.4) :
  let L := (C / r - (W + 2 * P) * P) / (W - 2 * P)
  L = 100 :=
by
  sorry

end NUMINAMATH_GPT_length_of_plot_l59_5932


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_square_l59_5946

theorem arithmetic_sequence_sum_square (a d : ℕ) :
  (∀ n : ℕ, ∃ k : ℕ, n * (a + (n-1) * d / 2) = k * k) ↔ (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_square_l59_5946


namespace NUMINAMATH_GPT_max_value_expression_l59_5935

theorem max_value_expression : ∃ (max_val : ℝ), max_val = (1 / 16) ∧ ∀ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → (a - b^2) * (b - a^2) ≤ max_val :=
by
  sorry

end NUMINAMATH_GPT_max_value_expression_l59_5935


namespace NUMINAMATH_GPT_evaluate_expression_l59_5910

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l59_5910


namespace NUMINAMATH_GPT_solve_B_share_l59_5918

def ratio_shares (A B C : ℚ) : Prop :=
  A = 1/2 ∧ B = 1/3 ∧ C = 1/4

def initial_capitals (total_capital : ℚ) (A_s B_s C_s : ℚ) : Prop :=
  A_s = 1/2 * total_capital ∧ B_s = 1/3 * total_capital ∧ C_s = 1/4 * total_capital

def total_capital_contribution (A_contrib B_contrib C_contrib : ℚ) : Prop :=
  A_contrib = 42 ∧ B_contrib = 48 ∧ C_contrib = 36

def B_share (B_contrib total_contrib profit : ℚ) : ℚ := 
  (B_contrib / total_contrib) * profit

theorem solve_B_share : 
  ∀ (A_s B_s C_s total_capital profit A_contrib B_contrib C_contrib total_contrib : ℚ),
  ratio_shares (1/2) (1/3) (1/4) →
  initial_capitals total_capital A_s B_s C_s →
  total_capital_contribution A_contrib B_contrib C_contrib →
  total_contrib = A_contrib + B_contrib + C_contrib →
  profit = 378 →
  B_s = (1/3) * total_capital →
  B_contrib = 48 →
  B_share B_contrib total_contrib profit = 108 := by 
    sorry

end NUMINAMATH_GPT_solve_B_share_l59_5918


namespace NUMINAMATH_GPT_carol_rectangle_length_l59_5958

theorem carol_rectangle_length :
  let j_length := 6
  let j_width := 30
  let c_width := 15
  let c_length := j_length * j_width / c_width
  c_length = 12 := by
  sorry

end NUMINAMATH_GPT_carol_rectangle_length_l59_5958


namespace NUMINAMATH_GPT_june_ride_time_l59_5966

theorem june_ride_time (dist1 time1 dist2 time2 : ℝ) (h : dist1 = 2 ∧ time1 = 8 ∧ dist2 = 5 ∧ time2 = 20) :
  (dist2 / (dist1 / time1) = time2) := by
  -- using the defined conditions
  rcases h with ⟨h1, h2, h3, h4⟩
  rw [h1, h2, h3, h4]
  -- simplifying the expression
  sorry

end NUMINAMATH_GPT_june_ride_time_l59_5966


namespace NUMINAMATH_GPT_find_d_l59_5940

theorem find_d (d : ℚ) (int_part frac_part : ℚ) 
  (h1 : 3 * int_part^2 + 19 * int_part - 28 = 0)
  (h2 : 4 * frac_part^2 - 11 * frac_part + 3 = 0)
  (h3 : frac_part ≥ 0 ∧ frac_part < 1)
  (h4 : d = int_part + frac_part) :
  d = -29 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l59_5940


namespace NUMINAMATH_GPT_no_odd_tens_digit_in_square_l59_5939

theorem no_odd_tens_digit_in_square (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n > 0) (h₃ : n < 100) : 
  (n * n / 10) % 10 % 2 = 0 := 
sorry

end NUMINAMATH_GPT_no_odd_tens_digit_in_square_l59_5939


namespace NUMINAMATH_GPT_dubblefud_red_balls_l59_5955

theorem dubblefud_red_balls (R B : ℕ) 
  (h1 : 2 ^ R * 4 ^ B * 5 ^ B = 16000)
  (h2 : B = G) : R = 6 :=
by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_dubblefud_red_balls_l59_5955


namespace NUMINAMATH_GPT_first_more_than_200_paperclips_day_l59_5903

-- Definitions based on the conditions:
def paperclips_on_day (k : ℕ) : ℕ :=
  3 * 2^k

-- The theorem stating the solution:
theorem first_more_than_200_paperclips_day :
  ∃ k : ℕ, paperclips_on_day k > 200 ∧ k = 7 :=
by
  use 7
  sorry

end NUMINAMATH_GPT_first_more_than_200_paperclips_day_l59_5903


namespace NUMINAMATH_GPT_factor_polynomial_l59_5904

theorem factor_polynomial (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := 
by 
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_factor_polynomial_l59_5904


namespace NUMINAMATH_GPT_painted_cells_possible_values_l59_5901

theorem painted_cells_possible_values (k l : ℕ) (hk : 2 * k + 1 > 0) (hl : 2 * l + 1 > 0) (h : k * l = 74) :
  (2 * k + 1) * (2 * l + 1) - 74 = 301 ∨ (2 * k + 1) * (2 * l + 1) - 74 = 373 := 
sorry

end NUMINAMATH_GPT_painted_cells_possible_values_l59_5901


namespace NUMINAMATH_GPT_machine_work_rates_l59_5995

theorem machine_work_rates :
  (∃ x : ℝ, (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2)) = 1 / x ∧ x = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_machine_work_rates_l59_5995


namespace NUMINAMATH_GPT_value_of_t_l59_5992

theorem value_of_t (x y t : ℝ) (hx : 2^x = t) (hy : 7^y = t) (hxy : 1/x + 1/y = 2) : t = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_GPT_value_of_t_l59_5992


namespace NUMINAMATH_GPT_coins_in_distinct_colors_l59_5945

theorem coins_in_distinct_colors 
  (n : ℕ)  (h1 : 1 < n) (h2 : n < 2010) : (∃ k : ℕ, 2010 = n * k) ↔ 
  ∀ i : ℕ, i < 2010 → (∃ f : ℕ → ℕ, ∀ j : ℕ, j < n → f (j + i) % n = j % n) :=
sorry

end NUMINAMATH_GPT_coins_in_distinct_colors_l59_5945


namespace NUMINAMATH_GPT_find_fx_l59_5984

variable {e : ℝ} {a : ℝ} (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (hodd : odd_function f)
variable (hdef : ∀ x, -e ≤ x → x < 0 → f x = a * x + Real.log (-x))

theorem find_fx (x : ℝ) (hx : 0 < x ∧ x ≤ e) : f x = a * x - Real.log x :=
by
  sorry

end NUMINAMATH_GPT_find_fx_l59_5984


namespace NUMINAMATH_GPT_binom_18_6_l59_5996

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end NUMINAMATH_GPT_binom_18_6_l59_5996


namespace NUMINAMATH_GPT_cos_2x_quadratic_l59_5985

theorem cos_2x_quadratic (x : ℝ) (a b c : ℝ)
  (h : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0)
  (h_a : a = 4) (h_b : b = 2) (h_c : c = -1) :
  4 * (Real.cos (2 * x)) ^ 2 + 2 * Real.cos (2 * x) - 1 = 0 := sorry

end NUMINAMATH_GPT_cos_2x_quadratic_l59_5985


namespace NUMINAMATH_GPT_car_transport_distance_l59_5954

theorem car_transport_distance
  (d_birdhouse : ℕ) 
  (d_lawnchair : ℕ) 
  (d_car : ℕ)
  (h1 : d_birdhouse = 1200)
  (h2 : d_birdhouse = 3 * d_lawnchair)
  (h3 : d_lawnchair = 2 * d_car) :
  d_car = 200 := 
by
  sorry

end NUMINAMATH_GPT_car_transport_distance_l59_5954


namespace NUMINAMATH_GPT_ball_weights_l59_5972

-- Define the weights of red and white balls we are going to use in our conditions and goal
variables (R W : ℚ)

-- State the conditions as hypotheses
axiom h1 : 7 * R + 5 * W = 43
axiom h2 : 5 * R + 7 * W = 47

-- State the theorem we want to prove, given the conditions
theorem ball_weights :
  4 * R + 8 * W = 49 :=
by
  sorry

end NUMINAMATH_GPT_ball_weights_l59_5972


namespace NUMINAMATH_GPT_chess_tournament_l59_5917

theorem chess_tournament (n games : ℕ) 
  (h_games : games = 81)
  (h_equation : (n - 2) * (n - 3) = 156) :
  n = 15 :=
sorry

end NUMINAMATH_GPT_chess_tournament_l59_5917


namespace NUMINAMATH_GPT_simplify_expression_l59_5900

theorem simplify_expression (x : ℝ) : (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l59_5900


namespace NUMINAMATH_GPT_sum_of_remainders_l59_5990

-- Definitions of the given problem
def a : ℕ := 1234567
def b : ℕ := 123

-- First remainder calculation
def r1 : ℕ := a % b

-- Second remainder calculation with the power
def r2 : ℕ := (2 ^ r1) % b

-- The proof statement
theorem sum_of_remainders : r1 + r2 = 29 := by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l59_5990


namespace NUMINAMATH_GPT_polynomial_transformation_l59_5981

noncomputable def f (x : ℝ) : ℝ := sorry

theorem polynomial_transformation (x : ℝ) :
  (f (x^2 + 2) = x^4 + 6 * x^2 + 4) →
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_transformation_l59_5981


namespace NUMINAMATH_GPT_remainder_3005_98_l59_5907

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end NUMINAMATH_GPT_remainder_3005_98_l59_5907


namespace NUMINAMATH_GPT_olivia_initial_quarters_l59_5964

theorem olivia_initial_quarters : 
  ∀ (spent_quarters left_quarters initial_quarters : ℕ),
  spent_quarters = 4 → left_quarters = 7 → initial_quarters = spent_quarters + left_quarters → initial_quarters = 11 :=
by
  intros spent_quarters left_quarters initial_quarters h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_olivia_initial_quarters_l59_5964


namespace NUMINAMATH_GPT_rr_sr_sum_le_one_l59_5959

noncomputable def rr_sr_le_one (r s : ℝ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : Prop :=
  r^r * s^s + r^s * s^r ≤ 1

theorem rr_sr_sum_le_one {r s : ℝ} (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : rr_sr_le_one r s h_pos_r h_pos_s h_sum :=
  sorry

end NUMINAMATH_GPT_rr_sr_sum_le_one_l59_5959


namespace NUMINAMATH_GPT_triangle_BC_length_l59_5961

theorem triangle_BC_length (A B C X : Type) 
  (AB AC : ℕ) (BX CX BC : ℕ)
  (h1 : AB = 100)
  (h2 : AC = 121)
  (h3 : ∃ x y : ℕ, x = BX ∧ y = CX ∧ AB = 100 ∧ x + y = BC)
  (h4 : x * y = 31 * 149 ∧ x + y = 149) :
  BC = 149 := 
by
  sorry

end NUMINAMATH_GPT_triangle_BC_length_l59_5961


namespace NUMINAMATH_GPT_contractor_absent_days_l59_5905

-- Definition of problem conditions
def total_days : ℕ := 30
def daily_wage : ℝ := 25
def daily_fine : ℝ := 7.5
def total_amount_received : ℝ := 620

-- Function to define the constraint equations
def equation1 (x y : ℕ) : Prop := x + y = total_days
def equation2 (x y : ℕ) : Prop := (daily_wage * x - daily_fine * y) = total_amount_received

-- The proof problem translation as Lean 4 statement
theorem contractor_absent_days (x y : ℕ) (h1 : equation1 x y) (h2 : equation2 x y) : y = 8 :=
by
  sorry

end NUMINAMATH_GPT_contractor_absent_days_l59_5905


namespace NUMINAMATH_GPT_jared_annual_earnings_l59_5993

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end NUMINAMATH_GPT_jared_annual_earnings_l59_5993


namespace NUMINAMATH_GPT_number_of_bouquets_l59_5982

theorem number_of_bouquets : ∃ n, n = 9 ∧ ∀ x y : ℕ, 3 * x + 2 * y = 50 → (x < 17) ∧ (x % 2 = 0 → y = (50 - 3 * x) / 2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_bouquets_l59_5982


namespace NUMINAMATH_GPT_badge_counts_l59_5960

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end NUMINAMATH_GPT_badge_counts_l59_5960


namespace NUMINAMATH_GPT_bella_truck_stamps_more_l59_5923

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end NUMINAMATH_GPT_bella_truck_stamps_more_l59_5923


namespace NUMINAMATH_GPT_jenny_hours_left_l59_5944

theorem jenny_hours_left
  (hours_research : ℕ)
  (hours_proposal : ℕ)
  (hours_total : ℕ)
  (h1 : hours_research = 10)
  (h2 : hours_proposal = 2)
  (h3 : hours_total = 20) :
  (hours_total - (hours_research + hours_proposal) = 8) :=
by
  sorry

end NUMINAMATH_GPT_jenny_hours_left_l59_5944


namespace NUMINAMATH_GPT_geom_sequence_a1_l59_5936

noncomputable def a_n (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n-1)

theorem geom_sequence_a1 {a1 q : ℝ} 
  (h1 : 0 < q)
  (h2 : a_n a1 q 4 * a_n a1 q 8 = 2 * (a_n a1 q 5)^2)
  (h3 : a_n a1 q 2 = 1) :
  a1 = (Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_geom_sequence_a1_l59_5936


namespace NUMINAMATH_GPT_earnings_proof_l59_5999

theorem earnings_proof (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 300) (h3 : C = 100) : A + C = 400 :=
sorry

end NUMINAMATH_GPT_earnings_proof_l59_5999


namespace NUMINAMATH_GPT_total_books_l59_5934

-- Define the number of books Stu has
def Stu_books : ℕ := 9

-- Define the multiplier for Albert's books
def Albert_multiplier : ℕ := 4

-- Define the number of books Albert has
def Albert_books : ℕ := Albert_multiplier * Stu_books

-- Prove that the total number of books is 45
theorem total_books:
  Stu_books + Albert_books = 45 :=
by 
  -- This is where the proof steps would go, but we skip it for now 
  sorry

end NUMINAMATH_GPT_total_books_l59_5934


namespace NUMINAMATH_GPT_steps_in_staircase_using_210_toothpicks_l59_5983

-- Define the conditions
def first_step : Nat := 3
def increment : Nat := 2
def total_toothpicks_5_steps : Nat := 55

-- Define required theorem
theorem steps_in_staircase_using_210_toothpicks : ∃ (n : ℕ), (n * (n + 2) = 210) ∧ n = 13 :=
by
  sorry

end NUMINAMATH_GPT_steps_in_staircase_using_210_toothpicks_l59_5983


namespace NUMINAMATH_GPT_square_area_l59_5911

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end NUMINAMATH_GPT_square_area_l59_5911


namespace NUMINAMATH_GPT_measured_diagonal_in_quadrilateral_l59_5976

-- Defining the conditions (side lengths and diagonals)
def valid_diagonal (side1 side2 side3 side4 diagonal : ℝ) : Prop :=
  side1 + side2 > diagonal ∧ side1 + side3 > diagonal ∧ side1 + side4 > diagonal ∧ 
  side2 + side3 > diagonal ∧ side2 + side4 > diagonal ∧ side3 + side4 > diagonal

theorem measured_diagonal_in_quadrilateral :
  let sides := [1, 2, 2.8, 5]
  let diagonal1 := 7.5
  let diagonal2 := 2.8
  (valid_diagonal 1 2 2.8 5 diagonal2) :=
sorry

end NUMINAMATH_GPT_measured_diagonal_in_quadrilateral_l59_5976


namespace NUMINAMATH_GPT_john_shots_l59_5913

theorem john_shots :
  let initial_shots := 30
  let initial_percentage := 0.60
  let additional_shots := 10
  let final_percentage := 0.58
  let made_initial := initial_percentage * initial_shots
  let total_shots := initial_shots + additional_shots
  let made_total := final_percentage * total_shots
  let made_additional := made_total - made_initial
  made_additional = 5 :=
by
  sorry

end NUMINAMATH_GPT_john_shots_l59_5913


namespace NUMINAMATH_GPT_shadow_area_greatest_integer_l59_5963

theorem shadow_area_greatest_integer (x : ℝ)
  (h1 : ∀ (a : ℝ), a = 1)
  (h2 : ∀ (b : ℝ), b = 48)
  (h3 : ∀ (c: ℝ), x = 1 / 6):
  ⌊1000 * x⌋ = 166 := 
by sorry

end NUMINAMATH_GPT_shadow_area_greatest_integer_l59_5963


namespace NUMINAMATH_GPT_find_x_l59_5962

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end NUMINAMATH_GPT_find_x_l59_5962


namespace NUMINAMATH_GPT_range_of_a_l59_5930

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l59_5930


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_a7_a3_a5_l59_5909

noncomputable def arithmetic_sequence_property (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_a1_a7_a3_a5 (a : ℕ → ℝ) (h_arith : arithmetic_sequence_property a)
  (h_cond : a 1 + a 7 = 10) : a 3 + a 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_a7_a3_a5_l59_5909


namespace NUMINAMATH_GPT_max_value_abs_cube_sum_l59_5922

theorem max_value_abs_cube_sum (x : Fin 5 → ℝ) (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (|x 0 - x 1|^3 + |x 1 - x 2|^3 + |x 2 - x 3|^3 + |x 3 - x 4|^3 + |x 4 - x 0|^3) ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_abs_cube_sum_l59_5922


namespace NUMINAMATH_GPT_ratio_induction_l59_5968

theorem ratio_induction (k : ℕ) (hk : k > 0) :
    (k + 2) * (k + 3) / (2 * (2 * k + 1)) = 1 := by
sorry

end NUMINAMATH_GPT_ratio_induction_l59_5968


namespace NUMINAMATH_GPT_athlete_total_heartbeats_l59_5969

theorem athlete_total_heartbeats (h : ℕ) (p : ℕ) (d : ℕ) (r : ℕ) : (h = 150) ∧ (p = 6) ∧ (d = 30) ∧ (r = 15) → (p * d + r) * h = 29250 :=
by
  sorry

end NUMINAMATH_GPT_athlete_total_heartbeats_l59_5969


namespace NUMINAMATH_GPT_find_share_of_A_l59_5953

variable (A B C : ℝ)
variable (h1 : A = (2/3) * B)
variable (h2 : B = (1/4) * C)
variable (h3 : A + B + C = 510)

theorem find_share_of_A : A = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_share_of_A_l59_5953


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l59_5989

theorem arithmetic_sequence_sum :
  (∀ (a : ℕ → ℤ),  a 1 + a 2 = 4 ∧ a 3 + a 4 = 6 → a 8 + a 9 = 10) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l59_5989


namespace NUMINAMATH_GPT_time_for_six_visits_l59_5933

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end NUMINAMATH_GPT_time_for_six_visits_l59_5933


namespace NUMINAMATH_GPT_arithmetic_seq_S11_l59_5978

def Sn (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1)) / 2 * d

theorem arithmetic_seq_S11 (a₁ d : ℤ)
  (h1 : a₁ = -11)
  (h2 : (Sn 10 a₁ d) / 10 - (Sn 8 a₁ d) / 8 = 2) :
  Sn 11 a₁ d = -11 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_S11_l59_5978


namespace NUMINAMATH_GPT_factor_polynomial_l59_5998

theorem factor_polynomial (x y z : ℂ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) := by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l59_5998


namespace NUMINAMATH_GPT_part_a_l59_5924

theorem part_a (a b : ℤ) (x : ℤ) :
  (x % 5 = a) ∧ (x % 6 = b) → x = 6 * a + 25 * b :=
by
  sorry

end NUMINAMATH_GPT_part_a_l59_5924
