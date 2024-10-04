import Mathlib

namespace camp_cedar_counselors_l638_638507

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h1 : boys = 40)
  (h2 : girls = 3 * boys)
  (h3 : total_children = boys + girls)
  (h4 : counselors = total_children / 8) : 
  counselors = 20 :=
by sorry

end camp_cedar_counselors_l638_638507


namespace triangle_is_isosceles_l638_638663

variables (A B C a b c : ℝ)
variables (triangle : c ≠ 0 ∧ a ≠ 0)

noncomputable def isosceles_triangle (A B C a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem triangle_is_isosceles 
  (h1 : triangle) 
  (h2 : b / a = (1 - cos B) / cos A) :
  isosceles_triangle A B C a b c :=
sorry

end triangle_is_isosceles_l638_638663


namespace find_fraction_l638_638958

theorem find_fraction {a b : ℕ} 
  (h1 : 32016 + (a / b) = 2016 * 3 + (a / b)) 
  (ha : a = 2016) 
  (hb : b = 2016^3 - 1) : 
  (b + 1) / a^2 = 2016 := 
by 
  sorry

end find_fraction_l638_638958


namespace roulette_P2007_gt_P2008_l638_638871

-- Define the roulette probability function based on the given conditions
noncomputable def roulette_probability : ℕ → ℝ
| 0 := 1
| n := (1 / 2007) * (List.foldl (λ acc k, acc + roulette_probability (n - k)) 0 (List.range 2007))

-- Define the theorem to prove P_{2007} > P_{2008}
theorem roulette_P2007_gt_P2008 : roulette_probability 2007 > roulette_probability 2008 :=
sorry

end roulette_P2007_gt_P2008_l638_638871


namespace team_win_requirement_l638_638881

theorem team_win_requirement 
  (first_games_won_percentage : ℕ → Prop)
  (remaining_games: ℕ) 
  (total_games_in_season: ℕ)
  (desired_win_percentage: ℕ) 
  (first_games_won: ℕ) 
  (total_games_won: ℕ) :
  (first_games_won_percentage 60) ∧ 
  remaining_games = 40 ∧ 
  total_games_in_season = 100 ∧ 
  desired_win_percentage = 75 ∧ 
  first_games_won = 36 ∧ 
  total_games_won = 75 → 
  sorry :=
begin
  -- To be filled in with the proof steps if necessary
  sorry
end

end team_win_requirement_l638_638881


namespace sum_floor_eq_l638_638565

theorem sum_floor_eq (n : ℕ) : ∑ k in (finset.range n), ⌊(n + 2^k) / (2^(k+1))⌋ = n := by sorry

end sum_floor_eq_l638_638565


namespace correct_statements_l638_638313

def α_terminal_side_same (k : ℤ) : Prop := 
  let α := (2 * k + 1) * 180
  let β := (4 * k ± 1) * 180
  α % 360 = β % 360

def M_subset_N : Prop :=
  let M := { x : ℤ | ∃ k : ℤ, x = 45 + k * 90 }
  let N := { y : ℤ | ∃ k : ℤ, y = 90 + k * 45 }
  ∀ x ∈ M, x ∈ N

theorem correct_statements :
  α_terminal_side_same ∧ M_subset_N :=
sorry

end correct_statements_l638_638313


namespace equation_of_perpendicular_line_through_point_l638_638312

theorem equation_of_perpendicular_line_through_point :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), (a = 3) ∧ (b = 1) ∧ (x - 2 * y - 3 = 0 → y = (-(1/2)) * x + 3/2) ∧ (2 * a + b - 7 = 0) := sorry

end equation_of_perpendicular_line_through_point_l638_638312


namespace greatest_divisor_remainders_l638_638355

/-- Greatest number that leaves the given remainders -/
theorem greatest_divisor_remainders :
  let d1 := 1557 - 7
  let d2 := 2037 - 5
  let gcd_d1_d2 := Nat.gcd d1 d2
  gcd_d1_d2 = 2 := 
by
  let d1 := 1557 - 7
  let d2 := 2037 - 5
  have h1 : d1 = 1550 := by rfl
  have h2 : d2 = 2032 := by rfl
  symm
  sorry

end greatest_divisor_remainders_l638_638355


namespace P2007_gt_P2008_l638_638869

namespace ProbabilityProblem

def probability (k : ℕ) : ℝ := sorry  -- Placeholder for the probability function

axiom probability_rec :
  ∀ n, probability n = (1 / 2007) * (∑ k in finset.range 2007, probability (n - (k + 1)))

axiom P0 :
  probability 0 = 1

theorem P2007_gt_P2008 : probability 2007 > probability 2008 := sorry

end ProbabilityProblem

end P2007_gt_P2008_l638_638869


namespace time_to_complete_project_alone_l638_638025

variable (A B : Type)
variable (day x : ℕ)
variable (work_rate_A : ℚ)
variable (work_rate_B : ℚ)

theorem time_to_complete_project_alone (hA : work_rate_A = 1 / x)
                                      (hB : work_rate_B = 1 / 30)
                                      (hAB_joint : ∀ (d : ℕ), d = 21 → 6 * (work_rate_A + work_rate_B) + 15 * work_rate_B = 1)
                                      (hx : ∀ (day : ℕ), day = 21 - 15 → true) :
  x = 20 := 
begin
  -- proof will go here
  sorry
end

end time_to_complete_project_alone_l638_638025


namespace sin_alpha_minus_beta_eq_sixteen_over_sixty_five_l638_638965

theorem sin_alpha_minus_beta_eq_sixteen_over_sixty_five
  (α β : ℝ)
  (hαβ_interval : α ∈ set.Ioo (π / 3) (5 * π / 6) ∧ β ∈ set.Ioo (π / 3) (5 * π / 6))
  (h1 : Real.sin (α + π / 6) = 4 / 5)
  (h2 : Real.cos (β - 5 * π / 6) = 5 / 13) :
  Real.sin (α - β) = 16 / 65 :=
sorry

end sin_alpha_minus_beta_eq_sixteen_over_sixty_five_l638_638965


namespace boundary_shadow_l638_638042

theorem boundary_shadow 
  (r : ℝ) (center : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ)
  (h_r : r = 2) (h_center : center = (0, 0, 2))
  (h_P : P = (0, -2, 3)) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, g x = (x^2 / 10) - (18 / 5)) := 
by
  use (λ x : ℝ, (x^2 / 10) - (18 / 5))
  intro x
  rfl

end boundary_shadow_l638_638042


namespace stella_profit_loss_l638_638747

theorem stella_profit_loss :
  let dolls := 6
  let clocks := 4
  let glasses := 8
  let vases := 3
  let postcards := 10
  let dolls_price := 8
  let clocks_price := 25
  let glasses_price := 6
  let vases_price := 12
  let postcards_price := 3
  let cost := 250
  let clocks_discount_threshold := 2
  let clocks_discount := 10 / 100
  let glasses_bundle := 3
  let glasses_bundle_price := 2 * glasses_price
  let sales_tax_rate := 5 / 100
  let dolls_revenue := dolls * dolls_price
  let clocks_revenue_full := clocks * clocks_price
  let clocks_discounts_count := clocks / clocks_discount_threshold
  let clocks_discount_amount := clocks_discounts_count * clocks_discount * clocks_discount_threshold * clocks_price
  let clocks_revenue := clocks_revenue_full - clocks_discount_amount
  let glasses_discount_quantity := glasses / glasses_bundle
  let glasses_revenue := (glasses - glasses_discount_quantity) * glasses_price
  let vases_revenue := vases * vases_price
  let postcards_revenue := postcards * postcards_price
  let total_revenue_without_discounts := dolls_revenue + clocks_revenue_full + glasses_revenue + vases_revenue + postcards_revenue
  let total_revenue_with_discounts := dolls_revenue + clocks_revenue + glasses_revenue + vases_revenue + postcards_revenue
  let sales_tax := sales_tax_rate * total_revenue_with_discounts
  let profit := total_revenue_with_discounts - cost - sales_tax
  profit = -17.25 := by sorry

end stella_profit_loss_l638_638747


namespace board_eventually_divisible_by_large_power_of_2_l638_638845

theorem board_eventually_divisible_by_large_power_of_2 :
  ∃ (n : ℕ), 2^(10_000_000) ∣ n :=
begin
  -- Initial conditions
  let initial_numbers : list ℕ := -- 100 natural numbers
  let odd_numbers_count : ℕ := 33, -- Exactly 33 of them are odd
  
  -- Process of adding sum of pairwise products each minute
  let add_pairwise_products (nums : list ℕ) : ℕ :=
    list.sum (list.map (λ (x : ℕ × ℕ), x.fst * x.snd) 
                 (list.product nums nums)),
  
  -- Note this approach is heuristic and can be formalized in details
  -- Our target is to prove eventual divisibility by \(2^{10,000,000}\)
  sorry
end

end board_eventually_divisible_by_large_power_of_2_l638_638845


namespace framing_required_l638_638842

-- Define the dimensions and conditions
def original_width := 5
def original_height := 7
def enlargement_factor := 4
def border_width := 3

-- Calculate dimensions after enlargement
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor

-- Calculate dimensions with borders
def bordered_width := enlarged_width + 2 * border_width
def bordered_height := enlarged_height + 2 * border_width

-- Calculate the perimeter of the bordered picture
def perimeter := 2 * (bordered_width + bordered_height)

-- Convert the perimeter to linear feet
def linear_feet := perimeter / 12

-- Prove that the minimum linear feet of framing required is 10 feet
theorem framing_required : linear_feet = 10 := by
  unfold original_width original_height enlargement_factor border_width
         enlarged_width enlarged_height bordered_width bordered_height
         perimeter linear_feet
  norm_num
  exact dec_trivial

end framing_required_l638_638842


namespace geometric_sequence_problem_l638_638214

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Geometric sequence definition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
def condition_1 : Prop := a 5 * a 8 = 6
def condition_2 : Prop := a 3 + a 10 = 5

-- Concluded value of q^7
def q_seven (q : ℝ) (a : ℕ → ℝ) : Prop := 
  q^7 = a 20 / a 13

theorem geometric_sequence_problem
  (h1 : is_geometric_sequence a q)
  (h2 : condition_1 a)
  (h3 : condition_2 a) :
  q_seven q a = (q = 3/2) ∨ (q = 2/3) :=
sorry

end geometric_sequence_problem_l638_638214


namespace find_x_eq_neg15_l638_638114

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l638_638114


namespace total_typing_cost_l638_638326

def typingCost (totalPages revisedOncePages revisedTwicePages : ℕ) (firstTimeCost revisionCost : ℕ) : ℕ := 
  let initialCost := totalPages * firstTimeCost
  let firstRevisionCost := revisedOncePages * revisionCost
  let secondRevisionCost := revisedTwicePages * (revisionCost * 2)
  initialCost + firstRevisionCost + secondRevisionCost

theorem total_typing_cost : typingCost 200 80 20 5 3 = 1360 := 
  by 
    rfl

end total_typing_cost_l638_638326


namespace workers_production_l638_638774

-- Define the conditions
def production_per_hour (workers barrels hours : ℕ) : ℕ := 
  barrels / hours

def production_per_worker_per_hour (workers barrels hours : ℕ) : ℕ := 
  production_per_hour workers barrels hours / workers

theorem workers_production (workers barrels hours : ℕ) :
  (production_per_worker_per_hour workers barrels hours) * 12 * 8 = 576 :=
by
  have h1 : production_per_hour 5 60 2 = 30 := by sorry
  have h2 : production_per_worker_per_hour 5 60 2 = 6 := by sorry
  have h3 : (production_per_worker_per_hour 5 60 2) * 12 * 8 = 576 := by sorry
  exact h3

end workers_production_l638_638774


namespace probability_record_3_l638_638738

namespace SashaDraw

def card_set := Finset.range 6

def draws_without_replacement := card_set.powerset.filter (λ s, s.card = 3)

def records_number_3 (s : Finset ℕ) : Prop :=
  if s.contains 0
  then (s.sum = 3)
  else ((s.sum / 3 : ℝ) = 3)

theorem probability_record_3 : 
  let favorable_outcomes := draws_without_replacement.filter records_number_3 in
  ((favorable_outcomes.card : ℝ) / (draws_without_replacement.card : ℝ)) = 1 / 5 :=
by
  sorry

end SashaDraw

end probability_record_3_l638_638738


namespace perpendicular_line_through_circle_center_l638_638539

theorem perpendicular_line_through_circle_center :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (y = m * x + b) → (x = -1 ∧ y = 0) ) ∧ m = 1 ∧ b = 1 ∧ (∀ (x y : ℝ), (y = x + 1) → (x - y + 1 = 0)) :=
sorry

end perpendicular_line_through_circle_center_l638_638539


namespace general_term_exists_sum_first_9_terms_sum_first_n_terms_c_l638_638211

-- Defining the arithmetic sequence conditions.
variables {a_n : ℕ → ℤ}
axiom a1_4_7 : a_n 1 + a_n 4 + a_n 7 = 9
axiom a3_6_9 : a_n 3 + a_n 6 + a_n 9 = 21

theorem general_term_exists : ∃ d : ℤ, ∃ a_1 : ℤ, ∀ n : ℕ, a_n n = a_1 + (n - 1) * d ∧ a_1 = -3 ∧ d = 2 :=
by sorry

theorem sum_first_9_terms : ∃ S_9 : ℤ, S_9 = ∑ i in finset.range 9, (2 * (i + 1) - 5) ∧ S_9 = 45 :=
by sorry

variables {c_n : ℕ → ℤ}
axiom c_def : ∀ n : ℕ, c_n n = 2 ^ (2 * (n - 1))

theorem sum_first_n_terms_c : ∃ T_n : ℤ, T_n = ∑ i in finset.range (n + 1), c_n (i + 1) ∧ T_n = (4 ^ (n + 1) - 1) / 3 :=
by sorry

end general_term_exists_sum_first_9_terms_sum_first_n_terms_c_l638_638211


namespace average_age_l638_638228

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l638_638228


namespace geometric_series_sum_y_equals_nine_l638_638903

theorem geometric_series_sum_y_equals_nine : 
  (∑' n : ℕ, (1 / 3) ^ n) * (∑' n : ℕ, (-1 / 3) ^ n) = ∑' n : ℕ, (1 / (9 ^ n)) :=
by
  sorry

end geometric_series_sum_y_equals_nine_l638_638903


namespace april_earnings_l638_638496

def price_per_rose := 7
def price_per_lily := 5
def initial_roses := 9
def initial_lilies := 6
def remaining_roses := 4
def remaining_lilies := 2

def total_roses_sold := initial_roses - remaining_roses
def total_lilies_sold := initial_lilies - remaining_lilies

def total_earnings := (total_roses_sold * price_per_rose) + (total_lilies_sold * price_per_lily)

theorem april_earnings : total_earnings = 55 := by
  sorry

end april_earnings_l638_638496


namespace specific_gravity_proof_l638_638891

noncomputable def specific_gravity_condition (M R h : ℝ) : ℝ :=
  let vol_cone := (1/3) * π * R^2 * M
  let vol_subcone := (1/3) * π * R^2 * (1/√3 * M)
  let water_density := (1/3) * π * R^2 * (M * √3 - h)
  vol_subcone / water_density

theorem specific_gravity_proof : 
  (∀ M R m, specific_gravity_condition M R m = 1 - (√3 / 9)) :=
by
  intros
  unfold specific_gravity_condition
  sorry

end specific_gravity_proof_l638_638891


namespace quadratic_completing_square_t_value_l638_638746

theorem quadratic_completing_square_t_value :
  ∃ q t : ℝ, 4 * x^2 - 24 * x - 96 = 0 → (x + q) ^ 2 = t ∧ t = 33 :=
by
  sorry

end quadratic_completing_square_t_value_l638_638746


namespace perimeter_of_rectangle_l638_638794

theorem perimeter_of_rectangle 
  (a b c : ℝ)
  (ha : a = 9)
  (hb : b = 12)
  (hc : c = 15)
  (right_triangle : a^2 + b^2 = c^2)
  (rectangle_length : ℝ)
  (h_length : rectangle_length = 5)
  (rectangle_area : (1/2) * a * b = 54)
  (h_area : rectangle_area = 54) : 
  (2 * (rectangle_length + rectangle_area / rectangle_length) = 31.6) :=
by
  sorry

end perimeter_of_rectangle_l638_638794


namespace quadratic_inequality_solution_l638_638170

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : 1 + 2 = b / a)
  (h3 : 1 * 2 = c / a) :
  ∀ x : ℝ, cx^2 + bx + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1 / 2 :=
by
  sorry

end quadratic_inequality_solution_l638_638170


namespace calculate_sum_l638_638971

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-2 : ℝ) (-1) then (1 / (2^x) - 2 * x - 4) else 0 -- only definition within bounds given.

axiom condition1 : ∀ x : ℝ, (x ∈ Set.Icc (-2 : ℝ) (-1)) → f 2 * x - 2 = f -2 * x - 2
axiom condition2 : ∀ x : ℝ, f (x - 3) + f (-x + 1) = 0
axiom condition3 : f (-2) = 4

theorem calculate_sum : ∑ k in List.range' 1 19, |f k| = 36 :=
by
  sorry

end calculate_sum_l638_638971


namespace inverse_proportion_decreasing_l638_638596

theorem inverse_proportion_decreasing (k : ℝ) :
  (∀ x : ℝ, x > 0 → (∃ y : ℝ, y = (k - 4) / x ∧ (∀ x1 > x, (k - 4) / x1 < y))) → k > 4 :=
begin
  sorry
end

end inverse_proportion_decreasing_l638_638596


namespace projection_eq_l638_638106

def plane (x y z : ℝ) : Prop := x + 2*y - z = 0

def vec_v : ℝ × ℝ × ℝ := (2, 3, 4)

def projection (v : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → Prop) : ℝ × ℝ × ℝ := 
  let n := (1, 2, -1) in
  v - ((( v.1 * n.1 + v.2 * n.2 + v.3 * n.3 ) / ( n.1 * n.1 + n.2 * n.2 + n.3 * n.3 )) × n)

theorem projection_eq : 
  projection (2, 3, 4) plane = (4/3, 5/3, 14/3) :=
by
  sorry

end projection_eq_l638_638106


namespace sculpture_and_base_height_l638_638900

def height_sculpture : ℕ := 2 * 12 + 10
def height_base : ℕ := 8
def total_height : ℕ := 42

theorem sculpture_and_base_height :
  height_sculpture + height_base = total_height :=
by
  -- provide the necessary proof steps here
  sorry

end sculpture_and_base_height_l638_638900


namespace length_of_strip_l638_638360

/-- Define the conversion factor from cubic kilometers to cubic meters --/
def cubicKmToCubicM (km³ : ℕ) : ℕ := km³ * 1000^3

/-- Define the conversion factor from meters to kilometers -/
def mToKm (m : ℕ) : ℕ := m / 1000

/-- The main theorem statement -/
theorem length_of_strip (V : ℕ) (hV : cubicKmToCubicM V = 1_000_000_000) : mToKm (cubicKmToCubicM V) = 1_000_000 := by
  sorry

end length_of_strip_l638_638360


namespace sequence_constant_condition_sequence_general_term_l638_638564

noncomputable def a_seq (x y : ℝ) : ℕ → ℝ
| 0     := x
| 1     := y
| (n+2) := (a_seq (n+1) * a_seq n + 1) / (a_seq (n+1) + a_seq n)

def is_constant_seq (s : ℕ → ℝ) : Prop :=
∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ → s n = s n₀

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

noncomputable def general_term (x y : ℝ) (n : ℕ) : ℝ :=
let F := fibonacci in
  ((x + 1) ^ F (n - 2) * (y + 1) ^ F (n - 1) + (x - 1) ^ F (n - 2) * (y - 1) ^ F (n - 1)) /
  ((x + 1) ^ F (n - 2) * (y + 1) ^ F (n - 1) - (x - 1) ^ F (n - 2) * (y - 1) ^ F (n - 1))

theorem sequence_constant_condition (x y : ℝ) :
  (|x| = 1 ∧ x + y ≠ 0) ∨ (|y| = 1 ∧ x + y ≠ 0) →
  is_constant_seq (a_seq x y) := sorry

theorem sequence_general_term (x y : ℝ) (n : ℕ) :
  a_seq x y n = general_term x y n := sorry

end sequence_constant_condition_sequence_general_term_l638_638564


namespace prove_a_equals_1_l638_638679

theorem prove_a_equals_1 
    (a b c d k m : ℤ) 
    (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) (h_odd_c : c % 2 = 1) (h_odd_d : d % 2 = 1)
    (h_ineq : 0 < a ∧ a < b ∧ b < c ∧ c < d)
    (h_ad_bc : a * d = b * c)
    (h_ad_eq : a + d = 2 ^ k)
    (h_bc_eq : b + c = 2 ^ m) : a = 1 :=
by
  sorry

end prove_a_equals_1_l638_638679


namespace solve_the_problem_l638_638265

noncomputable def problem : Prop :=
  ∃ a b : ℕ,
  a > 0 ∧
  b > 0 ∧
  (log 10 a + 3 * log 10 (Nat.gcd a b) = 90) ∧
  (log 10 b + 3 * log 10 (Nat.lcm a b) = 870) ∧
  let p := (Nat.factors a).length in
  let q := (Nat.factors b).length in
  (4 * p + 3 * q = 1484)

theorem solve_the_problem : problem :=
by
  sorry

end solve_the_problem_l638_638265


namespace simplify_expression_correct_l638_638740

noncomputable def simplify_expression (θ : ℝ) : ℝ :=
  (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) +
  (1 - Real.cos θ + Real.sin θ) / (1 + Real.cos θ + Real.sin θ)

theorem simplify_expression_correct (θ : ℝ) : simplify_expression θ = 2 * Real.csc θ :=
  sorry

end simplify_expression_correct_l638_638740


namespace coefficient_of_monomial_l638_638307

theorem coefficient_of_monomial : 
  ∀ (m n : ℝ), -((2 * Real.pi) / 3) * m * (n ^ 5) = -((2 * Real.pi) / 3) * m * (n ^ 5) :=
by
  sorry

end coefficient_of_monomial_l638_638307


namespace zero_pow_zero_is_meaningless_l638_638400

theorem zero_pow_zero_is_meaningless : (5 * 3 - 30 / 2 : ℝ) ^ 0 ≠ 1 := 
by
  -- Definitions and conditions
  let A : ℝ := 5 * 3 - 30 / 2
  have : A = 0 := by
    calc
      A = 5 * 3 - 30 / 2 : by rfl
        ... = 15 - 15    : by norm_num
        ... = 0         : by norm_num
  -- Assertion that original expression is meaningless
  have : A ^ 0 ≠ 1 := by
    rw [this]
    sorry
  exact this

end zero_pow_zero_is_meaningless_l638_638400


namespace glass_volume_l638_638426

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638426


namespace tile_count_l638_638474

theorem tile_count (room_length room_width tile_length tile_width : ℝ)
  (h1 : room_length = 10)
  (h2 : room_width = 15)
  (h3 : tile_length = 1 / 4)
  (h4 : tile_width = 3 / 4) :
  (room_length * room_width) / (tile_length * tile_width) = 800 :=
by
  sorry

end tile_count_l638_638474


namespace a3_value_l638_638938

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end a3_value_l638_638938


namespace find_p_l638_638177

def collinear (v₁ v₂ v₃ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v₃ = (v₁.1 + t * (v₂.1 - v₁.1), v₁.2 + t * (v₂.2 - v₁.2), v₁.3 + t * (v₂.3 - v₁.3))

theorem find_p
  (a : ℝ × ℝ × ℝ) 
  (b : ℝ × ℝ × ℝ)
  (ha : a = (2, -2, 3))
  (hb : b = (1, 4, 1))
  (p : ℝ × ℝ × ℝ)
  (hp : collinear a b p) : 
  p = (58 / 39, 40 / 39, 55 / 39) :=
sorry

end find_p_l638_638177


namespace students_participated_in_both_games_l638_638882

theorem students_participated_in_both_games
  (total_students : ℕ)
  (students_in_A : ℕ)
  (students_in_B : ℕ)
  (students_participating : total_students = 55)
  (students_A : students_in_A = 38)
  (students_B : students_in_B = 42) :
  let x := students_in_A + students_in_B - total_students 
  in x = 25 := 
by
  sorry

end students_participated_in_both_games_l638_638882


namespace shaded_fraction_correct_l638_638477

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ) in
  let r := (1/16 : ℚ) in
  a / (1 - r)

theorem shaded_fraction_correct :
  shaded_fraction = 4 / 15 :=
by
  sorry

end shaded_fraction_correct_l638_638477


namespace sphere_volume_surface_area_eq_radius_3_l638_638475

theorem sphere_volume_surface_area_eq_radius_3 :
  let r := 3 in
  let surface_area := 4 * Real.pi * r^2 in
  let volume := (4 / 3) * Real.pi * r^3 in
  surface_area = volume := by
  let r := 3
  let surface_area := 4 * Real.pi * r^2
  let volume := (4 / 3) * Real.pi * r^3
  sorry

end sphere_volume_surface_area_eq_radius_3_l638_638475


namespace roulette_P2007_gt_P2008_l638_638873

-- Define the roulette probability function based on the given conditions
noncomputable def roulette_probability : ℕ → ℝ
| 0 := 1
| n := (1 / 2007) * (List.foldl (λ acc k, acc + roulette_probability (n - k)) 0 (List.range 2007))

-- Define the theorem to prove P_{2007} > P_{2008}
theorem roulette_P2007_gt_P2008 : roulette_probability 2007 > roulette_probability 2008 :=
sorry

end roulette_P2007_gt_P2008_l638_638873


namespace p_minus_q_value_l638_638328

theorem p_minus_q_value :
  ∃ x y z : ℝ, (4 * x + 7 * y + z = 11) ∧ (3 * x + y + 5 * z = 15) ∧ 
  let p := 67, q := 17 in x + y + z = p / q ∧ (p - q = 50) :=
begin
  sorry
end

end p_minus_q_value_l638_638328


namespace average_sales_is_167_5_l638_638064

def sales_january : ℝ := 150
def sales_february : ℝ := 90
def sales_march : ℝ := 1.5 * sales_february
def sales_april : ℝ := 180
def sales_may : ℝ := 210
def sales_june : ℝ := 240
def total_sales : ℝ := sales_january + sales_february + sales_march + sales_april + sales_may + sales_june
def number_of_months : ℝ := 6

theorem average_sales_is_167_5 :
  total_sales / number_of_months = 167.5 :=
sorry

end average_sales_is_167_5_l638_638064


namespace initial_ratio_l638_638406

variables {p q : ℝ}

theorem initial_ratio (h₁ : p + q = 20) (h₂ : p / (q + 1) = 4 / 3) : p / q = 3 / 2 :=
sorry

end initial_ratio_l638_638406


namespace sum_of_corners_10x10_l638_638276

theorem sum_of_corners_10x10 : 
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  (top_left + top_right + bottom_left + bottom_right) = 202 :=
by
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  show top_left + top_right + bottom_left + bottom_right = 202
  sorry

end sum_of_corners_10x10_l638_638276


namespace value_of_3W5_l638_638911

-- Define the operation W
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- State the theorem to prove
theorem value_of_3W5 : W 3 5 = 23 := by
    sorry

end value_of_3W5_l638_638911


namespace percentage_length_more_than_breadth_l638_638770

-- Define the basic conditions
variables {C r l b : ℝ}
variable {p : ℝ}

-- Assume the conditions
def conditions (C r l b : ℝ) : Prop :=
  C = 400 ∧ r = 3 ∧ l = 20 ∧ 20 * b = 400 / 3

-- Define the statement that we want to prove
theorem percentage_length_more_than_breadth (C r l b : ℝ) (h : conditions C r l b) :
  ∃ (p : ℝ), l = b * (1 + p / 100) ∧ p = 200 :=
sorry

end percentage_length_more_than_breadth_l638_638770


namespace min_value_2x_plus_y_l638_638571

theorem min_value_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2y - 2 * x * y = 0) : 
  2 * x + y ≥ 9 / 2 :=
sorry

end min_value_2x_plus_y_l638_638571


namespace min_value_of_2x_plus_y_l638_638568

-- Define the conditions for the problem
variables {x y : ℝ}

-- Lean statement to show that the minimum value of 2x + y is 9/2 given the conditions
theorem min_value_of_2x_plus_y (hx : 0 < x) (hy : 0 < y) (h : x + 2y - 2 * x * y = 0) : 2 * x + y ≥ 9 / 2 :=
sorry

end min_value_of_2x_plus_y_l638_638568


namespace sqrt_D_rational_sometimes_not_l638_638697

-- Definitions and conditions
def D (a : ℤ) : ℤ := a^2 + (a + 2)^2 + (a * (a + 2))^2

-- The statement to prove
theorem sqrt_D_rational_sometimes_not (a : ℤ) : ∃ x : ℚ, x = Real.sqrt (D a) ∧ ¬(∃ y : ℤ, x = y) ∨ ∃ y : ℤ, Real.sqrt (D a) = y :=
by 
  sorry

end sqrt_D_rational_sometimes_not_l638_638697


namespace max_circumference_of_circle_inside_parabola_l638_638133

noncomputable def max_circle_circumference_inside_parabola : ℝ :=
  let C := { C : EuclideanSpace ℝ (Fin 2) | ∃ (b : ℝ), (∀ x y : ℝ, x^2 + (y - b)^2 = b^2 ∧ x^2 ≤ 4 * y ∧ (0, 0) ∈ C) } in
  let π := Real.pi in
  4 * π

theorem max_circumference_of_circle_inside_parabola
  : max_circle_circumference_inside_parabola = 4 * Real.pi := 
sorry

end max_circumference_of_circle_inside_parabola_l638_638133


namespace train_cross_pole_time_l638_638482

-- Definitions of the given problem
def train_speed_kmh : ℝ := 60
def train_length_m : ℝ := 250

-- Conversion from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Calculate time using the formula
def time_to_cross_pole (length_m speed_ms : ℝ) : ℝ := length_m / speed_ms

-- Main theorem
theorem train_cross_pole_time : time_to_cross_pole train_length_m (kmh_to_ms train_speed_kmh) = 15 := by
  sorry

end train_cross_pole_time_l638_638482


namespace find_angle_ACB_l638_638632

-- Define the given conditions and proof goals
theorem find_angle_ACB (A B C D : Point) 
  (h1 : angle B A C = 60)
  (h2 : 3 * dist B D = dist D C)
  (h3 : angle D A B = 30) :
  angle A C B = 90 := 
sorry

end find_angle_ACB_l638_638632


namespace combined_experience_is_correct_l638_638223

-- Define the conditions as given in the problem
def james_experience : ℕ := 40
def partner_less_years : ℕ := 10
def partner_experience : ℕ := james_experience - partner_less_years

-- The combined experience of James and his partner
def combined_experience : ℕ := james_experience + partner_experience

-- Lean statement to prove the combined experience is 70 years
theorem combined_experience_is_correct : combined_experience = 70 := by sorry

end combined_experience_is_correct_l638_638223


namespace cosine_of_angle_between_longer_edges_l638_638577

theorem cosine_of_angle_between_longer_edges (a : ℝ) (h1 : a > 0)
  (l1 : ℝ) (l2 : ℝ) (h2 : l1 = real.sqrt 3 * a) (h3 : l2 = real.sqrt 2 * a) :
  let cos_theta := (real.sqrt 6 / 3) in
  cos_theta = real.sqrt 6 / 3 := 
sorry

end cosine_of_angle_between_longer_edges_l638_638577


namespace compare_P2007_P2008_l638_638865

def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2007) * ∑ k in (finset.range 2007).image (λ x, x + n + 1 - 2007), P k

theorem compare_P2007_P2008 : P 2007 > P 2008 :=
sorry

end compare_P2007_P2008_l638_638865


namespace compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l638_638511

theorem compare_neg5_neg2 : -5 < -2 :=
by sorry

theorem compare_neg_third_neg_half : -(1/3) > -(1/2) :=
by sorry

theorem compare_absneg5_0 : abs (-5) > 0 :=
by sorry

end compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l638_638511


namespace find_a1_l638_638976

-- Define the range for natural numbers starting from 1 (ℕ≥1)
def n_star : Set ℕ := { n : ℕ // n > 0 }

-- Define the geometric sequence and its sum
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define a_1, and the condition a_(n+1) = a_1 * S_n + 1
axiom sum_condition : ∀ n : ℕ, n > 0 → a (n + 1) = a 1 * S n + 1

theorem find_a1 : a 1 = 1 := by
  sorry

end find_a1_l638_638976


namespace equation_of_AB_l638_638969

noncomputable def midpoint (A B : Point) : Point := 
{ x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def circle_center : Point := { x := 1, y := 0 }

def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

structure Line :=
(m : ℝ)
(c : ℝ)

def equation_of_line (L : Line) (x y : ℝ) : Prop :=
y = L.m * x + L.c

axiom midpoint_condition (A B : Point) (P : Point) : 
  midpoint A B = P

axiom line_through_points (A B : Point) : Line :=
{ m := (B.y - A.y) / (B.x - A.x), c := A.y - ((B.y - A.y) / (B.x - A.x)) * A.x }

def perpendicular_slope (m : ℝ) : ℝ := -1 / m

-- Given conditions
axiom point_P : Point := { x := 2, y := -1 }
axiom midpoint_chord : midpoint A B = point_P
axiom circle_eq : circle_center

-- Lean statement to show the equation of line AB
theorem equation_of_AB (A B : Point) (P : Point) (center : Point)
  (h1 : midpoint A B = P) (h2 : center = circle_center) :
    ∃ L : Line, equation_of_line L = λ x y, x - y - 3 = 0 :=
begin
  sorry
end

end equation_of_AB_l638_638969


namespace max_int_length_XY_l638_638675

theorem max_int_length_XY 
  (A B C : Point) 
  (h_collinear : collinear A B C)
  (h_AB : distance A B = 20) 
  (h_BC : distance B C = 18) 
  (r : ℝ) (h_r_pos : r > 0) 
  (ω : Circle B r)
  (ℓ₁ ℓ₂ : Line)
  (h_ℓ₁ : ℓ₁.tangent ω A)
  (h_ℓ₂ : ℓ₂.tangent ω C)
  (K : Point)
  (h_K : K ∈ ℓ₁ ∩ ℓ₂)
  (X Y : Point)
  (h_X_on_KA : X ∈ segment K A) 
  (h_Y_on_KC : Y ∈ segment K C)
  (h_XY_parallel_BC : parallel XY BC) 
  (h_XY_tangent_ω : tangent XY ω) :
  length XY = 35 := 
sorry

end max_int_length_XY_l638_638675


namespace f_odd_f_increasing_range_m_range_a_l638_638593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (x - x⁻¹)

variable {a : ℝ} (ha : a > 0) (ha_ne1 : a ≠ 1)

-- Odd function property
theorem f_odd : ∀ x : ℝ, f a (-x) = -f a x := sorry

-- Monotonicity
theorem f_increasing : ∀ x : ℝ, 0 < x → (f a x < f a (x + 1)) := sorry

-- Range of m such that f(1 - m) + f(1 - m^2) < 0 for x in (-1, 1)
theorem range_m (m : ℝ) (hm : 1 < m ∧ m < real.sqrt 2) : 
  f a (1 - m) + f a (1 - m^2) < 0 := sorry

-- Range of a such that f(x) - 4 < 0 for x in (-∞, 2)
theorem range_a (x : ℝ) (hx : x < 2) : 
  (f a x < 4) ↔ (2 - real.sqrt 3 < a ∧ a < 1) ∨ (1 < a ∧ a < 2 + real.sqrt 3) := sorry

end f_odd_f_increasing_range_m_range_a_l638_638593


namespace lattice_points_on_hyperbola_l638_638604

-- The hyperbola equation
def hyperbola_eq (x y : ℤ) : Prop :=
  x^2 - y^2 = 1800^2

-- The final number of lattice points lying on the hyperbola
theorem lattice_points_on_hyperbola : 
  ∃ (n : ℕ), n = 250 ∧ (∃ (x y : ℤ), hyperbola_eq x y) :=
sorry

end lattice_points_on_hyperbola_l638_638604


namespace find_p_value_l638_638762

theorem find_p_value :
  let a := 2021
  let p := 2 * a + 1
  let q := a * (a + 1)
  (a + 1 : ℚ) / a - (a : ℚ) / (a + 1) = p / q ∧ Nat.gcd p q = 1 :=
by
  let a := 2021
  let p := 2 * a + 1
  let q := a * (a + 1)
  sorry

end find_p_value_l638_638762


namespace area_of_cosine_polar_curve_l638_638004

noncomputable def area_of_polar_curve (r : ℝ → ℝ) (a b : ℝ) : ℝ :=
  1/2 * ∫ φ in a..b, (r φ)^2

theorem area_of_cosine_polar_curve :
  area_of_polar_curve (λ φ, 2 * Real.cos (6 * φ)) 0 (2 * Real.pi) = 2 * Real.pi :=
by
  sorry

end area_of_cosine_polar_curve_l638_638004


namespace glass_volume_correct_l638_638467

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638467


namespace first_girl_productivity_higher_l638_638385

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l638_638385


namespace investment_principal_l638_638340

theorem investment_principal (A r : ℝ) (n t : ℕ) (P : ℝ) : 
  r = 0.07 → n = 4 → t = 5 → A = 60000 → 
  A = P * (1 + r / n)^(n * t) →
  P = 42409 :=
by
  sorry

end investment_principal_l638_638340


namespace measure_of_angle_l638_638332

theorem measure_of_angle (x : ℝ) 
  (h₁ : 180 - x = 3 * x - 10) : x = 47.5 :=
by 
  sorry

end measure_of_angle_l638_638332


namespace f_1996x_l638_638690

noncomputable def f : ℝ → ℝ := sorry

axiom f_equation (x y : ℝ) : f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

theorem f_1996x (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end f_1996x_l638_638690


namespace polynomial_unique_l638_638105

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ)
  (h1 : p 3 = 10)
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p(x * y) - 3) : 
  p = λ x, x^2 + 1 :=
by 
  sorry

end polynomial_unique_l638_638105


namespace germination_percentage_in_second_plot_l638_638547

variables (seeds_plot1 seeds_plot2 seeds_germ_first_plot seeds_germ_total total_seeds seeds_germ_second_plot : ℕ)
variable (germ_first_percent germ_total_percent germ_second_percent : ℚ)

-- Given conditions
def conditions : Prop :=
(seeds_plot1 = 300) ∧
(seeds_plot2 = 200) ∧
(germ_first_percent = 0.25) ∧
(germ_total_percent = 0.27) ∧
(seeds_germ_first_plot = (germ_first_percent * seeds_plot1).toInt) ∧
(total_seeds = seeds_plot1 + seeds_plot2) ∧
(seeds_germ_total = (germ_total_percent * total_seeds).toInt) ∧
(seeds_germ_second_plot = seeds_germ_total - seeds_germ_first_plot) ∧
(germ_second_percent = (seeds_germ_second_plot : ℚ) / seeds_plot2 * 100)

-- Question to prove
theorem germination_percentage_in_second_plot : conditions 
→ germ_second_percent = 30 := by
  sorry

end germination_percentage_in_second_plot_l638_638547


namespace value_of_f_6_l638_638138

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom function_defined_on_real : ∀ x : ℝ, f x ∈ ℝ
axiom periodic_condition (x : ℝ) : f (x + 2) = -f (x)

-- Proof statement
theorem value_of_f_6 : f 6 = 0 :=
by
  sorry

end value_of_f_6_l638_638138


namespace glass_volume_correct_l638_638466

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638466


namespace tangential_tetrahedron_triangle_impossibility_l638_638137

theorem tangential_tetrahedron_triangle_impossibility (a b c d : ℝ) 
  (h : ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x > 0) :
  ¬ (∀ (x y z : ℝ) , (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    (y = a ∨ y = b ∨ y = c ∨ y = d) →
    (z = a ∨ z = b ∨ z = c ∨ z = d) → 
    x ≠ y → y ≠ z → z ≠ x → x + y > z ∧ x + z > y ∧ y + z > x) :=
sorry

end tangential_tetrahedron_triangle_impossibility_l638_638137


namespace num_pairs_satisfy_eqns_l638_638606

theorem num_pairs_satisfy_eqns :
  (∃ n : ℕ, n = 9 ∧
  ∀ (x y : ℝ), 
    (y^4 - y^2 = 0) ∧ 
    (x * y^3 - x * y = 0) ∧
    (x^3 * y - x * y = 0) ∧
    (x^4 - x^2 = 0) 
  → (x, y) ∈ ({0, 1, -1} : set ℝ) × ({0, 1, -1} : set ℝ)) :=
sorry

end num_pairs_satisfy_eqns_l638_638606


namespace f_f_one_third_l638_638982

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log 3 x else 2^x

theorem f_f_one_third : f (f (1 / 3)) = 1 / 2 :=
by
  sorry

end f_f_one_third_l638_638982


namespace productivity_comparison_l638_638391

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l638_638391


namespace tangent_circle_fixed_point_l638_638120

variables {α : Type*} [euclidean_geometry α]

/-- From point A to the circle ω, a tangent AD and an arbitrary secant intersect the circle at points B and C (with B lying between points A and C). Prove that the circle passing through points C and D and tangent to the line BD passes through a fixed point M different from D. -/
theorem tangent_circle_fixed_point
  (A B C D E M : α)
  {ω : circle α}
  (hA_tangent_D : tangent A D ω)
  (hA_secant_BC : secant A B C ω)
  (hA_tangent_E : tangent A E ω)
  (hM_midpoint_DE : midpoint M D E) :
  ∃ K, is_circumcircle K C D ∧ tangent K BD ∧ K M :=
sorry

end tangent_circle_fixed_point_l638_638120


namespace angle_between_vectors_l638_638246

variables (a b c : ℝ^3)
variable (θ : ℝ)

-- Conditions
def is_unit_vector (v : ℝ^3) := ‖v‖ = 1 

def vectors_and_condition : Prop :=
  is_unit_vector a ∧ is_unit_vector b ∧ is_unit_vector c ∧ (a + b + sqrt 2 • c = 0)

-- Question to answer
theorem angle_between_vectors (h: vectors_and_condition a b c) : θ = 90 :=
by {
  sorry
}

end angle_between_vectors_l638_638246


namespace surface_area_of_sphere_above_pentagon_is_correct_l638_638254

noncomputable def sphere_surface_area_above_pentagon_part : ℝ :=
  π * (5 * (Real.cos (Real.pi / 5)) - 3)

theorem surface_area_of_sphere_above_pentagon_is_correct (S : Set (ℝ × ℝ × ℝ))
  (P : Set (ℝ × ℝ × ℝ))
  (hS : ∀ (x y z : ℝ), (x, y, z) ∈ S ↔ x^2 + y^2 + z^2 = 1)
  (hP : ∀ (x y z : ℝ), (x, y, z) ∈ P ↔ z = 0 ∧ (x^2 + y^2 = 1) ∧
                              (pentagon ((x, y) : ℂ)))
  (pentagon : ℂ → Prop) :
  (sphere_surface_area_above_pentagon_part = π * (5 * Real.cos (Real.pi / 5) - 3)) :=
by
  sorry

end surface_area_of_sphere_above_pentagon_is_correct_l638_638254


namespace train_length_l638_638722

theorem train_length (sA sB t: ℝ) (same_speed: ∀ t, sA t = sB t) (speed_ratio: sA / sB = 1):
  sA * 5 + sB = 180 :=
by
  sorry

end train_length_l638_638722


namespace exists_three_distinct_div_l638_638239

theorem exists_three_distinct_div (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ m : ℕ, ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ abc ∣ (x * y * z) ∧ m ≤ x ∧ x < m + 2*c ∧ m ≤ y ∧ y < m + 2*c ∧ m ≤ z ∧ z < m + 2*c :=
by
  sorry

end exists_three_distinct_div_l638_638239


namespace first_girl_productivity_higher_l638_638387

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l638_638387


namespace problem_l638_638977

open Set

variable (U : Set ℝ) (P : Set ℝ) (M : Set ℝ)
variable (x : ℝ)

#check Real

noncomputable def P := {x : ℝ | x >= 3}
noncomputable def M := {x : ℝ | x < 4}

theorem problem (hU : U = univ) (hP : P = {x : ℝ | x >= 3}) (hM : M = {x : ℝ | x < 4}) :
  P ∩ (U \ M) = {x : ℝ | x >= 4} :=
by
  sorry

end problem_l638_638977


namespace angle_ACD_l638_638237

variables (A B C D : Type) [OrderedGeometry A B C D]

noncomputable def convex_quadrilateral (ABCD : ConvexQuadrilateral A B C D) : Prop :=
  (∠ (BAC) = 3 * ∠ (CAD)) ∧
  (AB = CD) ∧
  (∠ (ACD) = ∠ (CBD))

variable (α : ConvexQuadrilateral A B C D)
open_locale classical

theorem angle_ACD (h : convex_quadrilateral α) : ∠ (ACD) = 30 :=
  sorry

end angle_ACD_l638_638237


namespace max_height_of_rock_l638_638039

-- Definition of the height function
def height (t : ℝ) : ℝ := 150 * t - 15 * t^2

-- The maximum height the rock reaches
theorem max_height_of_rock : ∃ t : ℝ, height t = 375 := by
  use 5
  show height 5 = 375
  sorry

end max_height_of_rock_l638_638039


namespace ratio_of_men_to_women_l638_638334

theorem ratio_of_men_to_women (M W : ℕ) 
  (h1 : W = M + 2) 
  (h2 : M + W = 16) : M = 7 ∧ W = 9 ∧ M / gcd M W = 7 / 9 := 
by
  assume h1 h2
  sorry

end ratio_of_men_to_women_l638_638334


namespace proof_problem_l638_638588

-- Define the function f
def f (a : ℝ) : ℝ := ∫ x in 0..a, sin x

-- State the theorem to be proven
theorem proof_problem : f (f (π / 2)) = 1 - cos 1 :=
by 
  sorry

end proof_problem_l638_638588


namespace squares_of_numbers_are_rational_l638_638750

theorem squares_of_numbers_are_rational 
  (S : Finset ℝ) (h_distinct : S.card = 10)
  (h_non_zero : ∀ x ∈ S, x ≠ 0)
  (h_sum_or_product_rational : ∀ x y ∈ S, x ≠ y → (x + y ∈ ℚ ∨ x * y ∈ ℚ)) : 
  ∀ x ∈ S, x^2 ∈ ℚ :=
by
  sorry

end squares_of_numbers_are_rational_l638_638750


namespace rain_barrel_filled_percentage_l638_638407

def roof_side : ℝ := 5 -- meters
def rainfall_height_mm : ℝ := 6 -- millimeters
def rainfall_height_m : ℝ := rainfall_height_mm / 1000 -- meters
def barrel_diameter : ℝ := 0.5 -- meters
def barrel_radius : ℝ := barrel_diameter / 2 -- meters
def barrel_height : ℝ := 1 -- meter

def roof_area : ℝ := roof_side * roof_side -- 25 square meters
def rain_volume : ℝ := roof_area * rainfall_height_m -- cubic meters
def barrel_volume : ℝ := Real.pi * (barrel_radius ^ 2) * barrel_height -- cubic meters

def percentage_filled : ℝ := (rain_volume / barrel_volume) * 100

theorem rain_barrel_filled_percentage :
  abs (percentage_filled - 76.4) < 0.05 :=
sorry

end rain_barrel_filled_percentage_l638_638407


namespace tangent_line_at_1_maximum_value_t_inequality_ln_l638_638988

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1)

theorem tangent_line_at_1 : ∀ (x y : ℝ), y = f x → (x = 1) → ∃ m b, y = m * x + b ∧ m = 1/2 ∧ b = -1 := sorry

theorem maximum_value_t : ∀ (t : ℝ), (∀ x, x > 0 ∧ x ≠ 1 → f x - t / x > (Real.log x) / (x - 1)) → t ≤ -1 := sorry

theorem inequality_ln : ∀ (n : ℕ), n ≥ 2 → Real.log n < 1 + (∑ i in Finset.range (n + 1), 1 / i) - 1 / 2 - 1 / (2 * n) := sorry

end tangent_line_at_1_maximum_value_t_inequality_ln_l638_638988


namespace seats_per_row_l638_638056

-- Defining the conditions
def rows : ℕ := 150
def ticket_cost : ℕ := 10
def total_revenue : ℕ := 12000
def percent_sold : ℚ := 0.8

-- The number of seats that the opera house must have sold to make the revenue
def tickets_sold : ℕ := total_revenue / ticket_cost

-- The proof statement
theorem seats_per_row (S : ℕ) : 0.8 * (rows * S) = tickets_sold → S = 10 :=
by
  intro h,
  sorry

end seats_per_row_l638_638056


namespace sum_of_digits_of_N_l638_638483

open Nat

theorem sum_of_digits_of_N (T : ℕ) (hT : T = 3003) :
  ∃ N : ℕ, (N * (N + 1)) / 2 = T ∧ (digits 10 N).sum = 14 :=
by 
  sorry

end sum_of_digits_of_N_l638_638483


namespace product_sequence_eq_670_l638_638065

/-- Lean code statement for the given math problem -/
theorem product_sequence_eq_670 :
  (∏ k in Finset.range (2007), (k + 4) / (k + 3)) = 670 :=
by
  sorry

end product_sequence_eq_670_l638_638065


namespace find_x_l638_638991

open Matrix

-- Define the problem
def a : Fin 2 → ℤ := ![1, 1]
def b : Fin 2 → ℤ := ![2, 5]
def c (x : ℤ) : Fin 2 → ℤ := ![3, x]

-- The proof statement
theorem find_x (x : ℤ) (h : dot_product (fun i => 8 * a i - b i) (c x) = 30) : x = 4 :=
sorry

end find_x_l638_638991


namespace double_apply_of_four_l638_638947

def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 1 else -x + 3

theorem double_apply_of_four : f (f 4) = 0 := by
  sorry

end double_apply_of_four_l638_638947


namespace a_2004_minus_1_eq_zero_l638_638195

-- Define the variables and conditions
variables (m n : ℤ) (a : ℤ)
variable h1 : 3 * m * (x^a) * y + (-2) * n * (x^(4 * a - 3)) * y = c * (x^c') * y

-- State the theorem to be proven
theorem a_2004_minus_1_eq_zero (h1 : 3 * m * (x^a) * y + (-2) * n * (x^(4 * a - 3)) * y = c * (x^c') * y) : a^(2004) - 1 = 0 :=
sorry

end a_2004_minus_1_eq_zero_l638_638195


namespace f_2015_equals_2_l638_638953

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_2015_equals_2 (f_even : ∀ x : ℝ, f (-x) = f x)
    (f_shift : ∀ x : ℝ, f (-x) = f (2 + x))
    (f_log : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.log (3 * x + 1) / Real.log 2) :
    f 2015 = 2 :=
sorry

end f_2015_equals_2_l638_638953


namespace Danny_to_Steve_house_time_l638_638073

theorem Danny_to_Steve_house_time
  (t : ℝ)
  (h1 : ∀ t, Danny_to_Steve t → Steve_to_Danny (2 * t))
  (h2 : ∀ t, Halfway_Steve (t) = Halfway_Danny (t / 2) + 16.5) :
  t = 33 := 
sorry

end Danny_to_Steve_house_time_l638_638073


namespace combined_percentage_error_in_area_is_2_06_percent_l638_638055

def percentage_error_in_area (L W : ℝ) : ℝ := 
  let L' := 1.02 * L
  let W' := 1.03 * W
  let A := L * W
  let A' := L' * W'
  (A' - A) / A * 100

theorem combined_percentage_error_in_area_is_2_06_percent :
  ∀ (L W : ℝ), percentage_error_in_area L W = 2.06 := by
  sorry

end combined_percentage_error_in_area_is_2_06_percent_l638_638055


namespace glass_volume_230_l638_638433

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638433


namespace reciprocal_of_neg_three_l638_638403

theorem reciprocal_of_neg_three : (1:ℝ) / (-3:ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg_three_l638_638403


namespace hyperbola_same_asymptotes_l638_638078

theorem hyperbola_same_asymptotes :
  (∀ x y, (x^2 / 16) - (y^2 / 9) = 1 → (4 * y = 3 * x ∨ 4 * y = -3 * x)) ∧
  (∀ x y, (y^2 / 9) - (x^2 / 16) = 1 → (3 * x = 4 * y ∨ 3 * x = -4 * y)) :=
begin
  sorry
end

end hyperbola_same_asymptotes_l638_638078


namespace total_words_in_poem_l638_638288

theorem total_words_in_poem (s l w : ℕ) (h1 : s = 35) (h2 : l = 15) (h3 : w = 12) : 
  s * l * w = 6300 := 
by 
  -- the proof will be inserted here
  sorry

end total_words_in_poem_l638_638288


namespace coefficient_of_linear_term_l638_638306

theorem coefficient_of_linear_term :
  let f := λ x : ℝ, (x - 1) * (1/x + x) ^ 6,
  coeff : ℕ → ℝ := λ n, (polynomial.Taylor f).coeff (1, n) in
  coeff 1 = 20 :=
by
  let f := λ x : ℝ, (x - 1) * (λ x, (1/x) + x) x ^ 6
  have : coeff 1 = 20 := sorry
  exact this

end coefficient_of_linear_term_l638_638306


namespace average_age_is_35_l638_638226

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l638_638226


namespace no_singular_points_eventually_l638_638576

/-- 
  Given a graph where vertices are colored either red or blue, and a vertex is called a "singular point" 
  if more than half of its neighbors are of a different color. If the color of "singular points" is 
  repeatedly switched, then eventually no "singular points" will remain.
-/
theorem no_singular_points_eventually (G : Type) [graph G] (color : G → bool) :
  ∃ (N : ℕ), ∀ n ≥ N, ¬ ∃ (P : G), is_singular_point G color P n :=
sorry

/-- A predicate for identifying "singular points" -/
def is_singular_point (G : Type) [graph G] (color : G → bool) (P : G) (n : ℕ) : Prop :=
  let neighbors := {Q | Q ≠ P ∧ graph.adj G P Q}
  let opposite_color_neighbors := neighbors.filter (λ Q, color Q ≠ color P)
  2 * opposite_color_neighbors.card > neighbors.card

end no_singular_points_eventually_l638_638576


namespace expression_C_eq_seventeen_l638_638362

theorem expression_C_eq_seventeen : (3 + 4 * 5 - 6) = 17 := 
by 
  sorry

end expression_C_eq_seventeen_l638_638362


namespace value_of_expression_l638_638188

theorem value_of_expression (x y : ℝ) (h₁ : 3 ^ x = 6) (h₂ : 4 ^ y = 6) : 
  (2 / x) + (1 / y) = (Real.log 18)/(Real.log 6) :=
sorry

end value_of_expression_l638_638188


namespace car_mpg_l638_638026

open Nat

theorem car_mpg (x : ℕ) (h1 : ∀ (m : ℕ), m = 4 * (3 * x) -> x = 27) 
                (h2 : ∀ (d1 d2 : ℕ), d2 = (4 * d1) / 3 - d1 -> d2 = 126) 
                (h3 : ∀ g : ℕ, g = 14)
                : x = 27 := 
by
  sorry

end car_mpg_l638_638026


namespace linear_transformation_proof_l638_638550

theorem linear_transformation_proof (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) :
  ∃ (k b : ℝ), k = 4 ∧ b = -1 ∧ (y = k * x + b ∧ -1 ≤ y ∧ y ≤ 3) :=
by
  sorry

end linear_transformation_proof_l638_638550


namespace glass_volume_l638_638459

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638459


namespace lines_intersection_or_plane_l638_638207

theorem lines_intersection_or_plane
  (Lines : Set (Set Point)) : 
  (∀ l₁ l₂ ∈ Lines, ∃ O, O ∈ l₁ ∩ l₂) →
  (∃ O, ∀ l ∈ Lines, O ∈ l) ∨ 
  (∃ π, ∀ l ∈ Lines, l ⊆ π) :=
by
  sorry

end lines_intersection_or_plane_l638_638207


namespace juniper_final_bones_l638_638235

/--
Juniper initially has 4 bones. Her master gives her bones equal to 50% more than what she currently has. 
The neighbor's dog steals away 25% of Juniper's total bones. 
Prove that the final number of bones Juniper has after these transactions is 5.
-/
theorem juniper_final_bones : 
  let initial_bones := 4
  let master_gives := initial_bones / 2
  let total_after_master := initial_bones + master_gives
  let neighbor_steals := (total_after_master * 1 / 4).floor
  let final_bones := total_after_master - neighbor_steals
  final_bones = 5 := 
  by
    sorry

end juniper_final_bones_l638_638235


namespace product_sequence_l638_638813

theorem product_sequence : (∏ k in Finset.range (2010 - 3 + 1), (4 + k) / (3 + k)) = 670 :=
by
  sorry

end product_sequence_l638_638813


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l638_638595

noncomputable def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 4 * (n - 1)

def S (n : ℕ) : ℕ := 2 * n^2 - 2 * n

theorem geometric_sequence_general_term
    (a1 : ℕ := 2)
    (a4 : ℕ := 16)
    (h1 : a 1 = a1)
    (h2 : a 4 = a4)
    : ∀ n : ℕ, a n = a 1 * 2^(n-1) :=
by
  sorry

theorem arithmetic_sequence_sum
    (a2 : ℕ := 4)
    (a5 : ℕ := 32)
    (b2 : ℕ := a 2)
    (b9 : ℕ := a 5)
    (h1 : b 2 = b2)
    (h2 : b 9 = b9)
    : ∀ n : ℕ, S n = n * (n - 1) * 2 :=
by
  sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l638_638595


namespace isosceles_of_equal_bisectors_l638_638732

theorem isosceles_of_equal_bisectors
  (A B C L1 L3 : Type)
  [triangle : ∀ ⦃α β γ⦄, α ≠ β ∧ β ≠ γ ∧ γ ≠ α]
  (angle_bisector : ∀ ⦃P Q R T : Type⦄, Type)
  (AL1_bisects_BAC : angle_bisector A B C L1)
  (CL3_bisects_ACB : angle_bisector C A B L3)
  (equal_bisectors : ∀ {D E : Type}, angle_bisector D E A L1 → angle_bisector D E C L3 → AL1 = L3)
  (AB AC : Type)
  (equal_lengths : AB ≠ AC ∧ AL1 = L3):
  AB = AC :=
by
  sorry

end isosceles_of_equal_bisectors_l638_638732


namespace gallon_of_water_weighs_eight_pounds_l638_638222

theorem gallon_of_water_weighs_eight_pounds
  (pounds_per_tablespoon : ℝ := 1.5)
  (cubic_feet_per_gallon : ℝ := 7.5)
  (cost_per_tablespoon : ℝ := 0.50)
  (total_cost : ℝ := 270)
  (bathtub_capacity_cubic_feet : ℝ := 6)
  : (6 * 7.5) * pounds_per_tablespoon = 270 / cost_per_tablespoon / 1.5 :=
by
  sorry

end gallon_of_water_weighs_eight_pounds_l638_638222


namespace distance_between_incenter_and_excenter_l638_638515

-- Defining the side lengths of the triangle
def AB : ℝ := 10
def BC : ℝ := 11
def CA : ℝ := 12

-- Defining the semiperimeter of the triangle
def s : ℝ := (AB + BC + CA) / 2

-- Using Heron's formula to define the area of the triangle
def K : ℝ := Real.sqrt (s * (s - AB) * (s - BC) * (s - CA))

-- Defining the inradius
def r : ℝ := K / s

-- Assuming the given distance between the centers of the incircle and excircle
def distance_between_centers : ℝ := 12.6

-- The main theorem statement
theorem distance_between_incenter_and_excenter :
  ∃ I E : ℝ × ℝ, distance (I, E) = distance_between_centers := 
sorry

end distance_between_incenter_and_excenter_l638_638515


namespace glass_volume_230_l638_638445

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638445


namespace log_ratio_max_value_l638_638130

theorem log_ratio_max_value (x y : ℝ) (hx : x > 2) (hy : y > 2) (hxy : x > y) :
  ∃ c : ℝ, (c = log x y) ∧ (c > 0) ∧ (log y x = 1 / c) ∧ (2 - c - (1 / c)) ≤ 0 := 
sorry

end log_ratio_max_value_l638_638130


namespace product_of_possible_values_b_l638_638772

theorem product_of_possible_values_b :
  (∀ (y1 y2 : ℝ) (x1 : ℝ) (b : ℝ), y1 = 3 → y2 = 9 → x1 = -3 → (x1 - 6 = b ∨ x1 + 6 = b) → b = -9 ∨ b = 3) →
  -9 * 3 = -27 :=
by
  intros h
  have hb : -9 * 3 = -27 := rfl
  exact hb

end product_of_possible_values_b_l638_638772


namespace angle_B_of_isosceles_triangle_l638_638961

theorem angle_B_of_isosceles_triangle (A B C : ℝ) (h_iso : (A = B ∨ A = C) ∨ (B = C ∨ B = A) ∨ (C = A ∨ C = B)) (h_angle_A : A = 70) :
  B = 70 ∨ B = 55 :=
by
  sorry

end angle_B_of_isosceles_triangle_l638_638961


namespace area_correct_l638_638009

variable (A B C L : Type) [MetricSpace A]
variables (a b c bl al cl : ℝ)
variables (is_angle_bisector : (bl = 3 * Real.sqrt 10))
variables (AL_eq_2 : al = 2)
variables (CL_eq_3 : cl = 3)

noncomputable def area_of_triangle_ABC : ℝ :=
  let area := (15 * Real.sqrt 15) / 4 in
  area

theorem area_correct (h₁ : is_angle_bisector) (h₂ : AL_eq_2) (h₃ : CL_eq_3) :
  area_of_triangle_ABC A B C L a b c bl al cl is_angle_bisector AL_eq_2 CL_eq_3
  = (15 * Real.sqrt 15) / 4 :=
sorry

end area_correct_l638_638009


namespace domain_of_f_l638_638311

noncomputable def f : ℝ → ℝ := λ x, (x + 1 : ℝ) ^ 0

theorem domain_of_f :
  ∀ x : ℝ, (f x = (x + 1) ^ (0 : ℝ) → (x ∈ (-1, 2) ∪ (2, +∞))) :=
sorry

end domain_of_f_l638_638311


namespace range_of_m_nonempty_necessary_but_not_sufficient_condition_l638_638171

noncomputable def P (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
noncomputable def S (m x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem range_of_m (m : ℝ) (x : ℝ) :
  (-2 ≤ x ∧ x ≤ 10) → (S m x) :=
begin
  sorry
end

theorem nonempty {α : Type*} : set α → Prop 
  | ( ∅ ) := false 
  | _ := true

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∉ {y | P y} → x ∉ {y | S m y}) → (9 ≤ m) :=
by
  sorry

#check necessary_but_not_sufficient_condition

end range_of_m_nonempty_necessary_but_not_sufficient_condition_l638_638171


namespace water_speed_l638_638860

-- Define the conditions
def swimming_speed_still_water : ℝ := 4 -- km/h
def swimming_time : ℝ := 4 -- hours
def swimming_distance : ℝ := 8 -- km

-- Define the effective speed and the equation to find the water speed
theorem water_speed : ∃ v : ℝ, (swimming_speed_still_water - v) * swimming_time = swimming_distance → v = 2 := 
by
  -- Let v be the speed of the water
  let v : ℝ
  assume h : (swimming_speed_still_water - v) * swimming_time = swimming_distance
  -- Solve for v
  sorry

end water_speed_l638_638860


namespace breadth_of_added_rectangle_l638_638480

theorem breadth_of_added_rectangle 
  (s : ℝ) (b : ℝ) 
  (h_square_side : s = 8) 
  (h_perimeter_new_rectangle : 2 * s + 2 * (s + b) = 40) : 
  b = 4 :=
by
  sorry

end breadth_of_added_rectangle_l638_638480


namespace dog_weight_ratio_l638_638896

theorem dog_weight_ratio
  (w7 : ℕ) (r : ℕ) (w13 : ℕ) (w21 : ℕ) (w52 : ℕ):
  (w7 = 6) →
  (w13 = 12 * r) →
  (w21 = 2 * w13) →
  (w52 = w21 + 30) →
  (w52 = 78) →
  r = 2 :=
by 
  sorry

end dog_weight_ratio_l638_638896


namespace cylinder_lateral_surface_area_l638_638861

-- Define the radius and height as constants
def radius_cylinder : ℝ := 3
def height_cylinder : ℝ := 7

-- Define the formula for the curved surface area of the cylinder
def lateral_surface_area_of_cylinder (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- State the theorem to be proved: the lateral surface area of the cylinder
theorem cylinder_lateral_surface_area :
  lateral_surface_area_of_cylinder radius_cylinder height_cylinder = 42 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l638_638861


namespace Emily_first_three_cards_sum_is_14_l638_638921

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def Emily_stack_sum : ℕ :=
  let green_cards := {2, 3, 4, 5}
      red_cards := {6, 7, 8, 9}
      alternates_correctly (cards : List ℕ) : Prop :=
        ∀ i, i < cards.length - 1 →
          ((cards[i] ∈ green_cards ∧ cards[i+1] ∈ red_cards) ∨
           (cards[i] ∈ red_cards ∧ cards[i+1] ∈ green_cards))
      primes_pairs := [(2, 9), (3, 8), (4, 9)] -- based on solution pairs
      valid_stack : List ℕ := [2, 9, 3, 8, 4, 7, 5, 6] -- first valid stacking
    
  -- Function sums the first three cards in the stack
  let first_three_cards_sum (lst : List ℕ) : ℕ :=
    lst.take 3 |>.sum
    
  -- Ensure conditions and calculate the sum of the first three cards
  if alternates_correctly valid_stack ∧ 
     (valid_stack.pairwise (λ a b, (a, b) ∈ primes_pairs ∨ (b, a) ∈ primes_pairs)) then
    first_three_cards_sum valid_stack
  else 0

theorem Emily_first_three_cards_sum_is_14 : 
  Emily_stack_sum = 14 :=
by
  -- Proof goes here, currently skipped
  sorry

end Emily_first_three_cards_sum_is_14_l638_638921


namespace collinear_points_l638_638884

noncomputable def Triangle (α β γ : Type) := α × β × γ

def is_base_of_external_bisector (K A B C : Type) := sorry
def is_midpoint_of_arc (M A C : Type) := sorry
def is_on_angle_bisector (N C : Type) := sorry
def parallel (AN BM : Type) := sorry
def collinear (M N K : Type) := sorry

theorem collinear_points 
  (A B C K M N : Type) 
  (triangle_ABC : Triangle A B C) 
  (H1 : is_base_of_external_bisector K A B C)
  (H2 : is_midpoint_of_arc M A C)
  (H3 : is_on_angle_bisector N C)
  (H4 : parallel (AN N) (BM M)) :
  collinear M N K := 
sorry

end collinear_points_l638_638884


namespace find_f_f_of_sqrt_e_l638_638589

def f (x : ℝ) : ℝ := if x ≤ 1 then -x + 1 else Real.log x

theorem find_f_f_of_sqrt_e : f (f (Real.sqrt Real.exp 1)) = 1 / 2 := by
  sorry

end find_f_f_of_sqrt_e_l638_638589


namespace bilingual_point_3x_plus_1_bilingual_points_k_over_x_value_of_k_l638_638348

-- Part (1)
theorem bilingual_point_3x_plus_1 : ∃ x y : ℝ, y = 2 * x ∧ y = 3 * x + 1 ∧ (x, y) = (-1, -2) := 
by sorry

-- Part (2)
theorem bilingual_points_k_over_x (k : ℝ) (h : k ≠ 0) :
  (k > 0 → ∃ x y : ℝ, y = 2 * x ∧ y = k / x ∧ ((x, y) = (sqrt (2 * k) / 2, sqrt (2 * k)) ∨ (x, y) = (-sqrt (2 * k) / 2, -sqrt (2 * k)))) ∧
  (k < 0 → ∀ x y : ℝ, ¬(y = 2 * x ∧ y = k / x)) := 
by sorry

-- Part (3)
theorem value_of_k (n : ℝ) (h : 1 ≤ n ∧ n ≤ 3) :
  ∃ k m : ℝ, (∀ x y : ℝ, y = 2 * x ∧ y = (1 / 4) * x^2 + (n - k - 1) * x + m + k + 2 → 
    x^2 - 4 * (n - k - 1) * x + 4 * (m + k + 2) = 0 ∧ x = y / 2) ∧ (m = k) ∧ (k = 1 + sqrt 3 ∨ k = -1) :=
by sorry

end bilingual_point_3x_plus_1_bilingual_points_k_over_x_value_of_k_l638_638348


namespace glass_volume_230_l638_638440

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638440


namespace acute_triangle_trig_inequality_l638_638734

theorem acute_triangle_trig_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) (h_acute : A < π/2 ∧ B < π/2 ∧ C < π/2) :
  cos A + cos B + cos C < sin A + sin B + sin C :=
sorry

end acute_triangle_trig_inequality_l638_638734


namespace total_tiles_cost_is_2100_l638_638670

noncomputable def total_tile_cost : ℕ :=
  let length := 10
  let width := 25
  let tiles_per_sq_ft := 4
  let green_tile_percentage := 0.40
  let cost_per_green_tile := 3
  let cost_per_red_tile := 1.5
  let area := length * width
  let total_tiles := area * tiles_per_sq_ft
  let green_tiles := green_tile_percentage * total_tiles
  let red_tiles := total_tiles - green_tiles
  let cost_green := green_tiles * cost_per_green_tile
  let cost_red := red_tiles * cost_per_red_tile
  cost_green + cost_red

theorem total_tiles_cost_is_2100 : total_tile_cost = 2100 := by 
  sorry

end total_tiles_cost_is_2100_l638_638670


namespace first_girl_productivity_higher_l638_638383

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l638_638383


namespace pastries_sold_l638_638502

def initial_pastries : ℕ := 148
def pastries_left : ℕ := 45

theorem pastries_sold : initial_pastries - pastries_left = 103 := by
  sorry

end pastries_sold_l638_638502


namespace total_amount_spent_on_supplies_l638_638992

-- Define the costs and quantities
def cost_per_paper_haley : ℝ := 3.5
def cost_per_paper_sister : ℝ := 4.25
def cost_per_pen_haley : ℝ := 1.25
def cost_per_pen_sister : ℝ := 1.5

def quantity_paper_haley : ℕ := 2
def quantity_paper_sister : ℕ := 3
def quantity_pen_haley : ℕ := 5
def quantity_pen_sister : ℕ := 8

-- Define costs for Haley and Sister
def cost_paper_haley := quantity_paper_haley * cost_per_paper_haley
def cost_paper_sister := quantity_paper_sister * cost_per_paper_sister
def cost_pen_haley := quantity_pen_haley * cost_per_pen_haley
def cost_pen_sister := quantity_pen_sister * cost_per_pen_sister

-- Define total costs
def total_cost_haley := cost_paper_haley + cost_pen_haley
def total_cost_sister := cost_paper_sister + cost_pen_sister
def total_cost := total_cost_haley + total_cost_sister

-- Theorem statement
theorem total_amount_spent_on_supplies : total_cost = 38 := by
  -- Calculations
  have h1 : cost_paper_haley = 2 * 3.5 := rfl
  have h2 : cost_paper_sister = 3 * 4.25 := rfl
  have h3 : cost_pen_haley = 5 * 1.25 := rfl
  have h4 : cost_pen_sister = 8 * 1.5 := rfl
  have h5 : total_cost_haley = (2 * 3.5) + (5 * 1.25) := by rw [h1, h3]; rfl
  have h6 : total_cost_sister = (3 * 4.25) + (8 * 1.5) := by rw [h2, h4]; rfl
  have : total_cost = ((2 * 3.5) + (5 * 1.25)) + ((3 * 4.25) + (8 * 1.5)) := by rw [h5, h6]; rfl
  have : 38 = ((2 * 3.5) + (5 * 1.25)) + ((3 * 4.25) + (8 * 1.5)) := rfl
  rw this
  rfl


end total_amount_spent_on_supplies_l638_638992


namespace unique_triangle_shape_l638_638716

theorem unique_triangle_shape (ratio_of_two_sides_and_included_angle : Prop) 
    (ratios_of_three_angle_bisectors : Prop)
    (ratios_of_three_medians : Prop)
    (ratios_of_two_altitudes_and_bases : Prop)
    (two_angles_and_ratio_of_side_sum : Prop) : 
    ∃ triangle_shape, 
    ratios_of_three_angle_bisectors ∧
    ¬ratio_of_two_sides_and_included_angle ∧
    ¬ratios_of_three_medians ∧
    ¬ratios_of_two_altitudes_and_bases ∧
    ¬two_angles_and_ratio_of_side_sum :=
sorry

end unique_triangle_shape_l638_638716


namespace glass_volume_is_230_l638_638452

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638452


namespace cars_return_to_start_l638_638498

theorem cars_return_to_start (n : ℕ) (h : n > 0)
    (initial_positions : fin n → ℝ) (initial_directions : fin n → bool)
    (speed : ℝ) (h_speed : speed > 0)
    (h_period : ∀ (i : fin n), initial_positions i < 1)
    (h_meeting : ∀ t i j, (∃ t,
        initial_positions i + t * speed % 1 = initial_positions j + t * speed % 1) →
        initial_directions i ≠ initial_directions j) :
    ∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∀ i : fin n, (∃ t : ℝ, t = d → 
    initial_positions i + d * speed % 1 = initial_positions i) :=
begin
  -- The proof will go here
  sorry
end

end cars_return_to_start_l638_638498


namespace third_pedal_triangle_is_similar_to_initial_triangle_l638_638906

variables {P : Point}
variables {A0 B0 C0 : Point}
def H0 := Triangle.mk A0 B0 C0

-- Define Hn+1 from Hn and P
def pedal_triangle (P : Point) (Hn : Triangle) : Triangle :=
  let A := perpendicular_projection P Hn.bc
  let B := perpendicular_projection P Hn.ca
  let C := perpendicular_projection P Hn.ab
  Triangle.mk A B C

def H1 := pedal_triangle P H0
def H2 := pedal_triangle P H1
def H3 := pedal_triangle P H2

-- Theorem statement
theorem third_pedal_triangle_is_similar_to_initial_triangle
  (P : Point) (H0 : Triangle)
  (H1 := pedal_triangle P H0)
  (H2 := pedal_triangle P H1)
  (H3 := pedal_triangle P H2)
  [nondegenerate H0] [nondegenerate H1] [nondegenerate H2] [nondegenerate H3] :
  similar H3 H0 := sorry

end third_pedal_triangle_is_similar_to_initial_triangle_l638_638906


namespace balls_of_yarn_per_sweater_l638_638509

-- Define the conditions as constants
def cost_per_ball := 6
def sell_price_per_sweater := 35
def total_gain := 308
def number_of_sweaters := 28

-- Define a function that models the total gain given the number of balls of yarn per sweater.
def total_gain_formula (x : ℕ) : ℕ :=
  number_of_sweaters * (sell_price_per_sweater - cost_per_ball * x)

-- State the theorem which proves the number of balls of yarn per sweater
theorem balls_of_yarn_per_sweater (x : ℕ) (h : total_gain_formula x = total_gain): x = 4 :=
sorry

end balls_of_yarn_per_sweater_l638_638509


namespace productivity_comparison_l638_638375

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l638_638375


namespace price_of_each_sundae_l638_638363

theorem price_of_each_sundae 
    (num_ice_cream_bars : ℕ) 
    (num_sundaes : ℕ) 
    (total_price : ℝ) 
    (price_per_ice_cream_bar : ℝ) 
    (h_ice_cream_bars : num_ice_cream_bars = 125) 
    (h_sundaes : num_sundaes = 125) 
    (h_total_price : total_price = 250) 
    (h_price_per_ice_cream_bar : price_per_ice_cream_bar = 0.60) : 
    (price_per_sundae : ℝ) 
    (h_price_per_sundae : price_per_sundae = (total_price - (num_ice_cream_bars * price_per_ice_cream_bar)) / num_sundaes) 
    : price_per_sundae = 1.4 :=
by
    sorry

end price_of_each_sundae_l638_638363


namespace mid_of_XY_l638_638682

variables {A B C D M P X Q Y : Type} 
variables [Coord A] [Coord B] [Coord C] [Coord D]
variables [Point A] [Point B] [Point C] [Point D] [Point M] [Point P]
variables [Line A B] [Line C D] [Line P D] [Line P M] [Line D Q] [Line A C]

-- Given that AB is parallel to CD
axiom parallel_AB_CD : parallel (line A B) (line C D)

-- Given that M is the midpoint of segment AB
axiom is_midpoint_M_AB : midpoint M (segment A B)

-- Given P is a point on segment BC 
axiom point_P_on_BC : on P (segment B C)

-- X is the intersection of line PD and AB
axiom X_is_intersection_PD_AB : intersection X (line P D) (line A B)

-- Q is the intersection of line PM and AC
axiom Q_is_intersection_PM_AC : intersection Q (line P M) (line A C)

-- Y is the intersection of line DQ and AB
axiom Y_is_intersection_DQ_AB : intersection Y (line D Q) (line A B)

-- Prove that M is the midpoint of segment XY
theorem mid_of_XY : midpoint M (segment X Y) :=
sorry

end mid_of_XY_l638_638682


namespace odd_sum_probability_in_3x3_grid_l638_638635

theorem odd_sum_probability_in_3x3_grid :
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let odd_numbers := {1, 3, 5, 7, 9}
  let even_numbers := {2, 4, 6, 8}
  (probability (grid : Fin 3 × Fin 3 → ℤ) 
    (∀ i, (∑ j, grid (i, j)) ∈ odd_numbers ∧ ∀ j, (∑ i, grid (i, j)) ∈ odd_numbers ∧
          ∀ i j, grid (i, j) ∈ numbers ∧ ∀ i j k l, grid (i, j) ≠ grid (k, l))) = 1/21 :=
sorry

end odd_sum_probability_in_3x3_grid_l638_638635


namespace supplement_of_angle_l638_638192

theorem supplement_of_angle (θ : ℝ) 
  (h_complement: θ = 90 - 30) : 180 - θ = 120 :=
by
  sorry

end supplement_of_angle_l638_638192


namespace magpies_gather_7_trees_magpies_not_gather_6_trees_l638_638491

-- Define the problem conditions.
def trees (n : ℕ) := (∀ (i : ℕ), i < n → ∃ (m : ℕ), m = i * 10)

-- Define the movement condition for magpies.
def magpie_move (n : ℕ) (d : ℕ) :=
  (∀ (i j : ℕ), i < n ∧ j < n ∧ i ≠ j → ∃ (k : ℕ), k = d ∧ ((i + d < n ∧ j - d < n) ∨ (i - d < n ∧ j + d < n)))

-- Prove that all magpies can gather on one tree for 7 trees.
theorem magpies_gather_7_trees : 
  ∃ (i : ℕ), i < 7 ∧ trees 7 ∧ magpie_move 7 (i * 10) → True :=
by
  -- proof steps here, which are not necessary for the task
  sorry

-- Prove that all magpies cannot gather on one tree for 6 trees.
theorem magpies_not_gather_6_trees : 
  ∀ (i : ℕ), i < 6 ∧ trees 6 ∧ magpie_move 6 (i * 10) → False :=
by
  -- proof steps here, which are not necessary for the task
  sorry

end magpies_gather_7_trees_magpies_not_gather_6_trees_l638_638491


namespace glass_volume_230_l638_638444

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638444


namespace max_x_plus_y_l638_638696

theorem max_x_plus_y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 9) (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7 / 3 :=
sorry

end max_x_plus_y_l638_638696


namespace equal_angles_in_right_triangle_l638_638264

theorem equal_angles_in_right_triangle 
  {A B C M N K : Type} [EuclideanGeometry A B C M N K]
  (hC : angle A B M = 90)
  (hM : midpoint B C M)
  (hN : midpoint A B N)
  (hK : ∃ K, angle (KAM) = angle (KCA))
  : angle (KCM) = angle (KAM) := 
sorry

end equal_angles_in_right_triangle_l638_638264


namespace problem1_xy_xplusy_l638_638399

theorem problem1_xy_xplusy (x y: ℝ) (h1: x * y = 5) (h2: x + y = 6) : x - y = 4 ∨ x - y = -4 := 
sorry

end problem1_xy_xplusy_l638_638399


namespace binomial_coefficients_sum_binomial_sum_512_no_constant_term_largest_coefficient_term_not_4th_l638_638736

open Nat

theorem binomial_coefficients_sum : 
  let x : ℝ := 1
  let expansion := (x - 1/x)^9
  (Σ i in range 10, (choose 9 i) * (-1)^i * x^(9-2*i)) = 0 :=
sorry

theorem binomial_sum_512 : 
  ∑ i in range 10, choose 9 i = 512 :=
by
  simp only [Nat.choose_succ_self_left, choose]
  simp only [Nat.pow_succ, Nat.pow_zero, Nat.sum_range_succ]
  simp only [Nat.add_right_eq_self, eq_self_iff_true, Nat.sum_single]
  norm_num
  rw [← pow_add]
  exact 2^9 = 512
  split
      
theorem no_constant_term : 
  let expansion := (λ r, (choose 9 r) * (-1)^r * x^(9 - 2 * r))
  ∀ r : ℕ, 9 - 2*r ≠ 0 :=
by
  intro r
  simp only [expansion]
  simp only [Ne.def, Nat.ne_iff]
  intro h
  exact h

theorem largest_coefficient_term_not_4th : 
  let x := 1
  let expansion := (x - 1/x)^9
  let coeffs := λ r, choose 9 r * (-1)^r
  ∀ r : ℕ, coeffs 4 ≤ coeffs r :=
sorry

end binomial_coefficients_sum_binomial_sum_512_no_constant_term_largest_coefficient_term_not_4th_l638_638736


namespace XY_sum_l638_638210

theorem XY_sum (A B C D X Y : ℕ) 
  (h1 : A + B + C + D = 22) 
  (h2 : X = A + B) 
  (h3 : Y = C + D) 
  : X + Y = 4 := 
  sorry

end XY_sum_l638_638210


namespace sqrt_eq_power_of_two_l638_638898

theorem sqrt_eq_power_of_two : 
  (√2 : ℝ) = 2^(1/6 : ℝ) := 
sorry

end sqrt_eq_power_of_two_l638_638898


namespace transform_h_x_l638_638317

noncomputable def f : ℝ → ℝ := sorry

def h (x : ℝ) : ℝ := -f (5 - x)

theorem transform_h_x (x : ℝ) : 
  h x = -f (5 - x) := 
by
  -- We would provide the steps for the proof here.
  sorry

end transform_h_x_l638_638317


namespace like_terms_exponent_difference_l638_638613

theorem like_terms_exponent_difference {x y : ℕ} (hx : x = 4) (hy : y = 3) :
  (y - x) ^ 2023 = -1 := 
by
  rw [hx, hy]
  simp
  sorry

end like_terms_exponent_difference_l638_638613


namespace glass_volume_230_l638_638442

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638442


namespace committee_probability_l638_638752

/--
Suppose there are 24 members in a club: 12 boys and 12 girls.
A 5-person committee is chosen at random.
Prove that the probability of having at least 2 boys and at least 2 girls in the committee is 121/177.
-/
theorem committee_probability :
  let boys := 12
  let girls := 12
  let total_members := 24
  let committee_size := 5
  let all_ways := Nat.choose total_members committee_size
  let invalid_ways := 2 * Nat.choose boys committee_size + 2 * (Nat.choose boys 1 * Nat.choose girls 4)
  let valid_ways := all_ways - invalid_ways
  let probability := valid_ways / all_ways
  probability = 121 / 177 :=
by
  sorry

end committee_probability_l638_638752


namespace general_term_formula_exponential_seq_l638_638563

variable (n : ℕ)

def exponential_sequence (a1 r : ℕ) (n : ℕ) : ℕ := a1 * r^(n-1)

theorem general_term_formula_exponential_seq :
  exponential_sequence 2 3 n = 2 * 3^(n-1) :=
by
  sorry

end general_term_formula_exponential_seq_l638_638563


namespace six_digit_start_5_no_12_digit_perfect_square_l638_638828

theorem six_digit_start_5_no_12_digit_perfect_square :
  ∀ (n : ℕ), (500000 ≤ n ∧ n < 600000) → 
  (∀ (m : ℕ), n * 10^6 + m ≠ k^2) :=
by
  sorry

end six_digit_start_5_no_12_digit_perfect_square_l638_638828


namespace simplify_expression_correct_l638_638743

noncomputable def simplify_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :=
  ((a / b) * ((b - (4 * (a^6) / b^3)) ^ (1 / 3))
    - a^2 * ((b / a^6 - (4 / b^3)) ^ (1 / 3))
    + (2 / (a * b)) * ((a^3 * b^4 - 4 * a^9) ^ (1 / 3))) /
    ((b^2 - 2 * a^3) ^ (1 / 3) / b^2)

theorem simplify_expression_correct (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expr a b ha hb = (a + b) * ((b^2 + 2 * a^3) ^ (1 / 3)) :=
sorry

end simplify_expression_correct_l638_638743


namespace Xiaoming_winning_random_l638_638646

theorem Xiaoming_winning_random :
  ∀ (bag : Finset ℕ) (draw_one : ∀ (ball ∈ bag), Finset ℕ)
  (prob : (n : ℕ) × (k : ℕ) in ball, k / n),
  ∃ (prob : Prop), 
  bag.card > 0 ∧ 1 ≤ draw_one -> prob :=
sorry

end Xiaoming_winning_random_l638_638646


namespace evaluate_expression_l638_638088

theorem evaluate_expression : 
  (let a := 2023 - 1910 + 5 in (a ^ 2) / 121) = 114 + 70 / 121 := 
by
  let a := 2023 - 1910 + 5
  calc
    (a ^ 2) / 121 = sorry -- proof steps go here

end evaluate_expression_l638_638088


namespace two_digit_multiples_of_6_and_9_l638_638607

theorem two_digit_multiples_of_6_and_9 : ∃ n : ℕ, n = 5 ∧ (∀ k : ℤ, 10 ≤ k ∧ k < 100 ∧ (k % 6 = 0) ∧ (k % 9 = 0) → 
    k = 18 ∨ k = 36 ∨ k = 54 ∨ k = 72 ∨ k = 90) := 
sorry

end two_digit_multiples_of_6_and_9_l638_638607


namespace find_x_value_l638_638323

theorem find_x_value (a b x : ℤ) (h : a * b = (a - 1) * (b - 1)) (h2 : x * 9 = 160) :
  x = 21 :=
sorry

end find_x_value_l638_638323


namespace find_lambda_l638_638179

open Real

def vector (α : Type) := α × α

noncomputable def lambda (a b : vector ℝ) (λ : ℝ) := 
  (a.1 + λ*b.1, a.2 + 2*λ*b.2)

theorem find_lambda :
  let a : vector ℝ := (2, 3)
  let b : vector ℝ := (1, 2)
  ∃ λ : ℝ, (a.1 + λ * b.1 + a.2 + 2 * λ * b.2 = 0) → λ = -(5 / 3) :=
begin
  sorry
end

end find_lambda_l638_638179


namespace smallest_n_gt_100_l638_638844

def isFriendly (n : ℕ) (seq : List ℕ) : Prop :=
  ∀ i < n, 0 < i → seq[i] = 1 ∨ seq[i-1] = 1 ∨ (i+1 < n ∧ seq[i+1] = 1)

noncomputable def F (n : ℕ) : ℕ :=
  -- The code here would calculate the number of friendly sequences of length n.
  sorry

theorem smallest_n_gt_100 : ∃ n : ℕ, n ≥ 2 ∧ F n > 100 :=
  exists.intro 11 (and.intro (by norm_num) (by norm_num))

end smallest_n_gt_100_l638_638844


namespace P_2007_greater_P_2008_l638_638875

noncomputable def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2008) * ∑ k in finset.range 2008, P (n - k)

theorem P_2007_greater_P_2008 : P 2007 > P 2008 := 
sorry

end P_2007_greater_P_2008_l638_638875


namespace sin_minus_cos_eq_pm_one_l638_638551

theorem sin_minus_cos_eq_pm_one (α : ℝ) (h : sin α + cos α = 1) : sin α - cos α = 1 ∨ sin α - cos α = -1 :=
by
  sorry

end sin_minus_cos_eq_pm_one_l638_638551


namespace tic_tac_toe_ways_l638_638638

theorem tic_tac_toe_ways :
  let n := 4 in
  let k := 4 in
  let remaining := n * n - k in
  (2 * nat.choose remaining k + 8 * nat.choose remaining k = 4950) := 
  by sorry

end tic_tac_toe_ways_l638_638638


namespace cylindrical_tin_diameter_l638_638309

noncomputable def diameter_of_cylinder (h : ℝ) (V : ℝ) : ℝ :=
  2 * real.sqrt (V / (real.pi * h))

theorem cylindrical_tin_diameter :
  diameter_of_cylinder 5 125.00000000000001 ≈ 5.64 := 
by
  sorry

end cylindrical_tin_diameter_l638_638309


namespace ethan_hours_per_day_l638_638531

-- Define the known constants
def hourly_wage : ℝ := 18
def work_days_per_week : ℕ := 5
def total_earnings : ℝ := 3600
def weeks_worked : ℕ := 5

-- Define the main theorem
theorem ethan_hours_per_day :
  (∃ hours_per_day : ℝ, 
    hours_per_day = total_earnings / (weeks_worked * work_days_per_week * hourly_wage)) →
  hours_per_day = 8 :=
by
  sorry

end ethan_hours_per_day_l638_638531


namespace glass_volume_l638_638460

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638460


namespace range_of_a_l638_638167

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2 * Real.log x
noncomputable def h (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x y, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (1 / Real.exp 1) ≤ y ∧ y ≤ Real.exp 1 ∧ f x a = g x ∧ f y a = g y → x ≠ y) →
  1 < a ∧ a ≤ (1 / Real.exp 2) + 2 :=
sorry

end range_of_a_l638_638167


namespace people_in_room_l638_638182

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end people_in_room_l638_638182


namespace urea_production_number_of_moles_of_urea_l638_638523

def ammonia := ℝ -- represent moles of Ammonia (NH3)
def carbon_dioxide := ℝ -- represent moles of CO2
def water := ℝ -- represent moles of H2O
def ammonium_carbonate := ℝ -- represent moles of (NH4)2CO3
def ammonium_hydroxide := ℝ -- represent moles of NH4OH
def urea := ℝ -- represent moles of Urea (NH2-CO-NH2)

-- initial conditions
def init_ammonia : ammonia := 2
def init_carbon_dioxide : carbon_dioxide := 1
def init_water : water := 1

-- reaction definitions
def reaction_1 (nh3 : ammonia) (co2 : carbon_dioxide) : ammonium_carbonate :=
  if nh3 ≥ 2 * co2 then co2 else nh3 / 2

def reaction_2 (nh4_2co3 : ammonium_carbonate) (h2o : water) : (ammonium_hydroxide × carbon_dioxide) :=
  if h2o ≥ nh4_2co3 then (2 * nh4_2co3, nh4_2co3) else (2 * h2o, h2o)

def reaction_3 (nh4oh : ammonium_hydroxide) (co2 : carbon_dioxide) : urea :=
  if nh4oh ≥ co2 then co2 else nh4oh

theorem urea_production : urea :=
  let nh4_2co3 := reaction_1 init_ammonia init_carbon_dioxide
  let (nh4oh, co2) := reaction_2 nh4_2co3 init_water
  reaction_3 nh4oh co2

theorem number_of_moles_of_urea : urea_production = 1 :=
sorry

end urea_production_number_of_moles_of_urea_l638_638523


namespace angles_of_triangle_A_l638_638720

-- Define the problem's conditions
variables {A B C A' B' C' : Type}
variables (α β γ : ℝ) 
variables (h1 : α + β + γ = 2 * Real.pi)

-- Define vertices A', B', and C' and angles at these vertices
def isosceles_triangle_A'BC : Prop := -- some definition
def isosceles_triangle_AB'C : Prop := -- some definition
def isosceles_triangle_ABC' : Prop := -- some definition

-- Prove that the angles of the triangle A'B'C' are α/2, β/2, and γ/2
theorem angles_of_triangle_A'B'C' 
    (hA' : isosceles_triangle_A'BC)
    (hB' : isosceles_triangle_AB'C)
    (hC' : isosceles_triangle_ABC') :
  angles_of_triangle A' B' C' = (α/2, β/2, γ/2) :=
sorry

end angles_of_triangle_A_l638_638720


namespace nonnegative_difference_of_roots_l638_638811

theorem nonnegative_difference_of_roots :
  ∀ (x : ℝ), x^2 + 40 * x + 300 = -50 → (∃ a b : ℝ, x^2 + 40 * x + 350 = 0 ∧ x = a ∧ x = b ∧ |a - b| = 25) := 
by 
sorry

end nonnegative_difference_of_roots_l638_638811


namespace men_meet_at_9am_l638_638421

-- Define the start and end times for both men
def start_time_man1 : ℕ := 6 -- in hours
def end_time_man1 : ℕ := 10 -- in hours
def start_time_man2 : ℕ := 8 -- in hours
def end_time_man2 : ℕ := 12 -- in hours

-- Define the speeds and meeting time
def speed (D : ℝ) (t : ℕ) : ℝ := D / t
def relative_speed (D : ℝ) : ℝ := speed D 4 + speed D 4
def meeting_time (start_time : ℕ) (relative_speed : ℝ) (remaining_distance : ℝ) : ℕ :=
  start_time + (remaining_distance / relative_speed).toNat

theorem men_meet_at_9am (D : ℝ) : meeting_time start_time_man2 (relative_speed D) (D / 2) = 9 :=
by
  sorry

end men_meet_at_9am_l638_638421


namespace initial_values_of_N_l638_638041

theorem initial_values_of_N {N : ℕ} : 
  (machine_output (machine_output (machine_output (machine_output N))) = 10) ↔ (N = 3 ∨ N = 160) :=
sorry

def machine_output (N : ℕ) : ℕ :=
if N % 2 = 0 then N / 2 else 3 * N + 1

end initial_values_of_N_l638_638041


namespace units_digit_fib_cycle_length_60_l638_638751

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib n + fib (n+1)

-- Define the function to get the units digit (mod 10)
def units_digit_fib (n : ℕ) : ℕ :=
  (fib n) % 10

-- State the theorem about the cycle length of the units digits in Fibonacci sequence
theorem units_digit_fib_cycle_length_60 :
  ∃ k, k = 60 ∧ ∀ n, units_digit_fib (n + k) = units_digit_fib n := sorry

end units_digit_fib_cycle_length_60_l638_638751


namespace next_elements_l638_638908

-- Define the conditions and the question
def next_elements_in_sequence (n : ℕ) : String :=
  match n with
  | 1 => "О"  -- "Один"
  | 2 => "Д"  -- "Два"
  | 3 => "Т"  -- "Три"
  | 4 => "Ч"  -- "Четыре"
  | 5 => "П"  -- "Пять"
  | 6 => "Ш"  -- "Шесть"
  | 7 => "С"  -- "Семь"
  | 8 => "В"  -- "Восемь"
  | _ => "?"

theorem next_elements (n : ℕ) :
  next_elements_in_sequence 7 = "С" ∧ next_elements_in_sequence 8 = "В" := by
  sorry

end next_elements_l638_638908


namespace part1_part2_l638_638574

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | m - 3 ≤ x ∧ x ≤ m + 3}
noncomputable def C : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem part1 (m : ℝ) (h : A ∩ B m = C) : m = 5 :=
  sorry

theorem part2 (m : ℝ) (h : A ⊆ (B m)ᶜ) : m < -4 ∨ 6 < m :=
  sorry

end part1_part2_l638_638574


namespace determine_b_value_l638_638773

noncomputable def parabola_vertex_form (a q : ℝ) : (ℝ → ℝ) :=
  λ x, a * (x - q/2)^2 + q/2

theorem determine_b_value (a b c q : ℝ) (h1 : q ≠ 0)
  (h2 : ∀ x, parabola_vertex_form a q x = a * x^2 + b * x + c)
  (h3 : parabola_vertex_form a q 0 = -2 * q)
  (h4 : parabola_vertex_form a q (q/2) = q/2) :
  b = 10 :=
by
  sorry

end determine_b_value_l638_638773


namespace length_of_bridge_is_230_l638_638002

noncomputable def train_length : ℚ := 145
noncomputable def train_speed_kmh : ℚ := 45
noncomputable def time_to_cross_bridge : ℚ := 30
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 1000) / 3600
noncomputable def bridge_length : ℚ := (train_speed_ms * time_to_cross_bridge) - train_length

theorem length_of_bridge_is_230 :
  bridge_length = 230 :=
sorry

end length_of_bridge_is_230_l638_638002


namespace triangle_conditions_l638_638199

noncomputable def measure_angle_A (A : ℝ) (π : ℝ) : Prop :=
  A = π / 6

noncomputable def area_of_triangle (a b c A B C π : ℝ) : Prop :=
  let s := (a + b + c) / 2 in
  let area := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = sqrt (3) / 4

theorem triangle_conditions 
  (a b c A B C π : ℝ)
  (h1 : sin (A + π / 3) = 4 * sin (A / 2) * cos (A / 2))
  (h2 : sin B = sqrt (3) * sin C) 
  (h3 : a = 1) :
  measure_angle_A A π ∧ area_of_triangle a b c A B C π :=
by 
  sorry

end triangle_conditions_l638_638199


namespace glass_volume_l638_638463

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638463


namespace comparison_of_a_b_c_l638_638127

noncomputable def a : ℝ := 2018 ^ (1 / 2018)
noncomputable def b : ℝ := Real.logb 2017 (Real.sqrt 2018)
noncomputable def c : ℝ := Real.logb 2018 (Real.sqrt 2017)

theorem comparison_of_a_b_c :
  a > b ∧ b > c :=
by
  -- Definitions
  have def_a : a = 2018 ^ (1 / 2018) := rfl
  have def_b : b = Real.logb 2017 (Real.sqrt 2018) := rfl
  have def_c : c = Real.logb 2018 (Real.sqrt 2017) := rfl

  -- Sorry is added to skip the proof
  sorry

end comparison_of_a_b_c_l638_638127


namespace length_of_greater_segment_l638_638366

theorem length_of_greater_segment (x : ℤ) (h1 : (x + 2)^2 - x^2 = 32) : x + 2 = 9 := by
  sorry

end length_of_greater_segment_l638_638366


namespace triangle_BC_length_l638_638631

theorem triangle_BC_length  {A B C : Type} [Field B] [Field C]
  (angleA : B) (AB AC BC : C) 
  (h1 : angleA = 60) (h2 : AB = 2) (h3 : (1/2) * AB * AC * sin angleA = (Real.sqrt 3 / 2)) :
  BC = Real.sqrt 3 := 
begin
  sorry
end

end triangle_BC_length_l638_638631


namespace tan_double_angle_l638_638939

theorem tan_double_angle (α : ℝ) (h : sin α = -√3 * cos α) : tan (2 * α) = √3 := 
by
  sorry

end tan_double_angle_l638_638939


namespace part1_part2_l638_638560

noncomputable def sequence_sn (n : ℕ) : ℕ := 12 * n - n^2

theorem part1 :
  (∀ n : ℕ, a_n = sequence_sn n - sequence_sn (n-1) ∧ a_n = 13 - 2 * n) ∧
  (∀ n : ℕ, a_{n+1} - a_n = -2) :=
sorry

noncomputable def c_n (n : ℕ) : ℕ := 12 - (13 - 2 * n)

theorem part2 :
  (T_n : ℕ → ℕ) : (∀ n : ℕ, T_n n = ∑ i in finset.range n, 1 / (c_n i * c_n (i+1))) →
  (T_n n = n / (2 * n + 1)) :=
sorry

end part1_part2_l638_638560


namespace angie_age_l638_638802

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l638_638802


namespace sufficient_but_not_necessary_condition_l638_638394

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end sufficient_but_not_necessary_condition_l638_638394


namespace prime_sum_l638_638191

theorem prime_sum (m n : ℕ) (hm : Prime m) (hn : Prime n) (h : 5 * m + 7 * n = 129) :
  m + n = 19 ∨ m + n = 25 := by
  sorry

end prime_sum_l638_638191


namespace f_decreases_on_01e_l638_638158

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_decreases_on_01e : ∀ x ∈ Set.Ioo 0 (1 / Real.exp 1), Deriv f x < 0 := 
by 
  sorry

end f_decreases_on_01e_l638_638158


namespace scientific_notation_of_4212000_l638_638634

theorem scientific_notation_of_4212000 :
  4212000 = 4.212 * 10^6 :=
by
  sorry

end scientific_notation_of_4212000_l638_638634


namespace calculate_expression_l638_638506

theorem calculate_expression : 
  |(-3)| - 2 * Real.tan (Real.pi / 4) + (-1:ℤ)^(2023) - (Real.sqrt 3 - Real.pi)^(0:ℤ) = -1 :=
  by
  sorry

end calculate_expression_l638_638506


namespace shaded_fraction_is_4_over_15_l638_638478

-- Define the geometric series sum function
def geom_series_sum (a r : ℝ) (hr : |r| < 1) : ℝ := a / (1 - r)

-- The target statement for the given problem
theorem shaded_fraction_is_4_over_15 :
  let a := (1 / 4 : ℝ)
  let r := (1 / 16 : ℝ)
  geom_series_sum a r (by norm_num : |r| < 1) = (4 / 15 : ℝ) :=
by
  -- Proof is omitted with sorry
  sorry

end shaded_fraction_is_4_over_15_l638_638478


namespace min_tablets_needed_l638_638022

theorem min_tablets_needed 
  (medA : ℕ) (medB : ℕ) (medC : ℕ) 
  (hA : medA = 25) 
  (hB : medB = 30) 
  (hC : medC = 20) : 
  ∃ n, n = 55 ∧ (∀ x y z, 0 ≤ x ∧ x < 2 → 0 ≤ y ∧ y < 2 → 0 ≤ z ∧ z < 2 → x + y + z = n) :=
by 
  use 55
  split
  . refl
  . intros x y z hx hy hz
  sorry

end min_tablets_needed_l638_638022


namespace snakelike_integers_1000_9999_distinct_l638_638893

def is_snake_like (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∀ i : ℕ, i < digits.length - 1 → 
    (odd i → digits.nth i < digits.nth (i + 1)) ∧ 
    (even i → digits.nth i > digits.nth (i + 1))

def count_snake_like_integers (start : ℕ) (end_ : ℕ) : ℕ :=
  (start to end_).count (λ n, is_snake_like n ∧ (1 ≤ n.digits.length ≤ 4) ∧ digits.nodup)

theorem snakelike_integers_1000_9999_distinct :
  count_snake_like_integers 1000 9999 = 882 :=
sorry

end snakelike_integers_1000_9999_distinct_l638_638893


namespace intersect_parabola_line_l638_638194

noncomputable def y_x_line (p : ℝ) (x : ℝ) : ℝ := 2*x + p/2
noncomputable def x_parabola (p : ℝ) (y : ℝ) : ℝ := sqrt(2*p*y)

theorem intersect_parabola_line (p : ℝ) (A B : ℝ × ℝ) 
  (hp : p > 0)
  (hA : A.2 = y_x_line p A.1) 
  (hB : B.2 = y_x_line p B.1)
  (hAP : A.1^2 = 2 * p * A.2)
  (hBP : B.1^2 = 2 * p * B.2) :
  |A.2 + B.2 + p| = 10 * p :=
sorry

end intersect_parabola_line_l638_638194


namespace math_problem_l638_638351

theorem math_problem (a b : ℕ) (h₁ : a = 6) (h₂ : b = 6) : 
  (a^3 + b^3) / (a^2 - a * b + b^2) = 12 :=
by
  sorry

end math_problem_l638_638351


namespace vector_b_satisfies_conditions_l638_638686

open Matrix Vec

noncomputable def vector_a : Matrix (Fin 3) (Fin 1) ℝ := 
  ![3, 2, 4]

noncomputable def vector_b : Matrix (Fin 3) (Fin 1) ℝ := 
  ![1, 4, 1 / 2]

def dot_product (u v : Matrix (Fin 3) (Fin 1) ℝ) : ℝ := 
  (u 0 0 * v 0 0) + (u 1 0 * v 1 0) + (u 2 0 * v 2 0)

def cross_product (u v : Matrix (Fin 3) (Fin 1) ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  ![(u 1 0 * v 2 0 - u 2 0 * v 1 0), 
    (u 2 0 * v 0 0 - u 0 0 * v 2 0), 
    (u 0 0 * v 1 0 - u 1 0 * v 0 0)]

theorem vector_b_satisfies_conditions :
  dot_product vector_a vector_b = 14 ∧ 
  cross_product vector_a vector_b = ![-11, -5, 2] := 
  sorry

end vector_b_satisfies_conditions_l638_638686


namespace max_odd_sums_l638_638058

theorem max_odd_sums (f : Fin 998 → ℕ) (h_range : ∀ i, 1000 ≤ f i ∧ f i ≤ 1997) :
  ∃ g : Fin 998.succ → Fin 998, ∀ i : Fin 998.succ, (∑ j in range 3, f (g ((i+j) % 998))) % 2 = 1 :=
sorry

end max_odd_sums_l638_638058


namespace brandon_skittles_final_l638_638062
-- Conditions
def brandon_initial_skittles := 96
def brandon_lost_skittles := 9

-- Theorem stating the question and answer
theorem brandon_skittles_final : brandon_initial_skittles - brandon_lost_skittles = 87 := 
by
  -- Proof steps go here
  sorry

end brandon_skittles_final_l638_638062


namespace find_n_l638_638164

theorem find_n (n : ℝ) (h1 : ∀ x y : ℝ, (n + 1) * x^(n^2 - 5) = y) 
               (h2 : ∀ x > 0, (n + 1) * x^(n^2 - 5) > 0) :
               n = 2 :=
by
  sorry

end find_n_l638_638164


namespace roots_of_varying_signs_l638_638084

theorem roots_of_varying_signs :
  (∃ x : ℝ, (4 * x^2 - 8 = 40 ∧ x != 0) ∧
           (∃ y : ℝ, (3 * y - 2)^2 = (y + 2)^2 ∧ y != 0) ∧
           (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z1 = 0 ∨ z2 = 0) ∧ x^3 - 8 * x^2 + 13 * x + 10 = 0)) :=
sorry

end roots_of_varying_signs_l638_638084


namespace carlos_welfare_deduction_l638_638899

-- Define Carlos' hourly wage in dollars
def carlosHourlyWage : ℝ := 25

-- Define the welfare fund deduction rate as a percentage
def welfareFundRate : ℝ := 1.6 / 100

-- Define the conversion from dollars to cents
def dollarsToCents (d : ℝ) : ℝ := d * 100

-- Calculate the amount in cents per hour from Carlos' wages dedicated to the welfare fund
def amountDedicatedToWelfare : ℝ := (dollarsToCents carlosHourlyWage) * welfareFundRate

-- Statement to prove
theorem carlos_welfare_deduction : amountDedicatedToWelfare = 40 := 
  sorry

end carlos_welfare_deduction_l638_638899


namespace P2007_gt_P2008_l638_638868

namespace ProbabilityProblem

def probability (k : ℕ) : ℝ := sorry  -- Placeholder for the probability function

axiom probability_rec :
  ∀ n, probability n = (1 / 2007) * (∑ k in finset.range 2007, probability (n - (k + 1)))

axiom P0 :
  probability 0 = 1

theorem P2007_gt_P2008 : probability 2007 > probability 2008 := sorry

end ProbabilityProblem

end P2007_gt_P2008_l638_638868


namespace math_proof_problem_l638_638959

noncomputable def parabola_focus : Prop :=
  ∃ (p : ℝ), (∀ x y : ℝ, (y^2 = 2 * p * x) ↔ (1 = 1)) ∧ (∃ (f : ℝ × ℝ), f = (1/2, 0) ∧ (y^2 = 2 * x))

noncomputable def circle_through_origin : Prop :=
  ∀ E A B M N O : ℝ × ℝ,
    E = (2,2) → 
    (∃ p : ℝ, E ∈ parabola_focus) →
    ∃ l : ℝ × ℝ, l = (2,0) ∧ l intersects parabola_focus at (A, B) ∧
    ∃ y : ℝ, M = (something to do with A on line y = -2) ∧
    ∃ y : ℝ, N = (something to do with B on line y = -2) ∧
    O = (0,0) →
    circle diameter (M, N) passes through O

-- Combining both parts into a single definition
theorem math_proof_problem (E A B M N O : ℝ × ℝ) : parabola_focus ∧ circle_through_origin :=
by
  sorry

end math_proof_problem_l638_638959


namespace optimal_green_tiles_l638_638517

variable (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ)

def conditions (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ) :=
  n_indigo ≥ n_red + n_orange + n_yellow + n_green + n_blue ∧
  n_blue ≥ n_red + n_orange + n_yellow + n_green ∧
  n_green ≥ n_red + n_orange + n_yellow ∧
  n_yellow ≥ n_red + n_orange ∧
  n_orange ≥ n_red ∧
  n_red + n_orange + n_yellow + n_green + n_blue + n_indigo = 100

theorem optimal_green_tiles : 
  conditions n_red n_orange n_yellow n_green n_blue n_indigo → 
  n_green = 13 := by
    sorry

end optimal_green_tiles_l638_638517


namespace cylindrical_coordinates_conversion_l638_638909

theorem cylindrical_coordinates_conversion :
  ∃ (r θ z : ℝ), 
    (r = 6 * Real.sqrt 2) ∧ 
    (θ = 7 * Real.pi / 4) ∧ 
    (z = 10) ∧ 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ((6:ℝ, -6, 10) = (r * Real.cos θ, r * Real.sin θ, z)) :=
by
  sorry

end cylindrical_coordinates_conversion_l638_638909


namespace maximum_value_l638_638116

noncomputable def max_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : Prop :=
  (abcd * (a + b + c + d) / ((a + b)^3 * (b + c)^3) ≤ 4 / 9)

theorem maximum_value (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : max_value_inequality a b c d ha hb hc hd :=
sorry

end maximum_value_l638_638116


namespace prob_one_qualified_factory_a_prob_random_light_bulb_qualified_l638_638060

-- Define the conditions
def factory_a_market_share : ℝ := 0.6
def factory_b_market_share : ℝ := 0.4
def factory_a_qualification_rate : ℝ := 0.9
def factory_b_qualification_rate : ℝ := 0.8

-- Statements to prove

-- Probability that exactly one of two light bulbs from Factory A is qualified
theorem prob_one_qualified_factory_a :
  (C 2 1) * factory_a_qualification_rate * (1 - factory_a_qualification_rate) = 0.18 :=
sorry

-- Probability that a randomly purchased light bulb is qualified
theorem prob_random_light_bulb_qualified :
  (factory_a_market_share * factory_a_qualification_rate) + 
  (factory_b_market_share * factory_b_qualification_rate) = 0.86 :=
sorry

end prob_one_qualified_factory_a_prob_random_light_bulb_qualified_l638_638060


namespace factorization_l638_638090

theorem factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 :=
by sorry

end factorization_l638_638090


namespace Impossible_to_collect_all_water_in_one_bucket_l638_638059

/-- 
  Given 2018 buckets containing 1, 2, ..., 2018 liters of water respectively. 
  Arsenius can pour into the second bucket exactly as much water as is already 
  in the second bucket from the first bucket. All buckets are large enough to 
  hold all the water they get. It is impossible to collect all the water in one bucket.
-/
theorem Impossible_to_collect_all_water_in_one_bucket :
  let total_water := (2018 * 2019) / 2,
      buckets := List.range' 1 2018,
      pour_water (a b : ℕ) : ℕ := 2 * b in
  total_water % 2 = 1 → -- the total amount of water is odd
  (∀ steps, ∃ buckets_after_steps, List.sum buckets_after_steps ≠ total_water) := 
by
  sorry

end Impossible_to_collect_all_water_in_one_bucket_l638_638059


namespace value_of_a_if_1_in_S_l638_638598

variable (a : ℤ)
def S := { x : ℤ | 3 * x + a = 0 }

theorem value_of_a_if_1_in_S (h : 1 ∈ S a) : a = -3 :=
sorry

end value_of_a_if_1_in_S_l638_638598


namespace compare_P2007_P2008_l638_638863

def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2007) * ∑ k in (finset.range 2007).image (λ x, x + n + 1 - 2007), P k

theorem compare_P2007_P2008 : P 2007 > P 2008 :=
sorry

end compare_P2007_P2008_l638_638863


namespace part_I_part_II_l638_638142

open Complex

noncomputable def modulus_of_z (z : ℂ) : ℝ :=
  abs z

noncomputable def z_conjugate_equation (z : ℂ) : Prop :=
  z + 2*conj(z) = (5 + I) / (1 + I)

theorem part_I (z : ℂ) (hz : z_conjugate_equation z) : modulus_of_z z = sqrt 5 := by
  sorry

def z_point_first_quadrant (z : ℂ) (m : ℝ) : Prop :=
  let w := z * (2 - m * I)
  0 < w.re ∧ 0 < w.im

theorem part_II (m : ℝ) (hz : z = 1 + 2*I) : -1 < m ∧ m < 4 := by
  have hw : z_point_first_quadrant (1 + 2*I) m := by
    sorry
  sorry

end part_I_part_II_l638_638142


namespace ellipse_foci_on_y_axis_l638_638626

theorem ellipse_foci_on_y_axis (k : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x y, x^2 + k * y^2 = 2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ b^2 > a^2)
  → (0 < k ∧ k < 1) :=
sorry

end ellipse_foci_on_y_axis_l638_638626


namespace productivity_difference_l638_638381

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l638_638381


namespace jackson_pays_2100_l638_638668

def tile_cost (length : ℝ) (width : ℝ) (tiles_per_sqft : ℝ) (percent_green : ℝ) (cost_green : ℝ) (cost_red : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * percent_green
  let red_tiles := total_tiles - green_tiles
  let cost_green_total := green_tiles * cost_green
  let cost_red_total := red_tiles * cost_red
  cost_green_total + cost_red_total

theorem jackson_pays_2100 :
  tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by
  sorry

end jackson_pays_2100_l638_638668


namespace labor_cost_per_hour_l638_638234

theorem labor_cost_per_hour (total_repair_cost part_cost labor_hours : ℕ)
    (h1 : total_repair_cost = 2400)
    (h2 : part_cost = 1200)
    (h3 : labor_hours = 16) :
    (total_repair_cost - part_cost) / labor_hours = 75 := by
  sorry

end labor_cost_per_hour_l638_638234


namespace L2_possible_equations_l638_638698

-- Lean 4 statement for the provided mathematical proof problem.
def point := ℝ × ℝ

def L1 (x y : ℝ) : Prop := 6 * x - y + 6 = 0

def P : point := (-1, 0)
def Q : point := (0, 6)

def triangle_area (A B C : point) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def area_OPQ : ℝ := triangle_area (0, 0) P Q
def area_QRS (R S : point) : ℝ := triangle_area Q R S

theorem L2_possible_equations (m : ℝ) (b : ℝ) (L2 : ∀ x : ℝ, y = m * x + b) :
  L2 (1, 0) ∧ (area_OPQ = 6 * area_QRS (0, -m) (-(m + 6) / (6 - m), -12 * m / (6 - m))) →
  (L2 = (λ x : ℝ, -3 * x + 3) ∨ L2 = (λ x : ℝ, -10 * x + 10)) := 
sorry

end L2_possible_equations_l638_638698


namespace solve_for_x_l638_638109

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l638_638109


namespace probability_seventh_week_A_l638_638790

/-- Define the problem parameters -/
def codes : Finset ℕ := {0, 1, 2, 3}

/-- Define that code A is used in week 1 -/
def code_first_week : ℕ := 0

/-- Define the random selection rule for subsequent weeks -/
def prob_not_prev_code (prev_code : ℕ) : ℕ → ℚ :=
  λ code, if code ≠ prev_code then (1/3 : ℚ) else 0

/-- Define the probability of selecting code A in the seventh week -/
def probability_A_seventh_week (init_code : ℕ) : ℚ :=
  let prob_week : ℕ → ℚ → ℚ
    | 0, p := p
    | k + 1, p := 
      (1/3 : ℚ) * (1 - 
        prob_week k ((1/4 : ℚ) * (-1/3 : ℚ) ^ (k - 1) + 1/4 : ℚ)
      )
  in prob_week 6 1

/-- Formal statement of the problem --/
theorem probability_seventh_week_A : probability_A_seventh_week code_first_week = (61 / 243 : ℚ) :=
  sorry

end probability_seventh_week_A_l638_638790


namespace nitrogen_mass_percentage_in_ammonium_phosphate_l638_638915

def nitrogen_mass_percentage
  (molar_mass_N : ℚ)
  (molar_mass_H : ℚ)
  (molar_mass_P : ℚ)
  (molar_mass_O : ℚ)
  : ℚ :=
  let molar_mass_NH4 := molar_mass_N + 4 * molar_mass_H
  let molar_mass_PO4 := molar_mass_P + 4 * molar_mass_O
  let molar_mass_NH4_3_PO4 := 3 * molar_mass_NH4 + molar_mass_PO4
  let mass_N_in_NH4_3_PO4 := 3 * molar_mass_N
  (mass_N_in_NH4_3_PO4 / molar_mass_NH4_3_PO4) * 100

theorem nitrogen_mass_percentage_in_ammonium_phosphate
  (molar_mass_N : ℚ := 14.01)
  (molar_mass_H : ℚ := 1.01)
  (molar_mass_P : ℚ := 30.97)
  (molar_mass_O : ℚ := 16.00)
  : nitrogen_mass_percentage molar_mass_N molar_mass_H molar_mass_P molar_mass_O = 28.19 :=
by
  sorry

end nitrogen_mass_percentage_in_ammonium_phosphate_l638_638915


namespace max_purple_cards_l638_638270

-- Define conditions
def initial_red_cards : ℕ := 100
def initial_blue_cards : ℕ := 100

-- Exchange rules
def exchange_red_cards : ℕ := 2  -- 2 red cards for 1 blue card and 1 purple card
def exchange_blue_cards : ℕ := 3  -- 3 blue cards for 1 red card and 1 purple card

-- Question: Proving maximum number of purple cards
theorem max_purple_cards (initial_red initial_blue : ℕ) (exchange_r exchange_b : ℕ) : 
  initial_red = 100 → initial_blue = 100 → 
  exchange_r = 2 → exchange_b = 3 → 
  ∃ (max_purple : ℕ), max_purple = 138 :=
by
  intro h_red h_blue h_ex_r h_ex_b
  use 138
  sorry

end max_purple_cards_l638_638270


namespace trajectory_eqn_find_m_fixed_area_l638_638208

-- Problem 1: Proving the equation of the trajectory
theorem trajectory_eqn (x y : ℝ) (h : (sqrt((x + 1)^2 + y^2)) / abs(x + 4) = 1 / 2) :
  (x^2/4) + (y^2/3) = 1 := by
  sorry

-- Problem 2: Finding the value of m
theorem find_m (m : ℝ) (h1 : 0 < m) (h2 : m < 2) (N : ℝ × ℝ) (H : ∀ (x y : ℝ), (x, y) = N →
  (x - m)^2 + y^2 ≥ 1):
  m = 1 := by
  sorry

-- Problem 3: Proving the fixed area of the quadrilateral
theorem fixed_area (A B A1 B1 : ℝ × ℝ) (hA : pow((A.2), 2) = 3 * (1 - (pow(A.1, 2) / 4)))
  (hB : pow((B.2), 2) = 3 * (1 - (pow(B.1, 2) / 4))) 
  (hSlope : (A.2 * B.2) / (A.1 * B.1) = -3/4)
  (H: (A.1^2 + B.1^2 = 4)) : 
  let S := 4 * Real.sqrt(3) in
  S = 4 * Real.sqrt(3) := by 
  sorry

end trajectory_eqn_find_m_fixed_area_l638_638208


namespace circle_center_equal_coordinates_l638_638652

theorem circle_center_equal_coordinates (D E F : ℝ)
    (h1 : D^2 ≠ E^2) (h2 : E^2 > 4*F) :
    let center_x := -D / 2 := -E / 2 :=
    sorry

end circle_center_equal_coordinates_l638_638652


namespace triangle_a_range_l638_638662

def triangle_inequality_lemma (a b c: ℝ) (A B C: ℝ) : Prop :=
    A + B + C = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ a = b ∧ B = 60 ∧ b = √3 

theorem triangle_a_range (a: ℝ):
  triangle_inequality_lemma a :=
  0 < a ∧ a ≤√3 ∨ a = 2 :=
sorry

end triangle_a_range_l638_638662


namespace prove_equal_product_of_roots_sum_of_roots_l638_638169

section QuadraticEquation

variable {a b c : ℝ} (h : a ≠ 0)

def original_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

def transformed_eq (k : ℝ) (y : ℝ) : ℝ :=
  a * (y + k)^2 + b * (y + k) + c

def product_of_roots (f : ℝ → ℝ) : ℝ :=
  let r1 := (-b + Mathlib.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let r2 := (-b - Mathlib.sqrt (b^2 - 4 * a * c)) / (2 * a)
  r1 * r2

theorem prove_equal_product_of_roots :
  ∃ k, product_of_roots (original_eq) h = product_of_roots (transformed_eq k) h := sorry

theorem sum_of_roots (k : ℝ) :
  k = 0 → (product_of_roots original_eq h + product_of_roots (transformed_eq k) h = -2 * (b / a)) ∧
  k = -b / a → (product_of_roots original_eq h + product_of_roots (transformed_eq k) h = 0) := sorry

end QuadraticEquation

end prove_equal_product_of_roots_sum_of_roots_l638_638169


namespace centers_and_midpoints_form_square_l638_638952

-- Define the vertices of the triangle and points of the squares
variables {A B C M N P Q : Type} [AffineSpace B]
variables [vectorSpaceℝ B]
variables [point : has_coords A B]
variables [point : has_coords B B]
variables [point : has_coords C B]
variables [point : has_coords M B]
variables [point : has_coords N B]
variables [point : has_coords P B]
variables [point : has_coords Q B]

-- Given conditions
def is_square (A B C D : Type) [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace D] : Prop :=
∥B - A∥ = ∥C - B∥ ∧ ∥C - B∥ = ∥D - C∥ ∧ ∥D - C∥ = ∥A - D∥ ∧
angle B A D = 90˚ ∧ angle C B A = 90˚ ∧ angle D C B = 90˚ ∧ angle A D C = 90˚

-- The proof problem
theorem centers_and_midpoints_form_square :
  ∀ (A B C M N P Q : Type) [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace M] [AffineSpace N] [AffineSpace P] [AffineSpace Q]
  [vectorSpaceℝ B] [point : has_coords A B] [point : has_coords B B] [point : has_coords C B] [point : has_coords M B] [point : has_coords N B]
  [point : has_coords P B] [point : has_coords Q B],
  is_square (midpoint A M) (center_sq ABMN B) (midpoint M P) (center_sq BCPQ Q) :=
sorry

end centers_and_midpoints_form_square_l638_638952


namespace limit_calculation_l638_638155

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.log (3 * x) + 8 * x

theorem limit_calculation :
  (Real.limit (fun (Δx : ℝ) => (f (1 - 2 * Δx) - f 1) / Δx) 0 = -20) :=
by
  -- Derivation steps omitted
  sorry

end limit_calculation_l638_638155


namespace sum_of_elements_in_M_value_of_a_if_M_eq_N_l638_638709

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + a * Real.cos x

def M (a : ℝ) : Set ℝ :=
  {x | f x a = 0}

def N (a : ℝ) : Set ℝ :=
  {x | f (f x a) a = 0}

theorem sum_of_elements_in_M (a : ℝ) :
  ∑ x in (M a).toFinset, x = 0 :=
sorry

theorem value_of_a_if_M_eq_N :
  ∀ a : ℝ, M a = N a → a = 0 :=
sorry

end sum_of_elements_in_M_value_of_a_if_M_eq_N_l638_638709


namespace fourier_sine_transform_of_f_l638_638103

noncomputable def F (p : ℝ) : ℝ :=
if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℝ :=
sqrt (2 / Real.pi) * (1 - Real.cos x) / x

theorem fourier_sine_transform_of_f :
  (∀ p : ℝ, 0 < p ∧ p < 1 → (∫ x in 0..∞, f x * Real.sin (p * x) dx) = 1) ∧
  (∀ p : ℝ, 1 < p → (∫ x in 0..∞, f x * Real.sin (p * x) dx) = 0) :=
sorry

end fourier_sine_transform_of_f_l638_638103


namespace total_number_of_ways_l638_638337

-- Definitions corresponding to problem's conditions
def Balls := {A, B, C}
def Boxes := {2, 3, 4}

-- Total number of ways to place the balls such that box 1 has no balls is 27
theorem total_number_of_ways : fintype.card (Balls → Boxes) = 27 := 
by 
  sorry

end total_number_of_ways_l638_638337


namespace sqrt_meaningful_l638_638615

theorem sqrt_meaningful (x : ℝ) (h : x ≥ 3) : ∃ y : ℝ, y = sqrt (x - 3) :=
by
  sorry

end sqrt_meaningful_l638_638615


namespace glass_volume_230_l638_638432

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638432


namespace t_value_l638_638701

noncomputable def polynomial_roots (p : ℕ → ℂ) : Set ℂ := sorry

theorem t_value(a b c t : ℂ) (h1 : a + b + c = 6) (h2 : a * b + a * c + b * c = 8) (h3 : a * b * c = 2)
  (h4 : t = complex.sqrt a + complex.sqrt b + complex.sqrt c) : t^4 - 12 * t^2 - 4 * t = -4 := 
  sorry

end t_value_l638_638701


namespace integer_part_of_expression_l638_638321

noncomputable def expression := 
  10 * 75 * ∏ k in (finset.range 99).map (λ i, i+1), (1 + 1 / ((k: ℝ) * (k + 2)))

theorem integer_part_of_expression : (expression).toInt = 1 :=
by
  sorry

end integer_part_of_expression_l638_638321


namespace brick_width_l638_638848

-- Define the dimensions of the wall
def L_wall : Real := 750 -- length in cm
def W_wall : Real := 600 -- width in cm
def H_wall : Real := 22.5 -- height in cm

-- Define the dimensions of the bricks
def L_brick : Real := 25 -- length in cm
def H_brick : Real := 6 -- height in cm

-- Define the number of bricks needed
def n_bricks : Nat := 6000

-- Define the total volume of the wall
def V_wall : Real := L_wall * W_wall * H_wall

-- Define the volume of one brick
def V_brick (W : Real) : Real := L_brick * W * H_brick

-- Statement to prove
theorem brick_width : 
  ∃ W : Real, V_wall = V_brick W * (n_bricks : Real) ∧ W = 11.25 := by 
  sorry

end brick_width_l638_638848


namespace integral_solution_pairs_l638_638926

theorem integral_solution_pairs :
  { (p, q) : ℕ × ℕ // p > 0 ∧ q > 0 ∧
    ∃ a b c d : ℤ, (a + b = p ∧ a * b = q ∧ c + d = q ∧ c * d = p) } = 
  {(4, 4), (6, 5), (5, 6)} :=
begin
  sorry
end

end integral_solution_pairs_l638_638926


namespace n_mul_1_expression_l638_638075

def operation (n : ℕ) : ℕ := 
match n with
| 1 => 1
| (n+1) => 3 * (operation n)

theorem n_mul_1_expression (n : ℕ) : operation n = 3^(n-1) :=
by sorry

end n_mul_1_expression_l638_638075


namespace average_age_is_35_l638_638224

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l638_638224


namespace sufficient_but_not_necessary_condition_l638_638006

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ (|x| > 1 → (x > 1 ∨ x < -1)) ∧ ¬(|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l638_638006


namespace length_of_QZ_l638_638213

theorem length_of_QZ (AB_parallel_YZ : AB ∥ YZ) 
  (AZ : ℝ) (BQ : ℝ) (QY : ℝ) (hAZ : AZ = 60) 
  (hBQ : BQ = 15) (hQY : QY = 30) : QZ = 40 := 
sorry

end length_of_QZ_l638_638213


namespace circle_equation_trajectory_of_point_P_l638_638760

variables (M N A O : Point)
variables (x y : ℝ)

-- Defining points M, N, and the center of circle C
def Point (x y : ℝ) := (x, y)

-- Circle C passes through points M(5, 2) and N(3, 2)
def circle_through_points (M N : Point) : Prop :=
  ∃ (center : Point) (r : ℝ), (M ≠ N) ∧ (is_center_on_x_axis center) ∧
  (distance M center = r) ∧ (distance N center = r)

-- The center of circle C is on the x-axis
def is_center_on_x_axis : Point → Prop
| (x, 0) := true
| _ := false

-- Problem 1: Prove the equation of the circle C
theorem circle_equation :
  circle_through_points (Point 5 2) (Point 3 2) →
  ∃ center : Point, (center = (4, 0)) ∧ (∀ x y : ℝ, (x-4)^2 + y^2 = 5) :=
sorry

-- Problem 2: Prove the equation of the trajectory of point P
theorem trajectory_of_point_P (A P : Point) :
  (distance O A = distance A P) →
  (trajectory_of_point_C (Point 8 0) (Point 0 0)) :=
sorry

end circle_equation_trajectory_of_point_P_l638_638760


namespace sum_of_ages_l638_638671

variable (J L : ℝ)
variable (h1 : J = L + 8)
variable (h2 : J + 10 = 5 * (L - 5))

theorem sum_of_ages (J L : ℝ) (h1 : J = L + 8) (h2 : J + 10 = 5 * (L - 5)) : J + L = 29.5 := by
  sorry

end sum_of_ages_l638_638671


namespace fill_time_six_faucets_to_fill_25_gallon_l638_638338

variables {faucets : ℕ → ℤ} -- faucet count
variables {time : ℚ} -- time in minutes
variables {volume : ℚ} -- volume in gallons

-- Condition and definitions
def fill_time (f : ℕ) (v : ℚ) : ℚ := time
def faucet_rate (f : ℕ) (v : ℚ) : ℚ := v / time

axiom faucet3_100 (t : ℚ) : fill_time 3 100 = 6
axiom faucet_dispense_rate : ∀ (f : ℕ) (v : ℚ), faucet_rate f v = faucet_rate f 100

-- Goal
theorem fill_time_six_faucets_to_fill_25_gallon : 
  fill_time 6 25 * 60 = 45 :=
by {
  sorry 
}

end fill_time_six_faucets_to_fill_25_gallon_l638_638338


namespace simplify_T_l638_638700

theorem simplify_T (x : ℝ) : 
  let T := (x+2)^4 - 4*(x+2)^3 + 6*(x+2)^2 - 4*(x+2) + 1 in
  T = (x + 1)^4 :=
by
  sorry

end simplify_T_l638_638700


namespace tangent_point_intersect_two_points_l638_638986

noncomputable def f (x : ℝ) : ℝ := x^2 + x * Real.sin x + Real.cos x

theorem tangent_point (a b : ℝ) (h_tangent : ∀ x, deriv f x = (λ x, 2*x + x * Real.cos x) x) 
  (h_tangent_at_a : deriv f a = 0) 
  (h_b : b = f a) : a = 0 ∧ b = 1 :=
begin
  sorry,
end

theorem intersect_two_points (b : ℝ) (h_deriv : ∀ x, deriv f x = (λ x, 2*x + x * Real.cos x) x) 
  (h_minimum : f 0 = 1) : b > 1 ↔ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = b ∧ f x2 = b) :=
begin
  sorry,
end

end tangent_point_intersect_two_points_l638_638986


namespace total_paintable_area_is_2006_l638_638920

-- Define the dimensions of the bedrooms and the hallway
def bedroom_length := 14
def bedroom_width := 11
def bedroom_height := 9

def hallway_length := 20
def hallway_width := 7
def hallway_height := 9

def num_bedrooms := 4
def doorway_window_area := 70

-- Compute the areas of the bedroom walls and the hallway walls
def bedroom_wall_area : ℕ :=
  2 * (bedroom_length * bedroom_height) +
  2 * (bedroom_width * bedroom_height)

def paintable_bedroom_wall_area : ℕ :=
  bedroom_wall_area - doorway_window_area

def total_paintable_bedroom_area : ℕ :=
  num_bedrooms * paintable_bedroom_wall_area

def hallway_wall_area : ℕ :=
  2 * (hallway_length * hallway_height) +
  2 * (hallway_width * hallway_height)

-- Compute the total paintable area
def total_paintable_area : ℕ :=
  total_paintable_bedroom_area + hallway_wall_area

-- Theorem stating the total paintable area is 2006 sq ft
theorem total_paintable_area_is_2006 : total_paintable_area = 2006 := 
  by
    unfold total_paintable_area
    rw [total_paintable_bedroom_area, paintable_bedroom_wall_area, bedroom_wall_area]
    rw [hallway_wall_area]
    norm_num
    sorry -- Proof omitted

end total_paintable_area_is_2006_l638_638920


namespace sufficient_not_necessary_condition_l638_638904

variable (a b : ℝ)

theorem sufficient_not_necessary_condition (h1 : a > b) (h2 : b > 0) :
  (a > b > 0) → (1/a < 1/b) ∧ ¬((1/a < 1/b) → (a > b)) :=
by
  sorry

end sufficient_not_necessary_condition_l638_638904


namespace part1_inequality_solution_set_part2_inequality_l638_638591

-- Part 1: Proving the inequality solution set
theorem part1_inequality_solution_set (x : ℝ) : 
  let f := λ x : ℝ, abs (x + 1) in
  f (x + 8) ≥ 10 - f x → (x ≤ -10 ∨ x ≥ 0) :=
by sorry

-- Part 2: Proving the inequality with given conditions
theorem part2_inequality (x y : ℝ) (hx: abs x > 1) (hy: abs y < 1) : 
  let f := λ x : ℝ, abs (x + 1) in
  f y < abs x * f (y / x^2) :=
by sorry

end part1_inequality_solution_set_part2_inequality_l638_638591


namespace system_of_equations_solution_system_of_inequalities_solution_l638_638745

theorem system_of_equations_solution (x y : ℝ) :
  (3 * x - 4 * y = 1) → (5 * x + 2 * y = 6) → 
  x = 1 ∧ y = 0.5 := by
  sorry

theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x + 6 > 0) → (x - 2 < -x) → 
  -2 < x ∧ x < 1 := by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l638_638745


namespace interval_of_decrease_l638_638989

noncomputable def f (x : ℝ) : ℝ := abs (tan ((1 / 2) * x - (π / 6)))

theorem interval_of_decrease (k : ℤ) :
  ∀ x : ℝ, 2 * k * π - (2 * π / 3) < x ∧ x ≤ 2 * k * π + π / 3 → ((f x) ' < 0) :=
sorry

end interval_of_decrease_l638_638989


namespace number_solution_l638_638820

theorem number_solution : ∃ x : ℝ, x + 9 = x^2 ∧ x = (1 + Real.sqrt 37) / 2 :=
by
  use (1 + Real.sqrt 37) / 2
  simp
  sorry

end number_solution_l638_638820


namespace min_value_proof_l638_638657

open Classical

variable (a : Nat → ℕ)
variable (m n : ℕ)
variable {q : ℕ}

axiom geom_seq (q_pos : 0 < q) : ∀ (n : ℕ), a (n + 1) = a 1 * q^n

axiom condition1 (h1 : a 2016 = a 2015 + 2 * a 2014) : True

axiom condition2 (h2 : ∀ m n, a m * a n = 16 * a 1 ^ 2) : m + n = 6

noncomputable def min_value : ℚ := 
  let frac_sum := (4 / m : ℚ) + (1 / n : ℚ)
  frac_sum

theorem min_value_proof 
  (q_eq : q = 2)
  (mn_eq : m + n = 6) :
  ∀ m n, min_value m n = 3 / 2 := 
  by
    sorry

end min_value_proof_l638_638657


namespace sum_of_ages_l638_638829

theorem sum_of_ages (S F : ℕ) 
  (h1 : F - 18 = 3 * (S - 18)) 
  (h2 : F = 2 * S) : S + F = 108 := by 
  sorry

end sum_of_ages_l638_638829


namespace determine_m_and_roots_l638_638524

theorem determine_m_and_roots 
    (m : ℤ) 
    (p : Polynomial ℤ := Polynomial.X ^ 3 - (m^2 - m + 7) * Polynomial.X - (3 * m^2 - 3 * m - 6)) 
    (h: p.eval (-1) = 0) : 
    (m = 3 ∨ m = -2) ∧ (∀ m, m = 3 ∨ m = -2 → 
        let q : Polynomial ℤ := p / (Polynomial.X + 1);
        (q = Polynomial.X^2 - Polynomial.X - 12 ∧ 
            (q = (Polynomial.X + 3) * (Polynomial.X - 4))) :=
by sorry

end determine_m_and_roots_l638_638524


namespace tile_area_is_one_l638_638530

def area_of_each_tile
    (length width : ℝ)
    (num_tiles : ℕ)
    (fraction_tiled : ℝ) 
    (room_area : length * width = 240) 
    (tiled_area : fraction_tiled * (length * width) = 40) 
    (fraction_tiled_val : fraction_tiled = 1 / 6)
    (num_tiles_val : num_tiles = 40) :
    (tile_area : ℝ) := 
  sorry

-- Theorem statement to prove that the area of each tile is 1 square foot
theorem tile_area_is_one
    (length : ℝ)
    (width : ℝ) 
    (num_tiles : ℕ)
    (fraction_tiled : ℝ)
    (room_area : length * width = 240)
    (tiled_area : fraction_tiled * (length * width) = 40)
    (fraction_tiled_val : fraction_tiled = 1 / 6)
    (num_tiles_val : num_tiles = 40) :
    area_of_each_tile length width num_tiles fraction_tiled room_area tiled_area fraction_tiled_val num_tiles_val = 1 :=
  sorry

end tile_area_is_one_l638_638530


namespace ball_drawing_valid_combinations_l638_638202

theorem ball_drawing_valid_combinations :
  ∃ (W R B : ℕ), 
  (W ∈ {3, 4, 5, 6, 7}) ∧
  (R ∈ {2, 3, 4, 5}) ∧
  (B ∈ {0, 1, 2, 3}) ∧ 
  (W + R + B = 10) ∧ 
  (finset.card { (W, R, B) | (W ∈ {3, 4, 5, 6, 7}) ∧ 
                               (R ∈ {2, 3, 4, 5}) ∧ 
                               (B ∈ {0, 1, 2, 3}) ∧ 
                               (W + R + B = 10) } = 14) := 
sorry

end ball_drawing_valid_combinations_l638_638202


namespace total_votes_l638_638825

/-- Let V be the total number of votes. Define the votes received by the candidate and rival. -/
def votes_cast (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) : Prop :=
  votes_candidate = 40 * V / 100 ∧ votes_rival = votes_candidate + 2000 ∧ votes_candidate + votes_rival = V

/-- Prove that the total number of votes is 10000 given the conditions. -/
theorem total_votes (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) :
  votes_cast V votes_candidate votes_rival → V = 10000 :=
by
  sorry

end total_votes_l638_638825


namespace compound_interest_calculation_l638_638330

theorem compound_interest_calculation :
  let P_SI := 1750.0000000000018
  let r_SI := 0.08
  let t_SI := 3
  let r_CI := 0.10
  let t_CI := 2
  let SI := P_SI * r_SI * t_SI
  let CI (P_CI : ℝ) := P_CI * ((1 + r_CI) ^ t_CI - 1)
  (SI = 420.0000000000004) →
  (SI = (1 / 2) * CI P_CI) →
  P_CI = 4000.000000000004 :=
by
  intros P_SI r_SI t_SI r_CI t_CI SI CI h1 h2
  sorry

end compound_interest_calculation_l638_638330


namespace kibble_ratio_l638_638714

theorem kibble_ratio (x : ℕ) (morning evening afternoon remaining : ℕ) :
  let total_kibble_given := 5,
      mary_given := morning + evening,
      frank_given := afternoon + x,
      initial_kibble := 12,
      final_kibble := remaining
  in mary_given = 2 ∧ afternoon = 1 ∧ initial_kibble - remaining = total_kibble_given ∧ total_kibble_given - mary_given - afternoon = x
  → (x = 2) → (2 / 1 = 2) :=
by
sory

end kibble_ratio_l638_638714


namespace sqrt_eq_sum_iff_nonneg_l638_638526

theorem sqrt_eq_sum_iff_nonneg (a b : ℝ) : sqrt(a^2 + b^2 + 2 * a * b) = a + b ↔ a + b ≥ 0 :=
by
  sorry

end sqrt_eq_sum_iff_nonneg_l638_638526


namespace degree_of_monomial_l638_638761

def monomial := -((5 : ℚ) * a * (b ^ 3))/8

theorem degree_of_monomial (a b : ℚ) : 
  degree monomial = 4 := 
  sorry

end degree_of_monomial_l638_638761


namespace right_triangle_angle_ACB_l638_638687

theorem right_triangle_angle_ACB (A B C D E F : Point)
  (AD BE CF : Line)
  (hAD : is_altitude A D)
  (hBE : is_altitude B E)
  (hCF : is_altitude C F)
  (hBAC : angle A B C = 90)
  (vector_eq : 6 * (vector A D) + 3 * (vector B E) + 2 * (vector C F) = 0) :
  angle A C B = 90 := sorry

end right_triangle_angle_ACB_l638_638687


namespace beth_finishes_first_l638_638494

variable (x y : ℝ) -- x: representing the area of Beth's lawn, y: representing Andy's mowing rate
variable (A : Type) [LinearOrder A] [AdditiveGroup A]

-- Andy
noncomputable def AndysLawn : ℝ := 3 * x
noncomputable def AndysRate : ℝ := y
noncomputable def AndysTime : ℝ := AndysLawn / AndysRate + 1/2

-- Beth
noncomputable def BethsLawn : ℝ := x
noncomputable def BethsRate : ℝ := y / 2
noncomputable def BethsTime : ℝ := BethsLawn / BethsRate

-- Carlos
noncomputable def CarlosLawn : ℝ := 3 * x / 4
noncomputable def CarlosRate : ℝ := y / 4
noncomputable def CarlosTime : ℝ := CarlosLawn / CarlosRate

theorem beth_finishes_first
  (h_andys_lawn : AndysLawn = 3 * x)
  (h_beths_lawn : BethsLawn = x)
  (h_carlos_lawn : CarlosLawn = 3 * x / 4)
  (h_andys_rate : AndysRate = y)
  (h_beths_rate : BethsRate = y / 2)
  (h_carlos_rate : CarlosRate = y / 4)
  (h_andys_time : AndysTime = AndysLawn / AndysRate + 1/2)
  (h_beths_time : BethsTime = BethsLawn / BethsRate)
  (h_carlos_time : CarlosTime = CarlosLawn / CarlosRate) :
  BethsTime < CarlosTime ∧ BethsTime < AndysTime :=
by
  sorry

end beth_finishes_first_l638_638494


namespace option_b_correct_l638_638185

theorem option_b_correct (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3: a ≠ 1) (h4: b ≠ 1) (h5 : 0 < m) (h6 : m < 1) :
  m^a < m^b :=
sorry

end option_b_correct_l638_638185


namespace find_f_of_4_l638_638154

-- Define the function f under the given condition
def f : ℝ → ℝ := λ y, 
  if h : ∃ x : ℝ, 3 * x + 1 = y then 
    let ⟨x, hx⟩ := h in x^2 + 3 * x + 2 
  else 
    0 -- Define 0 for other values, as they are not specified

-- The theorem statement to prove
theorem find_f_of_4 : f 4 = 6 := 
  by 
    sorry

end find_f_of_4_l638_638154


namespace problem_statement_l638_638932

def k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ ∃ m : ℕ, m = k ∧ ∏ i in range (m + 1), (k * (i + 1) + 1) = N

def count_k_nice (n k : ℕ) : ℕ :=
  let upper_bound := n - 1
  in (upper_bound / k) + 1

def main_problem (limit : ℕ) : ℕ :=
  let count_6_nice := count_k_nice limit 6
  let count_9_nice := count_k_nice limit 9
  let count_18_nice := count_k_nice limit 18
  limit - (count_6_nice + count_9_nice - count_18_nice)

theorem problem_statement : main_problem 1200 = 934 := by
  /- The proof goes here -/
  sorry

end problem_statement_l638_638932


namespace volume_region_lte_six_l638_638015

theorem volume_region_lte_six :
  (volume {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p in 
    |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 ∧ 
    x ≤ y ∧ y ≤ z 
  } = 4) :=
sorry

end volume_region_lte_six_l638_638015


namespace average_minutes_run_per_day_l638_638499

theorem average_minutes_run_per_day (e : ℕ)
  (sixth_grade_avg : ℕ := 16)
  (seventh_grade_avg : ℕ := 18)
  (eighth_grade_avg : ℕ := 12)
  (sixth_graders : ℕ := 3 * e)
  (seventh_graders : ℕ := 2 * e)
  (eighth_graders : ℕ := e) :
  ((sixth_grade_avg * sixth_graders + seventh_grade_avg * seventh_graders + eighth_grade_avg * eighth_graders)
   / (sixth_graders + seventh_graders + eighth_graders) : ℕ) = 16 := 
by
  sorry

end average_minutes_run_per_day_l638_638499


namespace glass_volume_correct_l638_638471

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638471


namespace solve_trigonometric_system_l638_638294

theorem solve_trigonometric_system 
  (x y : ℝ) 
  (h1 : (sin x + cos x) / (sin y + cos y) + (sin y - cos y) / (sin x + cos x) = 1 / (sin (x + y) + cos (x - y)))
  (h2 : 2 * (sin x + cos x)^2 - (2 * cos y^2 + 1) = real.sqrt 3 / 2) :
  (∃ n : ℤ, x = 30 * real.pi / 180 + n * 180 * real.pi / 180 ∨ x = 60 * real.pi / 180 + n * 180 * real.pi / 180)
  ∧ (∃ m : ℤ, y = 15 * real.pi / 180 + m * 180 * real.pi / 180 ∨ y = 165 * real.pi / 180 + m * 180 * real.pi / 180) :=
by sorry

end solve_trigonometric_system_l638_638294


namespace boys_of_other_communities_l638_638641

axiom total_boys : ℕ
axiom muslim_percentage : ℝ
axiom hindu_percentage : ℝ
axiom sikh_percentage : ℝ

noncomputable def other_boy_count (total_boys : ℕ) 
                                   (muslim_percentage : ℝ) 
                                   (hindu_percentage : ℝ) 
                                   (sikh_percentage : ℝ) : ℝ :=
  let total_percentage := muslim_percentage + hindu_percentage + sikh_percentage
  let other_percentage := 1 - total_percentage
  other_percentage * total_boys

theorem boys_of_other_communities : 
    other_boy_count 850 0.44 0.32 0.10 = 119 :=
  by 
    sorry

end boys_of_other_communities_l638_638641


namespace concentric_circles_properties_l638_638174

noncomputable def concentric_circles 
  (R r : ℝ) (R_gt_r : R > r) 
  (P : ℝ × ℝ) (P_on_smaller_circle : P.1^2 + P.2^2 = r^2)
  (B : ℝ × ℝ) (B_on_larger_circle : B.1^2 + B.2^2 = R^2) 
  (C : ℝ × ℝ) (C_on_larger_circle : C.1^2 + C.2^2 = R^2)
  (A : ℝ × ℝ) (A_on_smaller_circle : A.1^2 + A.2^2 = r^2)
  (BP : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ := λ p b, (b.1 - p.1, b.2 - p.2))
  (perpendicular_through_P : ℝ × ℝ → ℝ × ℝ := λ p, (-p.2, p.1)) : ℝ :=
begin
  sorry,
end

theorem concentric_circles_properties
  (R r : ℝ) (R_gt_r : R > r) 
  (P : ℝ × ℝ) (P_on_smaller_circle : P.1^2 + P.2^2 = r^2)
  (B : ℝ × ℝ) (B_on_larger_circle : B.1^2 + B.2^2 = R^2) 
  (C : ℝ × ℝ) (C_on_larger_circle : C.1^2 + C.2^2 = R^2)
  (A : ℝ × ℝ) (A_on_smaller_circle : A.1^2 + A.2^2 = r^2)
  (BP : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ := λ p b, (b.1 - p.1, b.2 - p.2))
  (perpendicular_through_P : ℝ × ℝ → ℝ × ℝ := λ p, (-p.2, p.1)) :
  (|BP P C|^2 + |perpendicular_through_P P|^2 + |BP A B|^2 = 6R^2 + 2r^2) ∧
  (trajectory_midpoint_AB : set (ℝ × ℝ) := {(x, y) | (x + r/2)^2 + y^2 = (R/2)^2}) :=
sorry

end concentric_circles_properties_l638_638174


namespace solve_y_l638_638744

theorem solve_y (y : ℝ) (h : 5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4)) : y = 6561 := 
by 
  sorry

end solve_y_l638_638744


namespace fred_sheets_l638_638936

theorem fred_sheets (initial_sheets : ℕ) (received_sheets : ℕ) (given_sheets : ℕ) :
  initial_sheets = 212 → received_sheets = 307 → given_sheets = 156 →
  (initial_sheets + received_sheets - given_sheets) = 363 :=
by
  intros h_initial h_received h_given
  rw [h_initial, h_received, h_given]
  sorry

end fred_sheets_l638_638936


namespace find_x_eq_neg15_l638_638112

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l638_638112


namespace triangle_area_l638_638011

open Real

theorem triangle_area (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 3 * sqrt 10) (h₃ : c = 3) :
  let s := (8 + 12 + 5) / 2 in
  sqrt (s * (s - 8) * (s - 12) * (s - 5)) = 15 * sqrt 15 / 4 :=
by
  have h4 : 8 = 2,
  have h5 : 12 = 3 * sqrt 10,
  have h6 : 5 = 3,
  sorry

end triangle_area_l638_638011


namespace other_solution_l638_638964

theorem other_solution (x : ℚ) (h : x = 3/7 → 42 * x^2 + 2 * x + 31 = 73 * x + 4) : x = 3/2 :=
by
  have h₁ : 42 * (3/7:ℚ)^2 + 2 * (3/7:ℚ) + 31 = 73 * (3/7:ℚ) + 4 :=
  calc
    42 * (3/7)^2 + 2 * (3/7) + 31 = 42 * (9/49) + 6/7 + 31 : by { field_simp, linarith }
    ... = 6 * (3/7) + 31 : by { norm_num, field_simp, linarith }
    ... = 31 + 18/7 - 9 : by { norm_num, field_simp, linarith }
    ... = 22 : by { ring }
  sorry

end other_solution_l638_638964


namespace basketball_team_opponent_runs_l638_638409

theorem basketball_team_opponent_runs : 
  ∀ (team_scores : List ℕ) (opponent_scores : List ℕ),
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ∧
  (∀ n ∈ [1, 2, 3, 4, 5, 6], opponent_scores[team_scores.indexOf n] = n + 2) ∧
  (∀ n ∈ [7, 8, 9, 10, 11, 12], opponent_scores[team_scores.indexOf n] = n / 3) ∧
  opponent_scores.sum = 67 :=
begin
  sorry
end

end basketball_team_opponent_runs_l638_638409


namespace remaining_three_digit_numbers_l638_638608

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_invalid_number (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A ≠ B ∧ B ≠ 0 ∧ n = 100 * A + 10 * B + A

def count_valid_three_digit_numbers : ℕ :=
  let total_numbers := 900
  let invalid_numbers := 10 * 9
  total_numbers - invalid_numbers

theorem remaining_three_digit_numbers : count_valid_three_digit_numbers = 810 := by
  sorry

end remaining_three_digit_numbers_l638_638608


namespace glass_volume_correct_l638_638469

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638469


namespace function_behavior_l638_638710

theorem function_behavior (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ (x ∈ Ioo a b), f'''' x < 0) 
(h2 : m ≤ 2) (h3 : f = λ x, (1 / 6) * x^3 - (1 / 2) * m * x^2 + x) (m : ℝ) : 
∃ (x_max : ℝ), (a < x_max ∧ x_max < b ∧ ∀ (x ∈ Ioo a x_max ∪ Ioo x_max b), f x < f x_max) ∧ 
¬ ∃ (x_min : ℝ), (a < x_min ∧ x_min < b ∧ ∀ (x ∈ Ioo a x_min ∪ Ioo x_min b), f x > f x_min) :=
sorry

end function_behavior_l638_638710


namespace angle_B_values_l638_638200

variable {A B C : ℝ} -- Assume A, B, C are real numbers representing triangle angles
variable {a b c : ℝ} -- Assume a, b, c are real numbers representing sides of the triangle

-- Adding a condition for the angles to be within bounds of a triangle: 0 < B < pi
def triangle_ABC_condition (A B C : ℝ) : Prop := 
0 < B ∧ B < Real.pi -- Extra condition to handle the inherent properties of angles in a triangle

-- The given condition in the problem
def given_condition (a b c A B : ℝ) : Prop := 
sqrt 3 * a = 2 * b * Real.sin A

-- The target proof goal derived from question and solution
theorem angle_B_values (a b c A B C : ℝ) 
  (h1 : triangle_ABC_condition A B C) 
  (h2 : given_condition a b c A B) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 := 
sorry -- Skipping the proof with sorry

end angle_B_values_l638_638200


namespace find_m_l638_638414

theorem find_m 
  {x y : ℝ} 
  (hxy : ∀x y, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1)) 
  {a b : ℝ} 
  (ha : a = x + m * y) 
  (hb : b = y + m * x) 
  (hinteger_a : is_integer a) 
  (hinteger_b : is_integer b) 
  (positive_m : m > 0)
  (time_per_pair : ℝ := 5)
  (total_time : ℝ := 595) 
  : m = 11 :=
by 
  sorry

end find_m_l638_638414


namespace f1_is_F_f2_is_F_f3_is_F_number_of_F_functions_l638_638708

def is_F_function (f : ℝ → ℝ) : Prop :=
  ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), |f x| ≤ M * |x|

def f1 (x : ℝ) : ℝ := 2 * x

def f2 (x : ℝ) : ℝ := Real.sin x + Real.cos x

def f3 (x : ℝ) : ℝ :=
if x = 0 then 0 else if x > 0 then x else -x  -- This is a placeholder; f3 should be odd with the given property.

theorem f1_is_F : is_F_function f1 := sorry

theorem f2_is_F : is_F_function f2 := sorry

theorem f3_is_F : (∀ x1 x2 : ℝ, |f3 x1 - f3 x2| ≤ 2 * |x1 - x2|) → is_F_function f3 := sorry

theorem number_of_F_functions : 
  (is_F_function f1) ∧ (is_F_function f2) ∧ (∀ x1 x2 : ℝ, |f3 x1 - f3 x2| ≤ 2 * |x1 - x2|) → 
  (is_F_function f3) → 
  3 := sorry

end f1_is_F_f2_is_F_f3_is_F_number_of_F_functions_l638_638708


namespace ellipse_equation_sum_of_squares_l638_638146

noncomputable def circle_center : ℝ × ℝ := (√3, 0)

def tangent_condition (k : ℝ) : Prop :=
  Abs (√3 * k + √3 * k) / sqrt (k^2 + 1) = √3

theorem ellipse_equation 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : a^2 - b^2 = 3)
  (h4 : ∃ k, k > 0 ∧ tangent_condition k) :
  (a = 2 ∧ b = 1) ∧ (∀ x y : ℝ, (x^2 / a^2) + y^2 = 1 ↔ x^2 / 4 + y^2 = 1) :=
sorry

theorem sum_of_squares (x1 y1 x2 y2 : ℝ)
  (h1 : (x1^2 / 4 + y1^2 = 1) ∧ (x2^2 / 4 + y2^2 = 1))
  (h2 : (x1 * x2) + 4 * (y1 * y2) = 0) :
  x1^2 + x2^2 = 8 :=
sorry

end ellipse_equation_sum_of_squares_l638_638146


namespace geometric_sequence_value_l638_638654

theorem geometric_sequence_value 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_condition : a 4 * a 6 * a 8 * a 10 * a 12 = 32) :
  (a 10 ^ 2) / (a 12) = 2 :=
sorry

end geometric_sequence_value_l638_638654


namespace count_symmetric_figures_l638_638513

structure Pentomino := 
  (squares : Finset (ℕ × ℕ)) -- each pentomino is made up of five squares

def is_reflectionally_symmetric (p : Pentomino) : Prop := 
  -- condition defining reflectional symmetry
  sorry

def is_rotationally_symmetric_180 (p : Pentomino) : Prop := 
  -- condition defining 180-degree rotational symmetry
  sorry

def has_required_symmetry (p : Pentomino) : Prop :=
  is_reflectionally_symmetric p ∨ is_rotationally_symmetric_180 p

theorem count_symmetric_figures (figures : Finset Pentomino) (h_len : figures.card = 15) :
  (figures.filter has_required_symmetry).card = 8 :=
sorry

end count_symmetric_figures_l638_638513


namespace glass_volume_correct_l638_638464

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638464


namespace problem_geometric_description_of_set_T_l638_638999

open Complex

def set_T (a b : ℝ) : ℂ := a + b * I

theorem problem_geometric_description_of_set_T :
  {w : ℂ | ∃ a b : ℝ, w = set_T a b ∧
    (im ((5 - 3 * I) * w) = 2 * re ((5 - 3 * I) * w))} =
  {w : ℂ | ∃ a : ℝ, w = set_T a (-(13/5) * a)} :=
sorry

end problem_geometric_description_of_set_T_l638_638999


namespace max_full_pikes_l638_638841

theorem max_full_pikes (initial_pikes : ℕ) (pike_full_condition : ℕ → Prop) (remaining_pikes : ℕ) 
  (h_initial : initial_pikes = 30)
  (h_condition : ∀ n, pike_full_condition n → n ≥ 3)
  (h_remaining : remaining_pikes ≥ 1) :
    ∃ max_full : ℕ, max_full ≤ 9 := 
sorry

end max_full_pikes_l638_638841


namespace productivity_comparison_l638_638388

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l638_638388


namespace incorrect_statements_l638_638163

def f (x : ℝ) : ℝ := (2 - x) * Real.exp x
def f_prime (x : ℝ) : ℝ := (1 - x) * Real.exp x

theorem incorrect_statements :
  ¬ (f_prime 2 > 0) ∧
  ¬ (∀ x, x > 1 → ((1 - x) * Real.exp x > 0)) ∧
  ¬ (∀ a, f 1 = a → a < Real.exp 1) :=
by
  -- proving the statements
  sorry

end incorrect_statements_l638_638163


namespace difference_of_A_max_A_min_l638_638723

theorem difference_of_A_max_A_min (A A_max A_min : ℝ) 
  (hfrac1 : ∃ (m n : ℕ), odd n ∧ A = m / n ∧ IsSquare m) 
  (hfrac2 : ∃ (s t : ℕ), odd t ∧ A = s / t ∧ IsSquare s)
  (hA_not_int : ¬ ∃ (k : ℤ), A = k) 
  (hA_max : A_max = 2.31)
  (hA_min : A_min = 1.99) 
  : A_max - A_min = 1 :=
by
  sorry

end difference_of_A_max_A_min_l638_638723


namespace part_a_part_b_l638_638659

-- Define the conditions
variables {A B C : Point}
variable (A' : Point)
  (hA' : IsIntersection (PerpendicularBisector AB ) (AngleBisector <| Angle BAC) A')
variable (B' : Point)
  (hB' : IsIntersection (PerpendicularBisector BC ) (AngleBisector <| Angle ABC ) B')
variable (C' : Point)
  (hC' : IsIntersection (PerpendicularBisector CA ) (AngleBisector <| Angle BCA) C')

-- Part (a)
theorem part_a : (Equilateral △ABC) ↔ (A' = B') :=
  sorry

-- Part (b)
theorem part_b (hdistinct : A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A') :
  Angle B' A' C' = 90 - (1 / 2) * Angle BAC :=
  sorry

end part_a_part_b_l638_638659


namespace glass_volume_is_230_l638_638449

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638449


namespace sin_of_right_triangle_l638_638640

theorem sin_of_right_triangle (P Q R : Type) [RightAngleTriangle P Q R]
  (angle_Q : angle Q = 90)
  (PQ : length P Q = 15)
  (PR : length P R = 8) :
  sin_of_angle R = 8/17 := 
sorry

end sin_of_right_triangle_l638_638640


namespace num_prisms_crossed_by_diagonal_l638_638335

theorem num_prisms_crossed_by_diagonal :
  ∀ (a b c L : ℕ), 
  (a = 2) → (b = 3) → (c = 5) → (L = 90) → 
  let num_prisms_intersected := L / c - 1 + L / b - 1 + L / a - 1 - ((L / (a * b)) - 1) - ((L / (b * c)) - 1) - ((L / (a * c)) - 1) + (L / (a * b * c) - 1) 
  in
  num_prisms_intersected = 65 :=
by {
  intros a b c L ha hb hc hL,
  have ha : a = 2 := ha,
  have hb : b = 3 := hb,
  have hc : c = 5 := hc,
  have hL : L = 90 := hL,
  let num_prisms_intersected := (L / c - 1) + (L / b - 1) + (L / a - 1) - ((L / (a * b) - 1)) - ((L / (b * c) - 1)) - ((L / (a * c) - 1)) + (L / (a * b * c) - 1),
  show num_prisms_intersected = 65,
  sorry
}

end num_prisms_crossed_by_diagonal_l638_638335


namespace unoccupied_garden_area_is_correct_l638_638880

noncomputable def area_unoccupied_by_pond_trees_bench (π : ℝ) : ℝ :=
  let garden_area := 144
  let pond_area_rectangle := 6
  let pond_area_semi_circle := 2 * π
  let trees_area := 3
  let bench_area := 3
  garden_area - (pond_area_rectangle + pond_area_semi_circle + trees_area + bench_area)

theorem unoccupied_garden_area_is_correct : 
  area_unoccupied_by_pond_trees_bench Real.pi = 132 - 2 * Real.pi :=
by
  sorry

end unoccupied_garden_area_is_correct_l638_638880


namespace max_female_students_min_people_in_group_l638_638043

-- Problem 1: Given z = 4, the maximum number of female students is 6
theorem max_female_students (x y : ℕ) (h1 : x > y) (h2 : y > 4) (h3 : x < 8) : y <= 6 :=
sorry

-- Problem 2: The minimum number of people in the group is 12
theorem min_people_in_group (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : 2 * z > x) : 12 <= x + y + z :=
sorry

end max_female_students_min_people_in_group_l638_638043


namespace length_BC_l638_638661

theorem length_BC (AB AC AM : ℝ)
  (hAB : AB = 5)
  (hAC : AC = 7)
  (hAM : AM = 4)
  (M_midpoint_of_BC : ∃ (BM MC : ℝ), BM = MC ∧ ∀ (BC: ℝ), BC = BM + MC) :
  ∃ (BC : ℝ), BC = 2 * Real.sqrt 21 := by
  sorry

end length_BC_l638_638661


namespace geometric_series_mod_500_l638_638902

theorem geometric_series_mod_500 :
  let R : ℕ := 9
  let N : ℕ := 2002
  let M : ℕ := 500
  let S := finset.sum (finset.range (N + 1)) (λ k, R^k)
  (S % M) = 91 :=
by
  sorry

end geometric_series_mod_500_l638_638902


namespace line_bisects_circle_perpendicular_l638_638584

theorem line_bisects_circle_perpendicular :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, x^2 + y^2 + x - 2*y + 1 = 0 → l x = y)
               ∧ (∀ x y : ℝ, x + 2*y + 3 = 0 → x ∈ { x | ∃ k:ℝ, y = -1/2 * k + l x})
               ∧ (∀ x y : ℝ, l x = 2 * x - 2)) :=
sorry

end line_bisects_circle_perpendicular_l638_638584


namespace timber_volume_after_two_years_correct_l638_638415

-- Definitions based on the conditions in the problem
variables (a p b : ℝ) -- Assume a, p, and b are real numbers

-- Timber volume after one year
def timber_volume_one_year (a p b : ℝ) : ℝ := a * (1 + p) - b

-- Timber volume after two years
def timber_volume_two_years (a p b : ℝ) : ℝ := (timber_volume_one_year a p b) * (1 + p) - b

-- Prove that the timber volume after two years is equal to the given expression
theorem timber_volume_after_two_years_correct (a p b : ℝ) :
  timber_volume_two_years a p b = a * (1 + p)^2 - (2 + p) * b := sorry

end timber_volume_after_two_years_correct_l638_638415


namespace sum_first_six_terms_geometric_seq_l638_638197

theorem sum_first_six_terms_geometric_seq (a r : ℝ)
  (h1 : a + a * r = 12)
  (h2 : a + a * r + a * r^2 + a * r^3 = 36) :
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 84 :=
sorry

end sum_first_six_terms_geometric_seq_l638_638197


namespace numberOfCompanionSets_l638_638622

def isCompanionSet (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (1 / x) ∈ A

def M : Set ℝ := {-1, 0, 1/2, 1, 2}

def companionSubsets : Set (Set ℝ) :=
  {A ∈ (Set.powerset M) | isCompanionSet A ∧ A ≠ ∅}

theorem numberOfCompanionSets : Set.card companionSubsets = 7 := by
  sorry

end numberOfCompanionSets_l638_638622


namespace fraction_subtraction_proof_l638_638352

theorem fraction_subtraction_proof : 
  (21 / 12) - (18 / 15) = 11 / 20 := 
by 
  sorry

end fraction_subtraction_proof_l638_638352


namespace triangle_is_isosceles_l638_638633

theorem triangle_is_isosceles (A B C : ℝ) (h : Real.sin A = 2 * Real.cos B * Real.sin C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l638_638633


namespace heaviest_lightest_difference_total_deviation_total_selling_price_l638_638785

-- Define the standard weight and deviations
def standard_weight : ℕ := 25
def deviations : List ℚ := [1.5, -3, 2, -0.5, 1, -2, 2, -1.5, 1, 2.5]
def price_per_kg : ℕ := 3
def num_baskets : ℕ := 10

-- Question 1: Prove the difference between heaviest and lightest basket
theorem heaviest_lightest_difference :
  let max_excess := deviations.maximum -- Find the maximum deviation
  let min_excess := deviations.minimum -- Find the minimum deviation
  max_excess - min_excess = (5.5 : ℚ) := sorry

-- Question 2: Prove the total deviation from the standard weight
theorem total_deviation :
  deviations.sum = (3 : ℚ) := sorry

-- Question 3: Prove the total selling price for the 10 baskets
theorem total_selling_price :
  ((standard_weight * num_baskets) + deviations.sum) * price_per_kg = (759 : ℚ) := sorry

end heaviest_lightest_difference_total_deviation_total_selling_price_l638_638785


namespace glass_volume_l638_638424

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638424


namespace unique_root_value_of_a_l638_638117

   noncomputable def unique_root_condition (a : ℝ) : Prop :=
     ∃ t : ℝ, (∀ x : ℝ, ln (x - 2 * a) - 3 * (x - 2 * a) ^ 2 + 2 * a ≠ 0) ∧
              ln t - 3 * t ^ 2 + 2 * a = 0

   theorem unique_root_value_of_a : unique_root_condition ((ln 6 + 1) / 4) :=
   sorry
   
end unique_root_value_of_a_l638_638117


namespace pages_on_same_sheet_l638_638611

theorem pages_on_same_sheet (sheets pages_per_sheet page_limit : ℕ) (h_sheets : sheets = 15) 
(h_pages_per_sheet : pages_per_sheet = 4) (h_page_limit : page_limit = 60) : 
∃ a b c d : ℕ, a = 25 ∧ b = 26 ∧ c = 35 ∧ d = 36 ∧ a ∈ pages_per_sheet ∧ b ∈ pages_per_sheet ∧ c ∈ pages_per_sheet ∧ d ∈ pages_per_sheet := by
  sorry

end pages_on_same_sheet_l638_638611


namespace trigonometric_identity_application_l638_638401

theorem trigonometric_identity_application :
  2 * (Real.sin (35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) +
       Real.cos (35 * Real.pi / 180) * Real.cos (65 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end trigonometric_identity_application_l638_638401


namespace john_website_days_l638_638674

theorem john_website_days
  (monthly_visits : ℕ)
  (cents_per_visit : ℝ)
  (dollars_per_day : ℝ)
  (monthly_visits_eq : monthly_visits = 30000)
  (cents_per_visit_eq : cents_per_visit = 0.01)
  (dollars_per_day_eq : dollars_per_day = 10) :
  (monthly_visits / (dollars_per_day / cents_per_visit)) = 30 :=
by
  sorry

end john_website_days_l638_638674


namespace side_length_of_square_l638_638754

-- Define the conditions
def area_rectangle (length width : ℝ) : ℝ := length * width
def area_square (side : ℝ) : ℝ := side * side

-- Given conditions
def rect_length : ℝ := 2
def rect_width : ℝ := 8
def area_of_rectangle : ℝ := area_rectangle rect_length rect_width
def area_of_square : ℝ := area_of_rectangle

-- Main statement to prove
theorem side_length_of_square : ∃ (s : ℝ), s^2 = 16 ∧ s = 4 :=
by {
  -- use the conditions here
  sorry
}

end side_length_of_square_l638_638754


namespace roulette_P2007_gt_P2008_l638_638872

-- Define the roulette probability function based on the given conditions
noncomputable def roulette_probability : ℕ → ℝ
| 0 := 1
| n := (1 / 2007) * (List.foldl (λ acc k, acc + roulette_probability (n - k)) 0 (List.range 2007))

-- Define the theorem to prove P_{2007} > P_{2008}
theorem roulette_P2007_gt_P2008 : roulette_probability 2007 > roulette_probability 2008 :=
sorry

end roulette_P2007_gt_P2008_l638_638872


namespace eval_complex_fraction_expr_l638_638923

def complex_fraction_expr : ℚ :=
  2 + (3 / (4 + (5 / (6 + (7 / 8)))))

theorem eval_complex_fraction_expr : complex_fraction_expr = 137 / 52 :=
by
  -- we skip the actual proof but ensure it can build successfully.
  sorry

end eval_complex_fraction_expr_l638_638923


namespace number_of_four_digit_numbers_l638_638548

noncomputable def count_valid_four_digit_numbers : ℕ :=
  sorry

theorem number_of_four_digit_numbers :
  count_valid_four_digit_numbers = 324 :=
begin
  sorry
end

end number_of_four_digit_numbers_l638_638548


namespace Amy_chicken_soup_l638_638053

theorem Amy_chicken_soup (total_soups tomato_soups : ℕ) (h_total : total_soups = 9) (h_tomato : tomato_soups = 3) : total_soups - tomato_soups = 6 :=
by
  rw [h_total, h_tomato]
  norm_num

end Amy_chicken_soup_l638_638053


namespace carl_highway_miles_l638_638508

theorem carl_highway_miles
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (city_miles : ℕ)
  (gas_cost_per_gallon : ℕ)
  (total_cost : ℕ)
  (h1 : city_mpg = 30)
  (h2 : highway_mpg = 40)
  (h3 : city_miles = 60)
  (h4 : gas_cost_per_gallon = 3)
  (h5 : total_cost = 42)
  : (total_cost - (city_miles / city_mpg) * gas_cost_per_gallon) / gas_cost_per_gallon * highway_mpg = 480 := 
by
  sorry

end carl_highway_miles_l638_638508


namespace prob_abs_le_1_96_l638_638325

open ProbabilityTheory MeasureTheory

noncomputable def stdNormalDist : ProbabilityMeasure ℝ :=
  ProbMeasure.stdNormal

theorem prob_abs_le_1_96 :
  ∀ (ξ : ℝ →ₘ[stdNormalDist] ℝ), -- Random variable ξ follows a standard normal distribution
  (prob (ξ ≤ -1.96) stdNormalDist = 0.025) →
  (prob (|ξ| < 1.96) stdNormalDist = 0.950) :=
sorry

end prob_abs_le_1_96_l638_638325


namespace circles_area_outside_A_B_l638_638510

theorem circles_area_outside_A_B:
  let radius_A := 1
  let radius_B := 1
  let radius_C := 2
  let midpoint := (0, 0) -- placeholders for the midpoint of AB, assume it at origin for simplicity
  let center_A := (1, 0)
  let center_B := (-1, 0)
  let center_C := (0, 2) -- center of C above midpoint at a distance of its radius
  (tangent_point_A_B : dist center_A center_B = radius_A + radius_B)
  (tangent_midpoint_C : dist midpoint center_C = radius_C)
  (area_inside_C_but_outside_A_B : ℝ) :
  area_inside_C_but_outside_A_B = 3 * π + 4 := 
sorry

end circles_area_outside_A_B_l638_638510


namespace math_proof_problem_l638_638567

/-- 
  Given functions f(x) = 1/(a*x) + log x and g(x) = x / (e^(a*x)) + 1/2 - log a.
  (1) Prove that when a = 1, f(x) is monotonic decreasing on (0, 1) and 
      monotonic increasing on (1, +∞).
  (2) For a > 0, if graphs of y = f(x) and y = g(x) have exactly one common point, 
      prove that a = 2/e.
  (3) Given that there are exactly 3 tangents passing through (b, c) that are 
      tangent to y = f(x), prove that for b > e, (1/2)(1 - b/e) + c < f(b) < c.
-/
theorem math_proof_problem (a b c : ℝ) (e : ℝ := Real.exp 1) :
  (f : ℝ → ℝ := λ x, 1 / (a * x) + Real.log x) →
  (g : ℝ → ℝ := λ x, x / (Real.exp (a * x)) + 1/2 - Real.log a) →
  ((a = 1) → ((∀ x > 0, x < 1 → f x < f x + 0) ∧ (∀ x > 1, f x + 0 < f x))) →
  ((a > 0) → ∃ A : ℝ, A = 2 / e ∧ (∀ x : ℝ, f x = g x ↔ x = 1 / A)) →
  ((∃ b : ℝ, b > e ∧ (∃ c : ℝ, (1 / 2) * (1 - b / e) + c < f b ∧ f b < c))) :=
  sorry

end math_proof_problem_l638_638567


namespace frobenius_number_l638_638190

open Nat

theorem frobenius_number {p q m n x y : ℕ} (hc : coprime p q) :
  (m = p * q - p - q + 1) →
  (∀ n, n ≥ m → ∃ x y : ℕ, n = p * x + q * y) :=
sorry

end frobenius_number_l638_638190


namespace math_problem_l638_638558

noncomputable def minimum_segment_length (P : Point) (l1 l2 : Line) : ℝ :=
  let d := dist l1 l2
  in if perpendicular (direction_vector l1) (direction_vector l2) then d else sorry

noncomputable def segment_length_parallel_x_axis (P : Point) (l1 l2 : Line) : ℝ :=
  let x1 := x_coord (intersection l1 {y := P.y, passed_through := P})
  let x2 := x_coord (intersection l2 {y := P.y, passed_through := P})
  in abs (x2 - x1)

theorem math_problem (P : Point) (l1 l2 : Line) (d₁ d₂ : ℝ) : 
  P = ⟨2, 3⟩ → 
  l1 = ⟨3, 4, -7⟩ → 
  l2 = ⟨3, 4, 8⟩ → 
  minimum_segment_length P l1 l2 = 3 ∧ segment_length_parallel_x_axis P l1 l2 = 5 :=
by sorry

end math_problem_l638_638558


namespace right_triangle_median_equals_midsegment_l638_638371

theorem right_triangle_median_equals_midsegment
  (A B C M K N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace M] [MetricSpace K] [MetricSpace N]
  (hΔ : IsRightTriangle A B C)
  (hM : IsMidpoint M A B)
  (hK : IsMidpoint K A C)
  (hN : IsMidpoint N B C)
  : Distance C M = Distance K N := sorry

end right_triangle_median_equals_midsegment_l638_638371


namespace prob1_prob2_prob3_prob4_l638_638398

-- Problem (1)
theorem prob1 : 2^(-1) + | -1/4 | - 202 * 3^0 = -1/4 := by
  sorry

-- Problem (2)
theorem prob2 (x : ℝ) : 2 * x * x^2 - (-x^4)^2 / x^5 = x^3 := by
  sorry

-- Problem (3)
theorem prob3 (x : ℕ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := by
  sorry

-- Problem (4)
theorem prob4 : 
  let x := -1/2 in 
  (2 * x + 1) - 2 * (x - 2) * (x + 2) = 15/2 := by
  sorry

end prob1_prob2_prob3_prob4_l638_638398


namespace calculate_f_l638_638703

def f (a b c : ℝ) : ℝ := (c + a) / (c - b)

theorem calculate_f : f 2 (-3) (-1) = 1 / 2 :=
by
  sorry

end calculate_f_l638_638703


namespace compare_magnitudes_l638_638122

variable (a : ℝ)
variable (A B C D : ℝ)

-- Given conditions
def condition1 := 0 < a ∧ a < 1/2
def condition2 := A = (1 - a)^2
def condition3 := B = (1 + a)^2
def condition4 := C = 1 / (1 - a)
def condition5 := D = 1 / (1 + a)

-- Proof to be written later
theorem compare_magnitudes (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : D < A ∧ A < B ∧ B < C :=
by
  sorry

end compare_magnitudes_l638_638122


namespace sin_alpha_plus_pi_div_12_l638_638552

noncomputable def alpha : ℝ := sorry 

theorem sin_alpha_plus_pi_div_12 : 
  (∀ α : ℝ, 0 < α ∧ α < π → (sin α + cos α = real.sqrt 2 / 3) → sin (α + π / 12) = (real.sqrt 3 + 2 * real.sqrt 2) / 6) :=
by
  sorry

end sin_alpha_plus_pi_div_12_l638_638552


namespace inequality_holds_l638_638283

theorem inequality_holds (x : ℝ) (n : ℕ) (hn : 0 < n) : 
  Real.sin (2 * x)^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
sorry

end inequality_holds_l638_638283


namespace heaviest_vs_lightest_weight_diff_total_excess_or_shortfall_total_selling_price_l638_638783

-- Define the conditions
def standard_weight : ℝ := 25
def excess_shortfall : list ℝ := [1.5, -3, 2, -0.5, 1, -2, 2, -1.5, 1, 2.5]
def selling_price_per_kg : ℝ := 3

-- Define the properties to prove
theorem heaviest_vs_lightest_weight_diff :
  let max_excess := list.maximum excess_shortfall in
  let min_shortfall := list.minimum excess_shortfall in
  max_excess - min_shortfall = 5.5 := by sorry

theorem total_excess_or_shortfall :
  list.sum excess_shortfall = 3 := by sorry

theorem total_selling_price :
  let total_weight := 10 * standard_weight + list.sum excess_shortfall in
  total_weight * selling_price_per_kg = 759 := by sorry

end heaviest_vs_lightest_weight_diff_total_excess_or_shortfall_total_selling_price_l638_638783


namespace floor_sum_arith_eq_l638_638901

def a : ℝ := 0.5
def d : ℝ := 0.8
def n : ℕ := 126
def seq (k : ℕ) : ℝ := a + k * d
def floor_sum (n : ℕ) : ℕ := (Finset.range n).sum (λ k => ⌊seq k⌋)

theorem floor_sum_arith_eq :
  floor_sum n = 6300 := 
  sorry

end floor_sum_arith_eq_l638_638901


namespace area_of_common_part_of_rotated_rhombuses_l638_638897

theorem area_of_common_part_of_rotated_rhombuses (d1 d2 : ℝ) (h₁ : d1 = 4) (h₂ : d2 = 6) :
  let s: ℝ := (1/2) * d1 * d2,
      area := 4 * (3 - 0.6)
  in s = area :=
by
  simp only [h₁, h₂],
  have h3: s = 1/2 * 4 * 6 := by sorry,
  have h4: area = 4 * (3 - 0.6) := by sorry,
  rw [h3, h4]

end area_of_common_part_of_rotated_rhombuses_l638_638897


namespace f_has_exactly_one_zero_g_below_f_for_x_gt_1_l638_638984

noncomputable def f(x: ℝ) : ℝ := log x + (1 / 2) * (x - 1)^2

def g(a: ℝ, x: ℝ) : ℝ := if x > 1 then a * x - a else 0

theorem f_has_exactly_one_zero :
  ∃! x: ℝ, f x = 0 := sorry

theorem g_below_f_for_x_gt_1 (a : ℝ) :
  (∀ x: ℝ, x > 1 → g a x < f x) ↔ a ≤ 1 := sorry

end f_has_exactly_one_zero_g_below_f_for_x_gt_1_l638_638984


namespace number_of_true_propositions_is_two_l638_638981

def proposition1 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (1 + x) = f (1 - x)

def proposition2 : Prop :=
∀ x : ℝ, 2 * Real.sin x * Real.cos (abs x) -- minimum period not 1
  -- We need to define proper periodicity which is complex; so here's a simplified representation
  ≠ 2 * Real.sin (x + 1) * Real.cos (abs (x + 1))

def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

def proposition3 (k : ℝ) : Prop :=
∀ n : ℕ, n > 0 → increasing_sequence (fun n => n^2 + k * n + 2)

def condition (f : ℝ → ℝ) (k : ℝ) : Prop :=
proposition1 f ∧ proposition2 ∧ proposition3 k

theorem number_of_true_propositions_is_two (f : ℝ → ℝ) (k : ℝ) :
  condition f k → 2 = 2 :=
by
  sorry

end number_of_true_propositions_is_two_l638_638981


namespace correct_propositions_l638_638286

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_propositions :
  ¬ ∀ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * Real.pi ∧
  (∀ (x : ℝ), f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (- (Real.pi / 6)) = 0) ∧
  ¬ ∀ (x : ℝ), f x = f (-x - Real.pi / 6) :=
sorry

end correct_propositions_l638_638286


namespace find_xyz_l638_638074

noncomputable def floor_part (x : ℝ) : ℕ := ⌊x⌋
noncomputable def frac_part (x : ℝ) : ℝ := x - ⌊x⌋

def satisfies_conditions (x y z : ℝ) : Prop := 
  (0 < x) ∧ (0 < y) ∧ (0 < z) ∧
  (frac_part x + floor_part y + frac_part z = 2.9) ∧
  (frac_part y + floor_part z + frac_part x = 5.3) ∧
  (frac_part z + floor_part x + frac_part y = 4.0)

theorem find_xyz : ∃ (x y z : ℝ), satisfies_conditions x y z :=
  ⟨3.1, 2.7, 1.2, by sorry⟩

end find_xyz_l638_638074


namespace central_angle_sine_rational_l638_638303

theorem central_angle_sine_rational (R : ℝ) (BC : ℝ) (AD_bisect_BC : ∀ (AD AE : ℝ), AD / 2 = AE / 2 → BC / 2) :
    R = 10 → BC = 8 → (∀ (AD AE : ℝ), AD / 2 = AE / 2 ↔ AD = AE) →
    ∃ (m n : ℝ), m / n = 1 / √2 ∧ m * n = 1 :=
by
  sorry

end central_angle_sine_rational_l638_638303


namespace inequality_ineq_l638_638727

theorem inequality_ineq (x y : ℝ) (hx: x > Real.sqrt 2) (hy: y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
  sorry

end inequality_ineq_l638_638727


namespace weight_of_replaced_sailor_l638_638757

theorem weight_of_replaced_sailor (avg_increase : ℝ) (total_sailors : ℝ) (new_sailor_weight : ℝ) : 
  avg_increase = 1 ∧ total_sailors = 8 ∧ new_sailor_weight = 64 → 
  ∃ W, W = 56 :=
by
  intro h
  sorry

end weight_of_replaced_sailor_l638_638757


namespace problem_proof_l638_638599

-- Definition of the set A_n
def A_n (n : ℕ) := {x : ℕ | ∃ (k : Fin n → ℤ), (∀ i, k i = 1 ∨ k i = -1) ∧ 
                                           x = Finset.sum (Finset.range n) (λ i, k i * (2 ^ (i + 1))) ∧
                                           x > 0 }

-- Definition of the sum S_n
def S_n (n : ℕ) := Finset.sum {x // x ∈ A_n n} val

theorem problem_proof :
  (S_n 2 = 8) ∧
  (S_n 3 = 32) ∧
  (∀ n : ℕ, n ≥ 2 → S_n n = 2^(2*n - 1)) :=
by
  sorry

end problem_proof_l638_638599


namespace shirt_cost_l638_638807

-- Definitions and conditions
def num_ten_bills : ℕ := 2
def num_twenty_bills : ℕ := num_ten_bills + 1

def ten_bill_value : ℕ := 10
def twenty_bill_value : ℕ := 20

-- Statement to prove
theorem shirt_cost :
  (num_ten_bills * ten_bill_value) + (num_twenty_bills * twenty_bill_value) = 80 :=
by
  sorry

end shirt_cost_l638_638807


namespace budget_allocation_problem_l638_638413

theorem budget_allocation_problem
  (M F G I: ℝ) (deg_A: ℝ)
  (total_percent: ℝ)
  (H: ℝ) : 
  M = 13 ∧ F = 15 ∧ G = 29 ∧ I = 8 ∧ deg_A = 39.6 ∧ total_percent = 100 → 
  H = total_percent - (M + F + G + I + (deg_A / 360) * 100) :=
by 
  intros h
  cases h with M_eq h
  cases h with F_eq h
  cases h with G_eq h
  cases h with I_eq h
  cases h with deg_A_eq total_percent_eq
  rw [M_eq, F_eq, G_eq, I_eq, deg_A_eq, total_percent_eq]
  sorry

end budget_allocation_problem_l638_638413


namespace line_through_C_and_midpoint_cuts_one_third_l638_638003

theorem line_through_C_and_midpoint_cuts_one_third {A B C M N : Type*} [has_add A] [has_add B] [has_add C] [has_add M] [has_add N]
  (h_midpoint_M : (B + C) / 2 = M) -- M is the midpoint of BC
  (h_line_through_C : (C + M) / 2 = N) -- The line passing through C and M
  (h_segm_division : N = (A + B) / 3) -- N divides AB into ratio 1:3
  :
  (N = (2 * A + B) / 3) -> (distance A N) / (distance A B) = 1/3 :=
by
  sorry

end line_through_C_and_midpoint_cuts_one_third_l638_638003


namespace product_abc_l638_638271

theorem product_abc 
  (a b c : ℝ)
  (h1 : a + b + c = 1) 
  (h2 : 3 * (4 * a + 2 * b + c) = 15) 
  (h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a * b * c = -4 :=
by
  sorry

end product_abc_l638_638271


namespace probability_same_color_l638_638408

theorem probability_same_color :
  let red_marble_prob := (5 / 21) * (4 / 20) * (3 / 19)
  let white_marble_prob := (6 / 21) * (5 / 20) * (4 / 19)
  let blue_marble_prob := (7 / 21) * (6 / 20) * (5 / 19)
  let green_marble_prob := (3 / 21) * (2 / 20) * (1 / 19)
  red_marble_prob + white_marble_prob + blue_marble_prob + green_marble_prob = 66 / 1330 := by
  sorry

end probability_same_color_l638_638408


namespace find_fx_l638_638314

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

theorem find_fx (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = x * (x + 1) :=
by
  sorry

end find_fx_l638_638314


namespace wire_length_between_poles_l638_638759

theorem wire_length_between_poles :
  let x_dist := 20
  let y_dist := (18 / 2) - 8
  (x_dist ^ 2 + y_dist ^ 2 = 401) :=
by
  sorry

end wire_length_between_poles_l638_638759


namespace curve_is_line_l638_638538

theorem curve_is_line (θ : ℝ) (hθ : θ = π / 4) :
  ∃ L : set (ℝ × ℝ), is_line L :=
sorry

end curve_is_line_l638_638538


namespace smallest_odd_number_of_students_l638_638305

theorem smallest_odd_number_of_students :
  ∃ (n : ℕ), (∃ (m : ℕ), 0.505 * n ≤ m ∧ m < 0.515 * n) ∧ (∃ (k : ℕ), n = 2 * k + 1) ∧ n = 35 :=
by
  sorry

end smallest_odd_number_of_students_l638_638305


namespace g_2023_eq_3_l638_638692

noncomputable def g : ℝ → ℝ := sorry

axiom g_pos : ∀ x > 0, g x > 0
axiom g_eq : ∀ x y > 0, x > y → g(x - y) = sqrt(g(x * y) + 3)

theorem g_2023_eq_3 : g 2023 = 3 := by
  sorry

end g_2023_eq_3_l638_638692


namespace average_age_l638_638229

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l638_638229


namespace polygon_has_area_144_l638_638216

noncomputable def polygonArea (n_sides : ℕ) (perimeter : ℕ) (n_squares : ℕ) : ℕ :=
  let s := perimeter / n_sides
  let square_area := s * s
  square_area * n_squares

theorem polygon_has_area_144 :
  polygonArea 32 64 36 = 144 :=
by
  sorry

end polygon_has_area_144_l638_638216


namespace num_divisible_by_61875_l638_638525

theorem num_divisible_by_61875 : 
  let x : ℕ := 5 in
  let y : ℕ := 8 in
  let z : ℕ := 3 in
  let u : ℕ := 7 in
  let v : ℕ := 5 in
  let num := 5 * 10^9 + 6 * 10^8 + 1 * 10^7 + 8 * 10^6 + 0 * 10^5 + 6 * 10^4 + 4 * 10^3 + 3 * 10^2 + 7 * 10 + 5 in
  num = 5618064375 ∧ (num % 61875 = 0) :=
by {
  let x := 5,
  let y := 8,
  let z := 3,
  let u := 7,
  let v := 5,
  let num := 5 * 10^9 + 6 * 10^8 + 1 * 10^7 + 8 * 10^6 + 0 * 10^5 + 6 * 10^4 + 4 * 10^3 + 3 * 10^2 + 7 * 10 + 5,
  have h_num_val: num = 5618064375, sorry,
  have h_div_625: (5618064375 % 625 = 0), sorry,
  have h_div_9: ((5 + 6 + 1 + 8 + 0 + 6 + 4 + 3 + 7 + 5) % 9 = 0), sorry,
  have h_div_11: ((5 - 6 + 1 - 8 + 0 - 6 + 4 - 3 + 7 - 5) % 11 = 0), sorry,
  exact ⟨h_num_val, by { rw h_num_val, exact div_congr h_div_625 (mul_comm 36 1715) (by norm_num)⟩
}.

end num_divisible_by_61875_l638_638525


namespace total_stops_is_seven_l638_638030

-- Definitions of conditions
def initial_stops : ℕ := 3
def additional_stops : ℕ := 4

-- Statement to be proved
theorem total_stops_is_seven : initial_stops + additional_stops = 7 :=
by {
  -- this is a placeholder for the proof
  sorry
}

end total_stops_is_seven_l638_638030


namespace count_multiples_5_7_not_35_l638_638605

theorem count_multiples_5_7_not_35 : 
  let n := 600 in
  let count_5 := (n / 5) in
  let count_7 := (n / 7) in
  let count_35 := (n / 35) in
  (count_5 + count_7 - count_35) = 188 :=
by
  let n := 600
  let count_5 := n / 5
  let count_7 := n / 7
  let count_35 := n / 35
  have h1 : count_5 = 120 := by sorry
  have h2 : count_7 = 85 := by sorry
  have h3 : count_35 = 17 := by sorry
  have h4 : count_5 + count_7 - count_35 = 188 := by sorry
  exact h4

end count_multiples_5_7_not_35_l638_638605


namespace incorrect_statements_about_f_l638_638161

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := (1 - x) * Real.exp x

-- State the problem that some statements about f are incorrect
theorem incorrect_statements_about_f :
  ¬ (f' 2 > 0) ∧ ¬ ∀ x: ℝ, 1 < x → f' x > 0 := by
  sorry

end incorrect_statements_about_f_l638_638161


namespace angie_age_l638_638800

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l638_638800


namespace airplane_return_direction_l638_638493

theorem airplane_return_direction 
(initial_distance : ℕ) 
(initial_direction : ℕ) 
(return_direction : ℕ) 
(return_distance : ℕ) :
  initial_distance = 1200 →
  initial_direction = 30 →
  return_distance = 1200 →
  return_direction = 210 →
  return_direction = 180 + initial_direction ∧ initial_distance = return_distance
  :=
by
  intro h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  split; exact h4

end airplane_return_direction_l638_638493


namespace necessary_but_not_sufficient_condition_log2_m_l638_638859

theorem necessary_but_not_sufficient_condition_log2_m (m : ℝ) : (log 2 m < 1) → (0 ≤ m ∧ m < 3) :=
sorry

end necessary_but_not_sufficient_condition_log2_m_l638_638859


namespace probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l638_638349

noncomputable def diameter := 19 -- mm
noncomputable def side_length := 50 -- mm, side length of each square
noncomputable def total_area := side_length^2 -- 2500 mm^2 for each square
noncomputable def coin_radius := diameter / 2 -- 9.5 mm

theorem probability_completely_inside_square : 
  (side_length - 2 * coin_radius)^2 / total_area = 961 / 2500 :=
by sorry

theorem probability_partial_one_edge :
  4 * ((side_length - 2 * coin_radius) * coin_radius) / total_area = 1178 / 2500 :=
by sorry

theorem probability_partial_two_edges_not_vertex :
  (4 * ((diameter)^2 - (coin_radius^2 * Real.pi / 4))) / total_area = (4 * 290.12) / 2500 :=
by sorry

theorem probability_vertex :
  4 * (coin_radius^2 * Real.pi / 4) / total_area = 4 * 70.88 / 2500 :=
by sorry

end probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l638_638349


namespace general_formulas_sum_b_n_sum_d_n_l638_638581

noncomputable def a_n (n : ℕ) : ℕ := 2 ^ n
noncomputable def b_n (n : ℕ) : ℕ := n
noncomputable def d_n (n : ℕ) : ℚ :=
  if n % 2 = 1 then -(b_n n * a_n n ^ 2) / 2 else (b_n n * a_n n ^ 2) / 4

theorem general_formulas (n : ℕ) :
  ∀ n, a_n n = 2 ^ n ∧ b_n n = n :=
sorry

theorem sum_b_n (n : ℕ) :
  ∑ k in Finset.range (2 ^ n), b_n k = 2 ^ (2 * n - 1) - 2 ^ (n - 1) :=
sorry

theorem sum_d_n (n : ℕ) :
  ∑ i in Finset.range (2 * n), d_n i = (60 * n + 26) * 16 ^ n / 225 - 26 / 225 :=
sorry

end general_formulas_sum_b_n_sum_d_n_l638_638581


namespace average_age_l638_638227

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l638_638227


namespace area_enclosed_by_x2_y2_eq_abs_x_plus_abs_y_l638_638809

theorem area_enclosed_by_x2_y2_eq_abs_x_plus_abs_y :
  let A := setOf (fun p : ℝ × ℝ => p.1^2 + p.2^2 = |p.1| + |p.2|) in
  let area := measure (by volume) A in
  area = π + 2  :=
sorry

end area_enclosed_by_x2_y2_eq_abs_x_plus_abs_y_l638_638809


namespace maximum_squares_reachable_in_two_knight_moves_l638_638357

noncomputable def maximum_marked_squares : ℕ :=
  8

theorem maximum_squares_reachable_in_two_knight_moves (chessboard : matrix ℕ ℕ bool)
  (is_knight_move : ∀ (x1 y1 x2 y2 : ℕ), boolean := | KnightMove x1 y1 x2 y2 → True | _ → False):
  (∀ (x1 y1 x2 y2 : ℕ), KnightMove x1 y1 x2 y2 ∧ KnightMove x2 y2 x1 y1) → 
  (∃ marked_squares : list ℕ, (∀ (i j : ℕ), i ≠ j → KnightMove (marked_squares.get!(i)) (marked_squares.get!(j)) (marked_squares.get!(i+2)) ∧ KnightMove (marked_squares.get!(j)) (marked_squares.get!(i+2)) (marked_squares.get!(i))) ∧ marked_squares.length = 8) :=
sorry

end maximum_squares_reachable_in_two_knight_moves_l638_638357


namespace sequence_bounds_l638_638778

theorem sequence_bounds :
  ∃ α : ℝ, α = 1 / 3 ∧ ∀ n : ℕ, n > 0 →
  (λ a : ℕ → ℝ, ∀ n : ℕ, n > 0 → (a 1 = 1) → ((∀ k : ℕ, k > 0 → a (k + 1) = real.sqrt (a k ^ 2 + 1 / a k)) → 
  ∀ n : ℕ, (1/2 : ℝ) ≤ a n / n ^ α ∧ a n / n ^ α ≤ 2)))
:=
begin
  use 1 / 3,
  split,
  { refl },
  { intros n hn,
    sorry }
end

end sequence_bounds_l638_638778


namespace glass_volume_l638_638457

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638457


namespace glass_volume_l638_638428

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638428


namespace glass_volume_is_230_l638_638455

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638455


namespace rectangle_cos_angle_AOB_l638_638648

theorem rectangle_cos_angle_AOB (A B C D O : Type) [MetricSpace ℝ] [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D]
  (hABCD : Metric.rectangle A B C D) (hAOB : Metric.intersect_diagonals_at_midpoint A B C D O)
  (hAB : dist A B = 8) (hBC : dist B C = 15) : cos (metric.angle A O B) = 17 / 30 :=
sorry

end rectangle_cos_angle_AOB_l638_638648


namespace problem_solution_l638_638087

def problem_statement : Prop :=
  floor (ceil ((15 / 8) ^ 2) + 21 / 5) = 8

theorem problem_solution : problem_statement := 
  by 
    -- ensure all required mathematical properties and computations are provable
    sorry

end problem_solution_l638_638087


namespace intersection_points_after_adding_line_l638_638665

-- Defining the initial conditions
def f (k : ℕ) : ℕ := sorry

-- Lean statement to prove the assertion
theorem intersection_points_after_adding_line (k : ℕ) :
  f(k + 1) = f(k) + k :=
sorry

end intersection_points_after_adding_line_l638_638665


namespace range_of_theta_l638_638145

theorem range_of_theta (theta : ℝ) : (∀ x ∈ Icc (0 : ℝ) 1, x^2 * cos θ - x * (1 - x) + (1 - x)^2 * sin θ > 0) ↔ (π / 12 < θ ∧ θ < 5 * π / 12) :=
sorry

end range_of_theta_l638_638145


namespace find_P_values_l638_638954

theorem find_P_values (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_lt : x < y) :
  let P := (x^3 - y) / (1 + x * y) in P ∈ ({0} ∪ {n ∈ ℤ | 1 < n}) :=
by sorry

end find_P_values_l638_638954


namespace exists_coprime_in_consecutive_numbers_l638_638728

theorem exists_coprime_in_consecutive_numbers (n : ℤ) :
  ∃ k, k ∈ (n::(list.range 9).map (λ i, n + i + 1)) ∧
          ∀ m ∈ (n::(list.range 9).map (λ i, n + i + 1)), k ≠ m → Int.gcd k m = 1 :=
by
  sorry

end exists_coprime_in_consecutive_numbers_l638_638728


namespace simplify_trigonometric_expression_l638_638741

theorem simplify_trigonometric_expression (α : ℝ) :
    (cos (α - π/2) / sin (5/2 * π + α) * sin (α - π) * cos (2 * π - α)) = -sin(α) ^ 2 :=
by
  sorry

end simplify_trigonometric_expression_l638_638741


namespace find_ages_l638_638792

-- Define ages
variables (AgeMatt AgeLisa AgeAlex AgeJames : ℕ)

-- Define conditions as assumptions in Lean
axiom (H1 : AgeJames = 30) -- James turned 27 three years ago, so now he is 30
axiom (H2 : AgeMatt = 2 * (AgeJames + 5) - 5) -- In 5 years, Matt will be twice James' age
axiom (H3 : AgeLisa = (AgeJames + (2 * (AgeJames + 5) - 5)) + 5 - 4 - 5) -- In 5 years, Lisa will be 4 years younger than the combined ages of James and Matt
axiom (H4 : AgeAlex = 3 * AgeLisa + 9 - 9) -- In 9 years, Alex will be three times as old as Lisa is today

-- Lean proof statement
theorem find_ages : AgeMatt = 65 ∧ AgeLisa = 96 ∧ AgeAlex = 279 :=
by
  -- Due to the constraints of the task, we use sorry to skip the proof steps
  sorry

end find_ages_l638_638792


namespace convex_polygon_area_leq_l638_638038

theorem convex_polygon_area_leq (H P : ℝ) (hexagon_in_polygon : ∀ (H_area P_area : ℝ), ∃ (hexagon_verts : Fin 6 → ℝ × ℝ) (polygon_verts : Fin 6 → ℝ × ℝ), 
  hexagon_area = H ∧ polygon_area = P ∧ regular_hexagon hexagon_verts ∧ convex_polygon polygon_verts 
  ∧ inscribed hexagon_verts polygon_verts) : 
  P ≤ (3 / 2) * H :=
begin
  sorry
end

end convex_polygon_area_leq_l638_638038


namespace find_staff_age_l638_638832

theorem find_staff_age (n_students : ℕ) (avg_age_students : ℕ) (avg_age_with_staff : ℕ) (total_students : ℕ) :
  n_students = 32 →
  avg_age_students = 16 →
  avg_age_with_staff = 17 →
  total_students = 33 →
  (33 * 17 - 32 * 16) = 49 :=
by
  intros
  sorry

end find_staff_age_l638_638832


namespace productivity_comparison_l638_638392

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l638_638392


namespace relationship_among_a_b_c_l638_638941

noncomputable def a : ℝ := 0.6 ^ 0.6
noncomputable def b : ℝ := 0.6 ^ 1.5
noncomputable def c : ℝ := 1.5 ^ 0.6

theorem relationship_among_a_b_c : b < a ∧ a < c := 
by {
    sorry
}

end relationship_among_a_b_c_l638_638941


namespace integral_sin_plus_one_l638_638918

noncomputable def definite_integral : ℝ :=
  ∫ x in -1..1, sin x + 1

theorem integral_sin_plus_one :
  definite_integral = 2 - 2 * cos 1 :=
by
  sorry

end integral_sin_plus_one_l638_638918


namespace max_balls_identifiable_with_5_weighings_l638_638333

theorem max_balls_identifiable_with_5_weighings : 
  ∃ n : ℕ, (n ≤ 3^5) ∧ (n = 243) :=
by
  use 243
  split
  · exact Nat.le_refl 243
  · refl

end max_balls_identifiable_with_5_weighings_l638_638333


namespace range_of_a_range_of_lambda_l638_638310

noncomputable def f (x a : ℝ) := (x - 1)^2 + a * Real.log x

theorem range_of_a :
  (∀ x : ℝ, x > 1 / 4 → f x a = (x - 1)^2 + a * Real.log x) →
  (∃ x1 x2 : ℝ, x1 < x2 ∧ f x1 a = f x2 a ∧ ∀ x ∈ Ioo (1 / 4) 1, f' x a < 0) →
  (3 / 8 < a ∧ a < 1 / 2) :=
sorry

theorem range_of_lambda :
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1 + x2 = 1 ∧ x1 * x2 = a / 2) →
  (∀ x : ℝ, x ∈ Ioo (1 / 4) 1 → f x1 a = (x1 - 1)^2 + a * Real.log x1) →
  (f x2 a > λ * x1) →
  (λ < 1 - 2 / Real.sqrt Real.exp) :=
sorry

end range_of_a_range_of_lambda_l638_638310


namespace glass_volume_l638_638427

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638427


namespace quad_mono_decreasing_l638_638147

theorem quad_mono_decreasing {a : ℝ} :
  (∀ x y : ℝ, x < y → x < 3 → y < 3 → f y ≤ f x) ↔ a ≤ -6 :=
by
  let f : ℝ → ℝ
  | x => x^2 + a*x + 4
  sorry

end quad_mono_decreasing_l638_638147


namespace volume_tetrahedron_on_unit_cube_l638_638274

theorem volume_tetrahedron_on_unit_cube
  (M N P Q : ℝ^3)
  (MN_length : dist M N = 1/2)
  (PQ_length : dist P Q = 1/3)
  (d : ℝ) (d_def : d = √6 / 6)
  (alpha : ℝ) (alpha_def : α = π / 2) :
  volume_of_tetrahedron M N P Q d = (√6 / 216) :=
sorry

end volume_tetrahedron_on_unit_cube_l638_638274


namespace ratio_problem_l638_638614

theorem ratio_problem 
  (a b c d : ℚ)
  (h₁ : a / b = 8)
  (h₂ : c / b = 5)
  (h₃ : c / d = 1 / 3) : 
  d / a = 15 / 8 := 
by 
  sorry

end ratio_problem_l638_638614


namespace exists_subset_with_modulus_sum_ge_one_fourth_l638_638706

open Complex

theorem exists_subset_with_modulus_sum_ge_one_fourth
  (n : ℕ)
  (z : Fin n → ℂ)
  (h : (∑ k, ∥z k∥) = 1) :
  ∃ (s : Finset (Fin n)), ∥∑ i in s, z i∥ ≥ 1 / 4 := 
sorry

end exists_subset_with_modulus_sum_ge_one_fourth_l638_638706


namespace rotation_problem_l638_638651

theorem rotation_problem (P : ℝ × ℝ) (hP : P = (Real.sqrt 3 / 2, 1 / 2)) :
  let P' := (-(1 / 2), Real.sqrt 3 / 2) in
  rotate (Real.pi / 2) P = P' :=
sorry

end rotation_problem_l638_638651


namespace palindromic_polynomial_root_l638_638729

theorem palindromic_polynomial_root (a : ℕ → ℚ) (n : ℕ) (x1 : ℚ) : 
  (∀ k, a k = a (n - k)) →  -- The polynomial is palindromic
  (a n * x1^n + ∑ i in finset.range n, a i * x1^i = 0) →  -- x1 is a root of the polynomial
  a n * (1/x1)^n + ∑ i in finset.range n, a i * (1/x1)^i = 0 := -- 1/x1 is a root of the polynomial
by
  intros h_sym h_root
  sorry

end palindromic_polynomial_root_l638_638729


namespace find_y_plus_inv_y_l638_638967

theorem find_y_plus_inv_y (y : ℝ) (h : y^3 + 1 / y^3 = 110) : y + 1 / y = 5 :=
sorry

end find_y_plus_inv_y_l638_638967


namespace sum_f_eq_2007_sq_div_2_l638_638594

def f (x : ℝ) := x^3 / (1 + x^3)

theorem sum_f_eq_2007_sq_div_2 :
  (∑ k in Finset.range 2007, ∑ n in Finset.range 2007, f (k + 1) / f (n + 1)) = 2007^2 / 2 :=
by
  sorry

end sum_f_eq_2007_sq_div_2_l638_638594


namespace initial_time_french_fries_l638_638503

/-- Bill initially put the fries in the oven for 45 seconds given the total cooking time 
is 300 seconds and there are 255 seconds remaining. -/
theorem initial_time_french_fries :
  (total_cooking_time : ℕ) (remaining_time : ℕ)
  (h1 : total_cooking_time = 300)
  (h2 : remaining_time = 255) :
  total_cooking_time - remaining_time = 45 :=
by
  sorry

end initial_time_french_fries_l638_638503


namespace multiplication_factor_average_l638_638756

theorem multiplication_factor_average (a : ℕ) (b : ℕ) (c : ℕ) (F : ℝ) 
  (h1 : a = 7) 
  (h2 : b = 26) 
  (h3 : (c : ℝ) = 130) 
  (h4 : (a * b * F : ℝ) = a * c) :
  F = 5 := 
by 
  sorry

end multiplication_factor_average_l638_638756


namespace heaviest_vs_lightest_weight_diff_total_excess_or_shortfall_total_selling_price_l638_638784

-- Define the conditions
def standard_weight : ℝ := 25
def excess_shortfall : list ℝ := [1.5, -3, 2, -0.5, 1, -2, 2, -1.5, 1, 2.5]
def selling_price_per_kg : ℝ := 3

-- Define the properties to prove
theorem heaviest_vs_lightest_weight_diff :
  let max_excess := list.maximum excess_shortfall in
  let min_shortfall := list.minimum excess_shortfall in
  max_excess - min_shortfall = 5.5 := by sorry

theorem total_excess_or_shortfall :
  list.sum excess_shortfall = 3 := by sorry

theorem total_selling_price :
  let total_weight := 10 * standard_weight + list.sum excess_shortfall in
  total_weight * selling_price_per_kg = 759 := by sorry

end heaviest_vs_lightest_weight_diff_total_excess_or_shortfall_total_selling_price_l638_638784


namespace inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l638_638553

theorem inequality_d_over_c_lt_d_plus_4_over_c_plus_4
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : c > d)
  (h3 : d > 0) :
  (d / c) < ((d + 4) / (c + 4)) :=
by
  sorry

end inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l638_638553


namespace mutually_exclusive_events_l638_638119

theorem mutually_exclusive_events :
  let C1 := (5 : ℕ) -- 5 red balls
  let C2 := (5 : ℕ) -- 5 white balls
  ∀ (draw1 draw2 draw3 : ℕ → Prop),
  (draw1 = (λ n, n = 3) ∧ draw2 = (λ n, n ≥ 1) →
  (∃ n, draw1 n ∧ draw2 n → false)) :=
by
  intros
  let C1 := 5  -- 5 red balls
  let C2 := 5  -- 5 white balls
  assume H : draw1 = (λ n, n = 3) ∧ draw2 = (λ n, n ≥ 1)
  intro h
  have h1 := (λ n, draw1 n)
  have h2 := (λ n, draw2 n)
  sorry  -- skipping the proof

end mutually_exclusive_events_l638_638119


namespace expiration_date_of_blood_l638_638886

noncomputable def num_seconds_in_a_day : ℕ := 86400

noncomputable def num_seconds_in_factorial_twelve : ℕ := 12.factorial

theorem expiration_date_of_blood (donation_date : ℕ := 20210101) : 
  (donation_date + (num_seconds_in_factorial_twelve / num_seconds_in_a_day)) = 20351231 :=
by
  sorry

end expiration_date_of_blood_l638_638886


namespace xiao_ming_selection_l638_638889

/-- Among the 13 volleyball team members in the class, 7 tall players are selected
to participate in the school volleyball match. If the heights of these 13 team members
are all different, and the team member Xiao Ming wants to know if he can be selected,
he only needs to know the median. -/
theorem xiao_ming_selection
  (heights : list ℝ)
  (h_len : heights.length = 13)
  (h_distinct : heights.nodup)
  (xiao_ming_height : ℝ)
  (h_mem : xiao_ming_height ∈ heights) :
  (xiao_ming_height ≥ heights.sorted.nth_le 6 sorry) ↔
  (xiao_ming_height ∈ heights.sorted.reverse.take 7) :=
sorry

end xiao_ming_selection_l638_638889


namespace next_surprising_date_occurs_in_june_l638_638518

-- Define the months that can be surprising, by excluding those that repeat digits with the year 2XXX
def valid_months : List String := ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

-- Function to check if a date is surprising
def is_surprising_date (date : String) : Bool :=
  let digits := date.to_list
  digits.nodup

-- Define a function that checks if a month can contain a surprising date
def month_has_surprising_date (month : String) (year : String) : Bool :=
  valid_months.contains month ∧
  ∃ day, (1 ≤ day ∧ day ≤ 31) ∧ is_surprising_date (to_string day ++ "." ++ month ++ "." ++ year)

theorem next_surprising_date_occurs_in_june :
  ∃ year, (year.starts_with "2") ∧ month_has_surprising_date "06" year :=
by
  sorry

end next_surprising_date_occurs_in_june_l638_638518


namespace smallest_divisible_by_2022_l638_638817

theorem smallest_divisible_by_2022 (n : ℕ) (N : ℕ) :
  (N = 20230110) ∧ (∃ k : ℕ, N = 2023 * 10^n + k) ∧ N % 2022 = 0 → 
  ∀ M: ℕ, (∃ m : ℕ, M = 2023 * 10^n + m) ∧ M % 2022 = 0 → N ≤ M :=
sorry

end smallest_divisible_by_2022_l638_638817


namespace shaded_fraction_is_4_over_15_l638_638479

-- Define the geometric series sum function
def geom_series_sum (a r : ℝ) (hr : |r| < 1) : ℝ := a / (1 - r)

-- The target statement for the given problem
theorem shaded_fraction_is_4_over_15 :
  let a := (1 / 4 : ℝ)
  let r := (1 / 16 : ℝ)
  geom_series_sum a r (by norm_num : |r| < 1) = (4 / 15 : ℝ) :=
by
  -- Proof is omitted with sorry
  sorry

end shaded_fraction_is_4_over_15_l638_638479


namespace exists_closed_self_intersecting_polygonal_chain_l638_638370

structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Segment :=
(p1 : Point) (p2 : Point)

def midpoint (s : Segment) : Point :=
{x := (s.p1.x + s.p2.x) / 2, y := (s.p1.y + s.p2.y) / 2, z := (s.p1.z + s.p2.z) / 2}

structure PolygonalChain :=
(segments : List Segment)
(is_closed : (segments.head.p1 = segments.last.p2))
(self_intersects : ∀ s1 s2 ∈ segments, s1 ≠ s2 → midpoint s1 = midpoint s2)

theorem exists_closed_self_intersecting_polygonal_chain :
  ∃ chain : PolygonalChain, chain.is_closed ∧ chain.self_intersects :=
sorry

end exists_closed_self_intersecting_polygonal_chain_l638_638370


namespace target_function_l638_638835

-- Conditions
def log_condition (x y : ℝ) : Prop :=
  log (x - x^2 + 3) (y - 6) = log (x - x^2 + 3) ((abs (2 * x + 6) - abs (2 * x + 3)) / (3 * x + 7.5) * sqrt (x^2 + 5 * x + 6.25))

def orthogonal_tangents (a x0 y0 : ℝ) : Prop :=
  let zy := x0 + 2
  let uy := y0 - 2
  in (4 * a^2 * zy * uy = -1) ∧ a = -1 / (4 * uy)

-- Main statement to prove
theorem target_function (a x0 y0 : ℝ) (hx : log_condition x0 y0) (ha : orthogonal_tangents a x0 y0) :
  y0 = a * (x0 + 2)^2 + 2 :=
by
  sorry

end target_function_l638_638835


namespace whale_frenzy_total_plankton_l638_638486

theorem whale_frenzy_total_plankton (x : ℕ) :
  (x + 6 = 93) →
  (let first_hour := x in
   let second_hour := x + 3 in
   let third_hour := second_hour + 3 in
   let fourth_hour := third_hour + 3 in
   let fifth_hour := fourth_hour + 3 in
   first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 465) :=
begin
  intros h,
  have hx : x = 87 := by linarith,
  rw hx,
  let first_hour := 87,
  let second_hour := 87 + 3,
  let third_hour := second_hour + 3,
  let fourth_hour := third_hour + 3,
  let fifth_hour := fourth_hour + 3,
  calc
    first_hour + second_hour + third_hour + fourth_hour + fifth_hour
        = 87 + (87 + 3) + (87 + 3 + 3) + (87 + 3 + 3 + 3) + (87 + 3 + 3 + 3 + 3) : by refl
    ... = 87 + 90 + 93 + 96 + 99 : by norm_num
    ... = 465 : by norm_num,
end

end whale_frenzy_total_plankton_l638_638486


namespace crates_with_same_apples_count_l638_638481

theorem crates_with_same_apples_count :
  ∃ n, n ≥ 6 ∧ ∀ (crates : Finset ℕ), crates.card = 128 →
    (∀ x ∈ crates, 120 ≤ x ∧ x ≤ 144) →
    ∃ k ∈ crates, crates.filter (λ c, c = k).card ≥ n :=
by sorry

end crates_with_same_apples_count_l638_638481


namespace extremum_f_at_neg_four_thirds_monotonicity_g_l638_638590

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x) * Real.exp x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 
  let f_a_x := f a x
  ( f' a x * Real.exp x ) + ( f_a_x * Real.exp x)

theorem extremum_f_at_neg_four_thirds (a : ℝ) :
  f' a (-4/3) = 0 ↔ a = 1/2 := sorry

-- Assuming a = 1/2 from the previous theorem
theorem monotonicity_g :
  let a := 1/2
  ∀ x : ℝ, 
    ((x < -4 → g' a x < 0) ∧ 
     (-4 < x ∧ x < -1 → g' a x > 0) ∧
     (-1 < x ∧ x < 0 → g' a x < 0) ∧
     (x > 0 → g' a x > 0)) := sorry

end extremum_f_at_neg_four_thirds_monotonicity_g_l638_638590


namespace range_of_a_l638_638990

variable {x : ℝ} {a : ℝ}

def A : set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def B (a : ℝ) : set ℝ := { x | a < x ∧ x < a + 1 }

theorem range_of_a (h : B a ⊆ A) : 1 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l638_638990


namespace divisibility_of_a81_l638_638393

theorem divisibility_of_a81 
  (p : ℕ) (hp : Nat.Prime p) (hp_gt2 : 2 < p)
  (a : ℕ → ℕ) (h_rec : ∀ n, n * a (n + 1) = (n + 1) * a n - (p / 2)^4) 
  (h_a1 : a 1 = 5) :
  16 ∣ a 81 := 
sorry

end divisibility_of_a81_l638_638393


namespace determine_g_2023_l638_638694

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined_for_all_positive_real_numbers : ∀ x : ℝ, x > 0 → g x = g x
axiom g_positive : ∀ x : ℝ, x > 0 → g x > 0
axiom g_functional_equation : ∀ x y : ℝ, x > y → y > 0 → g (x - y) = sqrt (g (x * y) + 3)

theorem determine_g_2023 : g 2023 = 3 :=
by
  sorry

end determine_g_2023_l638_638694


namespace max_value_of_function_1_l638_638012

theorem max_value_of_function_1 (x : ℝ) (hx : x < 0) : 
  (y = (x^2 + x + 1) / x) → max_value y = -1 :=
sorry

end max_value_of_function_1_l638_638012


namespace find_least_positive_int_l638_638356

noncomputable def least_positive_int : Nat :=
  Nat.find (λ n, (∀ k : Nat, 1 ≤ k ∧ k ≤ n → k ∣ (n^2 + n)) ∧ (∃ k : Nat, 1 ≤ k ∧ k ≤ n ∧ ¬ (k ∣ (n^2 + n))))

theorem find_least_positive_int : least_positive_int = 3 :=
by
  sorry

end find_least_positive_int_l638_638356


namespace compare_P2007_P2008_l638_638864

def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2007) * ∑ k in (finset.range 2007).image (λ x, x + n + 1 - 2007), P k

theorem compare_P2007_P2008 : P 2007 > P 2008 :=
sorry

end compare_P2007_P2008_l638_638864


namespace value_of_a_plus_b_l638_638688

variable {x : ℝ}
variable {a b : ℝ}

def f (x : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 4

theorem value_of_a_plus_b (h : ∀ x, g (f x) = 4 * x + 3) : a + b = 11 / 3 := by
  sorry

end value_of_a_plus_b_l638_638688


namespace number_of_raised_beds_l638_638048

def length_feed := 8
def width_feet := 4
def height_feet := 1
def cubic_feet_per_bag := 4
def total_bags_needed := 16

theorem number_of_raised_beds :
  ∀ (length_feed width_feet height_feet : ℕ) (cubic_feet_per_bag total_bags_needed : ℕ),
    (length_feed * width_feet * height_feet) / cubic_feet_per_bag = 8 →
    total_bags_needed / (8 : ℕ) = 2 :=
by sorry

end number_of_raised_beds_l638_638048


namespace sum_of_squares_is_4_sqrt_2_l638_638625

-- Define the polar form of -256
def r : ℝ := 2^8
def theta : ℝ := 180

-- Define the general form for the roots of x^8 = r * cis(theta)
def x_k (k : ℕ) : ℂ :=
  (2 : ℂ) * complex.exp (complex.I * (real.pi / 4 + real.pi / 4 * k))

-- Define the condition for the real part being greater than zero.
def a_gt_zero (x : ℂ) : Prop := x.re > 0

-- Sum of the squares of solutions with a > 0
def sum_squares : ℂ :=
  ((2 : ℂ) * complex.exp (complex.I * real.pi / 8))^2 +
  ((2 : ℂ) * complex.exp (complex.I * 15 * real.pi / 8))^2

-- Complete statement for the proof problem
theorem sum_of_squares_is_4_sqrt_2 : sum_squares = 4 * real.sqrt 2 :=
by
  sorry

end sum_of_squares_is_4_sqrt_2_l638_638625


namespace find_FC_eq_l638_638943

noncomputable def find_FC (DE EB AD_fd_ratio : ℝ) : ℝ :=
  let FD := (3/4) * AD
      BE := 15
      AB := (1/3) * AD
      CA := 22.5
  in 16.875

theorem find_FC_eq :
  ∀ (DE EB AD_fd_ratio : ℝ),
    (DE = 9) →
    (EB = 6) →
    (AD_fd_ratio = 3/4) →
    (∃ FC,
      FC = find_FC DE EB AD_fd_ratio) :=
by
  intros DE EB AD_fd_ratio hDE hEB hFD
  use 16.875
  have hFC : find_FC DE EB AD_fd_ratio = 16.875 := sorry
  exact hFC

end find_FC_eq_l638_638943


namespace hundredth_day_N_minus_1_l638_638664

theorem hundredth_day_N_minus_1 (
    N : ℕ,
    is_leap_year : ∃ k : ℕ, N = 4 * k,
    day_250_N : ∃ (d : ℕ), d = 3 ∧ (250 % 7 = 3 ∧ d = 3),
    day_150_N_plus_1 : ∃ (d : ℕ), d = 3 ∧ (150 % 7 = 3 ∧ d = 3)
  ) : 
  ∃ (d : ℕ), d = 6 := -- (6 represents Saturday)
sorry

end hundredth_day_N_minus_1_l638_638664


namespace min_value_of_2x_plus_y_l638_638569

-- Define the conditions for the problem
variables {x y : ℝ}

-- Lean statement to show that the minimum value of 2x + y is 9/2 given the conditions
theorem min_value_of_2x_plus_y (hx : 0 < x) (hy : 0 < y) (h : x + 2y - 2 * x * y = 0) : 2 * x + y ≥ 9 / 2 :=
sorry

end min_value_of_2x_plus_y_l638_638569


namespace sum_of_averages_is_six_l638_638319

variable (a b c d e : ℕ)

def average_teacher : ℚ :=
  (5 * a + 4 * b + 3 * c + 2 * d + e) / (a + b + c + d + e)

def average_kati : ℚ :=
  (5 * e + 4 * d + 3 * c + 2 * b + a) / (a + b + c + d + e)

theorem sum_of_averages_is_six (a b c d e : ℕ) : 
    average_teacher a b c d e + average_kati a b c d e = 6 := by
  sorry

end sum_of_averages_is_six_l638_638319


namespace productivity_comparison_l638_638390

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l638_638390


namespace range_of_x_l638_638165

theorem range_of_x (x : ℝ) : x + 2 ≥ 0 ∧ x - 3 ≠ 0 → x ≥ -2 ∧ x ≠ 3 :=
by
  sorry

end range_of_x_l638_638165


namespace domain_of_f_l638_638913

open Real

def f (x : ℝ) : ℝ := (2 ^ x - 1) / sqrt (log (3 - 2 * x) / log (1 / 2) + 1)

theorem domain_of_f (x : ℝ) : 
  (∃ y, f y = f x) ↔ (1 / 2 < x ∧ x < 3 / 2) :=
by
  sorry

end domain_of_f_l638_638913


namespace merchant_loss_l638_638422

theorem merchant_loss :
  ∃ (x y : ℝ), 
  x * 1.1 = 216 ∧ 
  y * 0.9 = 216 ∧ 
  2 * 216 - (x + y) = -4 := 
begin
  sorry,
end

end merchant_loss_l638_638422


namespace total_tiles_cost_is_2100_l638_638669

noncomputable def total_tile_cost : ℕ :=
  let length := 10
  let width := 25
  let tiles_per_sq_ft := 4
  let green_tile_percentage := 0.40
  let cost_per_green_tile := 3
  let cost_per_red_tile := 1.5
  let area := length * width
  let total_tiles := area * tiles_per_sq_ft
  let green_tiles := green_tile_percentage * total_tiles
  let red_tiles := total_tiles - green_tiles
  let cost_green := green_tiles * cost_per_green_tile
  let cost_red := red_tiles * cost_per_red_tile
  cost_green + cost_red

theorem total_tiles_cost_is_2100 : total_tile_cost = 2100 := by 
  sorry

end total_tiles_cost_is_2100_l638_638669


namespace productivity_comparison_l638_638376

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l638_638376


namespace regular_pentagon_l638_638279

-- Definitions of the conditions
variables {ABCDE : Type} [polygon ABCDE]

def is_convex (P : polygon) : Prop := sorry
def equal_sides (P : polygon) : Prop := sorry
def angles (P : polygon) : list ℝ := sorry

-- The problem statement in Lean
theorem regular_pentagon (ABCDE : polygon) 
  (h_convex : is_convex ABCDE)
  (h_equal_sides : equal_sides ABCDE)
  (h_angles_sum : (angles ABCDE).sum = 540)
  (h_angle_order : list.sorted (≥) (angles ABCDE)) :
  ∃ (a : ℝ), angles ABCDE = [a, a, a, a, a] :=
sorry

end regular_pentagon_l638_638279


namespace subtraction_result_is_82_l638_638812

theorem subtraction_result_is_82:
  let a := 35 * 4
  let b := 6 * 3
  let total := 240
  total - (a + b) = 82 :=
by
  let a := 35 * 4
  let b := 6 * 3
  let total := 240
  have h1 : a = 140 := by rfl
  have h2 : b = 18 := by rfl
  have h3 : total = 240 := by rfl
  calc
    total - (a + b) = 240 - (140 + 18)   : by { simp [h1, h2, h3] }
                ... = 240 - 158          : by simp
                ... = 82                 : by simp

end subtraction_result_is_82_l638_638812


namespace max_value_of_log_xy_l638_638144

noncomputable theory
open Classical

variable {x y u : ℝ}

theorem max_value_of_log_xy (x_ge_one : x ≥ 1) (y_ge_one : y ≥ 1) (cond : log x ^ 2 + log y ^ 2 = log (10 * x ^ 2) + log (10 * y ^ 2)) :
  ∃ u, u = log (x * y) ∧ u ≤ 2 + 2 * Real.sqrt 2 :=
sorry

end max_value_of_log_xy_l638_638144


namespace dives_to_collect_pearls_l638_638016

theorem dives_to_collect_pearls :
  ∀ (collect_per_dive oyster_prob : ℝ) (total_pearls : ℕ),
    collect_per_dive = 16 →
    oyster_prob = 0.25 →
    total_pearls = 56 →
    total_pearls / (collect_per_dive * oyster_prob) = 14 :=
by
  intros collect_per_dive oyster_prob total_pearls h1 h2 h3
  calc
    total_pearls / (collect_per_dive * oyster_prob)
    = 56 / (16 * 0.25) : by rw [h1, h2, h3]
    ... = 56 / 4       : by norm_num
    ... = 14           : by norm_num

end dives_to_collect_pearls_l638_638016


namespace female_cow_milk_production_l638_638017

noncomputable def milk_per_female_cow 
  (total_cows : ℕ) 
  (male_cows : ℕ) 
  (total_milk : ℕ) 
  (males_percentage : ℝ)
  (females_percentage : ℝ) 
  (male_cows_from_total : males_percentage * total_cows = male_cows)
  (total_cows_ratio: males_percentage + females_percentage = 1) 
  (milk_produced : total_cows * females_percentage * total_milk = total_milk) : ℝ :=
total_milk / (females_percentage * total_cows)

theorem female_cow_milk_production : 
  ∀ (total_cows : ℕ) (male_cows : ℕ) (total_milk : ℕ)
    (males_percentage : ℝ) (females_percentage : ℝ),
  males_percentage = 0.40 →
  females_percentage = 0.60 →
  male_cows = 50 →
  total_milk = 150 →
  total_cows = 125 →
  milk_per_female_cow total_cows male_cows total_milk males_percentage females_percentage (by norm_num) (by norm_num) = 2 :=
by
  intros
  sorry

end female_cow_milk_production_l638_638017


namespace Amanda_lost_notebooks_l638_638888

theorem Amanda_lost_notebooks (initial_notebooks ordered additional_notebooks remaining_notebooks : ℕ)
  (h1 : initial_notebooks = 10)
  (h2 : ordered = 6)
  (h3 : remaining_notebooks = 14) :
  initial_notebooks + ordered - remaining_notebooks = 2 := by
sorry

end Amanda_lost_notebooks_l638_638888


namespace parallel_trans_l638_638974

-- Definitions of non-zero and parallel vectors.
variables {𝕜 : Type*} [Field 𝕜] {V : Type*} [AddCommGroup V] [Module 𝕜 V]

def nonzero_vector (v : V) : Prop := v ≠ 0

def parallel (u v : V) : Prop := ∃ k : 𝕜, u = k • v

-- The theorem
theorem parallel_trans {a b c : V} (ha : nonzero_vector a) (hb : nonzero_vector b) (hc : nonzero_vector c)
  (hab : parallel a b) (hac : parallel c a) : parallel c b :=
sorry

end parallel_trans_l638_638974


namespace domain_of_f_parity_of_f_range_of_f_l638_638157

def f (x : ℝ) : ℝ := log (1 - x) + log (1 + x) + x^4 - 2 * x^2

-- Define the domain condition for \( f(x) \)
theorem domain_of_f : ∀ x, (1 - x > 0 ∧ 1 + x > 0) ↔ (-1 < x ∧ x < 1) :=
  by sorry

-- Define the parity condition for \( f(x) \)
theorem parity_of_f : ∀ x, f(-x) = f(x) :=
  by sorry

-- Define the range condition for \( f(x) \)
theorem range_of_f : ∀ y, (∃ x, f(x) = y) ↔ y ≤ 0 :=
  by sorry

end domain_of_f_parity_of_f_range_of_f_l638_638157


namespace find_lambda_l638_638176

variables (λ : ℝ)
def a : ℝ × ℝ := (3,5)
def b : ℝ × ℝ := (2,4)
def c : ℝ × ℝ := (-3,-2)

theorem find_lambda (h : (a.1 + λ * b.1) * c.1 + (a.2 + λ * b.2) * c.2 = 0) : λ = -19/14 :=
by sorry

end find_lambda_l638_638176


namespace isosceles_pyramid_dihedral_false_l638_638621

-- Definitions to specify the conditions
structure Pyramid where
  apex : Point3D
  base : Set Point3D
  lateral_edges : Set (Point3D × Point3D)
  legs_equal : ∀ p, p ∈ lateral_edges → EdgeLength p = constant_length

-- Statement B rephrased
def dihedral_angles_are_equal_or_supp : Prop := 
  ∀ p q r s, dihedralAngle p q r = dihedralAngle p q s ∨
            dihedralAngle p q r + dihedralAngle p q s = π

-- The isosceles pyramid condition
def is_isosceles_pyramid (p: Pyramid) : Prop := 
  ∀ p1 p2, p1 ∈ p.base → p2 ∈ p.base → distance p.apex p1 = distance p.apex p2

-- The Lean statement for the proof
theorem isosceles_pyramid_dihedral_false (p: Pyramid) (h: is_isosceles_pyramid p) 
  (legs_eq: p.legs_equal) : ¬dihedral_angles_are_equal_or_supp :=
  sorry

end isosceles_pyramid_dihedral_false_l638_638621


namespace smallest_sum_of_squares_value_l638_638007

noncomputable def collinear_points_min_value (A B C D E P : ℝ): Prop :=
  let AB := 3
  let BC := 2
  let CD := 5
  let DE := 4
  let pos_A := 0
  let pos_B := pos_A + AB
  let pos_C := pos_B + BC
  let pos_D := pos_C + CD
  let pos_E := pos_D + DE
  let P := P
  let AP := (P - pos_A)
  let BP := (P - pos_B)
  let CP := (P - pos_C)
  let DP := (P - pos_D)
  let EP := (P - pos_E)
  let sum_squares := AP^2 + BP^2 + CP^2 + DP^2 + EP^2
  (sum_squares = 85.2)

theorem smallest_sum_of_squares_value : ∃ (A B C D E P : ℝ), collinear_points_min_value A B C D E P :=
sorry

end smallest_sum_of_squares_value_l638_638007


namespace ratio_of_speeds_l638_638346

variable (x y n : ℝ)

-- Conditions
def condition1 : Prop := 3 * (x - y) = n
def condition2 : Prop := 2 * (x + y) = n

-- Problem Statement
theorem ratio_of_speeds (h1 : condition1 x y n) (h2 : condition2 x y n) : x = 5 * y :=
by
  sorry

end ratio_of_speeds_l638_638346


namespace min_value_of_f_l638_638085

-- Define the function
def f (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2 + ((a + b + c)^2) / 3

-- Define the conditions
variable (a b c : ℝ)
variable (h : a - b + 2 * c = 3)

-- State that we want to find the minimum value of m
theorem min_value_of_f (h : a - b + 2 * c = 3) : ∃ m, m = a^2 + b^2 + c^2 ∧ m ≥ (3 / 2) ∧ 
  (∀ x, f x a b c ≥ (3 / 2)) := 
sorry

end min_value_of_f_l638_638085


namespace arrange_abc_l638_638972

def f (x : ℝ) : ℝ := 
  if x > 0 then sqrt (4^x + 1) - sqrt (4^x + 2) 
  else sqrt (4^(-x) + 1) - sqrt (4^(-x) + 2)

noncomputable def a : ℝ := f (Real.log 0.2 / Real.log 3)
noncomputable def b : ℝ := f (3^(-0.2))
noncomputable def c : ℝ := f (-3^(1.1))

theorem arrange_abc : c > a ∧ a > b :=
sorry

end arrange_abc_l638_638972


namespace glass_volume_230_l638_638437

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638437


namespace tangent_and_normal_lines_l638_638101

noncomputable def tangent_line (t : ℝ) : ℝ :=
  3 * t - 4

noncomputable def normal_line (t : ℝ) : ℝ :=
  -(t / 3) + 1

theorem tangent_and_normal_lines : 
  ∀ (t : ℝ), t = 1 → 
  ∃ (x y : ℝ), 
  x = (2 * t + t^2) / (1 + t^3) ∧ 
  y = (2 * t - t^2) / (1 + t^3) ∧ 
  tangent_line x = 3 * x - 4 ∧ 
  normal_line x = -(x / 3) + 1 := 
by {
  intro t,
  intro ht,
  use [((2 * t + t^2) / (1 + t^3)), ((2 * t - t^2) / (1 + t^3))],
  split; try {simp},
  split; try {simp},
  have ht1 : t = 1, exact ht, rw ht1,
  simp,
  have ht2 : t = 1, exact ht, rw ht2,
  exact sorry,
}

end tangent_and_normal_lines_l638_638101


namespace glass_volume_230_l638_638439

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638439


namespace total_rabbits_after_n_months_l638_638653

noncomputable def total_rabbits (α β : ℕ) (n : ℕ) : ℕ :=
  let a : ℕ → ℕ := λ k, if k = 0 then 1 else if k = 1 then α + 1 else a (k - 1) + α * b (k - 2) in
  let b : ℕ → ℕ := λ k, if k = 0 then 1 else if k = 1 then β + 1 else b (k - 1) + β * b (k - 2) in
  a n + b n

theorem total_rabbits_after_n_months (α β n : ℕ) : total_rabbits α β n = a n + b n := sorry

end total_rabbits_after_n_months_l638_638653


namespace ratio_XG_GY_l638_638218

-- conditions
variables (X Y Z E G Q : Type)
variables [AddCommGroup X] [AddCommGroup Y] [AddCommGroup Z] [AddCommGroup E] [AddCommGroup G] [AddCommGroup Q]
variables [VectorSpace ℝ X] [VectorSpace ℝ Y] [VectorSpace ℝ Z] [VectorSpace ℝ E] [VectorSpace ℝ G] [VectorSpace ℝ Q]
variables (x y z e g q : X)

-- given
axiom HXQ_QE : ∃ ρ : ℝ, ρ = 5 / 7 ∧ q = (ρ • x) + ((1 - ρ) • e)
axiom HGQ_QY : ∃ σ : ℝ, σ = 3 / 7 ∧ q = (σ • g) + ((1 - σ) • y)

-- to prove
theorem ratio_XG_GY : (∃ k : ℝ, k = 3 / 5 ∧ g = (k • x) + ((1 - k) • y)) := sorry

end ratio_XG_GY_l638_638218


namespace glass_volume_is_230_l638_638450

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638450


namespace ratio_equiv_solve_x_l638_638196

theorem ratio_equiv_solve_x (x : ℕ) (h : 3 / 12 = 3 / x) : x = 12 :=
sorry

end ratio_equiv_solve_x_l638_638196


namespace equivalent_weeks_l638_638610

def hoursPerDay := 24
def daysPerWeek := 7
def hoursPerWeek := daysPerWeek * hoursPerDay
def totalHours := 2016

theorem equivalent_weeks : totalHours / hoursPerWeek = 12 := 
by
  sorry

end equivalent_weeks_l638_638610


namespace lockers_toggle_l638_638826

theorem lockers_toggle (n : ℕ) :
  ∀ i : ℕ, (i ≤ n) → (is_perfect_square i ↔ locker_is_open_after_operations n i) :=
by
  -- Definitions and setup of initial conditions:
  let is_perfect_square := λ i : ℕ, ∃ k : ℕ, k * k = i
  let locker_is_open_after_operations := λ n i : ℕ, 
    let divisors := λ (k : ℕ) (i : ℕ), i % k = 0 in 
    (list.range_succ n).count (divisors ^~ i) % 2 = 1
  
  -- Skipping the formal proof:
  sorry

end lockers_toggle_l638_638826


namespace productivity_comparison_l638_638374

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l638_638374


namespace find_sum_F_J_l638_638071

noncomputable def R (z : ℂ) : ℂ := z^5 + 5*z^4 + 10*z^3 + 10*z^2 + 5*z + 1
def f (z : ℂ) : ℂ := 2 * complex.I * conj z

lemma vieta_sum_of_roots : ∑ z in (finset.univ : finset ℂ), z = -5 := sorry
lemma vieta_sum_of_products_of_roots_two_at_a_time : ∑ (s : finset ℂ) in (finset.powerset_len 2 (finset.univ : finset ℂ)), s.prod id = 10 := sorry
lemma vieta_product_of_roots : (finset.univ : finset ℂ).prod id = 1 := sorry

noncomputable def S (z : ℂ) : ℂ := z^5 + (2 * complex.I)^4 * 5 * z^4 + (2 * complex.I)^3 * 10 * conj 10 * z^3 + sorry
def F : ℂ := (2 * complex.I)^3 * conj 10 -- Calculating F
def J : ℂ := (2 * complex.I)^5 * conj 1 -- Calculating J

theorem find_sum_F_J : F + J = -48 * complex.I :=
by
  -- Calculate F and J separately and then sum them
  have hF : F = -80 * complex.I := sorry
  have hJ : J = 32 * complex.I := sorry
  calc
  F + J = -80 * complex.I + 32 * complex.I : by rw [hF, hJ]
       ... = -48 * complex.I : by ring

end find_sum_F_J_l638_638071


namespace exist_distinct_nums_with_equal_sum_of_digits_l638_638699

-- Define the sum of the digits function S
def S (n : ℕ) : ℕ :=
  n.toDigits 10 |>.sum

-- The proof problem statement
theorem exist_distinct_nums_with_equal_sum_of_digits (m n p : ℕ) 
  (h1 : m ≠ n) (h2 : n ≠ p) (h3 : m ≠ p) :
  m + S m = n + S n ∧ n + S n = p + S p := 
sorry

end exist_distinct_nums_with_equal_sum_of_digits_l638_638699


namespace distance_between_parallel_lines_l638_638623

theorem distance_between_parallel_lines 
  (l1 l2 : ℝ × ℝ × ℝ)
  (h1: l1 = ⟨2, -2, -6⟩)
  (h2: l2 = ⟨1, -1, 4⟩)
  (parallel : l1.1 * -1 = l2.1 * -2) :
  real.abs (l2.3 - l1.3) / real.sqrt (l1.1^2 + l1.2^2) = 7 * real.sqrt 2 / 2 :=
begin
  sorry
end

end distance_between_parallel_lines_l638_638623


namespace average_age_of_John_Mary_Tonya_is_35_l638_638231

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l638_638231


namespace distinct_symmetric_differences_l638_638966

variables {X : Type*} (𝓐 : set (set X)) (t : ℕ)

open set

def symmetric_difference (A B : set X) := (A \ B) ∪ (B \ A)

theorem distinct_symmetric_differences (h𝓐 : fintype 𝓐) (card_𝓐 : fintype.card 𝓐 = t) (ht : 2 ≤ t) :
  ∃ S : set (set X), (S ⊆ {symmetric_difference A B | A B ∈ 𝓐}) ∧ fintype.card S ≥ t :=
sorry

end distinct_symmetric_differences_l638_638966


namespace find_function_expression_find_range_of_m_l638_638650

-- Statement for Part 1
theorem find_function_expression (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) : 
  y = -1/2 * x - 2 := 
sorry

-- Statement for Part 2
theorem find_range_of_m (m x : ℝ) (hx : x > -2) (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) :
  (-x + m < -1/2 * x - 2) ↔ (m ≤ -3) := 
sorry

end find_function_expression_find_range_of_m_l638_638650


namespace pistachios_opened_shells_correct_l638_638020

-- Define the constants based on the conditions
def total_pistachios : ℕ := 80
def percent_with_shells : ℝ := 0.95
def percent_shells_opened : ℝ := 0.75

-- Define the expected result based on the correct answer
def pistachios_with_shells_and_opened_shell : ℕ := 57

-- Statement to prove
theorem pistachios_opened_shells_correct :
  (percent_shells_opened * (percent_with_shells * total_pistachios)).toNat = pistachios_with_shells_and_opened_shell :=
by
  sorry

end pistachios_opened_shells_correct_l638_638020


namespace glass_volume_l638_638461

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638461


namespace initial_students_l638_638642

variable (x : ℕ) -- let x be the initial number of students

-- each condition defined as a function
def first_round_rem (x : ℕ) : ℕ := (40 * x) / 100
def second_round_rem (x : ℕ) : ℕ := first_round_rem x / 2
def third_round_rem (x : ℕ) : ℕ := second_round_rem x / 4

theorem initial_students (x : ℕ) (h : third_round_rem x = 15) : x = 300 := 
by sorry  -- proof will be inserted here

end initial_students_l638_638642


namespace relatively_prime_b_count_l638_638821

theorem relatively_prime_b_count :
  let ℓ := 12 in
  let b_range := {b : ℕ | 1 < b ∧ b < 12} in
  let relatively_prime (b : ℕ) := Nat.gcd b ℓ = 1 in
  (set_of (λ b, (b ∈ b_range) ∧ relatively_prime b)).card = 3 :=
by
  sorry

end relatively_prime_b_count_l638_638821


namespace Eva_is_6_l638_638922

def ages : Set ℕ := {2, 4, 6, 8, 10}

def conditions : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a + b = 12 ∧
  b ≠ 2 ∧ b ≠ 10 ∧ a ≠ 2 ∧ a ≠ 10 ∧
  (∃ c d, c ∈ ages ∧ d ∈ ages ∧ c = 2 ∧ d = 10 ∧
           (∃ e, e ∈ ages ∧ e = 4 ∧
           ∃ eva, eva ∈ ages ∧ eva ≠ 2 ∧ eva ≠ 4 ∧ eva ≠ 8 ∧ eva ≠ 10 ∧ eva = 6))

theorem Eva_is_6 (h : conditions) : ∃ eva, eva ∈ ages ∧ eva = 6 := sorry

end Eva_is_6_l638_638922


namespace none_of_the_relations_hold_l638_638033

-- Given conditions
variable {α : Type*} [EuclideanGeometry α]
variables {A B C D : Point α}

-- The points A, B, and C form a triangle ABC
-- D is a point on BC; therefore, D is collinear with B and C.
axiom D_on_BC : collinear B C D

-- Let's state the relationships as propositions
def quadrilaterals_are_similar (ABDC ADBC : Quadrilateral α) : Prop := ∃ k, similar ABDC ADBC k
def quadrilaterals_are_congruent (ABDC ADBC : Quadrilateral α) : Prop := ∃ θ, congruent ABDC ADBC θ
def triangle_quadrilaterals_are_similar (ABD : Triangle α) (ABDC ADBC : Quadrilateral α) : Prop := (similar ABD ABDC) ∧ (similar ABD ADBC)
def quadrilaterals_equal_areas (ABDC ADBC : Quadrilateral α) : Prop := area ABDC = area ADBC

-- The proof statement
theorem none_of_the_relations_hold (ABD : Triangle α) (ABDC ADBC : Quadrilateral α) :
  ¬ quadrilaterals_are_similar ABDC ADBC ∧
  ¬ quadrilaterals_are_congruent ABDC ADBC ∧
  ¬ triangle_quadrilaterals_are_similar ABD ABDC ADBC ∧
  ¬ quadrilaterals_equal_areas ABDC ADBC :=
sorry

end none_of_the_relations_hold_l638_638033


namespace exists_n_consecutive_non_prime_or_prime_power_l638_638280

theorem exists_n_consecutive_non_prime_or_prime_power (n : ℕ) (h : n > 0) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ (Nat.Prime (seq i)) ∧ ¬ (∃ p k : ℕ, p.Prime ∧ k > 1 ∧ seq i = p ^ k)) :=
by
  sorry

end exists_n_consecutive_non_prime_or_prime_power_l638_638280


namespace at_least_one_greater_than_zero_l638_638248

noncomputable def a (x : ℝ) : ℝ := x^2 - 2 * x + (Real.pi / 2)
noncomputable def b (y : ℝ) : ℝ := y^2 - 2 * y + (Real.pi / 2)
noncomputable def c (z : ℝ) : ℝ := z^2 - 2 * z + (Real.pi / 2)

theorem at_least_one_greater_than_zero (x y z : ℝ) : (a x > 0) ∨ (b y > 0) ∨ (c z > 0) :=
by sorry

end at_least_one_greater_than_zero_l638_638248


namespace vector_on_line_at_t_neg3_l638_638857

theorem vector_on_line_at_t_neg3 : 
  ∃ (a : ℝ × ℝ) (d : ℝ × ℝ), 
    (a + (1 : ℝ) * d = (2, 5)) ∧ 
    (a + (4 : ℝ) * d = (5, -7)) ∧ 
    (a + (-3 : ℝ) * d = (-2, 21)) :=
sorry

end vector_on_line_at_t_neg3_l638_638857


namespace parallelogram_area_DECF_l638_638894

-- Define the points of the triangle and the parallelogram
variables {A B C D E F : Type} [linear_ordered_field A]

-- Assume point E divides segment AB into 3:1 ratio
def is_quarter_point (A B E : Type) [linear_ordered_field A] : Prop :=
  distance E B = 1 / 4 * distance A B

-- Define the area of triangle BEC
def area_triangle_BEC (A : Type) [linear_ordered_field A] := 30

theorem parallelogram_area_DECF
  (A B C D E F : Type) [linear_ordered_field A]
  (h1 : is_quarter_point A B (line_segment A B E))
  (h2 : area_triangle_BEC A = 30) :
  parallelogram_area D E C F = 240 :=
begin
  sorry
end

end parallelogram_area_DECF_l638_638894


namespace sector_area_correct_l638_638753

variable (r : ℝ)

def sector_arc_length (r : ℝ) : ℝ := (3 / 4) * Mathlib.pi * r

def sector_area (r : ℝ) : ℝ := (1 / 2) * Mathlib.pi * (3 / 4) * r * r

theorem sector_area_correct (h1 : sector_arc_length r = 3 * Mathlib.pi) : sector_area r = 6 * Mathlib.pi := 
    sorry

end sector_area_correct_l638_638753


namespace prepare_50_papers_l638_638206

-- Definitions based on the given conditions
structure Topic :=
  (id : Fin 25)
  (questions : Fin 8)

noncomputable def can_prepare_50_papers {T : Type} (topics : Fin 25 → T) (questions_per_topic : Fin 25 → Fin 8) : Prop :=
  ∃ (papers : Fin 50 → Fin 4 → T),
    (∀ (i : Fin 50), ∃ (t1 t2 t3 t4 : Fin 25), 
      papers i 0 = topics t1 ∧ papers i 1 = topics t2 ∧ papers i 2 = topics t3 ∧ papers i 3 = topics t4 ∧
      t1 ≠ t2 ∧ t1 ≠ t3 ∧ t1 ≠ t4 ∧ t2 ≠ t3 ∧ t2 ≠ t4 ∧ t3 ≠ t4) ∧
    (∀ (t : Fin 25) (q : Fin 8), ∃! (paper : Fin 50) (pos : Fin 4), papers paper pos = topics t) ∧
    (∀ (t1 t2 : Fin 25), t1 ≠ t2 → ∃ (paper : Fin 50), 
      ∃ (pos1 pos2 : Fin 4), papers paper pos1 = topics t1 ∧ papers paper pos2 = topics t2)

-- The theorem that we need to prove
theorem prepare_50_papers : can_prepare_50_papers (λ id, id) (λ id, id) :=
  sorry

end prepare_50_papers_l638_638206


namespace extreme_point_min_max_increasing_f_condition_l638_638156

noncomputable def f (x : ℝ) (a : ℝ) := x^3 - a * x^2 - 3 * x

theorem extreme_point_min_max (a : ℝ) (h : a = 4) :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ (f x a ≤ f 1 a) ∧ (f x a ≤ f 3 a ∧ f 3 a = -18) ∧ (f 1 a = -6) := 
begin
  sorry
end

theorem increasing_f_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → 3 * x^2 - 2 * a * x - 3 ≥ 0) ↔ a ≤ 0 :=
begin
  sorry
end

end extreme_point_min_max_increasing_f_condition_l638_638156


namespace glass_volume_230_l638_638446

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638446


namespace class_grades_l638_638501

theorem class_grades (boys girls n : ℕ) (h1 : girls = boys + 3) (h2 : ∀ (fours fives : ℕ), fours = fives + 6) (h3 : ∀ (threes : ℕ), threes = 2 * (fives + 6)) : ∃ k, k = 2 ∨ k = 1 :=
by
  sorry

end class_grades_l638_638501


namespace angle_equality_l638_638782

open EuclideanGeometry

variables {A B C D P : Point}
variables {α β γ δ : ℝ}

-- Definition of a parallelogram by its vertices
def parallelogram (A B C D : Point) : Prop :=
  collinear A B C ∧ collinear B C D ∧ collinear C D A ∧ collinear D A B

theorem angle_equality (parallelogram A B C D : parallelogram A B C D)
  (P_outside : ¬_inside_of_parallelogram P A B C D)
  (angle_PAB_eq_angle_PCB : ∠ P A B = ∠ P C B)
  (opposite_directions : ∠ P A B + ∠ P C B = π):
  ∠ A P B = ∠ D P C :=
by
  sorry

end angle_equality_l638_638782


namespace shoes_no_bad_pairings_l638_638919

theorem shoes_no_bad_pairings :
  ∃ m n : ℕ, Nat.Coprime m n ∧ m + n = 399 ∧ 
  (∃! (pairs : Finset (Fin 8 × Fin 8)), 
    ∀ k < 4, ∀ s : Finset (Fin 8 × Fin 8), s.card = k → 
        ¬ ∀ i ∈ s, (↑(i.1) = ↑(i.2))) →
  ∑ p in pairs, 1 = 7728 := sorry

end shoes_no_bad_pairings_l638_638919


namespace addition_puzzle_solution_l638_638660

theorem addition_puzzle_solution :
  ∃ I : ℕ, ∀ S X E L V : ℕ,
    S = 7 →
    X % 2 = 1 →
    S ≠ I ∧ S ≠ X ∧ S ≠ E ∧ S ≠ L ∧ S ≠ V ∧
    I ≠ X ∧ I ≠ E ∧ I ≠ L ∧ I ≠ V ∧
    X ≠ E ∧ X ≠ L ∧ X ≠ V ∧
    E ≠ L ∧ E ≠ V ∧
    L ≠ V ∧
    (S * 100 + I * 10 + X) + (S * 100 + I * 10 + X) = E * 1000 + L * 100 + E * 10 + V →
    I = 2 :=
begin
  sorry
end

end addition_puzzle_solution_l638_638660


namespace triangle_is_right_if_quadratic_vertex_on_x_axis_l638_638781

variables (a b c : ℝ)

/-- Prove that if the quadratic function y = x^2 - 2*(a + b)*x + c^2 + 2*ab has its vertex on the
    x-axis and a, b, c are the lengths of the sides of ΔABC, then ΔABC is a right triangle. -/
theorem triangle_is_right_if_quadratic_vertex_on_x_axis
  (h1 : (x : ℝ) -> (y : ℝ) = x^2 - 2 * (a + b) * x + c^2 + 2 * ab)
  (h2 : c^2 - a^2 - b^2 = 0) :
  a^2 + b^2 = c^2 :=
begin
  sorry
end

end triangle_is_right_if_quadratic_vertex_on_x_axis_l638_638781


namespace transformed_statistics_l638_638973

variables (n : ℕ) (x : Fin n → ℝ) (a b : ℝ) (x_bar s2 : ℝ) 

-- Assuming the given conditions
variables (h1 : Median x = a) 
variables (h2 : Mode x = b)
variables (h3 : Mean x = x_bar)
variables (h4 : Variance x = s2)

-- Define the new data set y_i = 7x_i - 9
def y (i : Fin n) := 7 * x i - 9

-- Define the new median, mode, mean, and variance variables
variables (a' b' x_bar' s2' : ℝ)

-- State the equivalent proof problem
theorem transformed_statistics :
  (Mode y = 7 * b - 9) ∧ (Mean y = 7 * x_bar - 9) :=
by
  sorry

end transformed_statistics_l638_638973


namespace digits_not_equal_l638_638308

-- Define a function that counts occurrences of a digit d in the concatenated sequence from 1 to n
def count_digit (d : ℕ) (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc, acc + (x.digits 10).count d) 0

theorem digits_not_equal (n : ℕ) : ¬ ∀ d, count_digit d n = count_digit 1 n :=
by sorry

end digits_not_equal_l638_638308


namespace shared_tetrahedron_volume_l638_638561

-- Definitions based on the conditions
structure Tetrahedron (α : Type) [OrderedField α] :=
(A B C D : Point α)

def centroid {α : Type} [OrderedField α] (A B C : Point α) : Point α :=
(A + B + C) / 3

def volume {α : Type} [OrderedField α] (t : Tetrahedron α) : α := sorry

def is_on_plane {α : Type} [OrderedField α] (p : Point α) (A B C : Point α) : Prop := sorry

-- The actual Lean statement of the proof problem
theorem shared_tetrahedron_volume {α : Type} [OrderedField α] 
  (ABC : Tetrahedron α) 
  (A1 B1 C1 D1 : Point α)
  (V : α)
  (hA1_on_plane_BCD : is_on_plane A1 ABC.B ABC.C ABC.D)
  (hB1_on_plane_CDA : is_on_plane B1 ABC.C ABC.D ABC.A)
  (hC1_on_plane_DAB : is_on_plane C1 ABC.D ABC.A ABC.B)
  (hD1_on_plane_ABC : is_on_plane D1 ABC.A ABC.B ABC.C)
  (hA1_centroid_BCD : A1 = centroid ABC.B ABC.C ABC.D)
  (hBD1_mid_AC : B1 = midpoint ABC.A ABC.D)
  (hCB1_mid_AD : C1 = midpoint ABC.A ABC.C)
  (hDC1_mid_AB : D1 = midpoint ABC.A ABC.B)
  (hV : volume ABC = V) :
  volume (sharedPartTetrahedrons ABC (mkTetrahedron A1 B1 C1 D1)) = (3 / 8) * V := 
sorry

noncomputable def sharedPartTetrahedrons {α : Type} [OrderedField α] 
  (t1 t2 : Tetrahedron α) : Tetrahedron α := sorry

noncomputable def midpoint {α : Type} [OrderedField α] (A B : Point α) : Point α := 
(A + B) / 2

-- Creating another tetrahedron based on points
def mkTetrahedron {α : Type} [OrderedField α] 
  (A1 B1 C1 D1 : Point α) : Tetrahedron α :=
{A := A1, B := B1, C := C1, D := D1}

end shared_tetrahedron_volume_l638_638561


namespace max_value_norm_diff_l638_638124

def vec3 := ℝ × ℝ × ℝ

def norm (v : vec3) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def sub (a b : vec3) : vec3 :=
  (a.1 - 2 * b.1, a.2 - 2 * b.2, a.3 - 2 * b.3)

def vec_a : vec3 := (1, 2, -2)

axiom norm_b : ℝ
axiom b_unit : norm_b = 1

noncomputable def max_norm_diff (a b : vec3) : ℝ :=
  real.sqrt (13 - 12 * real.cos (π))

theorem max_value_norm_diff : max_norm_diff vec_a (1, 1, 1) = 5 :=
  by {
    have h : norm_b = 1, from sorry,
    have angle_cos : real.cos (π) = -1, from sorry,
    rw [angle_cos],
    exact real.sqrt_eq_iff_eq_square.2 (or.inl rfl)
  }

end max_value_norm_diff_l638_638124


namespace divisible_by_7_of_sum_of_squares_l638_638628

theorem divisible_by_7_of_sum_of_squares (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 
    (7 ∣ a) ∧ (7 ∣ b) :=
sorry

end divisible_by_7_of_sum_of_squares_l638_638628


namespace polarEquationRepresentsCircleAndRay_l638_638656

noncomputable def polarEquationGraph (ρ θ : ℝ) : Prop :=
  (ρ - 3) * (θ - π / 2) = 0 ∧ ρ ≥ 0

theorem polarEquationRepresentsCircleAndRay :
  ∀ (ρ θ : ℝ), polarEquationGraph ρ θ → (ρ = 3 ∨ θ = π / 2) :=
by
  intros ρ θ h,
  sorry

end polarEquationRepresentsCircleAndRay_l638_638656


namespace glass_volume_230_l638_638447

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638447


namespace people_in_room_l638_638181

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end people_in_room_l638_638181


namespace part_1_part_2_l638_638689

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

-- Given conditions
variables (a b : ℝ) (h_ab : a ≠ 0 ∧ b ≠ 0)
variables (h_cond : ∀ x : ℝ, f a b x ≥ f a b (5 * Real.pi / 6))

-- Proof statements
theorem part_1 : f a b (Real.pi / 3) = 0 := sorry

theorem part_2 : ∀ x : ℝ, ∃ y : ℝ, y = f a b x ∧ (a, b) ∈ set.range (λ t : ℝ, (t, f a b t)) := sorry

end part_1_part_2_l638_638689


namespace linear_function_y_axis_intersection_l638_638767

theorem linear_function_y_axis_intersection (k : ℝ) (M : ℝ × ℝ) (hk : M = (-1, -2)) :
  (∃ k, ∀ x y, y = k * (x - 1) → y = -1 ↔ x = 0) :=
by {
  have h : k * (-1 - 1) = -2,
  { rw hk, },
  sorry
}

end linear_function_y_axis_intersection_l638_638767


namespace subset_size_condition_l638_638242

theorem subset_size_condition (p : ℕ) (hp : Prime p) (F : Finset ℕ) (hF : F = Finset.range p) 
  (A : Finset ℕ) (hA_prop : A ⊂ F)
  (hA_cond : ∀ a b ∈ A, (a * b + 1) % p ∈ A) : 
  if p = 2 then A.card = 0 else A.card = 1 :=
by
  sorry

end subset_size_condition_l638_638242


namespace angle_BCD_is_48_degrees_l638_638215

theorem angle_BCD_is_48_degrees
  (EB DC AB ED AC BD : Prop)
  (parallel_diameter : EB = DC)
  (parallel_abed : AB = ED)
  (parallel_acbd : AC = BD)
  (angle_ratio : ∃ x, angle AEB = 7 * x ∧ angle ABE = 8 * x)
  (right_angle_AEB : angle AEB = 90) :
  angle BCD = 48 :=
by 
  sorry

end angle_BCD_is_48_degrees_l638_638215


namespace calc_expression_l638_638836

theorem calc_expression : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end calc_expression_l638_638836


namespace trailing_zeros_sum_l638_638251

theorem trailing_zeros_sum :
  ∃ (n₁ n₂ n₃ n₄ : ℕ), n₁ > 6 ∧ n₂ > 6 ∧ n₃ > 6 ∧ n₄ > 6 ∧
  (let k₁ := (n₁ / 5 + n₁ / 25 + n₁ / 125 + n₁ / 625 + ...)
   in let k₂ := (n₂ / 5 + n₂ / 25 + n₂ / 125 + n₂ / 625 + ...)
   in let k₃ := (n₃ / 5 + n₃ / 25 + n₃ / 125 + n₃ / 625 + ...)
   in let k₄ := (n₄ / 5 + n₄ / 25 + n₄ / 125 + n₄ / 625 + ...),
  let t₁ := (3 * n₁ / 5 + 3 * n₁ / 25 + 3 * n₁ / 125 + 3 * n₁ / 625 + ...)
  in let t₂ := (3 * n₂ / 5 + 3 * n₂ / 25 + 3 * n₂ / 125 + 3 * n₂ / 625 + ...)
  in let t₃ := (3 * n₃ / 5 + 3 * n₃ / 25 + 3 * n₃ / 125 + 3 * n₃ / 625 + ...)
  in let t₄ := (3 * n₄ / 5 + 3 * n₄ / 25 + 3 * n₄ / 125 + 3 * n₄ / 625 + ...),
  4 * k₁ = t₁ ∧ 4 * k₂ = t₂ ∧ 4 * k₃ = t₃ ∧ 4 * k₄ = t₄ ∧
  n₁ + n₂ + n₃ + n₄ = 56) :=
sorry

end trailing_zeros_sum_l638_638251


namespace remove_one_piece_l638_638150

theorem remove_one_piece (pieces : Finset (Fin 8 × Fin 8)) (h_card : pieces.card = 15)
  (h_row : ∀ r : Fin 8, ∃ c, (r, c) ∈ pieces)
  (h_col : ∀ c : Fin 8, ∃ r, (r, c) ∈ pieces) :
  ∃ pieces' : Finset (Fin 8 × Fin 8), pieces'.card = 14 ∧ 
  (∀ r : Fin 8, ∃ c, (r, c) ∈ pieces') ∧ 
  (∀ c : Fin 8, ∃ r, (r, c) ∈ pieces') :=
sorry

end remove_one_piece_l638_638150


namespace circumcircles_concur_at_one_point_l638_638175

-- Definitions of points
variables (A B C D E F : Point)
variables (ABCD : Parallelogram A B C D)
variables (AECF : Parallelogram A E C F)
variables (common_diag : Diagonal A C)
variables (inside_E : InsideParallelogram E ABCD)
variables (inside_F : InsideParallelogram F ABCD)

-- Proof statement
theorem circumcircles_concur_at_one_point :
  ∃ P : Point, is_circumcenter P A E B ∧ 
               is_circumcenter P B F C ∧ 
               is_circumcenter P C E D ∧ 
               is_circumcenter P D F A :=
sorry

end circumcircles_concur_at_one_point_l638_638175


namespace convert_polar_to_rectangular_eq_l638_638775

noncomputable def polar_to_rectangular_circle_equation (ρ θ : ℝ) : Prop :=
  ∃ x y : ℝ, (x, y) = (ρ * cos θ, ρ * sin θ) ∧ ρ = 2 * cos (θ + π / 3)

-- Lean statement to prove
theorem convert_polar_to_rectangular_eq :
  polar_to_rectangular_circle_equation ρ θ →
  ∃ x y : ℝ, x^2 + y^2 - x + sqrt 3 * y = 0 :=
sorry

end convert_polar_to_rectangular_eq_l638_638775


namespace f_4_eq_4_f_n_expression_l638_638243

def P (n : ℕ) : Set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 ^ (n / 2) else 2 ^ ((n + 1) / 2)

theorem f_4_eq_4 : f 4 = 4 := by
  sorry

theorem f_n_expression (n : ℕ) : f n = 
  if even n then 2 ^ (n / 2)
  else 2 ^ ((n + 1) / 2) := by
  sorry

end f_4_eq_4_f_n_expression_l638_638243


namespace proof_problem_l638_638077

-- Define the conditions
def condition1 (x : ℝ) : Prop := 3 ≤ |x - 3| ∧ |x - 3| ≤ 6
def condition2 (x : ℝ) : Prop := x ≤ 8

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-3 ≤ x ∧ x ≤ 3) ∨ (6 ≤ x ∧ x ≤ 8)

theorem proof_problem :
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ solution_set x :=
by
  intros x
  split
  sorry

end proof_problem_l638_638077


namespace inv_f_of_6_l638_638942

def f (x : ℝ) : ℝ := 1 + x^2 + Real.log2 x

theorem inv_f_of_6 : f⁻¹ 6 = 2 :=
sorry

end inv_f_of_6_l638_638942


namespace prob_D_at_least_two_passing_grades_prob_only_one_C_D_receives_award_l638_638118

-- Conditions
variables (Contestant : Type) [fintype Contestant] [decidable_eq Contestant]
variables (A B C D : Contestant)
variables (competitions : fin 3) -- Three competitions
variables (passing : Contestant → competitions → Prop)

-- Each contestant has an equal chance of passing or failing each competition
axiom equal_chance_passing : ∀ (x : Contestant) (c : competitions), probability (passing x c) = 1 / 2

-- Questions to Prove

-- Question 1: The probability that contestant D gets at least two passing grades is 1/2
theorem prob_D_at_least_two_passing_grades : 
  probability ((set_of (λ (comp_set : set competitions), D ∈ comp_set ∧ comp_set.size ≥ 2) : set (set competitions))) = 1 / 2 := 
sorry

-- Question 2: The probability that only one of contestants C and D receives an award is 2/3
theorem prob_only_one_C_D_receives_award :
  probability (set_of (λ (award_set : set Contestant), (award_set.contains C ∧ ¬ award_set.contains D) ∨ (award_set.contains D ∧ ¬ award_set.contains C) ∧ award_set.size = 2) : set (set Contestant))) = 2 / 3 := sorry

end prob_D_at_least_two_passing_grades_prob_only_one_C_D_receives_award_l638_638118


namespace least_of_consecutive_odds_l638_638000

theorem least_of_consecutive_odds (avg : ℤ) (n : ℕ) (h_avg : avg = 414) (h_n : n = 102) : 
  ∃ F : ℤ, (∀ i : ℕ, 0 ≤ i ∧ i < n → F + 2 * i ∈ {x : ℤ | x % 2 = 1}) ∧ 
           ∑ i in finset.range n, (F + 2 * i) = avg * n ∧ 
           F = 313 :=
by sorry

end least_of_consecutive_odds_l638_638000


namespace not_all_vertices_equal_l638_638905

def Vertex := ℕ
inductive VertexLabel : Type
| A | B | C | D | E | F | G | H

structure CubeGraph where
  vertices: VertexLabel → Vertex
  edges: List (VertexLabel × VertexLabel)

def initialCubeGraph : CubeGraph := 
{ vertices := λ v, match v with
                     | VertexLabel.A | VertexLabel.C => 1
                     | _ => 0,
  edges := [(VertexLabel.A, VertexLabel.B), (VertexLabel.A, VertexLabel.D), (VertexLabel.A, VertexLabel.E),
            (VertexLabel.B, VertexLabel.C), (VertexLabel.B, VertexLabel.G), (VertexLabel.B, VertexLabel.F),
            (VertexLabel.C, VertexLabel.D), (VertexLabel.C, VertexLabel.H), (VertexLabel.C, VertexLabel.F),
            (VertexLabel.D, VertexLabel.G), (VertexLabel.D, VertexLabel.H), (VertexLabel.E, VertexLabel.G),
            (VertexLabel.E, VertexLabel.H), (VertexLabel.F, VertexLabel.G)] }

def incrementEdgeValues (g : CubeGraph) (v1 v2 : VertexLabel) : CubeGraph :=
{ vertices := λ v, 
    if v = v1 || v = v2 then g.vertices v + 1 else g.vertices v,
  ..g }

def S1 (g : CubeGraph) : Vertex :=
  g.vertices VertexLabel.A + g.vertices VertexLabel.C +
  g.vertices VertexLabel.F + g.vertices VertexLabel.H  

def S2 (g : CubeGraph) : Vertex :=
  g.vertices VertexLabel.B + g.vertices VertexLabel.D +
  g.vertices VertexLabel.E + g.vertices VertexLabel.G

theorem not_all_vertices_equal (g : CubeGraph) :
  (S1 initialCubeGraph - S2 initialCubeGraph) = 2 →
  ¬ ∃ k : Vertex, 
      ∀ v, g.vertices v = k :=
sorry

end not_all_vertices_equal_l638_638905


namespace correct_sum_l638_638737

theorem correct_sum (a b c n : ℕ) (h_m_pos : 100 * a + 10 * b + c > 0) (h_n_pos : n > 0)
    (h_err_sum : 100 * a + 10 * c + b + n = 128) : 100 * a + 10 * b + c + n = 128 := 
by
  sorry

end correct_sum_l638_638737


namespace conjugate_quadrant_l638_638946

noncomputable def find_quadrant (z : ℂ) : String :=
  if 0 < z.re ∧ 0 < z.im then "first"
  else if 0 < z.re ∧ z.im < 0 then "fourth"
  else if z.re < 0 ∧ 0 < z.im then "second"
  else "third"

theorem conjugate_quadrant (z : ℂ) (h : z + Complex.abs z = 3 - Complex.I * Real.sqrt 3) :
  find_quadrant (Complex.conj z) = "first" :=
sorry

end conjugate_quadrant_l638_638946


namespace joanne_first_hour_coins_l638_638533

theorem joanne_first_hour_coins 
  (X : ℕ)
  (H1 : 70 = 35 + 35)
  (H2 : 120 = X + 70 + 35)
  (H3 : 35 = 50 - 15) : 
  X = 15 :=
sorry

end joanne_first_hour_coins_l638_638533


namespace polynomial_division_remainder_l638_638402

-- Define the polynomials and the division result
def f := (λ x : ℝ, x^4 + 2)
def g := (λ x : ℝ, x^2 + 7*x + 3)
def remainder := (λ x : ℝ, -301*x - 136)

-- State the theorem about the remainder
theorem polynomial_division_remainder :
  ∀ x : ℝ, exists q : ℝ → ℝ, f x = g x * q x + remainder x :=
by
  sorry

end polynomial_division_remainder_l638_638402


namespace Moe_in_seat_3_l638_638236

def seat : Type := Fin 5

def Nancy_seat : seat := ⟨3, sorry⟩  -- Seat index starts from 0, thus seat #4 is index 3.

def Moe_sits_next_to_Nancy (m : seat) : Prop := abs (m.1 - Nancy_seat.1) = 1

def Larry_not_between_Moe_and_Nancy (m : seat) (l : seat) : Prop :=
  ¬ (m = (Nancy_seat.1 - 1) ∧ l = Nancy_seat) ∧
  ¬ (m = (Nancy_seat.1 + 1) ∧ l = Nancy_seat)

-- Final theorem statement to prove that Moe is sitting in seat #3
theorem Moe_in_seat_3 (m : seat) (l : seat) (o : seat) (p : seat) :
  Moe_sits_next_to_Nancy m ∧
  Larry_not_between_Moe_and_Nancy m l ∧
  ¬(m = Nancy_seat) ∧
  ¬(l = Nancy_seat) ∧
  ¬(o = Nancy_seat) ∧
  ¬(p = Nancy_seat) ∧
  distinct [m, l, o, p, Nancy_seat] →
  m = ⟨2, sorry⟩ := -- Index starts from 0, thus seat #3 is index 2.
sorry

end Moe_in_seat_3_l638_638236


namespace angies_age_l638_638796

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l638_638796


namespace count_palindromic_times_l638_638630

theorem count_palindromic_times : 
  let three_digit_palindromes := 3 * 6, -- Times like 101, 202, etc.
      four_digit_palindromes := 
        let f0 := 10, -- Times like 0000 to 0909
            f1 := 10, -- Times like 1010 to 1919
            f2 := 4 in -- Times like 2020 to 2323
        f0 + f1 + f2 in
  three_digit_palindromes + four_digit_palindromes = 42 :=
by
  sorry

end count_palindromic_times_l638_638630


namespace problem1_problem2_problem3_problem4_l638_638067

-- Problem 1: This statement should prove the equality.
theorem problem1 : 
  sqrt 50 + (2 / (sqrt 2 + 1)) - 4 * sqrt (1 / 2) + 2 * (sqrt 2 - 1) ^ 0 = 5 * sqrt 2 :=
sorry

-- Problem 2: This statement should prove the roots of the equation.
theorem problem2 (x : ℝ) : 
  (x - 5) * (x + 2) = 8 ↔ (x = 6 ∨ x = -3) :=
sorry

-- Problem 3: This statement should prove the roots of the equation.
theorem problem3 (x : ℝ) : 
  (x + 3) * (x - 2) = 2 * x + 6 ↔ (x = 4 ∨ x = -3) :=
sorry

-- Problem 4: This statement should prove the roots of the quadratic equation.
theorem problem4 (x : ℝ) : 
  (3 / 2) * x ^ 2 + 4 * x - 1 = 0 ↔ 
  (x = (-4 + sqrt 22) / 3 ∨ x = (-4 - sqrt 22) / 3) :=
sorry

end problem1_problem2_problem3_problem4_l638_638067


namespace feasible_transportation_plan_exists_maximize_weight_II_products_l638_638092

-- Define the weights and conditions of the packages
structure Package :=
  (label : String)
  (weight_I : ℕ)
  (weight_II : ℕ)
  (total_weight : ℕ)

def A := Package.mk "A" 5 1 6
def B := Package.mk "B" 3 2 5
def C := Package.mk "C" 2 3 5
def D := Package.mk "D" 4 3 7
def E := Package.mk "E" 3 5 8

def packages : list Package := [A, B, C, D, E]

-- Define the conditions
def max_weight : ℝ := 19.5
def min_weight_I : ℕ := 9
def max_weight_I : ℕ := 11

/-
Proof Problem 1: Feasible Transportation Plan
Prove that there exists a combination of packages that meet the following conditions:
1. The weight of Type I products is between 9 and 11 tons.
2. The total weight is less than or equal to 19.5 tons.
3. The combination meets the feasible transportation plans ABC, ABE, AD, ACD, BCD.
-/

theorem feasible_transportation_plan_exists : 
  ∃ (combination : list Package), 
    min_weight_I ≤ combination.sum (·.weight_I) ∧ 
    combination.sum (·.weight_I) ≤ max_weight_I ∧ 
    (combination.sum (·.total_weight) : ℝ) ≤ max_weight ∧ 
    (combination.map (·.label) = ["A", "B", "C"] ∨ 
     combination.map (·.label) = ["A", "B", "E"] ∨ 
     combination.map (·.label) = ["A", "D"] ∨ 
     combination.map (·.label) = ["A", "C", "D"] ∨ 
     combination.map (·.label) = ["B", "C", "D"]) := 
sorry

/-
Proof Problem 2: Maximize Weight of Type II Products
Prove that there exists a combination of packages that meet the following conditions:
1. The weight of Type I products is between 9 and 11 tons.
2. The total weight is less than or equal to 19.5 tons.
3. The weight of Type II products is maximized, resulting in either ABE or BCD.
-/

theorem maximize_weight_II_products : 
  ∃ (combination : list Package), 
    min_weight_I ≤ combination.sum (·.weight_I) ∧ 
    combination.sum (·.weight_I) ≤ max_weight_I ∧ 
    (combination.sum (·.total_weight) : ℝ) ≤ max_weight ∧ 
    (combination.map (·.label) = ["A", "B", "E"] ∨ 
     combination.map (·.label) = ["B", "C", "D"]) ∧ 
    combination.sum (·.weight_II) = 8 := 
sorry

end feasible_transportation_plan_exists_maximize_weight_II_products_l638_638092


namespace matrix_vect_mult_l638_638685

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Vector (Fin 2) ℝ)

-- Define the conditions as given in the problem
def Mv : Vector (Fin 2) ℝ := ![4, -1]
def Mw : Vector (Fin 2) ℝ := ![-2, 3]

theorem matrix_vect_mult {M : Matrix (Fin 2) (Fin 2) ℝ} 
                         {v w : Vector (Fin 2) ℝ} 
                         (h1 : M.mulVec v = Mv) 
                         (h2 : M.mulVec w = Mw) :
  M.mulVec (3 • v - w) = ![14, -6] := 
by
   sorry

end matrix_vect_mult_l638_638685


namespace product_of_last_two_digits_l638_638358

theorem product_of_last_two_digits (A B : ℕ) (h₁ : A + B = 17) (h₂ : 4 ∣ (10 * A + B)) :
  A * B = 72 := sorry

end product_of_last_two_digits_l638_638358


namespace savings_account_amount_l638_638049

-- Definitions and conditions from the problem
def checking_account_yen : ℕ := 6359
def total_yen : ℕ := 9844

-- Question we aim to prove - the amount in the savings account
def savings_account_yen : ℕ := total_yen - checking_account_yen

-- Lean statement to prove the equality
theorem savings_account_amount : savings_account_yen = 3485 :=
by
  sorry

end savings_account_amount_l638_638049


namespace fibonacci_determinant_identity_l638_638612

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![Nat.fib (n + 1), Nat.fib n], ![Nat.fib n, Nat.fib (n - 1)]]

theorem fibonacci_determinant_identity :
  ∀ n : ℕ, Matrix.det (fibonacci_matrix n) = (-1 : ℤ)^n :=
by sorry

example : Nat.fib 100 * Nat.fib 102 - Nat.fib 101 ^ 2 = -1 := 
by {
  have f_identity := fibonacci_determinant_identity 101,
  simp [fibonacci_matrix, Matrix.det] at f_identity,
  exact f_identity,
}

end fibonacci_determinant_identity_l638_638612


namespace santana_brothers_birthday_l638_638289

theorem santana_brothers_birthday (b : ℕ) (oct : ℕ) (nov : ℕ) (dec : ℕ) (c_presents_diff : ℕ) :
  b = 7 → oct = 1 → nov = 1 → dec = 2 → c_presents_diff = 8 → (∃ M : ℕ, M = 3) :=
by
  sorry

end santana_brothers_birthday_l638_638289


namespace sequence_no_consecutive_010101_l638_638217

theorem sequence_no_consecutive_010101 :
  ∀ (x : ℕ → ℕ),
    x 1 = 1 → x 2 = 0 → x 3 = 1 → x 4 = 0 → x 5 = 1 → x 6 = 0 →
    (∀ n, x (n + 6) = (x n + x (n + 1) + x (n + 2) + x (n + 3) + x (n + 4) + x (n + 5)) % 10) →
    ¬ ∃ n, x n = 0 ∧ x (n + 1) = 1 ∧ x (n + 2) = 0 ∧ x (n + 3) = 1 ∧ x (n + 4) = 0 ∧ x (n + 5) = 1 := 
begin 
  sorry 
end

end sequence_no_consecutive_010101_l638_638217


namespace find_dividend_l638_638365

theorem find_dividend (D Q R dividend : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) (h4 : dividend = D * Q + R) :
  dividend = 5336 :=
by
  -- We will complete the proof using the provided conditions
  sorry

end find_dividend_l638_638365


namespace probability_x_gt_8y_l638_638725

theorem probability_x_gt_8y :
  (let rect_area := 2016 * 2016;
       tri_area := 1/2 * 2016 * 252 in
   (tri_area / rect_area) = 127 / 2032) :=
by sorry

end probability_x_gt_8y_l638_638725


namespace paper_cutting_l638_638791

theorem paper_cutting (n : ℕ) (h : n ≥ 60) : ∃ x y : ℕ, 2 * x + 3 * y = 15 ∧ ∀ m : ℕ, m ≥ 60 → (∃ i : ℕ, i ≥ 1 ∧ (m + i) ∈ {m + 7, m + 11}) :=
by
  sorry

end paper_cutting_l638_638791


namespace math_problem_l638_638257

theorem math_problem
  (n : ℕ) (d : ℕ)
  (h1 : d ≤ 9)
  (h2 : 3 * n^2 + 2 * n + d = 263)
  (h3 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) :
  n + d = 11 := 
sorry

end math_problem_l638_638257


namespace sum_difference_first_2500_even_odd_l638_638353

open Nat

/-- 
  The difference between the sum of the first 2500 even counting numbers
  each increased by 5 and the sum of the first 2500 odd counting numbers is 15000.
-/
theorem sum_difference_first_2500_even_odd :
  let even_plus_five_sum := (2500 / 2) * (7 + 5005)
  let odd_sum := (2500 / 2) * (1 + 4999)
  even_plus_five_sum - odd_sum = 15000 := by
sory

end sum_difference_first_2500_even_odd_l638_638353


namespace Q3_volume_is_313_l638_638557

theorem Q3_volume_is_313 (Q : ℕ → ℚ) 
  (hQ0 : Q 0 = 1) 
  (hQ_next : ∀ i, Q (i + 1) = Q i + (6 * (1 / (4 * 24^i)))) : 
  ∃ p q : ℕ, p + q = 313 ∧ Q 3 = p / q ∧ nat.coprime p q :=
begin
  sorry
end

end Q3_volume_is_313_l638_638557


namespace An_integer_and_parity_l638_638368

theorem An_integer_and_parity (k : Nat) (h : k > 0) : 
  ∀ n ≥ 1, ∃ A : Nat, 
   (A = 1 ∨ (∀ A' : Nat, A' = ( (A * n + 2 * (n+1) ^ (2 * k)) / (n+2)))) 
  ∧ (A % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 2) := 
by 
  sorry

end An_integer_and_parity_l638_638368


namespace sin_of_angle_in_fourth_quadrant_l638_638151

theorem sin_of_angle_in_fourth_quadrant 
  (α : Real) 
  (h₁ : 3 * Real.pi / 2 < α)
  (h₂ : α < 2 * Real.pi)
  (h₃ : Real.tan (Real.pi - α) = 5 / 12) : 
  Real.sin(α) = -5 / 13 := 
sorry

end sin_of_angle_in_fourth_quadrant_l638_638151


namespace puzzle_solution_l638_638618

theorem puzzle_solution (a b c : ℕ) (p q r : ℕ) (x y z : ℕ) (result_w : ℕ) :
  ((5 * 3 = 15) ∧ (5 * 2 = 10) ∧ ((5 + 3 + 2) * 2 = 40) ∧ (5 + 3 + 2 = 151022)) →
  ((9 * 2 = 18) ∧ (9 * 4 = 36) ∧ ((9 + 2 + 4) * 2 = 60) ∧ (9 + 2 + 4 = 183652)) →
  (a = 7) ∧ (b = 2) ∧ (c = 5) →
  (a * b = 14) ∧ (a * c = 35) ∧ ((a + b + c) * 2 = 56) →
  (result_w = 143556) :=
begin
  sorry
end

end puzzle_solution_l638_638618


namespace range_of_function_l638_638166

noncomputable def y (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_function : set.range (y) = set.Icc 1 17 := 
by sorry

end range_of_function_l638_638166


namespace tangent_circle_intersects_l638_638173

variables {α : Type*} [MetricSpace α]

-- Define the circles and various points as described in the problem
variables (O₁ O₂ : α) (Q P X A B : α)
variables (Ω₁ Ω₂ : Set α) -- Ω₁ and Ω₂ are the sets of points on circles O₁ and O₂, respectively

-- Define the conditions given in the problem
def intersects_at_two_points (Ω₁ Ω₂ : Set α) (Q : α) : Prop :=
  Q ∈ Ω₁ ∧ Q ∈ Ω₂ ∧ ∃ R ≠ Q, R ∈ Ω₁ ∧ R ∈ Ω₂

def point_on_circle (P : α) (Ω : Set α) : Prop := P ∈ Ω

def point_inside_circle (P : α) (Ω : Set α) : Prop := 
  MetricSpace.dist P (MetricSpace.center Ω) < MetricSpace.radius Ω

def line_intersects_circle_at_point (line : Set α) (Ω : Set α) (X : α) : Prop :=
  ∃ d : ℝ, line = {p | ∃ t : ℝ, p = P + t • (Q - P)} ∧ X ∈ Ω ∧ X ≠ Q ∧ ∃ R, R ∈ line ∧ R ∈ Ω ∧ R ≠ X

def tangent_at_point (Ω : Set α) (X : α) (line : Set α) : Prop :=
  ∃ t : ℝ, line = {p | ∃ t : ℝ, p = X + t • (n (X - MetricSpace.center Ω))} ∧ MetricSpace.dist X (MetricSpace.center Ω) = MetricSpace.radius Ω

def line_parallel_to (line₁ line₂ : Set α) (P : α) : Prop :=
  ∃ d : ℝ, line₁ = {p | ∃ t : ℝ, p = P + t • d} ∧ line₂ = {p | ∃ k : ℝ, p = A + k • (B - A)} ∧ (B - A) = d

def circle_passing_through_and_tangent (A B : α) (l : Set α) : Set α :=
  {p | ∃ c : α, MetricSpace.dist c A = MetricSpace.dist c B ∧ ∃ R : ℝ, circle (c, R) ∧ MetricSpace.dist p l = R}

-- Prove the final condition
theorem tangent_circle_intersects (O : α) (Ω₁ : Set α) (A B : α) (l : Set α) (tangent_to_l : tangent_circle (A, B, l)) :
  tangent_to_l (A, B, l) → tangent_at_point O₁ (A ∩ B) Ω₁ :=
sorry

end tangent_circle_intersects_l638_638173


namespace glass_volume_l638_638458

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638458


namespace Zelda_success_prob_l638_638824

variable {X Y Z : Prop}

-- Given conditions
def P_X : ℝ := 1 / 5
def P_Y : ℝ := 1 / 2
def P_XY_not_Z : ℝ := 0.0375

-- Proof problem
theorem Zelda_success_prob : ∃ P_Z : ℝ, P_X * P_Y * (1 - P_Z) = P_XY_not_Z ∧ P_Z = 0.625 :=
by
  use 0.625
  split
  sorry -- Proving part (condition 1)
  rfl -- Proving part (condition 2)

end Zelda_success_prob_l638_638824


namespace shaded_fraction_correct_l638_638476

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ) in
  let r := (1/16 : ℚ) in
  a / (1 - r)

theorem shaded_fraction_correct :
  shaded_fraction = 4 / 15 :=
by
  sorry

end shaded_fraction_correct_l638_638476


namespace tom_to_luke_ratio_l638_638793

theorem tom_to_luke_ratio (Tom Luke Anthony : ℕ) 
  (hAnthony : Anthony = 44) 
  (hTom : Tom = 33) 
  (hLuke : Luke = Anthony / 4) : 
  Tom / Nat.gcd Tom Luke = 3 ∧ Luke / Nat.gcd Tom Luke = 1 := 
by
  sorry

end tom_to_luke_ratio_l638_638793


namespace path_count_A_to_B_l638_638994

-- Define the labeled points.
inductive Point
| A | B | C | D | E | F | G | H

-- Define the possible segments as pairs of points.
inductive Segment
| AC | AD | AE | BD | BE | CD | CF | DA | DC | DE | DF | DG | EG | EH | FC | FE | GB | HF | HF | HG

-- Define a path from A to B as a list of segments that does not revisit points.
def Path (p : Point) (q : Point) : Type :=
  { l : List Segment // l.head = Segment.pq ∧ l.last = Segment.pq ∧ l.NodupSegments }

-- Define the problem: there are exactly 10 such paths.
theorem path_count_A_to_B : ∃ l : List (Path Point.A Point.B), l.length = 10 := by
  sorry

end path_count_A_to_B_l638_638994


namespace find_g_function_l638_638315

noncomputable def g (x : ℝ) : ℝ := 5^x - 3^x

theorem find_g_function : 
  (∀ x y : ℝ, g(x + y) = 5^y * g x + 3^x * g y) ∧ 
  g 1 = 2 → g = fun x => 5^x - 3^x :=
by
  sorry

end find_g_function_l638_638315


namespace sum_of_digits_of_T_l638_638068

def is_seven_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ∈ {1,2,3,4,5,6,7,8,9} ∧ b ∈ {0,1,2,3,4,5,6,7,8,9} ∧
  c ∈ {0,1,2,3,4,5,6,7,8,9} ∧ d ∈ {0,1,2,3,4,5,6,7,8,9} ∧ n = 1000001 * a + 100010 * b + 10010 * c + 1000 * d

def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ {1,2,3,4,5,6,7,8,9} ∧ b ∈ {0,1,2,3,4,5,6,7,8,9} ∧
  c ∈ {0,1,2,3,4,5,6,7,8,9} ∧ n = 10001 * a + 1010 * b + 100 * c

def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ {1,2,3,4,5,6,7,8,9} ∧ b ∈ {0,1,2,3,4,5,6,7,8,9} ∧ n = 1001 * a + 110 * b

def is_three_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ {1,2,3,4,5,6,7,8,9} ∧ b ∈ {0,1,2,3,4,5,6,7,8,9} ∧ n = 101 * a + 10 * b

def T : ℕ :=
  (finset.range 9000).sum (λ i, 1000001 * 5 + 100010 * 5 + 10010 * 5 + 1000 * 5) +
  (finset.range 900).sum (λ i, 10001 * 5 + 1010 * 5 + 100 * 5) +
  (finset.range 90).sum (λ i, 1001 * 5 + 110 * 5) +
  (finset.range 90).sum (λ i, 101 * 5 + 10 * 5)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.foldr (λ d s, d + s) 0

theorem sum_of_digits_of_T : sum_of_digits T = 31 := sorry

end sum_of_digits_of_T_l638_638068


namespace sufficient_but_not_necessary_condition_l638_638395

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end sufficient_but_not_necessary_condition_l638_638395


namespace number_of_students_attending_exactly_two_clubs_l638_638416

noncomputable def num_students : ℕ := 800
noncomputable def frac_arithmetic : ℚ := 7 / 12
noncomputable def frac_biology : ℚ := 11 / 24
noncomputable def frac_chemistry : ℚ := 13 / 16
noncomputable def frac_all_three : ℚ := 3 / 32

noncomputable def num_arithmetic : ℕ := (frac_arithmetic * num_students).to_nat
noncomputable def num_biology : ℕ := (frac_biology * num_students).to_nat
noncomputable def num_chemistry : ℕ := (frac_chemistry * num_students).to_nat
noncomputable def num_all_three : ℕ := (frac_all_three * num_students).to_nat

theorem number_of_students_attending_exactly_two_clubs :
  let X := num_arithmetic + num_biology + num_chemistry - num_students - 2 * num_all_three in
  X = 534 := by
  let X := num_arithmetic + num_biology + num_chemistry - num_students - 2 * num_all_three
  have h : X = 534 := sorry
  exact h

end number_of_students_attending_exactly_two_clubs_l638_638416


namespace distinct_intersection_points_l638_638076

theorem distinct_intersection_points : 
  ∃! (x y : ℝ), (x + 2*y = 6 ∧ x - 3*y = 2) ∨ (x + 2*y = 6 ∧ 4*x + y = 14) :=
by
  -- proof would be here
  sorry

end distinct_intersection_points_l638_638076


namespace simple_interest_time_l638_638931

-- Definitions based on given conditions
def SI : ℝ := 640           -- Simple interest
def P : ℝ := 4000           -- Principal
def R : ℝ := 8              -- Rate
def T : ℝ := 2              -- Time in years (correct answer to be proved)

-- Theorem statement
theorem simple_interest_time :
  SI = (P * R * T) / 100 := 
by 
  sorry

end simple_interest_time_l638_638931


namespace find_ff10_l638_638983
-- Importing the required math library

-- Defining the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1
  else log 10 x

-- Stating the theorem to prove that f(f(10)) = 2
theorem find_ff10 : f (f 10) = 2 := by
  sorry

end find_ff10_l638_638983


namespace brad_running_speed_l638_638269

variable (dist_between_homes : ℕ)
variable (maxwell_speed : ℕ)
variable (time_maxwell_walks : ℕ)
variable (maxwell_start_time : ℕ)
variable (brad_start_time : ℕ)

#check dist_between_homes = 94
#check maxwell_speed = 4
#check time_maxwell_walks = 10
#check brad_start_time = maxwell_start_time + 1

theorem brad_running_speed (dist_between_homes : ℕ) (maxwell_speed : ℕ) (time_maxwell_walks : ℕ) (maxwell_start_time : ℕ) (brad_start_time : ℕ) :
  dist_between_homes = 94 →
  maxwell_speed = 4 →
  time_maxwell_walks = 10 →
  brad_start_time = maxwell_start_time + 1 →
  (dist_between_homes - maxwell_speed * time_maxwell_walks) / (time_maxwell_walks - (brad_start_time - maxwell_start_time)) = 6 :=
by
  intros
  sorry

end brad_running_speed_l638_638269


namespace tank_capacity_percentage_l638_638831

-- Definitions based on conditions
def height_A : ℝ := 10
def circumference_A : ℝ := 8
def height_B : ℝ := 8
def circumference_B : ℝ := 10
def pi : ℝ := Real.pi

-- Prove that the capacity of Tank A is 80% of the capacity of Tank B
theorem tank_capacity_percentage : 
  let radius_A := circumference_A / (2 * pi)
  let radius_B := circumference_B / (2 * pi)
  let volume_A := pi * (radius_A ^ 2) * height_A
  let volume_B := pi * (radius_B ^ 2) * height_B
  (volume_A / volume_B) * 100 = 80 := 
by
  let radius_A := circumference_A / (2 * pi)
  let radius_B := circumference_B / (2 * pi)
  let volume_A := pi * (radius_A ^ 2) * height_A
  let volume_B := pi * (radius_B ^ 2) * height_B
  have h_radius_A : radius_A = circumference_A / (2 * pi) := rfl
  have h_radius_B : radius_B = circumference_B / (2 * pi) := rfl
  have h_volume_A : volume_A = pi * (radius_A ^ 2) * height_A := rfl
  have h_volume_B : volume_B = pi * (radius_B ^ 2) * height_B := rfl
  have h_ratio : (volume_A / volume_B) * 100 = 80 :=
    by sorry
  exact h_ratio

end tank_capacity_percentage_l638_638831


namespace magic_ink_combinations_l638_638488

def herbs : ℕ := 4
def essences : ℕ := 6
def incompatible_herbs : ℕ := 3

theorem magic_ink_combinations :
  herbs * essences - incompatible_herbs = 21 := 
  by
  sorry

end magic_ink_combinations_l638_638488


namespace xia_sheets_left_l638_638489

def stickers_left (initial : ℕ) (shared : ℕ) (per_sheet : ℕ) : ℕ :=
  (initial - shared) / per_sheet

theorem xia_sheets_left :
  stickers_left 150 100 10 = 5 :=
by
  sorry

end xia_sheets_left_l638_638489


namespace find_m_plus_n_l638_638837

theorem find_m_plus_n:
  let GY := 27 / 2 in
  ∃ (m n : ℕ), GY = m / n ∧ Nat.coprime m n ∧ m + n = 29 := by
sorry -- The proof is omitted per the instructions

end find_m_plus_n_l638_638837


namespace area_of_triangle_ABC_l638_638795

noncomputable def radius_of_inscribed_circle (area : ℝ) : ℝ := sqrt(area / π)

theorem area_of_triangle_ABC :
  ∃ (x : ℝ), ∀ (BC AC : ℝ) (r : ℝ), 
    AC = 2 * BC ∧
    π * r^2 = 4 * π ∧
    r = radius_of_inscribed_circle (4 * π) ∧
    r = (BC + AC - sqrt(BC^2 + AC^2)) / 2 ∧
    BC = x ∧
    AC = 2 * x ∧
    x = 2 * (sqrt(5) - 3) →
    (1/2) * BC * AC = 56 - 24 * sqrt(5) :=
by
  sorry

end area_of_triangle_ABC_l638_638795


namespace glass_volume_is_230_l638_638448

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638448


namespace square_area_l638_638097

theorem square_area (side : ℕ) (h : side = 19) : side * side = 361 := by
  sorry

end square_area_l638_638097


namespace correct_propositions_l638_638582

-- Define the objects in the space.
variables (a b l : Line) (α β γ : Plane)

-- Define propositions to prove.

-- Proposition 1
def prop1 (h1 : a ⊥ α) (h2 : b ⊥ β) (h3 : l ⊥ γ) (h4 : a // b ∧ b // l) : Prop :=
  α // β ∧ β // γ

-- Proposition 2
def prop2 (h1 : α ⊥ γ) (h2 : β ⊥ γ) (h3 : α ∩ β = l) : Prop :=
  l ⊥ γ

-- Proposition 4
def prop4 (h1 : a ∦ b) (h2 : a ⊥ α) (h3 : b ⊥ β) (h4 : l ⊥ a) (h5 : l ⊥ b)
          (h6 : ¬l ⊂ α) (h7 : ¬l ⊂ β) : Prop :=
  ∃ m, (α ∩ β = m) ∧ m // l

-- The main theorem combining all propositions.
theorem correct_propositions (h1 : a ⊥ α) (h2 : b ⊥ β) (h3 : l ⊥ γ)
                             (h4 : a // b ∧ b // l) (h5 : α ⊥ γ)
                             (h6 : β ⊥ γ) (h7 : α ∩ β = l)
                             (h8 : a ∦ b) (h9 : l ⊥ a) (h10 : l ⊥ b)
                             (h11 : ¬l ⊂ α) (h12 : ¬l ⊂ β) :
  prop1 a b l α β γ h1 h2 h3 h4 ∧ prop2 α β γ l h5 h6 h7 ∧ prop4 a b l α β γ h8 h9 h10 h11 h12 :=
sorry

end correct_propositions_l638_638582


namespace domain_of_logarithmic_function_l638_638521

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, 2 - x > 0 ↔ x < 2 := 
by
  intro x
  sorry

end domain_of_logarithmic_function_l638_638521


namespace division_with_remainder_l638_638822

theorem division_with_remainder (d q r n : ℕ) : 
  d = 6 → q = 8 → n = d * q + r → r < d → (r = 5 ∧ n = 53) :=
by
  intros hd hq hn hr
  rw [hd, hq] at hn hr
  have : r < 6 := hr
  have : n = 6 * 8 + r := hn
  sorry

end division_with_remainder_l638_638822


namespace product_of_all_possible_values_l638_638520

section
variables {b x : ℝ}
def g (x : ℝ) (b : ℝ) : ℝ := b / (3 * x - 4)
def g_inv (y : ℝ) : ℝ := sorry -- definition of inverse function

theorem product_of_all_possible_values :
  (g 3 b = g_inv (b + 2) → b = 28 / 9) := by
  sorry
end

end product_of_all_possible_values_l638_638520


namespace cyclic_quadrilateral_cyclic_quad_l638_638683

noncomputable def is_cyclic (s : set P) := ∃ (C: circle P), ∀ p ∈ s, p ∈ C 

variables 
  {P : Type*} 
  [MetricSpace P] 
  [NormedAddTorsor ℝ P] -- Typically used for Euclidean geometry.

variables 
  (A B C D E F T M K N : P) 
  (omega : set P)

-- Given definitions and conditions.
def cyclic_quadrilateral := is_cyclic {B, C, E, D}

def intersect (l1 l2 : set P) (P : P) := (P ∈ l1 ∧ P ∈ l2 ∧ ∃! P' ∈ l1 ∩ l2, P' = P)

def midpoint (K : P) (B C : P) := dist B K = dist K C ∧ 2 * dist B K = dist B C

def reflection (M A N : P) := dist M A = dist M N ∧ (∃ l, M ∈ l ∧ A ∈ l ∧ N ∈ l)

theorem cyclic_quadrilateral_cyclic_quad
  (hBCED_cyclic : cyclic_quadrilateral)
  (h_inter_C_B_E_D_A : intersect (ray C B) (ray E D) A)
  (h_line_D_parallel_BC : ∃ l, D ∈ l ∧ l ∥ (line B C))
  (h_inter_line_D_omega_F : F ≠ D ∧ intersect (l) (omega) F)
  (h_inter_AF_omega_T : T ≠ F ∧ intersect (segment A F) (omega) T)
  (h_inter_ET_BC_M : intersect (line E T) (line B C) M)
  (h_midpoint_K_BC : midpoint K B C)
  (h_reflection_A_M_N : reflection M A N) :
  is_cyclic {D, N, K, E} :=
sorry

end cyclic_quadrilateral_cyclic_quad_l638_638683


namespace angie_age_l638_638804

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l638_638804


namespace find_circle_center_l638_638537

theorem find_circle_center (x y : ℝ) : 
    x^2 + 4 * x + y^2 - 6 * y = 24 → 
    (∃ x₀ y₀ : ℝ, x₀ = -2 ∧ y₀ = 3) :=
by
  intro h
  use (-2, 3)
  split
  · rfl
  · rfl
  -- Requires proof steps which we skip here
  sorry

end find_circle_center_l638_638537


namespace contractor_absent_days_l638_638851

theorem contractor_absent_days (x y : ℕ) (h1 : x + y = 30) (h2 : 25 * x - 7.5 * y = 360) : y = 12 :=
sorry

end contractor_absent_days_l638_638851


namespace minimum_value_l638_638975

theorem minimum_value (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + 1 / b = 2) :
  ∃ c, (c = 4) ∧ (∀ a b, 0 < a → 0 < b → a + 1 / b = 2 → ∃ min, min = (2 / a) + (2 * b) ∧ min = 4) :=
begin
  sorry
end

end minimum_value_l638_638975


namespace sum_of_x_coordinates_of_other_vertices_l638_638997

theorem sum_of_x_coordinates_of_other_vertices {x1 y1 x2 y2 x3 y3 x4 y4: ℝ} 
    (h1 : (x1, y1) = (2, 12))
    (h2 : (x2, y2) = (8, 3))
    (midpoint_eq : (x1 + x2) / 2 = (x3 + x4) / 2) 
    : x3 + x4 = 10 := 
by
  have h4 : (2 + 8) / 2 = 5 := by norm_num
  have h5 : 2 * 5 = 10 := by norm_num
  sorry

end sum_of_x_coordinates_of_other_vertices_l638_638997


namespace translation_preserves_geometry_l638_638490

-- Let's define the proof problem
theorem translation_preserves_geometry (seg1 seg2 : LineSegment) (ang1 ang2 : Angle) :
  (translated seg1) = seg2 ∧ (translated ang1) = ang2 
  → 
  ((parallel seg1 seg2 ∨ collinear seg1 seg2) ∧ length seg1 = length seg2 ∧ angle_measure ang1 = angle_measure ang2) := 
  sorry

end translation_preserves_geometry_l638_638490


namespace new_salary_statistics_l638_638028

variable {ι : Type} [Fintype ι] [DecidableEq ι]
variable (salaries : ι → ℝ) (mean : ℝ) (variance : ℝ)
variable (increase : ℝ := 100)
variable (n : ℕ := 10) [Fact (Fintype.card ι = n)]

-- Definitions for the problem conditions
def average_salaries (salaries : ι → ℝ) : ℝ :=
  (∑ i, salaries i) / (Fintype.card ι)

def variance_salaries (salaries : ι → ℝ) : ℝ :=
  (∑ i, (salaries i - average_salaries salaries) ^ 2) / ((Fintype.card ι) - 1)

-- Given the conditions
def condition_average (h_avg : average_salaries salaries = mean) : Prop := h_avg
def condition_variance (h_var : variance_salaries salaries = variance) : Prop := h_var

-- Proof statement
theorem new_salary_statistics 
  (h_avg : average_salaries salaries = mean)
  (h_var : variance_salaries salaries = variance) :
  average_salaries (λi, salaries i + increase) = mean + increase ∧
  variance_salaries (λi, salaries i + increase) = variance :=
by sorry

end new_salary_statistics_l638_638028


namespace average_paper_tape_length_l638_638037

-- Define the lengths of the paper tapes as given in the conditions
def red_tape_length : ℝ := 20
def purple_tape_length : ℝ := 16

-- State the proof problem
theorem average_paper_tape_length : 
  (red_tape_length + purple_tape_length) / 2 = 18 := 
by
  sorry

end average_paper_tape_length_l638_638037


namespace triangle_transformation_l638_638885

-- Define initial coordinates of the triangle vertices
def A₀ := (1, 2)
def B₀ := (4, 2)
def C₀ := (1, 5)

-- Define transformation functions
def rotate90Clockwise (p : Int × Int) : Int × Int :=
  (p.snd, -p.fst)

def reflectOverX (p : Int × Int) : Int × Int :=
  (p.fst, -p.snd)

def translateUp (p : Int × Int) (d : Int) : Int × Int :=
  (p.fst, p.snd + d)

def rotate180 (p : Int × Int) : Int × Int :=
  (-p.fst, -p.snd)

theorem triangle_transformation :
  let A₁ := rotate90Clockwise A₀;
  let B₁ := rotate90Clockwise B₀;
  let C₁ := rotate90Clockwise C₀;
  let A₂ := reflectOverX A₁;
  let B₂ := reflectOverX B₁;
  let C₂ := reflectOverX C₁;
  let A₃ := translateUp A₂ 3;
  let B₃ := translateUp B₂ 3;
  let C₃ := translateUp C₂ 3;
  let A₄ := rotate180 A₃;
  let B₄ := rotate180 B₃;
  let C₄ := rotate180 C₃;
  A₄ = (-2, -4) ∧ B₄ = (-2, -7) ∧ C₄ = (-5, -4) :=
by {
  sorry
}

end triangle_transformation_l638_638885


namespace minimum_sum_of_numbers_l638_638749

theorem minimum_sum_of_numbers 
  (a : Fin 10 → ℕ)
  (h_distinct : Function.Injective a)
  (h_even_product : ∀ S : Finset (Fin 10), S.card = 5 → (∃ x ∈ S, a x % 2 = 0))
  (h_sum_odd : ∑ i, a i % 2 = 1) :
  ∑ i, a i = 65 := 
sorry

end minimum_sum_of_numbers_l638_638749


namespace worker_distance_when_heard_explosion_l638_638046

-- Define the conditions
def timer_seconds : ℕ := 40
def worker_speed_yd_per_sec : ℕ := 5
def speed_of_sound_ft_per_sec : ℕ := 1100

-- Define the conversion factor
def yards_to_feet (y : ℕ) : ℕ := 3 * y

-- Prove that the worker runs approximately 203 yards when he hears the explosion
theorem worker_distance_when_heard_explosion :
  let t := 40.55 in
  let distance_in_feet := 15 * t in
  let distance_in_yards := distance_in_feet / 3 in
  distance_in_yards ≈ 203 :=
by
  sorry

end worker_distance_when_heard_explosion_l638_638046


namespace diana_shopping_for_newborns_l638_638082

-- Define the number of children, toddlers, and the toddler-to-teenager ratio
def total_children : ℕ := 40
def toddlers : ℕ := 6
def toddler_to_teenager_ratio : ℕ := 5

-- The result we need to prove
def number_of_newborns : ℕ := 
  total_children - (toddlers + toddler_to_teenager_ratio * toddlers) = 4

-- Define the proof statement
theorem diana_shopping_for_newborns : 
  total_children = 40 ∧ toddlers = 6 ∧ toddler_to_teenager_ratio = 5 → 
  number_of_newborns := 4 :=
by sorry

end diana_shopping_for_newborns_l638_638082


namespace equal_segment_ratios_l638_638277

structure Triangle (P : Type) :=
(point_A : P) (point_B : P) (point_C : P)

variables {P : Type} [EuclideanGeometry P]

def segment_length (p q : P) : ℝ := dist p q

theorem equal_segment_ratios 
  (A B C A1 A2 B1 B2 C1 C2 : P) 
  (h1 : Collinear A B C) 
  (h2 : segment_length A1 B2 = segment_length B1 C2) 
  (h3 : segment_length B1 C2 = segment_length C1 A2)
  (h4 : segment_length C1 A2 = segment_length A1 B2)
  (h5 : ∀ (p q r : P), angle p q r = 60) :
  -- sides equal due to 'equal lengths' and 'angle 60' implies equilateral similarity
  ((segment_length A1 A2 / segment_length B C) = 
   (segment_length B1 B2 / segment_length A C) ∧
   (segment_length B1 B2 / segment_length A C) = 
   (segment_length C1 C2 / segment_length A B)) :=
sorry

end equal_segment_ratios_l638_638277


namespace glass_volume_is_230_l638_638453

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638453


namespace find_a_l638_638125

noncomputable def a_value_given_conditions : ℝ :=
  let A := 30 * Real.pi / 180
  let C := 105 * Real.pi / 180
  let B := 180 * Real.pi / 180 - A - C
  let b := 8
  let a := (b * Real.sin A) / Real.sin B
  a

theorem find_a :
  a_value_given_conditions = 4 * Real.sqrt 2 :=
by
  -- We assume that the value computation as specified is correct
  -- hence this is just stating the problem.
  sorry

end find_a_l638_638125


namespace cube_mono_increasing_l638_638996

theorem cube_mono_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

end cube_mono_increasing_l638_638996


namespace star_7_5_eq_11_div_4_l638_638616

-- Defining the operation
def star (a b : ℕ) : ℝ := (a * b - (a - b)) / (a + b)

-- The theorem we want to prove
theorem star_7_5_eq_11_div_4 : star 7 5 = 11 / 4 := 
by
  sorry

end star_7_5_eq_11_div_4_l638_638616


namespace angle_BDC_is_15_degrees_l638_638298

-- Define the problem statement according to the conditions.
theorem angle_BDC_is_15_degrees
  (ABC ACD : Triangle)
  (AB AC AD : ℝ)
  (BAC : ℝ)
  (h_congruent : congruent ABC ACD)
  (h_equal_sides1 : AB = AC)
  (h_equal_sides2 : AC = AD)
  (h_angle_BAC : BAC = 30) :
  ∃ BDC : ℝ, BDC = 15 := 
sorry

end angle_BDC_is_15_degrees_l638_638298


namespace sum_of_areas_is_471_l638_638473

noncomputable def sum_of_unique_right_triangles_areas : ℕ :=
  let unique_triangles := 
    { (a, b) | 
      a > 0 ∧ b > 0 ∧ 
      (a * b) = 6 * (a + b) ∧ 
      ∃ d, (a-6) * (b-6) = 36 
    } in
  ∑ (a, b) in unique_triangles, a * b / 2

theorem sum_of_areas_is_471 : sum_of_unique_right_triangles_areas = 471 := by
  sorry

end sum_of_areas_is_471_l638_638473


namespace lily_remaining_milk_l638_638267

def initial_milk : ℚ := (11 / 2)
def given_away : ℚ := (17 / 4)
def remaining_milk : ℚ := initial_milk - given_away

theorem lily_remaining_milk : remaining_milk = 5 / 4 :=
by
  -- Here, we would provide the proof steps, but we can use sorry to skip it.
  exact sorry

end lily_remaining_milk_l638_638267


namespace sequence_inequality_l638_638678

noncomputable def a : Nat → ℚ
| 0     => 1 / 2
| (n+1) => - a n + 1 / (2 - a n)

theorem sequence_inequality (n : ℕ) :
  (n / (2 * (∑ i in Finset.range n, a i)) - 1) ^ n ≤ 
  ((∑ i in Finset.range n, a i) / n) ^ n * 
  ∏ i in Finset.range n, (1 / a i - 1) :=
by
  sorry

end sequence_inequality_l638_638678


namespace seq_arithmetic_find_a_n_l638_638950

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
variable (h1 : a 1 = 1)
variable (h2 : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0)
variable (h3 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i)

-- Question 1: Prove that the sequence {1 / S n} is an arithmetic sequence.
theorem seq_arithmetic (n : ℕ) (hn : n ≥ 1) : (1 / S n) - (1 / S (n - 1)) = 2 :=
by 
  sorry

-- Question 2: Find a_n.
theorem find_a_n (n : ℕ) : a n = 
  if n = 1 then 1 else -2 / ((2 * n - 1) * (2 * n - 3)) :=
by 
  sorry

end seq_arithmetic_find_a_n_l638_638950


namespace square_root_of_a_minus_b_l638_638586

theorem square_root_of_a_minus_b :
  (∃ (a b : ℝ), (a + 3) + (2 * a - 6) = 0 ∧ b = -8 ∧ sqrt (a - b) = 3 ∨ sqrt (a - b) = -3) :=
by
  sorry

end square_root_of_a_minus_b_l638_638586


namespace Raj_wins_with_probability_l638_638636

def RajBoxes : Type := ℕ 
def box1 (n : RajBoxes) := n = 1  -- Box 1 has two red balls
def box2 (n : RajBoxes) := n = 2  -- Box 2 has one red ball and one blue ball
def box3 (n : RajBoxes) := n = 3  -- Box 3 has two blue balls

def prior (n : RajBoxes) : ℚ := 1 / 3 -- Prior probabilities are equal for any box

def P_red (n : RajBoxes) (b : n = 1 ∨ n = 2 ∨ n = 3) : ℚ :=
  if box1 n then 1
  else if box2 n then 1 / 2
  else if box3 n then 0
  else 0

def posterior (n : RajBoxes) (red : ℚ) (b : n = 1 ∨ n = 2 ∨ n = 3) : ℚ :=
  -- Using Bayes' theorem
  if box1 n then (1 * prior n) / red
  else if box2 n then ((1 / 2) * prior n) / red
  else if box3 n then (0 * prior n) / red
  else 0

def winningProbability (posterior1 : ℚ) (posterior2 : ℚ) : ℚ :=
  -- Probability of winning by predicting two balls of the same color and drawing from the same box
  posterior1 * 1 + posterior2 * (1 / 2)

theorem Raj_wins_with_probability (optimal_play : winningProbability (posterior 1 (1 / 2) sorry) (posterior 2 (1 / 2) sorry)) :
  optimal_play = 5 / 6 :=
by
sorry

end Raj_wins_with_probability_l638_638636


namespace glass_volume_230_l638_638441

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638441


namespace hypotenuse_length_l638_638345

-- Definitions based on conditions
variables (A B C X Y : Type) [metric_space X Y]
variable (dist : X → Y → ℝ)
variable (AX XB AY YC BY CX : ℝ)

-- Given conditions
def is_right_triangle (A B C : Type) [metric_space A B C] :=
  ∃ (dist : A → B → ℝ) (h : dist A B ≠ 0 ∧ dist A C ≠ 0 ∧ dist B C ≠ 0),
  dist B C ^ 2 = dist A B ^ 2 + dist A C ^ 2 -- Pythagorean Theorem

axiom ratio_AX_XB : AX / XB = 1 / 3
axiom ratio_AY_YC : AY / YC = 1 / 3
axiom length_BY : BY = 12
axiom length_CX : CX = 18

-- Define the length of the hypotenuse BC
noncomputable def BC_length : ℝ :=
  let x := (4 / 3) * CX in
  let y := 4 * BY in
  real.sqrt (x^2 + y^2)

-- Proof problem
theorem hypotenuse_length : BC_length AX XB AY YC BY CX = 24 * real.sqrt 5 :=
sorry  -- Proof not included

end hypotenuse_length_l638_638345


namespace units_digit_of_result_is_eight_l638_638768

def three_digit_number_reverse_subtract (a b c : ℕ) : ℕ :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  original - reversed

theorem units_digit_of_result_is_eight (a b c : ℕ) (h : a = c + 2) :
  (three_digit_number_reverse_subtract a b c) % 10 = 8 :=
by
  sorry

end units_digit_of_result_is_eight_l638_638768


namespace constant_term_in_expansion_l638_638840

theorem constant_term_in_expansion :
  let f := (3 * x + 2) * (2 * x - 3),
      expanded_f := 6 * x^2 - 5 * x - 6
  in (3 * (2 * 1) + 2 * (-3)) = -6 :=
by sorry

end constant_term_in_expansion_l638_638840


namespace diana_shopping_for_newborns_l638_638081

-- Define the number of children, toddlers, and the toddler-to-teenager ratio
def total_children : ℕ := 40
def toddlers : ℕ := 6
def toddler_to_teenager_ratio : ℕ := 5

-- The result we need to prove
def number_of_newborns : ℕ := 
  total_children - (toddlers + toddler_to_teenager_ratio * toddlers) = 4

-- Define the proof statement
theorem diana_shopping_for_newborns : 
  total_children = 40 ∧ toddlers = 6 ∧ toddler_to_teenager_ratio = 5 → 
  number_of_newborns := 4 :=
by sorry

end diana_shopping_for_newborns_l638_638081


namespace P_2007_greater_P_2008_l638_638876

noncomputable def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2008) * ∑ k in finset.range 2008, P (n - k)

theorem P_2007_greater_P_2008 : P 2007 > P 2008 := 
sorry

end P_2007_greater_P_2008_l638_638876


namespace simplify_expression_correct_l638_638742

variable (a b x y : ℝ) (i : ℂ)

noncomputable def simplify_expression (a b x y : ℝ) (i : ℂ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (i^2 = -1) → (a * x + b * i * y) * (a * x - b * i * y) = a^2 * x^2 + b^2 * y^2

theorem simplify_expression_correct (a b x y : ℝ) (i : ℂ) :
  simplify_expression a b x y i := by
  sorry

end simplify_expression_correct_l638_638742


namespace december_28_is_saturday_l638_638527

def days_per_week := 7

def thanksgiving_day : Nat := 28

def november_length : Nat := 30

def december_28_day_of_week : Nat :=
  (thanksgiving_day % days_per_week + november_length + 28 - thanksgiving_day) % days_per_week

theorem december_28_is_saturday :
  (december_28_day_of_week = 6) :=
by
  sorry

end december_28_is_saturday_l638_638527


namespace probability_three_digit_even_l638_638843

namespace ProbabilityProblem

def digits := {2, 3, 4, 5, 6}

def is_even (n : ℕ) : Prop := n % 2 = 0

def three_digit_numbers := {
  (a, b, c) | a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c
}

def even_three_digit_numbers := {
  (a, b, c) | (a, b, c) ∈ three_digit_numbers ∧ is_even c
}

theorem probability_three_digit_even :
  (even_three_digit_numbers.card : ℚ) / (three_digit_numbers.card : ℚ) = 3 / 5 :=
sorry

end ProbabilityProblem

end probability_three_digit_even_l638_638843


namespace all_positive_integers_appear_as_a_i_l638_638702

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+1) := if h : ∃ m, ∀ k < n + 1, m ≠ a k ∧ nat.gcd m (a n) > 1 
           then classical.some h 
           else 0 -- we assume that gcd(n, n+1) > 1 is always true for n >= 3

theorem all_positive_integers_appear_as_a_i (n : ℕ) : ∃ i, a i = n :=
sorry

end all_positive_integers_appear_as_a_i_l638_638702


namespace hillary_descending_rate_l638_638993

def baseCampDistance : ℕ := 4700
def hillaryClimbingRate : ℕ := 800
def eddyClimbingRate : ℕ := 500
def hillaryStopShort : ℕ := 700
def departTime : ℕ := 6 -- time is represented in hours from midnight
def passTime : ℕ := 12 -- time is represented in hours from midnight

theorem hillary_descending_rate :
  ∃ r : ℕ, r = 1000 := by
  sorry

end hillary_descending_rate_l638_638993


namespace solve_system_eqns_l638_638293

theorem solve_system_eqns 
  {a b c : ℝ} (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
  {x y z : ℝ} 
  (h4 : a^3 + a^2 * x + a * y + z = 0)
  (h5 : b^3 + b^2 * x + b * y + z = 0)
  (h6 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + bc + ca ∧ z = -abc :=
by {
  sorry
}

end solve_system_eqns_l638_638293


namespace problem_solution_l638_638998

open Nat

theorem problem_solution (n : ℕ) (h1 : (∀ x : ℕ, (x+1)^n = x^n + ∑ i in range n, (choose n i) * (x ^ i))):
  (choose n 3 = 669 * (choose n 2)) → n = 2009 :=
by
  sorry

end problem_solution_l638_638998


namespace angie_age_l638_638799

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l638_638799


namespace min_cubes_to_fill_box_l638_638418

theorem min_cubes_to_fill_box : 
  let length := 7
  let width := 18
  let height := 3
  let V_box := length * width * height
  let V_cube := 8 -- side length has to be rounded down to be practical: side = 2 cm, V_cube = side³
  let min_cubes := (V_box + V_cube - 1) / V_cube -- to round up
  V_box = 378 → V_cube = 8 → min_cubes = 48 :=
by 
  intros length width height
  let V_box := length * width * height
  let V_cube := 8
  have V_box_calc : V_box = 378 := by sorry
  have V_cube_calc : V_cube = 8 := by sorry
  have min_cubes_calc : min_cubes = 48 := by sorry
  exact min_cubes_calc

end min_cubes_to_fill_box_l638_638418


namespace glass_volume_is_230_l638_638451

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638451


namespace original_selling_price_l638_638024

def cost_price (SP_gain: ℝ) (gain: ℝ) : ℝ :=
  SP_gain / (1 + gain / 100)

def original_SP (CP: ℝ) (loss: ℝ) : ℝ :=
  CP * (1 - loss / 100)

theorem original_selling_price (SP_gain: ℝ) (gain: ℝ) (loss: ℝ) (original_SP_RS: ℝ) : 
  SP_gain = 1100 → gain = 10 → loss = 20 → original_SP_RS = 800 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  have h4: cost_price 1100 10 = 1000 := by sorry
  have h5: original_SP 1000 20 = 800 := by sorry
  rw [← h4, ← h5]
  exact rfl

end original_selling_price_l638_638024


namespace acute_triangle_cannot_divide_into_two_obtuse_l638_638220

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

theorem acute_triangle_cannot_divide_into_two_obtuse (A B C A1 B1 C1 A2 B2 C2 : ℝ) 
  (h_acute : is_acute_triangle A B C) 
  (h_divide : A + B + C = 180 ∧ A1 + B1 + C1 = 180 ∧ A2 + B2 + C2 = 180)
  (h_sum : A1 + A2 = A ∧ B1 + B2 = B ∧ C1 + C2 = C) :
  ¬ (is_obtuse_triangle A1 B1 C1 ∧ is_obtuse_triangle A2 B2 C2) :=
sorry

end acute_triangle_cannot_divide_into_two_obtuse_l638_638220


namespace jackson_pays_2100_l638_638667

def tile_cost (length : ℝ) (width : ℝ) (tiles_per_sqft : ℝ) (percent_green : ℝ) (cost_green : ℝ) (cost_red : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * percent_green
  let red_tiles := total_tiles - green_tiles
  let cost_green_total := green_tiles * cost_green
  let cost_red_total := red_tiles * cost_red
  cost_green_total + cost_red_total

theorem jackson_pays_2100 :
  tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by
  sorry

end jackson_pays_2100_l638_638667


namespace median_of_data_set_l638_638136

-- Defining the given data set
def data_set : List ℕ := [12, 16, 20, 23, 20, 15, 23]

-- Define what it means to be the median
def is_median (lst : List ℕ) (median : ℕ) : Prop :=
  let sorted_lst := lst.sort (<=)
  let n := sorted_lst.length
  2 * n / 2 < n ∧ sorted_lst[n / 2] = median

-- Now, we state the theorem
theorem median_of_data_set : is_median data_set 20 :=
by
  sorry

end median_of_data_set_l638_638136


namespace train_crossing_time_l638_638018

theorem train_crossing_time (length_of_train : ℝ) (speed_kmh : ℝ) :
  length_of_train = 180 →
  speed_kmh = 72 →
  (180 / (72 * (1000 / 3600))) = 9 :=
by 
  intros h1 h2
  sorry

end train_crossing_time_l638_638018


namespace seq_div_l638_638777

noncomputable theory

open Nat

def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 1
  | k+2   => 5 * seq (k+1) - seq k - 1

theorem seq_div (n : ℕ) : seq n ∣ (seq (n + 1))^2 + (seq (n + 1)) + 1 := by
  sorry

end seq_div_l638_638777


namespace average_carnations_l638_638344

theorem average_carnations (c1 c2 c3 n : ℕ) (h1 : c1 = 9) (h2 : c2 = 14) (h3 : c3 = 13) (h4 : n = 3) :
  (c1 + c2 + c3) / n = 12 :=
by
  sorry

end average_carnations_l638_638344


namespace prove_triangle_conditions_l638_638962

-- Definitions and conditions
variables {a b c : ℝ} {A B C : ℝ}

-- Condition for angle A
def angle_condition : Prop := a * cos C + (c - 2 * b) * cos A = 0

-- Triangle definition
def is_triangle (A B C : ℝ) : Prop := 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π

-- Definition of correct angle A
def correct_angle_A (A : ℝ) : Prop := A = π / 3

-- Definition of maximum area given a = 2
def max_area_given_a (a b c : ℝ) (A : ℝ) : Prop := a = 2 → by exact (1 / 2) * b * c * sin A ≤ sqrt 3

-- Theorem to prove
theorem prove_triangle_conditions (h1 : is_triangle A B C) (h2 : angle_condition) :
  correct_angle_A A ∧ ∀ b c, max_area_given_a a b c A := 
by sorry

end prove_triangle_conditions_l638_638962


namespace binary_to_decimal_110011_l638_638072

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ bit pos, bit * (2 ^ pos)) (List.range b.length) |>.sum

def target_binary : List ℕ := [1, 1, 0, 0, 1, 1]
def expected_decimal : ℕ := 51

theorem binary_to_decimal_110011 :
  binary_to_decimal target_binary = expected_decimal :=
by
  sorry

end binary_to_decimal_110011_l638_638072


namespace acute_triangle_angle_side_l638_638014

theorem acute_triangle_angle_side 
    (a b C : ℝ) 
    (hab : (x^2 - 2 * real.sqrt 3 * x + 2) = 0) 
    (hA_B : 2 * real.sin (A + B) - real.sqrt 3 = 0)
    (a_plus_b : a + b = 2 * real.sqrt 3)
    (ab_product : a * b = 2)
    (angle_A : A = C) 
    (angle_B : B = C) : 
    (C = real.pi / 3) ∧ (c = real.sqrt 6) :=
begin
    sorry,
end

end acute_triangle_angle_side_l638_638014


namespace sum_first_ten_multiples_of_nine_l638_638818

theorem sum_first_ten_multiples_of_nine :
  let a := 9
  let d := 9
  let n := 10
  let S_n := n * (2 * a + (n - 1) * d) / 2
  S_n = 495 := 
by
  sorry

end sum_first_ten_multiples_of_nine_l638_638818


namespace angle_C_is_108_l638_638285

theorem angle_C_is_108
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : B < C)
  (h3 : C < D)
  (h4 : D < E)
  (h5 : B - A = C - B)
  (h6 : C - B = D - C)
  (h7 : D - C = E - D)
  (angle_sum : A + B + C + D + E = 540) :
  C = 108 := 
sorry

end angle_C_is_108_l638_638285


namespace foxes_hunt_duration_l638_638204

variable (initial_weasels : ℕ) (initial_rabbits : ℕ) (remaining_rodents : ℕ)
variable (foxes : ℕ) (weasels_per_week : ℕ) (rabbits_per_week : ℕ)

def total_rodents_per_week (weasels_per_week rabbits_per_week foxes : ℕ) : ℕ :=
  foxes * (weasels_per_week + rabbits_per_week)

def initial_rodents (initial_weasels initial_rabbits : ℕ) : ℕ :=
  initial_weasels + initial_rabbits

def total_rodents_caught (initial_rodents remaining_rodents : ℕ) : ℕ :=
  initial_rodents - remaining_rodents

def weeks_hunted (total_rodents_caught total_rodents_per_week : ℕ) : ℕ :=
  total_rodents_caught / total_rodents_per_week

theorem foxes_hunt_duration
  (initial_weasels := 100) (initial_rabbits := 50) (remaining_rodents := 96)
  (foxes := 3) (weasels_per_week := 4) (rabbits_per_week := 2) :
  weeks_hunted (total_rodents_caught (initial_rodents initial_weasels initial_rabbits) remaining_rodents) 
                 (total_rodents_per_week weasels_per_week rabbits_per_week foxes) = 3 :=
by
  sorry

end foxes_hunt_duration_l638_638204


namespace train_speed_l638_638417

theorem train_speed (jogger_speed_kmh : ℕ) (initial_lead_m : ℕ) (train_length_m : ℕ) (time_to_pass_s : ℕ) :
  jogger_speed_kmh = 9 →
  initial_lead_m = 200 →
  train_length_m = 200 →
  time_to_pass_s = 40 →
  let jogger_speed_ms := jogger_speed_kmh * 1000 / 3600,
      distance_covered_by_jogger := jogger_speed_ms * time_to_pass_s,
      total_distance_to_cover := initial_lead_m + train_length_m + distance_covered_by_jogger,
      train_speed_ms := total_distance_to_cover / time_to_pass_s
  in train_speed_ms * 3600 / 1000 = 45 :=
by
  intros h1 h2 h3 h4
  let jogger_speed_ms := jogger_speed_kmh * 1000 / 3600
  let distance_covered_by_jogger := jogger_speed_ms * time_to_pass_s
  let total_distance_to_cover := initial_lead_m + train_length_m + distance_covered_by_jogger
  let train_speed_ms := total_distance_to_cover / time_to_pass_s
  sorry

end train_speed_l638_638417


namespace roses_ratio_l638_638115

noncomputable def last_year_roses : ℕ := 12
noncomputable def this_year_roses : ℕ := last_year_roses / 2
noncomputable def rose_price : ℕ := 3
noncomputable def total_spent : ℕ := 54
noncomputable def bouquet_roses : ℕ := total_spent / rose_price

theorem roses_ratio {last_year_roses this_year_roses bouquet_roses : ℕ} :
  last_year_roses = 12 →
  this_year_roses = last_year_roses / 2 →
  total_spent = 54 →
  rose_price = 3 →
  bouquet_roses = total_spent / rose_price →
  bouquet_roses / last_year_roses = 3 / 2 :=
by
  intros h1 h2 h3 h4 h5
  have : last_year_roses = 12 := h1
  have : this_year_roses = 6 := h2
  have : total_spent = 54 := h3
  have : rose_price = 3 := h4
  have : bouquet_roses = 18 := h5
  sorry

end roses_ratio_l638_638115


namespace Masha_initial_ball_count_l638_638715

theorem Masha_initial_ball_count (r w n p : ℕ) (h1 : r + n * w = 101) (h2 : p * r + w = 103) (hn : n ≠ 0) :
  r + w = 51 ∨ r + w = 68 :=
  sorry

end Masha_initial_ball_count_l638_638715


namespace parallelogram_of_equilateral_l638_638369

noncomputable theory

-- Define the type for points in the plane
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Defining a Quadrilateral
structure Quadrilateral :=
(A B C D : Point)
(convex : true)  -- Just a placeholder for convex quadrilateral condition

-- Placeholder for the equilateral triangles construction condition
def equilateral_condition (quad : Quadrilateral) : Prop :=
  true  -- Detailed geometric condition is placeholder

-- Placeholder for the equilateral triangles being inside/outside
def external_internal_condition (quad : Quadrilateral) : Prop :=
  true  -- Detailed geometric condition is placeholder

-- Define our main theorem for the problem
theorem parallelogram_of_equilateral (
  quad : Quadrilateral,
  h₁ : equilateral_condition quad,
  h₂ : external_internal_condition quad
) : true :=  -- Placeholder for quadrilateral M1 M2 M3 M4 is a parallelogram
sorry

end parallelogram_of_equilateral_l638_638369


namespace milk_production_days_l638_638187

theorem milk_production_days (y : ℕ) (h1 : y > 0):
  (y+4) * y * (y+1) * (y+6) = (y+6) * (y+1) * y * (y+4) * (y+2) / (y * y+1) :=
begin
  sorry
end

end milk_production_days_l638_638187


namespace quadratic_roots_identity_l638_638934

theorem quadratic_roots_identity :
  (∀ x : ℝ, x^2 + 2 * x - 8 = 0 → 
    ∃ x1 x2 : ℝ, x^2 + 2 * x - 8 = 0 ∧ x = x1 ∨ x = x2) →
  (∃ x1 x2 : ℝ, (x1 + x2 = -2) ∧ (x1 * x2 = -8) ∧
    (x2 / x1 + x1 / x2 = -5 / 2)) :=
by {
  assume h,
  obtain ⟨x1, x2, h1, h2, h3⟩ := h,
  sorry
}

end quadratic_roots_identity_l638_638934


namespace productivity_difference_l638_638382

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l638_638382


namespace stratified_sampling_l638_638879

theorem stratified_sampling
  (students_class1 : ℕ)
  (students_class2 : ℕ)
  (formation_slots : ℕ)
  (total_students : ℕ)
  (prob_selected: ℚ)
  (selected_class1 : ℕ)
  (selected_class2 : ℕ)
  (h1 : students_class1 = 54)
  (h2 : students_class2 = 42)
  (h3 : formation_slots = 16)
  (h4 : total_students = students_class1 + students_class2)
  (h5 : prob_selected = formation_slots / total_students)
  (h6 : selected_class1 = students_class1 * prob_selected)
  (h7 : selected_class2 = students_class2 * prob_selected)
  : selected_class1 = 9 ∧ selected_class2 = 7 := by
  sorry

end stratified_sampling_l638_638879


namespace setup_time_correct_l638_638284

theorem setup_time_correct :
  ∃ S : ℝ, (S + 4 + 1 + 6 = 12) ∧ S = 1 := 
begin
  use 1,
  split,
  { linarith, },
  { refl, }
end

end setup_time_correct_l638_638284


namespace isosceles_triangle_angle_solution_vertex_angle_l638_638779

theorem isosceles_triangle_angle_solution (x : ℝ) (hx : x > 0 ∧ x < 90) :
  (∀ a b c : ℝ, (a = tan x ∧ b = tan x ∧ c = tan (5 * x)) →
                (4 * x = degree_to_radian (vertex_angle x)) →
                (isosceles_triangle a b c → x = 20)) :=
by
  sorry

theorem vertex_angle (x : ℝ) : ℝ :=
degree_to_radian (4 * x)

def isosceles_triangle (a b c : ℝ) : Prop :=
(a = b) ∨ (b = c) ∨ (c = a)

def degree_to_radian (d : ℝ) : ℝ :=
d * real.pi / 180

end isosceles_triangle_angle_solution_vertex_angle_l638_638779


namespace heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l638_638827

namespace PolygonColoring

/-- Define a regular n-gon and its coloring -/
def regular_ngon (n : ℕ) : Type := sorry

def isosceles_triangle {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

def same_color {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

/-- Part (a) statement -/
theorem heptagon_isosceles_triangle_same_color : 
  ∀ (p : regular_ngon 7), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (b) statement -/
theorem octagon_no_isosceles_triangle_same_color :
  ∃ (p : regular_ngon 8), ¬∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (c) statement -/
theorem general_ngon_isosceles_triangle_same_color :
  ∀ (n : ℕ), (n = 5 ∨ n = 7 ∨ n ≥ 9) → 
  ∀ (p : regular_ngon n), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

end PolygonColoring

end heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l638_638827


namespace truncated_cone_sphere_radius_l638_638484

theorem truncated_cone_sphere_radius :
  ∀ (r1 r2 h : ℝ), 
  r1 = 24 → 
  r2 = 6 → 
  h = 20 → 
  ∃ r, 
  r = 17 * Real.sqrt 2 / 2 := by
  intros r1 r2 h hr1 hr2 hh
  sorry

end truncated_cone_sphere_radius_l638_638484


namespace sum_of_coefficients_factorization_l638_638535

theorem sum_of_coefficients_factorization (x y : ℤ): 
  let expr := 8 * x ^ 8 - 243 * y ^ 8 in
  let f := (2 * x ^ 4 - 3 * y ^ 4) * (2 * x ^ 4 + 3 * y ^ 4) in
  (∀(x y : ℤ), (sum_of_integers (coefficients f)) = 4) :=
by
  sorry

end sum_of_coefficients_factorization_l638_638535


namespace magnitude_of_T_l638_638245

-- Defining the complex number expressions
def complex_term1 := (1 : ℂ) + (1 : ℂ) * complex.i
def complex_term2 := (1 : ℂ) - (1 : ℂ) * complex.i

-- Defining T based on the problem statement
def T := (complex_term1) ^ 19 + (complex_term1) ^ 19 - (complex_term2) ^ 19

-- Statement of the theorem to be proven
theorem magnitude_of_T : |T| = 2 ^ (9.5) * real.sqrt 5 := by
  sorry

end magnitude_of_T_l638_638245


namespace election_result_l638_638203

def totalVotes : ℕ := 12000
def geoffPercentage : ℕ := 1
def additionalVotes : ℕ := 5000

theorem election_result (geoff_votes : ℕ) (needed_votes : ℕ) (winning_percentage : ℕ) :
    geoff_votes = (geoffPercentage * totalVotes) / 100 →
    needed_votes = geoff_votes + additionalVotes →
    winning_percentage = (needed_votes * 100) / totalVotes →
    winning_percentage = 42 :=
begin
  sorry
end

end election_result_l638_638203


namespace first_digit_of_1998_digit_number_l638_638132

theorem first_digit_of_1998_digit_number :
  ∃ (num : ℕ), 
  num < 10^1998 ∧
  last_digit num = 1 ∧
  (∀ i : ℕ, i < 1997 → ((num / 10^i) % 100) % 17 = 0 ∨ ((num / 10^i) % 100) % 23 = 0) →
  first_digit num = 9 :=
by
  sorry

end first_digit_of_1998_digit_number_l638_638132


namespace infinitly_many_n_f_n_divides_n_l638_638677

-- Definition of sigma, the sum of divisors of natural number x.
def σ (x : ℕ) : ℕ := x.divisors.sum

-- Definition of f, the number of natural numbers m ≤ n where σ(m) is odd.
def f (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ m, σ m % 2 = 1).card

-- The main theorem: there are infinitely many natural numbers n such that f(n) | n.
theorem infinitly_many_n_f_n_divides_n :
  ∃ᶠ n in at_top, f(n) ∣ n :=
sorry

end infinitly_many_n_f_n_divides_n_l638_638677


namespace integral_value_l638_638624

-- Define the constant term condition
def constant_term_condition (a : ℝ) : Prop :=
  ∀ r : ℕ, r = 3 → -binomial (6 : ℕ) r * a^3 = -160

-- The main statement to prove the integral value
theorem integral_value (a : ℝ)
  (h : constant_term_condition a) : 
  ∫ x in 0..a, (3 * x^2 - 1) = 6 :=
by
  sorry

end integral_value_l638_638624


namespace sum_fractions_geq_n_over_2n_minus_1_l638_638261

theorem sum_fractions_geq_n_over_2n_minus_1
  (n : ℕ) (x : Fin n → ℝ)
  (hx_pos : ∀ i, 0 < x i)
  (hx_sum : (Finset.univ.sum x) = 1) :
  (Finset.univ.sum (λ i, x i / (2 - x i))) ≥ n / (2 * n - 1) := 
sorry

end sum_fractions_geq_n_over_2n_minus_1_l638_638261


namespace glass_volume_230_l638_638436

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638436


namespace tetrahedron_volume_correct_l638_638649

noncomputable def tetrahedron_volume (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABD_ABC : ℝ) : ℝ :=
  let h_ABD := (2 * area_ABD) / AB
  let h := h_ABD * Real.sin angle_ABD_ABC
  (1 / 3) * area_ABC * h

theorem tetrahedron_volume_correct:
  tetrahedron_volume 3 15 12 (Real.pi / 6) = 20 :=
by
  sorry

end tetrahedron_volume_correct_l638_638649


namespace possible_initial_positions_l638_638404

-- Definitions based on the conditions
def valid_jump (k n : ℕ) : Prop :=
  k - n = 8 ∨ k - n = 5 ∨ k - n = -5 ∨ k - n = -8

def grasshopper_problem : Prop :=
  ∀ (path : List ℕ), (∀ i ∈ path, 1 ≤ i ∧ i ≤ 12) ∧ (∀ i j, i ≠ j → i ∈ path → j ∈ path → valid_jump i j) →
  path.length = 12 → (path.head = some 5 ∨ path.head = some 8)

theorem possible_initial_positions :
  grasshopper_problem :=
sorry

end possible_initial_positions_l638_638404


namespace number_of_ordered_quadruples_l638_638541

-- Nonnegative real numbers
def NNReal := {x : ℝ // 0 ≤ x}

-- The conditions given in the problem
def condition1 (a b c d : NNReal) : Prop :=
  a.val^2 + b.val^2 + c.val^2 + d.val^2 = 9

def condition2 (a b c d : NNReal) : Prop :=
  (a.val + b.val + c.val + d.val) * (a.val^3 + b.val^3 + c.val^3 + d.val^3) = 3 * (a.val^2 + b.val^2 + c.val^2 + d.val^2)^2

-- Proving the number of solutions is 15
theorem number_of_ordered_quadruples : 
  {p : NNReal × NNReal × NNReal × NNReal // condition1 p.1 p.2.1 p.2.2.1 p.2.2.2 ∧ condition2 p.1 p.2.1 p.2.2.1 p.2.2.2}.finite ∧
  {p : NNReal × NNReal × NNReal × NNReal // condition1 p.1 p.2.1 p.2.2.1 p.2.2.2 ∧ condition2 p.1 p.2.1 p.2.2.1 p.2.2.2}.card = 15 :=
sorry

end number_of_ordered_quadruples_l638_638541


namespace average_age_of_John_Mary_Tonya_is_35_l638_638230

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l638_638230


namespace problem_l638_638949

def a_seq : Nat → Nat
| 1 => 11
| (n + 2) => 
  let a_n := a_seq (n + 1) in
  if a_n % 2 = 1 then 3 * a_n + 5
  else let k := Nat.find (λ k, (a_n / 2^k) % 2 = 1) in a_n / 2^k

theorem problem :
  a_seq 100 = 62 ∧
  (∃ m : ℕ, ∀ n : ℕ, n > m → (a_seq n = 1 ∨ a_seq n = 5)) :=
begin
  sorry,
end

end problem_l638_638949


namespace dot_product_eq_half_l638_638180

noncomputable def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2
  
theorem dot_product_eq_half :
  vector_dot_product (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
                     (Real.cos (85 * Real.pi / 180), Real.cos (5 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end dot_product_eq_half_l638_638180


namespace house_orderings_l638_638808

-- Define the houses as elements
inductive House : Type
| Blue
| Yellow
| Green
| Red
| Orange

open House

-- Predicate functions for conditions
def b_before_y (lst : List House) := lst.indexOf Blue < lst.indexOf Yellow
def g_not_next_to_y (lst : List House) := (lst.indexOf Green ≠ lst.indexOf Yellow - 1 ∧ lst.indexOf Green ≠ lst.indexOf Yellow + 1)
def r_before_o (lst : List House) := lst.indexOf Red < lst.indexOf Orange
def b_not_next_to_g (lst : List House) := (lst.indexOf Blue ≠ lst.indexOf Green - 1 ∧ lst.indexOf Blue ≠ lst.indexOf Green + 1)

-- The main statement
theorem house_orderings : 
  Finset.card {
        lst : List House | lst ~ List.enumFrom 0 5 ∧ 
        b_before_y lst ∧ 
        g_not_next_to_y lst ∧ 
        r_before_o lst ∧ 
        b_not_next_to_g lst 
    } = 6 :=
by sorry

end house_orderings_l638_638808


namespace magnitude_of_z_l638_638555

-- Given:
-- z is a complex number such that z * i^2023 = 1 + i
-- We To prove:
-- |z| = sqrt 2

theorem magnitude_of_z (z : ℂ) (h : z * (complex.I ^ 2023) = 1 + complex.I) : complex.abs z = real.sqrt 2 := by
  sorry

end magnitude_of_z_l638_638555


namespace no_integer_roots_l638_638839

theorem no_integer_roots 
  (a b c : ℤ) (ha : a ≠ 0) (h0 : f 0 = c ∧ c % 2 = 1) (h1 : f 1 = a + b + c ∧ (a + b + c) % 2 = 1) :
  ¬ ∃ t : ℤ, a * t^2 + b * t + c = 0 := 
sorry

end no_integer_roots_l638_638839


namespace sum_of_c_n_l638_638580

noncomputable def a_n (n : ℕ) : ℝ := 12 * (-1/2)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := log2(3 / a_n (2*n + 3))
noncomputable def c_n (n : ℕ) : ℝ := 4 / (b_n n * b_n (n + 1))
noncomputable def T_n (n : ℕ) : ℝ := ∑ k in Finset.range n, c_n (k + 1)

theorem sum_of_c_n (n : ℕ) : T_n n = n / (n + 1) :=
by
  sorry

end sum_of_c_n_l638_638580


namespace glass_volume_l638_638430

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638430


namespace find_a_equals_two_l638_638620

noncomputable def a := ((7 + 4 * Real.sqrt 3) ^ (1 / 2) - (7 - 4 * Real.sqrt 3) ^ (1 / 2)) / Real.sqrt 3

theorem find_a_equals_two : a = 2 := 
sorry

end find_a_equals_two_l638_638620


namespace correct_operation_D_l638_638823

variable {a b x m n : ℝ}

theorem correct_operation_D :
  (m + n) * (-m + n) = -m^2 + n^2 :=
by calc
  (m + n) * (-m + n) = m * (-m) + m * n + n * (-m) + n * n : by simp [mul_add, add_mul]
                   ... = -m^2 + mn - mn + n^2             : by ring
                   ... = -m^2 + n^2                       : by ring

end correct_operation_D_l638_638823


namespace number_of_people_knowing_secret_by_following_monday_l638_638717

theorem number_of_people_knowing_secret_by_following_monday :
  let monday := 1 in
  let tuesday := monday + 1 * 2 in
  let wednesday := tuesday + 2 * 2 in
  let thursday := wednesday + 4 * 1 in
  let friday := thursday + 4 * 2 in
  let saturday := friday + 8 * 2 in
  let sunday := saturday + 16 * 2 in
  let following_monday := sunday + 32 * 2 in
  following_monday = 132 :=
by
  sorry

end number_of_people_knowing_secret_by_following_monday_l638_638717


namespace deepak_age_l638_638327

theorem deepak_age : ∀ (R D : ℕ), (R / D = 4 / 3) ∧ (R + 6 = 18) → D = 9 :=
by
  sorry

end deepak_age_l638_638327


namespace John_final_push_time_l638_638830

theorem John_final_push_time :
  ∃ t : ℝ, John_behind = 14 ∧ John_ahead = 2 ∧ John_speed = 4.2 ∧ Steve_speed = 3.7 ∧ 
  (John_behind + John_ahead) / John_speed = t ∧ t = 80 / 21 :=
by
  let John_behind := 14
  let John_ahead := 2
  let John_speed := 4.2
  let Steve_speed := 3.7
  let t := (John_behind + John_ahead) / John_speed
  use t
  have t_eq : t = 80 / 21 := by
    calc
      t = (John_behind + John_ahead) / John_speed := rfl
      ... = 16 / 4.2 := by norm_num [John_behind, John_ahead, John_speed]
      ... = 80 / 21 := by norm_num
  exact ⟨rfl, rfl, rfl, rfl, rfl, t_eq⟩

end John_final_push_time_l638_638830


namespace exists_polynomial_l638_638238

noncomputable theory
open_locale big_operators

variable (X0 : ℕ)
variable (xi : ℕ × ℕ → ℕ)
variable (epsilon : ℕ → ℕ)
variable (l : ℕ)
variable (X : ℕ → ℕ)
variable (M : ℕ → ℕ)

axiom indep_vars : ∀ i j k : ℕ, (X0, xi (i, j), epsilon k) are_independent
axiom xi_same_dist : ∀ i j : ℕ, xi (i, j) same_dist_as xi (1, 1)
axiom epsilon_same_dist : ∀ k : ℕ, epsilon k same_dist_as epsilon 1
axiom expec_xi : ∀ i j : ℕ, (E (xi (1, 1)) = 1)
axiom expec_X0_lt_infty : ∀ i : ℕ, (E (X0^l) < ∞)
axiom expec_xi_lt_infty : ∀ i j : ℕ, (E (xi (1, 1)^l) < ∞)
axiom expec_epsilon_lt_infty : ∀ k : ℕ, (E (epsilon 1^l) < ∞)

def X : ℕ → ℕ
| 0     := X0
| (n+1) := epsilon n + ∑ j in finset.range (X n), xi (n, j)

def M : ℕ → ℕ
| n     := X n - X (n - 1) - (E (epsilon n))

theorem exists_polynomial {l : ℕ} : 
  ∃ P : polynomial ℕ, degree P ≤ l/2 ∧ ∀ n ∈ ℕ, E((M n)^l) = P.eval n :=
sorry

end exists_polynomial_l638_638238


namespace range_f_for_x_ge_2_l638_638766

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x - 1)

theorem range_f_for_x_ge_2 :
  set.range (λ x, f x) (Ici 2) = set.Ioc 2 7 :=
sorry

end range_f_for_x_ge_2_l638_638766


namespace banana_price_reduction_l638_638849

theorem banana_price_reduction :
  ∃ P : ℝ, P > 0 ∧ (40.00001 / 3 - 40.00001 / P = 64 / 12) →
  (100 * (P - 3) / P) = 40 :=
begin
  sorry
end

end banana_price_reduction_l638_638849


namespace smallest_difference_possible_l638_638587

theorem smallest_difference_possible (digits : Finset ℕ)
  (h_digits : digits = {0, 3, 4, 7, 8}) :
  (∀ a b : ℕ, 
    (∃ x y z u v : ℕ, 
      digits = {x, y, z, u, v} ∧ 
      x < y ∧ y < z ∧ 
      u ≠ 0 ∧ a = 100 * x + 10 * y + z ∧ 
      b = 10 * u + v ∧ 
      (digits.erase x).erase y = {z, u, v} )
    → (a - b ≥ 339)) :=
begin
  sorry
end

end smallest_difference_possible_l638_638587


namespace geometric_sequence_solution_l638_638655

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, a n = a1 * r ^ (n - 1)

theorem geometric_sequence_solution :
  ∀ (a : ℕ → ℝ),
    (geometric_sequence a) →
    (∃ a2 a18, a2 + a18 = -6 ∧ a2 * a18 = 4 ∧ a 2 = a2 ∧ a 18 = a18) →
    a 4 * a 16 + a 10 = 6 :=
by
  sorry

end geometric_sequence_solution_l638_638655


namespace lagrange_mean_value_ex_l638_638201

theorem lagrange_mean_value_ex (a b : ℝ) (h1 : a < b) :
  ∃ ξ ∈ set.Ioo a b, 
    ∀ g : ℝ → ℝ, 
    (∀ x ∈ set.Icc a b, continuous_at g x) → 
    (∀ x ∈ set.Ioo a b, differentiable_at ℝ g x) →
    g b - g a = (deriv g ξ) * (b - a) → 
    g(1) - g(0) = (deriv g ξ) * (1 - 0) → g(0) = 1 → g(1) = e :=
begin
  use (real.log (real.exp 1 - 1)),
  split,
  {
    -- Proof that (0 < log(exp 1 - 1) < 1) omitted
    sorry
  },
  {
    intros g h_cont h_diff h_eq,
    -- Proof of congruence omitted
    sorry
  }
end

end lagrange_mean_value_ex_l638_638201


namespace relationship_among_abc_l638_638963

theorem relationship_among_abc :
  let a := 6^0.3
  let b := Real.log 0.6 / Real.log 0.3
  let c := Real.log (Real.sin 1) / Real.log 6
  a > b ∧ b > c :=
by
  let a := 6^0.3
  let b := Real.log 0.6 / Real.log 0.3
  let c := Real.log (Real.sin 1) / Real.log 6
  have h1 : a = 6^0.3 := rfl
  have h2 : b = Real.log 0.6 / Real.log 0.3 := rfl
  have h3 : c = Real.log (Real.sin 1) / Real.log 6 := rfl
  sorry

end relationship_among_abc_l638_638963


namespace angle_between_vectors_l638_638960

variables {ℝ : Type} [field ℝ] [has_zero ℝ] [has_one ℝ]

open_locale real_inner_product_space

noncomputable theory

variables (a b : ℝ × ℝ × ℝ)

def is_nonzero (v : ℝ × ℝ × ℝ) : Prop := v ≠ (0, 0, 0)
def is_perpendicular (v w : ℝ × ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

theorem angle_between_vectors
  (h1 : is_nonzero a)
  (h2 : is_nonzero b)
  (h3 : is_perpendicular (a.1 - 2 * b.1, a.2 - 2 * b.2, a.3 - 2 * b.3) a)
  (h4 : is_perpendicular (b.1 - 2 * a.1, b.2 - 2 * a.2, b.3 - 2 * a.3) b) :
  real.angle_between a b = π / 3 :=
by {
  sorry
}

end angle_between_vectors_l638_638960


namespace line_through_center_and_perpendicular_l638_638099

def center_of_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 1

def perpendicular_to_line (slope : ℝ) : Prop :=
  slope = 1

theorem line_through_center_and_perpendicular (x y : ℝ) :
  center_of_circle x y →
  perpendicular_to_line 1 →
  (x - y + 1 = 0) :=
by
  intros h_center h_perpendicular
  sorry

end line_through_center_and_perpendicular_l638_638099


namespace price_per_pound_of_tomatoes_l638_638290

theorem price_per_pound_of_tomatoes :
  let eggplant_cost := 5 * 2 in
  let zucchini_cost := 4 * 2 in
  let onion_cost := 3 * 1 in
  let basil_cost := 2 * 2.5 in
  let total_known_cost := eggplant_cost + zucchini_cost + onion_cost + basil_cost in
  let total_revenue := 4 * 10 in
  let remaining_amount := total_revenue - total_known_cost in
  4 * p = remaining_amount → p = 3.50 :=
by
  let eggplant_cost := 5 * 2
  let zucchini_cost := 4 * 2
  let onion_cost := 3 * 1
  let basil_cost := 2 * 2.5
  let total_known_cost := eggplant_cost + zucchini_cost + onion_cost + basil_cost
  let total_revenue := 4 * 10
  let remaining_amount := total_revenue - total_known_cost
  intro h
  sorry

end price_per_pound_of_tomatoes_l638_638290


namespace find_a_values_l638_638209

def point := (ℝ × ℝ)
def distsqr (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem find_a_values : 
  ∀ (a : ℝ),
    (∃ (P : point), distsqr P (1,0) * 4 = distsqr P (4,0) ∧ (P.1 - a)^2 + P.2^2 = 1) ↔ 
    a ∈ { -3, -1, 1, 3 } :=
by
  sorry

end find_a_values_l638_638209


namespace g_2023_eq_3_l638_638691

noncomputable def g : ℝ → ℝ := sorry

axiom g_pos : ∀ x > 0, g x > 0
axiom g_eq : ∀ x y > 0, x > y → g(x - y) = sqrt(g(x * y) + 3)

theorem g_2023_eq_3 : g 2023 = 3 := by
  sorry

end g_2023_eq_3_l638_638691


namespace individual_is_sane_l638_638721

-- Definitions based on the conditions
def isSane (person : Type) : Prop := 
  ∀ (p : person), p.answers "Are you a mindless vampire?" = "no"

def isTransylvanian (person : Type) : Prop :=
  ∀ (p : person), p.answers "Are you a mindless vampire?" = "yes"

def determined_type (p : person, answer : String) : Prop :=
  answer ≠ "yes" -- Because if answer is "yes", it leaves ambiguity.

-- The proof problem
theorem individual_is_sane (person : Type) (p : person) (answer : String) 
  (h_sane : isSane person) 
  (h_transylvanian : isTransylvanian person) 
  (h_determine : determined_type p answer) : 
  p ∈ isSane person := 
sorry

end individual_is_sane_l638_638721


namespace same_color_probability_correct_l638_638019

noncomputable def prob_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green / total) * (green / total) +
  (red / total) * (red / total) +
  (blue / total) * (blue / total)

theorem same_color_probability_correct :
  prob_same_color 5 7 3 = 83 / 225 :=
by
  sorry

end same_color_probability_correct_l638_638019


namespace matrix_addition_correct_l638_638107

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 4, -2], ![5, -3, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![ -3,  2, -4], ![ 1, -6,  3], ![-2,  4,  0]]

def expectedSum : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![-1,  1, -1], ![ 1, -2,  1], ![ 3,  1,  1]]

theorem matrix_addition_correct :
  A + B = expectedSum := by
  sorry

end matrix_addition_correct_l638_638107


namespace triangle_inscribed_circumcircles_l638_638838

theorem triangle_inscribed_circumcircles (P Q R X Y Z : Type*)
  (XY YR XR XP QZ QY PZ : ℝ)
  (h1 : XY = 26)
  (h2 : YZ = 28)
  (h3 : XZ = 27)
  (h4 : PZ = YR)
  (h5 : XR = QY)
  (h6 : XP = QZ) :
  QZ = 27 / 2 ∧ (∃ m n : ℕ, m = 27 ∧ n = 2 ∧ Nat.gcd m n = 1 ∧ m + n = 29) :=
by {
  have h7 : 2 * QZ = 27 := sorry,
  have h8 : QZ = 27 / 2 := sorry,
  have m := 27,
  have n := 2,
  have gcd_m_n := Nat.gcd_eq_one_iff_coprime.mpr (by norm_num),
  exact ⟨h8, ⟨m, n, by norm_num, by norm_num, gcd_m_n, by norm_num⟩⟩,
}

end triangle_inscribed_circumcircles_l638_638838


namespace area_ratios_correct_l638_638514

-- Define the given conditions
def main_square_area (s : ℝ) : ℝ := s ^ 2
def triangle_I_area (s : ℝ) : ℝ := s ^ 2 / 2
def circle_II_area (s : ℝ) : ℝ := π * (s / 2) ^ 2
def square_III_area (s : ℝ) : ℝ := (s / 2) ^ 2

-- Define the ratio calculations
def triangle_I_ratio (s : ℝ) : ℝ := triangle_I_area s / main_square_area s
def circle_II_ratio (s : ℝ) : ℝ := circle_II_area s / main_square_area s
def square_III_ratio (s : ℝ) : ℝ := square_III_area s / main_square_area s

-- State the theorem
theorem area_ratios_correct (s : ℝ) (hs : 0 < s) :
  triangle_I_ratio s = 1 / 2 ∧
  circle_II_ratio s = π / 4 ∧
  square_III_ratio s = 1 / 4 :=
by
  -- Proof of the theorem would go here
  sorry

end area_ratios_correct_l638_638514


namespace inequality_of_sums_l638_638955

theorem inequality_of_sums (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_ineq : a > b ∧ b > c ∧ c > d) :
  (a + b + c + d)^2 > a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 :=
by
  sorry

end inequality_of_sums_l638_638955


namespace monotonic_increasing_interval_l638_638186

-- Define the function
def f (x : ℝ) : ℝ := sin x - x * cos x

-- Define the interval
def interval := set.Icc 0 (2 * real.pi)

-- Prove the function is increasing on some sub-interval of [0, 2π]
theorem monotonic_increasing_interval : 
  ∃ (a b : ℝ), (interval a) → (interval b) → a < b → ∀ x ∈ Ico a b, 0 < (deriv f x) :=
begin
  sorry
end

end monotonic_increasing_interval_l638_638186


namespace remainder_7_pow_93_mod_12_l638_638814

theorem remainder_7_pow_93_mod_12 : 7 ^ 93 % 12 = 7 := 
by
  -- the sequence repeats every two terms: 7, 1, 7, 1, ...
  sorry

end remainder_7_pow_93_mod_12_l638_638814


namespace max_brownie_cakes_l638_638423

theorem max_brownie_cakes (m n : ℕ) (h : (m-2)*(n-2) = (1/2)*m*n) :  m * n ≤ 60 :=
sorry

end max_brownie_cakes_l638_638423


namespace min_value_f_interval_l638_638985

noncomputable def f (ω x : ℝ) : ℝ :=
  sin(ω * x) - 2 * sqrt(3) * sin(ω * x / 2)^2 + sqrt(3)

theorem min_value_f_interval (ω : ℝ) (hω : ω > 0)
  (h_dist : ∀ x1 x2 : ℝ, f ω x1 = 0 → f ω x2 = 0 → x1 ≠ x2 → |x1 - x2| = π/2) :
  ∃ x ∈ Icc (0:ℝ) (π / 2:ℝ), f ω x = -sqrt(3) := sorry

end min_value_f_interval_l638_638985


namespace only_possible_functions_l638_638096

noncomputable def f (x : ℝ) : ℝ

axiom functional_eqn :
  ∀ x y : ℝ, f (x * f (x + y)) = f (y * f x) + x^2

theorem only_possible_functions :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end only_possible_functions_l638_638096


namespace P_2007_greater_P_2008_l638_638878

noncomputable def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2008) * ∑ k in finset.range 2008, P (n - k)

theorem P_2007_greater_P_2008 : P 2007 > P 2008 := 
sorry

end P_2007_greater_P_2008_l638_638878


namespace glass_volume_230_l638_638443

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l638_638443


namespace standard_equation_of_circle_l638_638331

-- Definitions based on problem conditions
def center : ℝ × ℝ := (-1, 2)
def radius : ℝ := 2

-- Lean statement of the problem
theorem standard_equation_of_circle :
  ∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = radius ^ 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end standard_equation_of_circle_l638_638331


namespace angles_property_l638_638198

-- Definitions and conditions
def is_triangle (A B C : Point) := 
  ¬(A = B ∨ B = C ∨ C = A) ∧ 
  ¬collinear A B C

def angle_bisector (A B C : Point) (D : Point) := 
  D ∈ Line A B ∧ D ∈ Line A C ∧ 
  ∠ BAC = 2 * ∠ BAD

-- Lean statement (no proof)
theorem angles_property 
  (A B C F F' P : Point)
  (h_triangle : is_triangle A B C)
  (h_AB_AC : distance A B > distance A C)
  (h_angle_bisector_AF : angle_bisector A B C F)
  (h_angle_bisector_AF' : angle_bisector A B C F')
  (h_P_semicircle : Point_on_semicircle P F F')
  (h_P_in_triangle : Point_in_triangle P A B C) :
  ∠ APB - ∠ ACB = ∠ APC - ∠ ABC :=
sorry

end angles_property_l638_638198


namespace smallest_positive_integer_n_mean_squares_l638_638542

theorem smallest_positive_integer_n_mean_squares :
  ∃ n : ℕ, n > 1 ∧ (∃ m : ℕ, (n * m ^ 2 = (n + 1) * (2 * n + 1) / 6) ∧ Nat.gcd (n + 1) (2 * n + 1) = 1 ∧ n = 337) :=
sorry

end smallest_positive_integer_n_mean_squares_l638_638542


namespace meal_preppers_activity_setters_count_l638_638854

-- Definitions for the problem conditions
def num_friends : ℕ := 6
def num_meal_preppers : ℕ := 3

-- Statement of the theorem
theorem meal_preppers_activity_setters_count :
  (num_friends.choose num_meal_preppers) = 20 :=
by
  -- Proof would go here
  sorry

end meal_preppers_activity_setters_count_l638_638854


namespace tangent_line_to_circle_l638_638627

theorem tangent_line_to_circle (m : ℝ) :
  (m = -4 ∨ m = 6) ↔
  (∃ (t : ℝ), (2 * t, 1 - 4 * t) ∈ set.prod {x | x = 2 * t} {y | y = 1 - 4 * t} ∧ ∀ (θ : ℝ), (sqrt 5 * cos θ, m + sqrt 5 * sin θ) ∈ set.prod {x | x = sqrt 5 * cos θ} {y | y = m + sqrt 5 * sin θ} ∧
  (2 * sqrt 5 + (m + sqrt 5 * sin θ) - 1 = 0)) :=
by
  sorry

end tangent_line_to_circle_l638_638627


namespace sixth_term_geometric_sequence_l638_638815

theorem sixth_term_geometric_sequence (a r : ℚ) (h_a : a = 16) (h_r : r = 1/2) : 
  a * r^(5) = 1/2 :=
by 
  rw [h_a, h_r]
  sorry

end sixth_term_geometric_sequence_l638_638815


namespace polar_form_product_l638_638916

def complex_polar_form (r1 theta1 r2 theta2 : ℝ) : ℝ × ℝ :=
  (r1 * r2, (theta1 + theta2) % 360)

theorem polar_form_product :
  (complex_polar_form 4 45 5 120) = (20, 165) :=
by
  sorry

end polar_form_product_l638_638916


namespace no_queue_after_manual_checking_l638_638528

variable (a1 a2 : Int) (x : Int)

def arithmetic_seq_sum (n a d : Int) : Int := n * (2 * a + (n - 1) * d) / 2

theorem no_queue_after_manual_checking
  (h1 : a1 = 59) -- Initial number of students arriving at 7:31 AM
  (h2 : a2 = 40 * x + 12 * (x - 4)) -- Capacity of the detection points
  (h3 : x ≥ 4) -- Manual checking starts at x = 4 minutes
  : arithmetic_seq_sum x a1 (-2) ≤ h2 -> x = 8 := by
  sorry

end no_queue_after_manual_checking_l638_638528


namespace equal_areas_of_triangles_l638_638676

/-- Definition of the geometrical setup in the problem. -/
structure Trapezoid :=
(A B C D M N : Point)
(parallel_AD_BC : Parallel A D B C)
(M_in_trapezoid : InTrapezoid M A B C D)
(N_in_triangle_BMC : InTriangle N B M C)
(parallel_AM_CN : Parallel A M C N)
(parallel_BM_DN : Parallel B M D N)

theorem equal_areas_of_triangles
  (trapezoid : Trapezoid) :
  Area (Triangle trapezoid.A trapezoid.B trapezoid.N) =
  Area (Triangle trapezoid.C trapezoid.D trapezoid.M) :=
sorry

end equal_areas_of_triangles_l638_638676


namespace evaluate_x_squared_plus_y_squared_l638_638583

theorem evaluate_x_squared_plus_y_squared (x y : ℚ) (h1 : x + 2 * y = 20) (h2 : 3 * x + y = 19) : x^2 + y^2 = 401 / 5 :=
sorry

end evaluate_x_squared_plus_y_squared_l638_638583


namespace diana_shops_for_newborns_l638_638079

theorem diana_shops_for_newborns (total_children : ℕ) (num_toddlers : ℕ) (teenager_ratio : ℕ) (num_teens : ℕ) (num_newborns : ℕ)
    (h1 : total_children = 40) (h2 : num_toddlers = 6) (h3 : teenager_ratio = 5) (h4 : num_teens = teenager_ratio * num_toddlers) 
    (h5 : num_newborns = total_children - num_teens - num_toddlers) : 
    num_newborns = 4 := sorry

end diana_shops_for_newborns_l638_638079


namespace log_ordering_l638_638123

theorem log_ordering {x a b c : ℝ} (h1 : 1 < x) (h2 : x < 10) (ha : a = Real.log x^2) (hb : b = Real.log (Real.log x)) (hc : c = (Real.log x)^2) :
  a > c ∧ c > b :=
by
  sorry

end log_ordering_l638_638123


namespace volume_of_pyramid_l638_638036

theorem volume_of_pyramid 
  (QR RS : ℝ) (PT : ℝ) 
  (hQR_pos : 0 < QR) (hRS_pos : 0 < RS) (hPT_pos : 0 < PT)
  (perp1 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * QR) * (x * y) = 0)
  (perp2 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * RS) * (x * y) = 0) :
  QR = 10 -> RS = 5 -> PT = 9 -> 
  (1/3) * QR * RS * PT = 150 :=
by
  sorry

end volume_of_pyramid_l638_638036


namespace problem_l638_638250

open Real

noncomputable def a : ℝ := sqrt 0.3
noncomputable def b : ℝ := sqrt 0.4
noncomputable def c : ℝ := log 3 0.6

theorem problem (a_def : a = sqrt 0.3) (b_def : b = sqrt 0.4) (c_def : c = log 3 0.6) : 
  c < a ∧ a < b := by
  sorry

end problem_l638_638250


namespace problem1_problem2_l638_638579

variable (α : ℝ)

axiom tan_alpha_condition : Real.tan (Real.pi + α) = -1/2

-- Problem 1 Statement
theorem problem1 
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) : 
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) / 
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3 * Real.pi / 2 - α)) = -7/9 := 
sorry

-- Problem 2 Statement
theorem problem2
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) :
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := 
sorry

end problem1_problem2_l638_638579


namespace find_k_l638_638834

variable {G : Type} [Graph G] {v : G} 

def number_of_vertices (k : ℕ) : ℕ := 12 * k
def number_of_edges (k : ℕ) : ℕ := 3 * k + 6
def common_neighbors (x y : G) : ℕ -- this would require actual graph-theory definitions

theorem find_k (k : ℕ) (h1 : number_of_vertices k = 12 * k) 
  (h2 : ∀ v : G, degree v = 3 * k + 6) 
  (h3 : ∀ x y : G, common_neighbors x y = common_neighbors x y)
  (h4 : number_of_edges k = 3 * k + 6) 
  (h5 : ∃ (k'), (k' = 3)) : 
  k = 3 :=
by
  -- Here we would have the detailed proof
  sorry

end find_k_l638_638834


namespace area_BDE_l638_638219

variable (α γ : ℝ) (H : ℝ) (S : ℝ)
variable (hα_pos : 0 < α)
variable (hγ_pos : 0 < γ)
variable (hα_gt_γ: α > γ)
variable (S_pos : 0 < S)

theorem area_BDE {S : ℝ} (α γ : ℝ) (hα_gt_γ : α > γ) (S_pos : 0 < S) :
  ∃ (A B C D E : Type), 
    area_BDE = S * (Real.sin (α - γ)) / (2 * Real.sin (α + γ)) := by
  sorry

end area_BDE_l638_638219


namespace minimize_y_l638_638070

noncomputable def y (x a b k : ℝ) : ℝ :=
  (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y (a b k : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b k ≤ y x' a b k) ∧ x = (a + b - k / 2) / 2 :=
by
  have x := (a + b - k / 2) / 2
  use x
  sorry

end minimize_y_l638_638070


namespace floor_a2021_eq_63_l638_638597

noncomputable def seq (n : ℕ) : ℕ → ℝ
| 1 => 1
| 2 => 2
| (n+3) => ((seq (n+2))^2 + 1) / ((seq (n+1))^2 + 1) * (seq (n+1))

theorem floor_a2021_eq_63 : (⌊seq 2021⌋ : ℕ) = 63 :=
by
  sorry

end floor_a2021_eq_63_l638_638597


namespace percentage_of_x_plus_y_l638_638189

variables (x y P : ℝ)

-- Conditions
def condition1 : 0.30 * (x - y) = (P / 100) * (x + y) := sorry
def condition2 : y = 0.20 * x := sorry

-- Problem statement
theorem percentage_of_x_plus_y (h1 : condition1 x y P) (h2 : condition2 x y) : P = 20 :=
sorry

end percentage_of_x_plus_y_l638_638189


namespace distance_centers_triangle_l638_638045

noncomputable def distance_between_centers (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let circumradius := (a * b * c) / (4 * K)
  let hypotenuse := by
    by_cases hc : a * a + b * b = c * c
    exact c
    by_cases hb : a * a + c * c = b * b
    exact b
    by_cases ha : b * b + c * c = a * a
    exact a
    exact 0
  let oc := hypotenuse / 2
  Real.sqrt (oc * oc + r * r)

theorem distance_centers_triangle :
  distance_between_centers 7 24 25 = Real.sqrt 165.25 := sorry

end distance_centers_triangle_l638_638045


namespace part_I_part_II_l638_638263

variable (a b : ℝ)
hypothesis (ha : a > 0) (hb : b > 0) (h : a^2 * b + a * b^2 = 2)

-- Part I
theorem part_I : a^3 + b^3 ≥ 2 := sorry

variable (h_ab : a^3 + b^3 ≥ 2)

-- Part II
theorem part_II : (a + b) * (a^5 + b^5) ≥ 4 := sorry

end part_I_part_II_l638_638263


namespace max_area_of_region_S_l638_638639

-- Define the radii of the circles
def radii : List ℕ := [2, 4, 6, 8]

-- Define the function for the maximum area of region S given the conditions
def max_area_region_S : ℕ := 75

-- Prove the maximum area of region S is 75π
theorem max_area_of_region_S {radii : List ℕ} (h : radii = [2, 4, 6, 8]) 
: max_area_region_S = 75 := by 
  sorry

end max_area_of_region_S_l638_638639


namespace walking_speed_percentage_l638_638858

theorem walking_speed_percentage (T U : ℝ) (h1 : U = 72.00000000000001) (h2 : T - U = 24) : 
  let S := (U * T / 96)
  in S / U = 0.75 :=
by
  -- Since T = U + 24, we can find a direct substitution:
  have h3 : T = U + 24 := by linarith [h2]
  calc
    U * 96 ≈ U * (U + 24 + 24)
    _ = U * (U + 24) -- by simplification
    ... sorry

end walking_speed_percentage_l638_638858


namespace inequality_proof_l638_638128

noncomputable def a : ℝ := Real.logBase 0.6 0.5
noncomputable def b : ℝ := Real.ln 0.5
noncomputable def c : ℝ := 0.6 ^ (1 / 2 : ℝ)

theorem inequality_proof : a > c ∧ c > b :=
by
  sorry

end inequality_proof_l638_638128


namespace min_cost_25_puzzles_l638_638500

/-- The minimum cost for buying exactly 25 puzzles given that each puzzle costs $10 and each box of 6 puzzles costs $50. -/
theorem min_cost_25_puzzles :
  (∃ (box_cost : ℕ) (single_cost : ℕ) (total_puzzles : ℕ) (min_cost : ℕ),
   box_cost = 50 ∧
   single_cost = 10 ∧
   total_puzzles = 25 ∧
   min_cost = 210 ∧
   ∀ (n_boxes : ℕ), n_boxes = total_puzzles / 6 →
     let total_boxes_cost := n_boxes * box_cost,
         remaining_puzzles := total_puzzles - n_boxes * 6,
         remaining_puzzles_cost := remaining_puzzles * single_cost in
     min_cost = total_boxes_cost + remaining_puzzles_cost) :=
sorry

end min_cost_25_puzzles_l638_638500


namespace exists_three_boxes_with_sufficient_parts_l638_638336

open Function

theorem exists_three_boxes_with_sufficient_parts
  (n : ℕ) (boxes : Fin n → ℕ)
  (h₁ : n = 7)
  (h₂ : ∑ i, boxes i = 100)
  (h₃ : Pairwise (≠) (boxes)) :
  ∃ (i j k : Fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ boxes i + boxes j + boxes k ≥ 50 :=
by
  sorry

end exists_three_boxes_with_sufficient_parts_l638_638336


namespace light_path_correct_and_eval_p_plus_q_l638_638262

-- Define the dimensions and initial conditions
def AB : ℝ := 13
def EF : ℝ := 13 -- Redundant in a rectangular prism, used definitionally.
def BC : ℝ := 15
def FG : ℝ := 15 -- Redundant in a rectangular prism, used definitionally.
def AE : ℝ := 10
def EH : ℝ := 10 -- Redundant in context but defined for clarity
def Q_distance_AB : ℝ := 9
def Q_distance_AE : ℝ := 6

-- Define the coordinates of interest
def E : (ℝ × ℝ × ℝ) := (0, 15, 10)
def Q : (ℝ × ℝ × ℝ) := (9, 6, 10)
def E0 : (ℝ × ℝ × ℝ) := (0, 15, 0)

-- Calculate the distance
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def light_path_length : ℝ :=
  distance Q E0

-- Prove the final answer evaluating p + q in our context
theorem light_path_correct_and_eval_p_plus_q :
  light_path_length = Real.sqrt 262 ∧ 1 + 262 = 263 :=
by
  -- This where the proof would go
  sorry

end light_path_correct_and_eval_p_plus_q_l638_638262


namespace collinear_prob_l638_638853

theorem collinear_prob : 
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  let fa := filter (λ (mn : ℕ × ℕ), mn.2 = 2 * mn.1) outcomes in
  (fa.length : ℚ) / (outcomes.length : ℚ) = 1 / 12 := by
  sorry

end collinear_prob_l638_638853


namespace p_range_l638_638695

noncomputable def p (x : ℝ) : ℝ :=
  if is_prime (floor x) then x + 1 
  else 
    let y := greatest_prime_factor (floor x)
    in p y + (x + 1 - floor x)

def greatest_prime_factor (n : ℕ) : ℕ := sorry -- Assume this definition exists
def is_prime (n : ℕ) : Prop := sorry -- Assume this definition exists

theorem p_range : set.range p = {x | (3 ≤ x ∧ x ≤ 7) ∨ (8 ≤ x ∧ x < 9)} :=
by
  sorry -- Proof skipped

end p_range_l638_638695


namespace range_of_d_l638_638559

theorem range_of_d
  (a b c x₁ x₂: ℝ)
  (h_quad : a * x₁^2 + 2 * b * x₁ + c = 0)
  (h_quad2 : a * x₂^2 + 2 * b * x₂ + c = 0)
  (h_roots_eq : x₁ + x₂ = -2 * b / a)
  (h_product : x₁ * x₂ = c / a)
  (h_a_gt_b : a > b)
  (h_b_gt_c : b > c)
  (h_sum_zero : a + b + c = 0) :
  sqrt 3 < abs (x₁ - x₂) ∧ abs (x₁ - x₂) < sqrt 12 :=
sorry

end range_of_d_l638_638559


namespace total_students_involved_l638_638252

-- Definitions of the given variables and conditions
variables (B G T : ℕ) (y z : ℕ)
-- Condition 1: 98 boys represent y% of B
def cond1 := 98 = y * B / 100
-- Condition 2: Boys make up 50% of the total school population
def cond2 := B = T / 2
-- Condition 3: Girls make up the remaining 50% of the total school population
def cond3 := G = T / 2
-- Condition 4: z% of the girls are involved in the same activity
def cond4 := z * G / 100

-- Definitions of intermediate steps
def total := 98 + 98 * z / y

-- The theorem we want to prove
theorem total_students_involved (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : total = 98 + 98 * z / y :=
sorry

end total_students_involved_l638_638252


namespace pythagorean_triple_fits_l638_638862

theorem pythagorean_triple_fits 
  (k : ℤ) (n : ℤ) : 
  (∃ k, (n = 5 * k ∨ n = 12 * k ∨ n = 13 * k) ∧ 
      (n = 62 ∨ n = 96 ∨ n = 120 ∨ n = 91 ∨ n = 390)) ↔ 
      (n = 120 ∨ n = 91) := by 
  sorry

end pythagorean_triple_fits_l638_638862


namespace problem_statement_l638_638681

-- Define A as the number of four-digit odd numbers
def A : ℕ := 4500

-- Define B as the number of four-digit multiples of 3
def B : ℕ := 3000

-- The main theorem stating the sum A + B equals 7500
theorem problem_statement : A + B = 7500 := by
  -- The exact proof is omitted using sorry
  sorry

end problem_statement_l638_638681


namespace sheet_margin_width_l638_638485

-- Define the problem parameters and conditions
theorem sheet_margin_width
  (width : ℕ)
  (height : ℕ)
  (side_margin : ℕ)
  (percent_used : ℚ)
  (area_used : ℚ)
  (top_bottom_margin : ℚ) :
  width = 20 →
  height = 30 →
  side_margin = 2 →
  percent_used = 0.64 →
  area_used = percent_used * (width * height) →
  top_bottom_margin = 
    (width - 2 * side_margin) * 
    ((height : ℚ) - 2 * top_bottom_margin) →
  top_bottom_margin = 3 :=
by
  intros _ h1 _ h2 _ h3 _ _ 
  sorry

end sheet_margin_width_l638_638485


namespace total_team_points_l638_638205

theorem total_team_points :
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  (A + B + C + D + E + F + G + H = 22) :=
by
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  sorry

end total_team_points_l638_638205


namespace correct_propositions_l638_638980

def proposition1 : Prop := 
  ∀ (A B : ℝ), (sin A = sin B) → (A = B) -- the inverse proposition is true

def proposition2 : Prop := 
  ∀ (P : ℝ × ℝ), (dist P (-4, 0) + dist P (4, 0) = 8) → P = (-4, 0) ∨ P = (4, 0)

def proposition3 : Prop := 
  ¬(p ∧ q) → (¬p ∨ ¬q) -- correct statement of the proposition

def proposition4 : Prop := 
  ∀ x : ℝ, (x^2 - 3 * x > 0) → x > 4 -- necessary but not sufficient

def proposition5 : Prop := 
  ∃ (m : ℝ) (m_pos : m > 0), (m ≠ 0) ∧ 
  (m^2 = 9) ∧ (eccentricity (conic_section (1 / m) 1) = sqrt 6 / 3)

theorem correct_propositions 
  : proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 ∧ proposition5 := sorry

end correct_propositions_l638_638980


namespace sin_is_odd_l638_638666

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem sin_is_odd : is_odd_function sin :=
by
  sorry

end sin_is_odd_l638_638666


namespace solve_lambda_l638_638562

open Real

noncomputable def foci_eccentricity_and_conditions :
  ∃ (a b c : ℝ), 
  (a > b ∧ b > 0) ∧ 
  (a^2 = 2) ∧ 
  (c = 1) ∧ 
  (b = 1) ∧ 
  (c / a = sqrt 2 / 2) ∧ 
  (shortest_line_segment_length = sqrt 2 - 1) := sorry

theorem solve_lambda :
  let ellipse := (λ x y : ℝ, x^2 / 2 + y^2 = 1),
      F1 := (0, -1),
      F2 := (0, 1),
      e := sqrt 2 / 2,
      P := (-1 / 2, sqrt 14 / 4) in
  (∃ (P : ℝ × ℝ), on_ellipse ellipse P) →
  (λ F1 A, ∃ (A : ℝ × ℝ), ∃ (λ : ℝ), 
    (λ > 0) ∧ 
    (vector P F1 = 2 * vector F1 A) ∧ 
    (vector P F2 = λ * vector F2 B)) →
  λ = 4 :=
sorry

end solve_lambda_l638_638562


namespace glass_volume_230_l638_638435

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638435


namespace correct_expression_l638_638492

theorem correct_expression :
  (2 + Real.sqrt 3 ≠ 2 * Real.sqrt 3) ∧ 
  (Real.sqrt 8 - Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6) ∧ 
  (Real.sqrt 27 / Real.sqrt 3 ≠ 9) := 
by
  sorry

end correct_expression_l638_638492


namespace anna_money_left_eur_l638_638057

noncomputable def total_cost_usd : ℝ := 4 * 1.50 + 7 * 2.25 + 3 * 0.75 + 3.00 * 0.80
def sales_tax_rate : ℝ := 0.075
def exchange_rate : ℝ := 0.85
def initial_amount_usd : ℝ := 50

noncomputable def total_cost_with_tax_usd : ℝ := total_cost_usd * (1 + sales_tax_rate)
noncomputable def total_cost_eur : ℝ := total_cost_with_tax_usd * exchange_rate
noncomputable def initial_amount_eur : ℝ := initial_amount_usd * exchange_rate

noncomputable def money_left_eur : ℝ := initial_amount_eur - total_cost_eur

theorem anna_money_left_eur : abs (money_left_eur - 18.38) < 0.01 := by
  -- Add proof steps here
  sorry

end anna_money_left_eur_l638_638057


namespace intersection_of_A_and_B_l638_638575

noncomputable def set_A : set ℝ := {x | x < 2}
noncomputable def set_B : set ℝ := {y | y > -1}

theorem intersection_of_A_and_B : ∀ x, x ∈ set_A ∩ set_B ↔ -1 < x ∧ x < 2 :=
by
  sorry  -- Proof goes here

end intersection_of_A_and_B_l638_638575


namespace total_chairs_l638_638852

-- Define the conditions as constants
def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6
def dining_room_chairs : ℕ := 8
def outdoor_patio_chairs : ℕ := 12

-- State the goal to prove
theorem total_chairs : 
  living_room_chairs + kitchen_chairs + dining_room_chairs + outdoor_patio_chairs = 29 := 
by
  -- The proof is not required as per instructions
  sorry

end total_chairs_l638_638852


namespace angle_sum_155_l638_638411

theorem angle_sum_155
  (AB AC DE DF : ℝ)
  (h1 : AB = AC)
  (h2 : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h3 : angle_BAC = 20)
  (h4 : angle_EDF = 30) :
  ∃ (angle_DAC angle_ADE : ℝ), angle_DAC + angle_ADE = 155 :=
by
  sorry

end angle_sum_155_l638_638411


namespace no_solution_exists_l638_638266

theorem no_solution_exists : 
  ¬ ∃ (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    45 * x = (35 / 100) * 900 ∧
    y^2 + x = 100 ∧
    z = x^3 * y - (2 * x + 1) / (y + 4) :=
by
  sorry

end no_solution_exists_l638_638266


namespace ratio_of_dogs_to_cats_l638_638063

-- Definition of conditions
def total_animals : Nat := 21
def cats_to_spay : Nat := 7
def dogs_to_spay : Nat := total_animals - cats_to_spay

-- Ratio of dogs to cats
def dogs_to_cats_ratio : Nat := dogs_to_spay / cats_to_spay

-- Statement to prove
theorem ratio_of_dogs_to_cats : dogs_to_cats_ratio = 2 :=
by
  -- Proof goes here
  sorry

end ratio_of_dogs_to_cats_l638_638063


namespace tamika_greater_probability_l638_638301

-- Define the sets of numbers.
def tamika_set : set ℕ := {7, 11, 12}
def carlos_set : set ℕ := {4, 6, 7}

-- Define the function to calculate all possible sums from Tamika's set.
def tamika_possible_sums : set ℕ := {18, 19, 23} -- {a + b | a, b ∈ tamika_set, a ≠ b}

-- Define the function to calculate all possible products from Carlos' set.
def carlos_possible_products : set ℕ := {24, 28, 42} -- {a * b | a, b ∈ carlos_set, a ≠ b}

-- Define the function to count the pairs where Tamika's sum is greater than Carlos's product.
def count_successful_pairs :=
  (tamika_possible_sums.product carlos_possible_products).count (λ p, p.1 > p.2)

-- Define the total number of possible pairs.
def total_pairs : ℕ := 9

-- Prove that the probability Tamika's result is greater than Carlos's result is 1/9.
theorem tamika_greater_probability : count_successful_pairs = 1 / total_pairs := by
  sorry

end tamika_greater_probability_l638_638301


namespace mass_of_man_proof_l638_638846

def volume_displaced (L B h : ℝ) : ℝ :=
  L * B * h

def mass_of_man (V ρ : ℝ) : ℝ :=
  ρ * V

theorem mass_of_man_proof :
  ∀ (L B h ρ : ℝ), L = 9 → B = 3 → h = 0.01 → ρ = 1000 →
  mass_of_man (volume_displaced L B h) ρ = 270 :=
by
  intros L B h ρ L_eq B_eq h_eq ρ_eq
  rw [L_eq, B_eq, h_eq, ρ_eq]
  unfold volume_displaced
  unfold mass_of_man
  simp
  sorry

end mass_of_man_proof_l638_638846


namespace max_square_test_plots_l638_638035

theorem max_square_test_plots (length width fence : ℕ)
  (h_length : length = 36)
  (h_width : width = 66)
  (h_fence : fence = 2200) :
  ∃ (n : ℕ), n * (11 / 6) * n = 264 ∧
      (36 * n + (11 * n - 6) * 66) ≤ 2200 := sorry

end max_square_test_plots_l638_638035


namespace rainfall_ratio_l638_638529

theorem rainfall_ratio (R1 R2 : ℕ) (hR2 : R2 = 24) (hTotal : R1 + R2 = 40) : 
  R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l638_638529


namespace approx_average_price_l638_638233

-- Definitions
def num_large_bottles : ℕ := 1325
def price_per_large_bottle : ℝ := 1.89
def discount_rate : ℝ := 0.05
def num_small_bottles : ℕ := 750
def price_per_small_bottle : ℝ := 1.38

-- Total cost for large bottles
def total_cost_large_bottles : ℝ := num_large_bottles * price_per_large_bottle
-- Discount for large bottles
def discount_large_bottles : ℝ := total_cost_large_bottles * discount_rate
-- Discounted total for large bottles
def discounted_total_large_bottles : ℝ := total_cost_large_bottles - discount_large_bottles
-- Total cost for small bottles
def total_cost_small_bottles : ℝ := num_small_bottles * price_per_small_bottle
-- Total cost for all bottles
def total_cost_all_bottles : ℝ := discounted_total_large_bottles + total_cost_small_bottles
-- Total number of bottles
def total_num_bottles : ℕ := num_large_bottles + num_small_bottles
-- Average price per bottle
def average_price_per_bottle : ℝ := total_cost_all_bottles / total_num_bottles

-- Proposition to prove
theorem approx_average_price :
  abs (average_price_per_bottle - 1.645) < 0.01 :=
by linarith

end approx_average_price_l638_638233


namespace simplify_expression_l638_638291

theorem simplify_expression (r : ℝ) (h1 : r^2 ≠ 0) (h2 : r^4 > 16) :
  ( ( ( (r^2 + 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 + 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ)
    - ( (r^2 - 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 - 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ) ) ^ 2 )
  / ( r^2 - (r^4 - 16) ^ (1 / 2 : ℝ) )
  = 2 * r ^ (-(2 / 3 : ℝ)) := by
  sorry

end simplify_expression_l638_638291


namespace first_chapter_length_l638_638410

theorem first_chapter_length (total_pages : ℕ) (second_chapter_pages : ℕ) (third_chapter_pages : ℕ)
  (h : total_pages = 125) (h2 : second_chapter_pages = 35) (h3 : third_chapter_pages  = 24) :
  total_pages - second_chapter_pages - third_chapter_pages = 66 :=
by
  -- Construct the proof using the provided conditions
  sorry

end first_chapter_length_l638_638410


namespace perpendicular_line_eqn_length_AQ_radius_circle_l638_638047

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 6)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (14, 0)

-- Define the midpoint P of AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def P : ℝ × ℝ := midpoint A B

-- Define the slope of a line given two points
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the equation of the line perpendicular to AB passing through P
noncomputable def linePerpendicularTo (A B P : ℝ × ℝ) (m_AB : ℝ) : ℝ → ℝ := 
  λ x, -1 / m_AB * x + (P.2 + 1 / m_AB * P.1)

-- Check the equation of the perpendicular line to AB through P
theorem perpendicular_line_eqn :
  let m_AB := slope A B in
  linePerpendicularTo A B P m_AB = (λ x, -1/3 * x + 10/3) := sorry

-- Define the point Q on BC
noncomputable def Q : ℝ × ℝ := (10, 0)

-- Define the distance between two points
def distance (A B : ℝ × ℝ): ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Check the distance AQ
theorem length_AQ :
  distance A Q = 10 := sorry

-- Define the circumcenter of triangle
noncomputable def circumcenter (A B C : ℝ × ℝ): ℝ × ℝ := 
  let P1 := (7, 0)
  let P2 := intersection (linePerpendicularTo A B (midpoint A B) (slope A B)) (λ x, P1.1)
  in P2

-- Define the radius of the circumcircle
noncomputable def circumradius (A B C : ℝ × ℝ) : ℝ := 
  distance A (circumcenter A B C)

-- Check the radius of circumcircle
theorem radius_circle :
  circumradius A B C = 5 * Real.sqrt 2 := sorry

end perpendicular_line_eqn_length_AQ_radius_circle_l638_638047


namespace smallest_base_not_9_l638_638917

def digit_sum (n : ℕ) (b : ℕ) : ℕ :=
  n.digits b |>.sum

noncomputable def smallest_base (n : ℕ) : ℕ :=
  (@Finset.filter ℕ (λ b, digit_sum ((b + 2) ^ 3) b ≠ 9)
                 (@Finset.range (n + 1))).min' (by sorry)

theorem smallest_base_not_9 : smallest_base 10 = 6 :=
  by sorry

end smallest_base_not_9_l638_638917


namespace tenths_digit_of_five_twelfths_l638_638354

-- Define the problem in Lean
theorem tenths_digit_of_five_twelfths : (decimal_place (\frac{5}{12}) 1) = 4 := 
by sorry

end tenths_digit_of_five_twelfths_l638_638354


namespace whose_number_is_larger_l638_638347

theorem whose_number_is_larger
    (vasya_prod : ℕ := 4^12)
    (petya_prod : ℕ := 2^25) :
    petya_prod > vasya_prod :=
    by
    sorry

end whose_number_is_larger_l638_638347


namespace max_distance_MN_l638_638979

-- Ellipse definition
def is_on_ellipse (x y : ℝ) : Prop := (y / 4) + x^2 = 1

-- Definition of point P on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := is_on_ellipse P.1 P.2

-- Lines through P parallel to l1: y = 2x and l2: y = -2x
def line_parallel_l1_through_P (P : ℝ × ℝ) (x : ℝ) : ℝ := 2 * (x - P.1) + P.2
def line_parallel_l2_through_P (P : ℝ × ℝ) (x : ℝ) : ℝ := -2 * (x - P.1) + P.2

-- Intersection points M and N
def intersection_M (P : ℝ × ℝ) : ℝ × ℝ :=
  let x_M := (2 * P.1 - P.2) / 4
  let y_M := -2 * x_M
  (x_M, y_M)

def intersection_N (P : ℝ × ℝ) : ℝ × ℝ :=
  let x_N := (2 * P.1 + P.2) / 4
  let y_N := 2 * x_N
  (x_N, y_N)

-- Distance between M and N
def distance_MN (M N : ℝ × ℝ) : ℝ :=
  real.sqrt (((N.1 - M.1)^2) + ((N.2 - M.2)^2))

-- Main theorem: Maximum distance |MN| is 2
theorem max_distance_MN (P : ℝ × ℝ) (hp : point_on_ellipse P) :
  ∀ (M N : ℝ × ℝ), M = intersection_M P → N = intersection_N P → distance_MN M N ≤ 2 :=
by
  sorry

end max_distance_MN_l638_638979


namespace find_constants_l638_638095

theorem find_constants (x : ℝ) : ∃ (C D : ℝ), 
  (x ≠ 9 ∧ x ≠ -4) → 
  (5 * x + 7) / (x ^ 2 - 5 * x - 36) = (C / (x - 9)) + (D / (x + 4)) ∧ C = 4 ∧ D = 1 :=
by
  use [4, 1]
  intros h
  have h1 : x ^ 2 - 5 * x - 36 = (x - 9) * (x + 4), by sorry
  rw h1
  suffices : 5 * x + 7 = 4 * (x + 4) + 1 * (x - 9), by sorry
  sorry

end find_constants_l638_638095


namespace x_gt_1_sufficient_not_necessary_x_squared_gt_1_l638_638396

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end x_gt_1_sufficient_not_necessary_x_squared_gt_1_l638_638396


namespace productivity_comparison_l638_638389

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l638_638389


namespace solve_inequality_l638_638295

open Real

theorem solve_inequality (a : ℝ) :
  ((a < 0 ∨ a > 1) → (∀ x, a < x ∧ x < a^2 ↔ (x - a) * (x - a^2) < 0)) ∧
  ((0 < a ∧ a < 1) → (∀ x, a^2 < x ∧ x < a ↔ (x - a) * (x - a^2) < 0)) ∧
  ((a = 0 ∨ a = 1) → (∀ x, ¬((x - a) * (x - a^2) < 0))) :=
by
  sorry

end solve_inequality_l638_638295


namespace range_of_a_l638_638320

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 6| - |x - 4| ≤ a^2 - 3 * a) ↔ a ∈ set.Iic (-2) ∪ set.Ici 5 :=
by
  sorry

end range_of_a_l638_638320


namespace green_pill_cost_l638_638051

variable (x : ℝ) -- cost of a green pill in dollars
variable (y : ℝ) -- cost of a pink pill in dollars
variable (total_cost : ℝ) -- total cost for 21 days

theorem green_pill_cost
  (h1 : x = y + 2) -- a green pill costs $2 more than a pink pill
  (h2 : total_cost = 819) -- total cost for 21 days is $819
  (h3 : ∀ n, n = 21 ∧ total_cost / n = (x + y)) :
  x = 20.5 :=
by
  sorry

end green_pill_cost_l638_638051


namespace angie_age_l638_638801

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l638_638801


namespace pencils_left_after_operations_l638_638532

theorem pencils_left_after_operations :
  ∀ (initial_pencils received_pencils pack_count pencils_per_pack given_away_pencils : ℕ),
    initial_pencils = 51 →
    received_pencils = 6 →
    pack_count = 4 →
    pencils_per_pack = 12 →
    given_away_pencils = 8 →
    initial_pencils + received_pencils + (pack_count * pencils_per_pack) - given_away_pencils = 97 :=
by
  intros initial_pencils received_pencils pack_count pencils_per_pack given_away_pencils
  assume h_initial h_received h_pack_count h_pencils_per_pack h_given_away
  sorry

end pencils_left_after_operations_l638_638532


namespace triangle_inequality_l638_638719

variables {A B C D F E : Type}
variables [metric_space E] [add_group E] [has_dist E E]

-- Definitions of points and conditions
def is_triangle (a b c : E) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a
def is_on_segment (p a b : E) : Prop := dist a p + dist p b = dist a b
def is_midpoint (m x y : E) : Prop := dist x m = dist m y ∧ 2 * dist x m = dist x y

-- Main statement to be proved
theorem triangle_inequality
  (A B C D F E : E)
  (h_triangle : is_triangle A B C)
  (h_D : is_on_segment D A B)
  (h_F : is_on_segment F B C)
  (h_E : is_midpoint E D F) :
  dist A D + dist F C ≤ dist A E + dist E C :=
sorry

end triangle_inequality_l638_638719


namespace area_correct_l638_638008

variable (A B C L : Type) [MetricSpace A]
variables (a b c bl al cl : ℝ)
variables (is_angle_bisector : (bl = 3 * Real.sqrt 10))
variables (AL_eq_2 : al = 2)
variables (CL_eq_3 : cl = 3)

noncomputable def area_of_triangle_ABC : ℝ :=
  let area := (15 * Real.sqrt 15) / 4 in
  area

theorem area_correct (h₁ : is_angle_bisector) (h₂ : AL_eq_2) (h₃ : CL_eq_3) :
  area_of_triangle_ABC A B C L a b c bl al cl is_angle_bisector AL_eq_2 CL_eq_3
  = (15 * Real.sqrt 15) / 4 :=
sorry

end area_correct_l638_638008


namespace ABC_product_l638_638069

theorem ABC_product :
  (∀ x : ℝ, x^3 - 3 * x^2 - 4 * x + 12 = (x - 2) * (x + 2) * (x - 3)) →
  (∃ A B C : ℝ,
    (∀ x : ℝ, x^2 - 16 = A * (x + 2) * (x - 3) + B * (x - 2) * (x - 3) + C * (x - 2) * (x + 2)) ∧
    A * B * C = -63 / 25) :=
begin
  intro h,
  use [3, 3/5, -7/5],
  split,
  { intro x,
    rw [h],
    sorry
  },
  { norm_num }
end

end ABC_product_l638_638069


namespace sum_b_first_5_terms_eq_93_l638_638637

noncomputable def a : ℕ → ℝ
noncomputable def b : ℕ → ℝ

axiom geom_seq (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, a n > 0) ∧ 
  (a 2 * a 4 = 16) ∧ 
  (a 6 = 32) ∧ 
  (∀ n, b n = a n + a (n + 1))

theorem sum_b_first_5_terms_eq_93 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ)
  (h : geom_seq a b) :
  let S5 := b 1 + b 2 + b 3 + b 4 + b 5 in
  S5 = 93 :=
sorry

end sum_b_first_5_terms_eq_93_l638_638637


namespace wendy_sold_40_apples_in_the_morning_l638_638350

theorem wendy_sold_40_apples_in_the_morning 
  (price_apple price_orange : ℝ)
  (morning_oranges afternoon_apples afternoon_oranges : ℕ)
  (total_sales : ℝ)
  (morning_apples : ℕ) :
  price_apple = 1.50 ∧
  price_orange = 1 ∧
  morning_oranges = 30 ∧
  afternoon_apples = 50 ∧
  afternoon_oranges = 40 ∧
  total_sales = 205 →
  1.50 * morning_apples + 1.50 * 50 + 1 * 30 + 1 * 40 = 205 →
  morning_apples = 40 :=
by
  intros h_cond h_eq
  sorry

end wendy_sold_40_apples_in_the_morning_l638_638350


namespace inequality_proof_l638_638259

open Real

-- All conditions as given in the problem
variables {n : ℕ} (a : Fin (n + 2) → ℝ)

-- Define the assumptions
def condition1 : Prop := n ≥ 2
def condition2 : Prop := ∀ i, 0 < a i
def condition3 : Prop := ∀ k : ℕ, 1 ≤ k → k ≤ n → a (k + 1) - a k = a (n + 1) - a n ∧ a (k + 1) - a k ≥ 0

-- The main theorem to be proved
theorem inequality_proof (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  ∑ k in Finset.range (n - 1) + 2, 1 / a (k + 2) ^ 2 ≤ 
  (n - 1 : ℝ) / 2 * (a 1 * a n + a 2 * a (n + 1)) / (a 1 * a 2 * a n * a (n + 1)) :=
by
  sorry

end inequality_proof_l638_638259


namespace glass_volume_correct_l638_638468

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638468


namespace triangle_construction_l638_638516

noncomputable def exists_triangle_with_angle_bisector (a b m : ℝ) : Prop :=
  ∃ (A B C D : Type)
    [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (dist : A → A → ℝ)
    (AB AC BC : ℝ) (CD : A → A → ℝ)
    (A_ne_B : A ≠ B) (A_ne_C : A ≠ C) (B_ne_C : B ≠ C)
    (on_AB : D ∈ line_segment A B)
    (angle_bisector : CD D = dist A C ∧ CD D = dist B C) ,
    AB = dist A B ∧ 
    AC = dist A C ∧ 
    BC = dist B C ∧
    AB = c ∧
    AC = b ∧
    BC = a ∧
    CD = m

-- The theorem to prove the existence:
theorem triangle_construction (a b m: ℝ) : exists_triangle_with_angle_bisector a b m :=
sorry

end triangle_construction_l638_638516


namespace prove_distance_from_radical_axis_l638_638748

noncomputable def distance_from_circumcenter_to_radical_axis 
    (a b c R : ℝ) 
    (O is_circumcenter : Prop) -- O is circumcenter of triangle ABC
    (area_condition : ∀ S (ABC OAB OAC OBC : Set ℝ), (S OAB + S OAC) / 2 = S OBC) :
    ℝ :=
  \frac{a^2}{\sqrt{9R^2 - (a^2 + b^2 + c^2)}}

theorem prove_distance_from_radical_axis
    (a b c R : ℝ)
    (O is_circumcenter : Prop)
    (area_condition : ∀ S (ABC OAB OAC OBC : Set ℝ), (S OAB + S OAC) / 2 = S OBC) :
    distance_from_circumcenter_to_radical_axis a b c R O is_circumcenter area_condition 
    = \frac{a^2}{\sqrt{9R^2 - (a^2 + b^2 + c^2)}} :=
by
  sorry

end prove_distance_from_radical_axis_l638_638748


namespace a_n_is_arithmetic_T_n_sum_l638_638134

-- Definition of the sequence and conditions.
def sequence_a (n : ℕ) : ℕ := sorry

def sum_S (n : ℕ) : ℕ := sorry

axiom a2_minus_a1 : sequence_a 2 - sequence_a 1 = 1
axiom arithmetic_S : ∀ n, n ≥ 2 → 2 * sum_S(n) = sum_S(n-1) - 1 + sum_S(n + 1)

-- Problem 1
theorem a_n_is_arithmetic (n : ℕ) : ∀ (n : ℕ), sequence_a (n + 1) - sequence_a n = 1 :=
sorry

-- Problem 2
def sequence_b (n : ℕ) : ℚ := 1 / (sequence_a n * sequence_a (n + 1))

theorem T_n_sum (n : ℕ) : 
  let T_n := ∑ i in finset.range n, sequence_b (i + 1) in
  T_n = n / (n + 1) :=
sorry

end a_n_is_arithmetic_T_n_sum_l638_638134


namespace maximum_value_f_zeros_l638_638987

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if 1 < x then k * x + 1
  else 0

theorem maximum_value_f_zeros (k : ℝ) (x1 x2 : ℝ) :
  0 < k ∧ ∀ x, f x k = 0 ↔ x = x1 ∨ x = x2 → x1 ≠ x2 →
  x1 > 0 → x2 > 0 → -1 < k ∧ k < 0 →
  (x1 = -1 / k) ∧ (x2 = 1 / (1 + Real.sqrt (1 + k))) →
  ∃ y, (1 / x1) + (1 / x2) = y ∧ y = 9 / 4 := sorry

end maximum_value_f_zeros_l638_638987


namespace prove_parallel_BK_AE_l638_638556

variables (A B C D E K : Type) [convex_pentagon A B C D E]
variables [parallel AE CD] [eq AB BC]
variables [angle_bisectors_intersect K A C]

theorem prove_parallel_BK_AE : parallel BK AE :=
sorry

end prove_parallel_BK_AE_l638_638556


namespace width_of_box_l638_638023

theorem width_of_box 
(length depth num_cubes : ℕ)
(h_length : length = 49)
(h_depth : depth = 14)
(h_num_cubes : num_cubes = 84)
: ∃ width : ℕ, width = 42 := 
sorry

end width_of_box_l638_638023


namespace smallest_int_square_eq_3x_plus_72_l638_638816

theorem smallest_int_square_eq_3x_plus_72 :
  ∃ x : ℤ, x^2 = 3 * x + 72 ∧ (∀ y : ℤ, y^2 = 3 * y + 72 → x ≤ y) :=
sorry

end smallest_int_square_eq_3x_plus_72_l638_638816


namespace balance_the_scale_l638_638273

theorem balance_the_scale (w1 : ℝ) (w2 : ℝ) (book_weight : ℝ) (h1 : w1 = 0.5) (h2 : w2 = 0.3) :
  book_weight = w1 + 2 * w2 :=
by
  sorry

end balance_the_scale_l638_638273


namespace sqrt_meaningful_range_l638_638184

-- Define the condition
def sqrt_condition (x : ℝ) : Prop := 1 - 3 * x ≥ 0

-- State the theorem
theorem sqrt_meaningful_range (x : ℝ) (h : sqrt_condition x) : x ≤ 1 / 3 :=
sorry

end sqrt_meaningful_range_l638_638184


namespace complex_arithmetic_l638_638505

-- Definitions of the complex numbers involved in the question
def z1 : ℂ := 1 - 1*I
def z2 : ℂ := -3 + 2*I
def z3 : ℂ := 4 - 6*I

-- The statement we need to prove
theorem complex_arithmetic : (z1 - z2 + z3) = 8 - 9*I :=
by sorry

end complex_arithmetic_l638_638505


namespace x_gt_1_sufficient_not_necessary_x_squared_gt_1_l638_638397

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end x_gt_1_sufficient_not_necessary_x_squared_gt_1_l638_638397


namespace daycare_initial_ratio_l638_638061

-- Defining the initial number of infants as a parameter
def initial_ratio_toddlers_to_infants (I : ℕ) : Prop :=
  let T := 42 in
  let I_new := I + 12 in
  let new_ratio_valid := (7 * I_new = 5 * T) in
  new_ratio_valid ∧ (T / Nat.gcd T I) = 7 ∧ (I / Nat.gcd T I) = 3

-- Now, we state the theorem which uses above definition.
theorem daycare_initial_ratio:
  ∃ I : ℕ, initial_ratio_toddlers_to_infants I :=
begin
  sorry
end

end daycare_initial_ratio_l638_638061


namespace choose_four_pairwise_coprime_l638_638281

noncomputable def three_digit_natural (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def are_pairwise_coprime (S : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1

theorem choose_four_pairwise_coprime
  (S : Set ℕ)
  (h1 : ∀ n ∈ S, three_digit_natural n)
  (h2 : S.card ≥ 4)
  (h3 : are_pairwise_coprime S)
  : ∃ T ⊆ S, T.card = 4 ∧ are_pairwise_coprime T :=
sorry

end choose_four_pairwise_coprime_l638_638281


namespace find_a_l638_638927

theorem find_a (a x y : ℝ) 
  (h1 : (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0) 
  (h2 : (x + 2)^2 + (y + 4)^2 = a) 
  (h3 : ∃! x y, (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0 ∧ (x + 2)^2 + (y + 4)^2 = a) :
  a = 9 ∨ a = 23 + 4 * Real.sqrt 15 :=
sorry

end find_a_l638_638927


namespace triangle_inequality_l638_638995

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality (a b c : ℝ) (P : ℝ) (hP : P = triangle_area a b c) : 
  (a * b + b * c + c * a) / (4 * P) ≥ Real.sqrt 3 :=
by
  sorry

end triangle_inequality_l638_638995


namespace length_of_second_train_l638_638806

-- Define the conditions as constants
def speed_train1_kmph : ℝ := 42
def speed_train2_kmph : ℝ := 36
def length_train1_meters : ℝ := 120
def clear_time_seconds : ℝ := 18.460061656605934

-- Convert speeds from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

def speed_train1_mps : ℝ := kmph_to_mps speed_train1_kmph
def speed_train2_mps : ℝ := kmph_to_mps speed_train2_kmph

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

-- Calculate the total distance covered when the trains clear each other
def total_distance_covered : ℝ := relative_speed_mps * clear_time_seconds

-- Define the length of the second train
def length_train2_meters : ℝ := total_distance_covered - length_train1_meters

-- The statement to prove
theorem length_of_second_train : length_train2_meters = 278.3333333335 :=
by
  -- Proof can be provided here later
  sorry

end length_of_second_train_l638_638806


namespace area_of_S_l638_638519

-- Define a four-presentable complex number and the set S
def is_four_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), abs w = 4 ∧ z = w - 2 / w

def S : set ℂ := {z | is_four_presentable z}

-- Statement of the theorem
theorem area_of_S : (π * (9 / 2) * (7 / 2)) = (63 / 4) * π :=
by sorry

end area_of_S_l638_638519


namespace imaginary_part_l638_638769

theorem imaginary_part :
  complex.im (5 * complex.I / (1 + 2 * complex.I)) = 1 :=
by {
  sorry
}

end imaginary_part_l638_638769


namespace inequality_of_monomials_l638_638372

theorem inequality_of_monomials {n : ℕ} (A B : Fin n → ℝ) (hA : ∀ i j, i ≤ j → A i ≥ A j) (hB : ∀ i j, i ≤ j → B i ≥ B j) :
  (∑ i in Finset.range n, A i * B i) ≥ (∑ i in Finset.range n, A i * B (n - i - 1)) :=
sorry

end inequality_of_monomials_l638_638372


namespace triangle_area_l638_638010

open Real

theorem triangle_area (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 3 * sqrt 10) (h₃ : c = 3) :
  let s := (8 + 12 + 5) / 2 in
  sqrt (s * (s - 8) * (s - 12) * (s - 5)) = 15 * sqrt 15 / 4 :=
by
  have h4 : 8 = 2,
  have h5 : 12 = 3 * sqrt 10,
  have h6 : 5 = 3,
  sorry

end triangle_area_l638_638010


namespace magnitude_of_z_l638_638143

theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  |((1 - i) / (1 + i) : ℂ)| = 1 := 
by 
  sorry

end magnitude_of_z_l638_638143


namespace volume_of_scaled_surface_area_cube_l638_638819

theorem volume_of_scaled_surface_area_cube 
  (V₁ : ℝ) (s₁ : ℝ) (A₁ : ℝ) (s₂ : ℝ) (A₂ : ℝ) (V₂ : ℝ) :
  V₁ = 8 → s₁ = real.cbrt V₁ → A₁ = 6 * s₁^2 →
  A₂ = 3 * A₁ → s₂ = real.sqrt (A₂ / 6) → V₂ = s₂^3 →
  V₂ = 24 * real.sqrt 3 := 
by
  intros
  sorry

end volume_of_scaled_surface_area_cube_l638_638819


namespace circles_externally_tangent_l638_638148

def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := real.sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2)

theorem circles_externally_tangent :
  ∀ (O1 O2 : ℝ × ℝ) (r1 r2 : ℝ),
    O1 = (0, 8) →
    O2 = (-6, 0) →
    r1 = 6 →
    r2 = 2 →
    dist O1.1 O1.2 O2.1 O2.2 = r1 + r2 → 
    (dist O1.1 O1.2 O2.1 O2.2 > r1 + r2) :=

sorry


end circles_externally_tangent_l638_638148


namespace solutions_to_cube_eq_27_l638_638543

theorem solutions_to_cube_eq_27 (z : ℂ) : 
  (z^3 = 27) ↔ (z = 3 ∨ z = (Complex.mk (-3 / 2) (3 * Real.sqrt 3 / 2)) ∨ z = (Complex.mk (-3 / 2) (-3 * Real.sqrt 3 / 2))) :=
by sorry

end solutions_to_cube_eq_27_l638_638543


namespace compare_P2007_P2008_l638_638866

def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2007) * ∑ k in (finset.range 2007).image (λ x, x + n + 1 - 2007), P k

theorem compare_P2007_P2008 : P 2007 > P 2008 :=
sorry

end compare_P2007_P2008_l638_638866


namespace metal_rods_per_sheet_l638_638850

theorem metal_rods_per_sheet :
  (∀ (metal_rod_for_sheets metal_rod_for_beams total_metal_rod num_sheet_per_panel num_panel num_rod_per_beam),
    (num_rod_per_beam = 4) →
    (total_metal_rod = 380) →
    (metal_rod_for_beams = num_panel * (2 * num_rod_per_beam)) →
    (metal_rod_for_sheets = total_metal_rod - metal_rod_for_beams) →
    (num_sheet_per_panel = 3) →
    (num_panel = 10) →
    (metal_rod_per_sheet = metal_rod_for_sheets / (num_panel * num_sheet_per_panel)) →
    metal_rod_per_sheet = 10
  ) := sorry

end metal_rods_per_sheet_l638_638850


namespace f_f_neg1_l638_638764

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2 else -x + 1

theorem f_f_neg1 : f (f (-1)) = 6 := sorry

end f_f_neg1_l638_638764


namespace truncated_pyramid_lateral_surface_area_l638_638318

-- Define the conditions
def H : ℝ := 3 -- height in cm
def V : ℝ := 38 -- volume in cm^3
def S1_to_S2 : ℝ := 4 / 9 -- ratio of areas

-- Define the statement to be proven
def lateral_surface_area : Prop :=
  (∃ (S1 S2 : ℝ), S1_to_S2 = S1 / S2 ∧ V = H / 3 * (S1 + S2 + real.sqrt (S1 * S2)) ∧
  ∃ (b a : ℝ), S1 = b^2 ∧ S2 = a^2 ∧
  (let p1 := 4 * b in
   let p2 := 4 * a in
   let ON := a / 2 in
   let O1K := b / 2 in
   let PN := ON - O1K in
   let KN := real.sqrt (H^2 + PN^2) in
   (p1 + p2) / 2 * KN =  10 * real.sqrt 19))

-- Main theorem stating the proof problem
theorem truncated_pyramid_lateral_surface_area : lateral_surface_area :=
begin
  sorry -- place holder for the actual proof
end

end truncated_pyramid_lateral_surface_area_l638_638318


namespace part1_a2_part1_a3_part2_general_formula_l638_638135

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| n + 1 => (n + 1) * n / 2

noncomputable def S (n : ℕ) : ℚ := (n + 2) * a n / 3

theorem part1_a2 : a 2 = 3 := sorry

theorem part1_a3 : a 3 = 6 := sorry

theorem part2_general_formula (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

end part1_a2_part1_a3_part2_general_formula_l638_638135


namespace incorrect_statements_about_f_l638_638160

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := (1 - x) * Real.exp x

-- State the problem that some statements about f are incorrect
theorem incorrect_statements_about_f :
  ¬ (f' 2 > 0) ∧ ¬ ∀ x: ℝ, 1 < x → f' x > 0 := by
  sorry

end incorrect_statements_about_f_l638_638160


namespace inequality_direction_change_l638_638050

theorem inequality_direction_change :
  ∃ (a b c : ℝ), (a < b) ∧ (c < 0) ∧ (a * c > b * c) :=
by
  sorry

end inequality_direction_change_l638_638050


namespace number_of_M_partitions_l638_638296

def A := { i : ℕ | 1 ≤ i ∧ i ≤ 2002 }

def M := {1001, 2003, 3005}

def is_M_free (B : Set ℕ) : Prop :=
  ∀ a b ∈ B, a ≠ b → a + b ∉ M

def is_M_partition (A1 A2 : Set ℕ) : Prop :=
  A1 ∪ A2 = A ∧ A1 ∩ A2 = ∅ ∧ is_M_free A1 ∧ is_M_free A2

theorem number_of_M_partitions :
  ∃ n : ℕ, n = 2^501 ∧ ∀ (A1 A2 : Set ℕ), is_M_partition A1 A2 → true :=
sorry

end number_of_M_partitions_l638_638296


namespace average_of_integers_between_1_5_and_1_3_is_20_l638_638810

theorem average_of_integers_between_1_5_and_1_3_is_20 :
  let s := {N : ℕ | 15 < N ∧ N < 25} in
    (∑ N in s, N) / s.card = 20 :=
by
  sorry

end average_of_integers_between_1_5_and_1_3_is_20_l638_638810


namespace fred_sheets_l638_638937

theorem fred_sheets (initial_sheets : ℕ) (received_sheets : ℕ) (given_sheets : ℕ) :
  initial_sheets = 212 → received_sheets = 307 → given_sheets = 156 →
  (initial_sheets + received_sheets - given_sheets) = 363 :=
by
  intros h_initial h_received h_given
  rw [h_initial, h_received, h_given]
  sorry

end fred_sheets_l638_638937


namespace isosceles_of_equal_bisectors_l638_638733

theorem isosceles_of_equal_bisectors
  (A B C L1 L3 : Type)
  [triangle : ∀ ⦃α β γ⦄, α ≠ β ∧ β ≠ γ ∧ γ ≠ α]
  (angle_bisector : ∀ ⦃P Q R T : Type⦄, Type)
  (AL1_bisects_BAC : angle_bisector A B C L1)
  (CL3_bisects_ACB : angle_bisector C A B L3)
  (equal_bisectors : ∀ {D E : Type}, angle_bisector D E A L1 → angle_bisector D E C L3 → AL1 = L3)
  (AB AC : Type)
  (equal_lengths : AB ≠ AC ∧ AL1 = L3):
  AB = AC :=
by
  sorry

end isosceles_of_equal_bisectors_l638_638733


namespace problem1_problem2_problem3_l638_638292
noncomputable def root1 := 11
noncomputable def root2 := -13

theorem problem1 : (root1 + 1)^2 - 144 = 0 ∧ (root2 + 1)^2 - 144 = 0 :=
by
  split
  -- Proof for root1
  { calc
    (root1 + 1)^2 - 144 = (11 + 1)^2 - 144 : by rw [root1]
    ... = 12^2 - 144 : by norm_num
    ... = 144 - 144 : by norm_num
    ... = 0 : by norm_num }
  -- Proof for root2
  { calc
    (root2 + 1)^2 - 144 = (-13 + 1)^2 - 144 : by rw [root2]
    ... = (-12)^2 - 144 : by norm_num
    ... = 144 - 144 : by norm_num
    ... = 0 : by norm_num }

noncomputable def root3 := 1
noncomputable def root4 := 3

theorem problem2 : root3^2 - 4*root3 + 3 = 0 ∧ root4^2 - 4*root4 + 3 = 0 :=
by
  split
  -- Proof for root3
  { calc
    root3^2 - 4*root3 + 3 = 1^2 - 4*1 + 3 : by rw [root3]
    ... = 1 - 4 + 3 : by norm_num
    ... = 0 : by norm_num }
  -- Proof for root4
  { calc
    root4^2 - 4*root4 + 3 = 3^2 - 4*3 + 3 : by rw [root4]
    ... = 9 - 12 + 3 : by norm_num
    ... = 0 : by norm_num }

noncomputable def root5 := (-5 + Real.sqrt 21) / 2
noncomputable def root6 := (-5 - Real.sqrt 21) / 2

theorem problem3 : root5^2 + 5*root5 + 1 = 0 ∧ root6^2 + 5*root6 + 1 = 0 :=
by
  split
  -- Proof for root5
  { sorry } -- Proof details skipped
  -- Proof for root6
  { sorry } -- Proof details skipped

end problem1_problem2_problem3_l638_638292


namespace diff_x_y_l638_638629

theorem diff_x_y (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 :=
sorry

end diff_x_y_l638_638629


namespace terminal_point_of_angle_l638_638940

theorem terminal_point_of_angle {α : ℝ} (h1 : Real.sin α = - (3 / 5)) (h2 : Real.cos α = 4 / 5) :
  (∃ k > 0, (4 * k, -3 * k) = (4, -3)) :=
by
  use 1
  simp
  sorry

end terminal_point_of_angle_l638_638940


namespace sequence_value_l638_638658

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 3) ∧ (11 - 5 = 6) ∧ (20 - 11 = 9) ∧ (x - 20 = 12) → x = 32 := 
by intros; sorry

end sequence_value_l638_638658


namespace solve_for_x_l638_638111

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l638_638111


namespace eq_sets_M_N_l638_638329

def setM : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def setN : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem eq_sets_M_N : setM = setN := by
  sorry

end eq_sets_M_N_l638_638329


namespace glass_volume_l638_638456

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638456


namespace max_elements_in_S_is_23_l638_638244

-- Define the problem
def is_subset (S : set ℕ) : Prop :=
  ∀ x ∈ S, x ≥ 1 ∧ x ≤ 50

def no_pair_sum_divisible_by_7 (S : set ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → ¬((x + y) % 7 = 0)

-- State the theorem
theorem max_elements_in_S_is_23
  (S : set ℕ) (h1 : is_subset S) (h2 : no_pair_sum_divisible_by_7 S) :
  ∃ T : set ℕ, T ⊆ S ∧ set.card T = 23 :=
sorry

end max_elements_in_S_is_23_l638_638244


namespace not_divisible_by_4_8_16_32_l638_638680

def x := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬ (x % 4 = 0) ∧ ¬ (x % 8 = 0) ∧ ¬ (x % 16 = 0) ∧ ¬ (x % 32 = 0) := 
by 
  sorry

end not_divisible_by_4_8_16_32_l638_638680


namespace isosceles_if_angle_bisectors_eq_l638_638731

theorem isosceles_if_angle_bisectors_eq (A B C P Q : Type) 
  [is_triangle A B C] 
  [angle_bisector B A P] 
  [angle_bisector C A Q] 
  (h1 : (BP : ℝ) = (CQ : ℝ)) : 
  AB = AC :=
begin
  sorry
end

end isosceles_if_angle_bisectors_eq_l638_638731


namespace inequality_bc_gt_ac_l638_638554

theorem inequality_bc_gt_ac (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c + b + a = 0) : bc > ac :=
  sorry

end inequality_bc_gt_ac_l638_638554


namespace max_value_of_expression_l638_638684

open Complex

theorem max_value_of_expression (α β : ℂ) (hβ : complex.abs β = 2) (hαβ : conj(α) * β ≠ 1) :
  ∃ M : ℝ, (∀ (α β : ℂ), complex.abs β = 2 ∧ conj(α) * β ≠ 1 → complex.abs ((β - α) / (2 - conj(α) * β)) ≤ M) ∧ M = 1 / 4 :=
sorry

end max_value_of_expression_l638_638684


namespace roulette_P2007_gt_P2008_l638_638874

-- Define the roulette probability function based on the given conditions
noncomputable def roulette_probability : ℕ → ℝ
| 0 := 1
| n := (1 / 2007) * (List.foldl (λ acc k, acc + roulette_probability (n - k)) 0 (List.range 2007))

-- Define the theorem to prove P_{2007} > P_{2008}
theorem roulette_P2007_gt_P2008 : roulette_probability 2007 > roulette_probability 2008 :=
sorry

end roulette_P2007_gt_P2008_l638_638874


namespace possible_row_lengths_l638_638412

theorem possible_row_lengths (x : ℕ) (hx : 6 ≤ x ∧ x ≤ 25) (h : x ∣ 90) : 
  x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15 ∨ x = 18 :=
by
  sorry

example : 
  finset.card (finset.filter (λ x, 6 ≤ x ∧ x ≤ 25) (finset.divisors 90)) = 5 :=
by
  sorry

end possible_row_lengths_l638_638412


namespace sqrt_a_minus_b_squared_eq_one_l638_638726

noncomputable def PointInThirdQuadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b < 0

noncomputable def DistanceToYAxis (a : ℝ) : Prop :=
  abs a = 5

noncomputable def BCondition (b : ℝ) : Prop :=
  abs (b + 1) = 3

theorem sqrt_a_minus_b_squared_eq_one
    (a b : ℝ)
    (h1 : PointInThirdQuadrant a b)
    (h2 : DistanceToYAxis a)
    (h3 : BCondition b) :
    Real.sqrt ((a - b) ^ 2) = 1 := 
  sorry

end sqrt_a_minus_b_squared_eq_one_l638_638726


namespace find_base_l638_638617

theorem find_base (b : ℕ) (h : (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 2 * b + 5) : b = 7 :=
sorry

end find_base_l638_638617


namespace fixed_point_graph_l638_638930

theorem fixed_point_graph (m : ℝ) : ∀ m : ℝ, ∃ c d : ℝ, (c, d) = (-3, 45) ∧ d = 5 * c^2 + m * c + 3 * m :=
by
  use -3, 45
  split
  { refl }
  { sorry }

end fixed_point_graph_l638_638930


namespace angle_between_vectors_l638_638139

noncomputable def vector_magnitude {α : Type*} [inner_product_space ℝ α] (v : α) : ℝ :=
  real.sqrt (inner_product_space.is_R_or_C_inner_self ℝ v)

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors {a b : V} (ha : vector_magnitude a ≠ 0) (hb : vector_magnitude b ≠ 0)
  (h : vector_magnitude a = vector_magnitude b) :
  inner_product_space.angle (a + b) (a - b) = real.pi / 2 :=
begin
  -- The detailed proof for this theorem can be added here
  sorry
end

end angle_between_vectors_l638_638139


namespace original_employees_l638_638855

theorem original_employees (X : ℝ) (h1 : 0.725 * X = 3223) : X ≈ 4446 :=
by sorry

end original_employees_l638_638855


namespace factor_expression_l638_638089

variable (x y : ℝ)

theorem factor_expression : 3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := 
by 
  sorry

end factor_expression_l638_638089


namespace tank_capacity_l638_638364

theorem tank_capacity (C : ℝ) :
  (C / 10 - 960 = C / 18) → C = 21600 := by
  intro h
  sorry

end tank_capacity_l638_638364


namespace surface_integral_solution_l638_638108

-- Definitions of the functions φ and ψ
def φ (x y z : ℝ) : ℝ := x^2 + y^2 + x + z
def ψ (x y z : ℝ) : ℝ := x^2 + y^2 + 2 * z + x

-- Surface and volume specifications
def Σ (R H : ℝ) : Set (ℝ × ℝ × ℝ) := 
  {p | (∃ (z : ℝ), (z = 0 ∨ z = H) ∧ (p.1^2 + p.2^2 = R^2))}

def V (R H : ℝ) : Set (ℝ × ℝ × ℝ) := 
  {p | p.1^2 + p.2^2 ≤ R^2 ∧ 0 ≤ p.3 ∧ p.3 ≤ H}

-- Statement of Green's identity application and the result
theorem surface_integral_solution (R H : ℝ) (hR : R > 0) (hH : H > 0) :
  let I := ∫ x in Σ R H, (φ x.1 x.2 x.3 * (∂ ψ / ∂ n) - ψ x.1 x.2 x.3 * (∂ φ / ∂ n)) in
  I = -2 * π * R^2 * H^2 := 
by
  -- The detailed proof steps will be filled in here.
  sorry

end surface_integral_solution_l638_638108


namespace blocks_needed_l638_638487

-- Definitions and conditions
def volume_of_rectangular_prism (l w h : ℝ) : ℝ := l * w * h
def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def round_up (n : ℝ) : ℕ := if n.ceil = n then n.to_nat else n.ceil.to_nat

-- Given conditions for the block dimensions
def length := 8
def width := 3
def height := 1.5

-- Given conditions for the sculpture's dimensions
def sculpture_diameter := 5
def sculpture_radius := sculpture_diameter / 2
def sculpture_height := 9

-- Volumes
def block_volume := volume_of_rectangular_prism length width height
def sculpture_volume := volume_of_cylinder sculpture_radius sculpture_height

-- Number of blocks required
def num_blocks := round_up (sculpture_volume / block_volume)

-- Prove the number of blocks needed
theorem blocks_needed : num_blocks = 5 := by
  sorry

end blocks_needed_l638_638487


namespace roots_range_l638_638776

theorem roots_range (b : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + b = 0 → 0 < x) ↔ 0 < b ∧ b ≤ 1 :=
sorry

end roots_range_l638_638776


namespace productivity_difference_l638_638380

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l638_638380


namespace problem_statement_l638_638140

theorem problem_statement (n : ℕ) (a b : Fin n → ℝ) 
  (h1 : (∑ i : Fin n, (a i) ^ 2) ≤ 1) 
  (h2 : (∑ i : Fin n, (b i) ^ 2) ≤ 1) : 
  (1 - ∑ i : Fin n, a i * b i) ^ 2 ≥ (1 - ∑ i : Fin n, (a i) ^ 2) * (1 - ∑ i : Fin n, (b i) ^ 2) := 
by
  sorry

end problem_statement_l638_638140


namespace John_Mary_chickens_diff_l638_638302

variable (Ray_chickens : ℕ)
variable (John_Ray_diff : ℕ)
variable (Ray_Mary_diff : ℕ)
variable (John_Mary_diff : ℕ)

-- Define the given conditions
def Ray_chickens_eq : Prop := Ray_chickens = 10
def John_Ray_eq : Prop := ∀ R, Ray_chickens = R → John_Ray_diff = R + 11
def Ray_Mary_eq : Prop := ∀ R M, Ray_chickens = R → M = R + 6 → ∃ J, John_Mary_diff = (R + 11) - M

-- The final statement to prove John took 5 more chickens than Mary
theorem John_Mary_chickens_diff (h1 : Ray_chickens_eq) (h2 : John_Ray_eq Ray_chickens) (h3 : Ray_Mary_eq Ray_chickens (Ray_chickens + 6)) :
  John_Mary_diff = 5 :=
by
  sorry

end John_Mary_chickens_diff_l638_638302


namespace isosceles_triangle_base_angles_l638_638645

theorem isosceles_triangle_base_angles (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B ∨ B = C ∨ C = A) (h₃ : A = 80 ∨ B = 80 ∨ C = 80) :
  A = 50 ∨ B = 50 ∨ C = 50 ∨ A = 80 ∨ B = 80 ∨ C = 80 := 
by
  sorry

end isosceles_triangle_base_angles_l638_638645


namespace solve_for_a_l638_638765

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a * x

theorem solve_for_a (a : ℝ) :
  f (f 0 a) a = 4 * a → a = 2 :=
by
  sorry

end solve_for_a_l638_638765


namespace cyclic_quadrilateral_l638_638339

theorem cyclic_quadrilateral {α : Type*} [euclidean_geometry α]
  (A B K L M N P : α) (c₁ c₂ : circle α) :
  P ∈ (chord A B) →
  P ∈ c₁ ∧ P ∈ c₂ →
  is_chord P K M c₁ →
  is_chord P L N c₂ →
  is_cyclic_quad K L M N :=
sorry

end cyclic_quadrilateral_l638_638339


namespace number_difference_l638_638031

theorem number_difference
  (n : ℕ)
  (h₁ : n = 15)
  (h₂ : n * 13 = n + 180) :
  (n * 13) - n = 180 :=
by
  rw h₁ at h₂
  rw h₁
  rw h₂
  simp
  sorry

end number_difference_l638_638031


namespace small_rectangular_prisms_intersect_diagonal_l638_638957

def lcm (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

def inclusion_exclusion (n : Nat) : Nat :=
  n / 2 + n / 3 + n / 5 - n / (2 * 3) - n / (3 * 5) - n / (5 * 2) + n / (2 * 3 * 5)

theorem small_rectangular_prisms_intersect_diagonal :
  ∀ (a b c : Nat) (L : Nat), a = 2 → b = 3 → c = 5 → L = 90 →
  lcm a b c = 30 → 3 * inclusion_exclusion (lcm a b c) = 66 :=
by
  intros
  sorry

end small_rectangular_prisms_intersect_diagonal_l638_638957


namespace max_value_of_expression_l638_638522

theorem max_value_of_expression (x : ℝ) : 
  let expression := 11 - 8 * Real.cos x - 2 * (Real.sin x)^2
  in ∃ b, b = 19 ∧ ∀ a, a = expression → a ≤ b := by
  sorry

end max_value_of_expression_l638_638522


namespace welders_that_left_first_day_l638_638405

-- Definitions of conditions
def welders := 12
def days_to_complete_order := 3
def days_remaining_work_after_first_day := 8
def work_done_first_day (r : ℝ) := welders * r * 1
def total_work (r : ℝ) := welders * r * days_to_complete_order

-- Theorem statement
theorem welders_that_left_first_day (r : ℝ) : 
  ∃ x : ℝ, 
    (welders - x) * r * days_remaining_work_after_first_day = total_work r - work_done_first_day r 
    ∧ x = 9 :=
by
  sorry

end welders_that_left_first_day_l638_638405


namespace angies_age_l638_638798

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l638_638798


namespace disinfectant_prices_min_indoor_disinfectant_l638_638083

theorem disinfectant_prices :
  ∃ x y : ℕ, y - x = 30 ∧ 2 * x + 3 * y = 340 ∧ x = 50 ∧ y = 80 :=
by
  existsi 50
  existsi 80
  split
  sorry
  split
  sorry
  split
  refl
  refl

theorem min_indoor_disinfectant (m : ℕ) :
  50 * m + 80 * (200 - m) ≤ 14000 → m ≥ 67 :=
by 
  intro h
  have h1 : 50 * m + 16000 - 80 * m ≤ 14000 := by 
    linarith
  have h2 : -30 * m ≤ -2000 := by 
    linarith
  have h3 : m ≥ 2000 / 30 := by 
    linarith
  have h4 : 2000 / 30 = 66 + 2 / 3 := by
    norm_num
  linarith

end disinfectant_prices_min_indoor_disinfectant_l638_638083


namespace tamika_greater_probability_l638_638300

noncomputable def tamika_results : set ℕ :=
  {19, 20, 21}

noncomputable def carlos_results : set ℕ :=
  {5, 11, 25}

def favorable_pairs_count : ℕ :=
  finset.card (finset.filter (λ (p : ℕ × ℕ), p.1 > p.2)
    (finset.product (tamika_results.to_finset) (carlos_results.to_finset)))

def total_pairs_count : ℕ :=
  finset.card (finset.product (tamika_results.to_finset) (carlos_results.to_finset))

def probability_tamika_greater : ℚ :=
  favorable_pairs_count / total_pairs_count

theorem tamika_greater_probability : probability_tamika_greater = 7 / 9 :=
by
  -- Proof goes here
  sorry

end tamika_greater_probability_l638_638300


namespace max_area_of_triangle_l638_638322

-- Define the parameters and the conditions
variables {AB BC AC : ℝ} (B : ℝ)
def area_of_triangle (s : ℝ) := 1 / 2 * AB * BC * Real.sin B

-- Given conditions
axiom AB_is_2 : AB = 2
axiom AC_is_sqrt3_BC : AC = Real.sqrt 3 * BC

-- Define the maximum area of the triangle
noncomputable def max_area_triangle_ABC (BC : ℝ) :=
  let sin_B := Real.sqrt (1 - (Real.cos B)^2) in
  area_of_triangle 2 BC sin_B

-- Statement: Given the conditions, the maximum area of the triangle is sqrt(3)
theorem max_area_of_triangle : ∀ B BC, AB = 2 → AC = Real.sqrt 3 * BC →
  max_area_triangle_ABC BC = Real.sqrt 3 :=
begin
  sorry
end

end max_area_of_triangle_l638_638322


namespace triangle_area_ellipse_l638_638978

open Real

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

theorem triangle_area_ellipse 
  (a : ℝ) (b : ℝ) 
  (h1 : a = sqrt 2) (h2 : b = 1) 
  (F1 F2 : ℝ × ℝ) 
  (hfoci : ellipse_foci a b = (F1, F2))
  (hF2 : F2 = (sqrt 3, 0))
  (A B : ℝ × ℝ)
  (hA : A = (0, -1))
  (hB : B = (0, -1))
  (h_inclination : ∃ θ, θ = pi / 4 ∧ (B.1 - A.1) / (B.2 - A.2) = tan θ) :
  F1 = (-sqrt 3, 0) → 
  1/2 * (B.1 - A.1) * (B.2 - A.2) = 4/3 :=
sorry

end triangle_area_ellipse_l638_638978


namespace unique_students_count_l638_638787

theorem unique_students_count
  (O B C J : ℕ) -- Number of students in Orchestra, Band, Choir, Jazz Ensemble
  (OB OC BC BJ OJ CJ OBC OBCJ : ℕ) -- Number of overlapping students
  (hO : O = 25)
  (hB : B = 40)
  (hC : C = 30)
  (hJ : J = 15)
  (hOB : OB = 5)
  (hOC : OC = 6)
  (hBC : BC = 4)
  (hBJ : BJ = 3)
  (hOJ : OJ = 2)
  (hCJ : CJ = 4)
  (hOBC : OBC = 3)
  (hOBCJ : OBCJ = 1) : 
  (O + B + C + J - OB - OC - BC - BJ - OJ - CJ + OBC + OBCJ = 90) :=
begin
  sorry
end

end unique_students_count_l638_638787


namespace optimal_messenger_strategy_l638_638275

variable (p : ℝ) (h : 0 < p ∧ p < 1)

theorem optimal_messenger_strategy :
  if p < 1/3 then
    (sending_four_messengers : true) -- Stand-in for definition that checks probability computation for four messengers
  else if 1/3 ≤ p then
    (sending_two_messengers : true) -- Stand-in for definition that checks probability computation for two messengers
  else
    false :=
by
  sorry

end optimal_messenger_strategy_l638_638275


namespace arithmetic_sequence_problem_l638_638212

theorem arithmetic_sequence_problem 
  (a : ℕ → ℚ) 
  (a1 : a 1 = 1 / 3) 
  (a2_a5 : a 2 + a 5 = 4) 
  (an : ∃ n, a n = 33) :
  ∃ n, a n = 33 ∧ n = 50 := 
by 
  sorry

end arithmetic_sequence_problem_l638_638212


namespace Felicity_family_store_visits_l638_638924

theorem Felicity_family_store_visits
  (lollipop_stick : ℕ := 1)
  (fort_total_sticks : ℕ := 400)
  (fort_completion_percent : ℕ := 60)
  (weeks_collected : ℕ := 80)
  (sticks_collected : ℕ := (fort_total_sticks * fort_completion_percent) / 100)
  (store_visits_per_week : ℕ := sticks_collected / weeks_collected) :
  store_visits_per_week = 3 := by
  sorry

end Felicity_family_store_visits_l638_638924


namespace probability_of_six_event_l638_638367

-- The faces of the die are represented as natural numbers from 1 to 6.
-- The outcomes of the three rolls are represented by a tuple (a1, a2, a3).

noncomputable def probability_event : ℝ :=
  let outcomes := {(a1, a2, a3) | a1, a2, a3 ∈ {1, 2, 3, 4, 5, 6}} in
  let favorable_event := {(a1, a2, a3) ∈ outcomes | 
                          abs (a1 - a2) + abs (a2 - a3) + abs (a3 - a1) = 6} in
  (favorable_event.to_finset.card : ℝ) / (outcomes.to_finset.card : ℝ)

-- The desired probability is proved as follows:
theorem probability_of_six_event : probability_event = 1/4 := 
by sorry

end probability_of_six_event_l638_638367


namespace average_age_is_35_l638_638225

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l638_638225


namespace transform_g_neg_abs_l638_638907

def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then -2 * x
  else if 0 < x ∧ x ≤ 2 then x^2 - 4
  else 0 -- This handles cases not specified in the problem

theorem transform_g_neg_abs (x : ℝ) :
  g (-|x|) = if x ≤ 0 then 2 * x else x^2 - 4 :=
by
  sorry

end transform_g_neg_abs_l638_638907


namespace determine_g_2023_l638_638693

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined_for_all_positive_real_numbers : ∀ x : ℝ, x > 0 → g x = g x
axiom g_positive : ∀ x : ℝ, x > 0 → g x > 0
axiom g_functional_equation : ∀ x y : ℝ, x > y → y > 0 → g (x - y) = sqrt (g (x * y) + 3)

theorem determine_g_2023 : g 2023 = 3 :=
by
  sorry

end determine_g_2023_l638_638693


namespace third_pipe_empty_time_l638_638805

theorem third_pipe_empty_time :
  let A_rate := 1/60
  let B_rate := 1/75
  let combined_rate := 1/50
  let third_pipe_rate := combined_rate - (A_rate + B_rate)
  let time_to_empty := 1 / third_pipe_rate
  time_to_empty = 100 :=
by
  sorry

end third_pipe_empty_time_l638_638805


namespace factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l638_638534

theorem factorize_x3_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

theorem factorize_a3b_minus_2a2b_plus_ab (a b : ℝ) : a^3 * b - 2 * a^2 * b + a * b = a * b * (a - 1)^2 :=
sorry

end factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l638_638534


namespace pet_shop_new_total_l638_638032

theorem pet_shop_new_total : 
  ∃ (kittens hamsters birds puppies : ℕ),
    kittens = 3 * (hamsters / 2) ∧
    birds = hamsters + 30 ∧
    puppies = birds / 4 ∧
    150 = kittens + hamsters + birds + puppies ∧
    5 ≤ puppies ∧
    let new_puppies := puppies - 5 in
    let new_birds := birds + 10 in
    155 = kittens + hamsters + new_birds + new_puppies :=
by
  sorry

end pet_shop_new_total_l638_638032


namespace heaviest_lightest_difference_total_deviation_total_selling_price_l638_638786

-- Define the standard weight and deviations
def standard_weight : ℕ := 25
def deviations : List ℚ := [1.5, -3, 2, -0.5, 1, -2, 2, -1.5, 1, 2.5]
def price_per_kg : ℕ := 3
def num_baskets : ℕ := 10

-- Question 1: Prove the difference between heaviest and lightest basket
theorem heaviest_lightest_difference :
  let max_excess := deviations.maximum -- Find the maximum deviation
  let min_excess := deviations.minimum -- Find the minimum deviation
  max_excess - min_excess = (5.5 : ℚ) := sorry

-- Question 2: Prove the total deviation from the standard weight
theorem total_deviation :
  deviations.sum = (3 : ℚ) := sorry

-- Question 3: Prove the total selling price for the 10 baskets
theorem total_selling_price :
  ((standard_weight * num_baskets) + deviations.sum) * price_per_kg = (759 : ℚ) := sorry

end heaviest_lightest_difference_total_deviation_total_selling_price_l638_638786


namespace unique_midpoint_of_isosceles_right_triangle_l638_638644

-- Define the isosceles right triangle and conditions
structure IsoscelesRightTriangle (A B C : Type) :=
(ab : ℝ)
(ac : ℝ)
(bc : ℝ)
(m_ab_ac : ab = ac)

-- Lean 4 statement for the proof problem
theorem unique_midpoint_of_isosceles_right_triangle (A B C P: Type) 
    [isosceles_rt_abc : IsoscelesRightTriangle A B C] 
    (is_on_side : (P = midpoint B C) ∨ (P = midpoint A C) ∨ (P = midpoint A B)) :
    ∃! (P : Type), rtriangle P A B ∧ rtriangle P A C :=
by
    sorry

end unique_midpoint_of_isosceles_right_triangle_l638_638644


namespace work_completion_time_l638_638847

theorem work_completion_time (days_B days_C days_all : ℝ) (h_B : days_B = 5) (h_C : days_C = 12) (h_all : days_all = 2.2222222222222223) : 
    (1 / ((days_all / 9) * 10) - 1 / days_B - 1 / days_C)⁻¹ = 60 / 37 := by 
  sorry

end work_completion_time_l638_638847


namespace min_value_2x_plus_y_l638_638570

theorem min_value_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2y - 2 * x * y = 0) : 
  2 * x + y ≥ 9 / 2 :=
sorry

end min_value_2x_plus_y_l638_638570


namespace jen_age_when_son_born_l638_638672

theorem jen_age_when_son_born (S : ℕ) (Jen_present_age : ℕ) 
  (h1 : S = 16) (h2 : Jen_present_age = 3 * S - 7) : 
  Jen_present_age - S = 25 :=
by {
  sorry -- Proof would be here, but it is not required as per the instructions.
}

end jen_age_when_son_born_l638_638672


namespace number_of_possible_values_b_l638_638297

theorem number_of_possible_values_b : 
  ∃ n : ℕ, n = 2 ∧ 
    (∀ b : ℕ, b ≥ 2 → (b^3 ≤ 256) ∧ (256 < b^4) ↔ (b = 5 ∨ b = 6)) :=
by {
  sorry
}

end number_of_possible_values_b_l638_638297


namespace find_linear_function_l638_638129

-- Define a function f that is linear
variable {f : ℝ → ℝ}

-- Define the condition that f is a linear function
axiom linear_f (x : ℝ) : f x = k * x + b

-- Given condition f[f(x)] = x + 2
axiom ff_condition (x : ℝ) : f (f x) = x + 2

-- Statement to prove
theorem find_linear_function (x k b : ℝ) (h1 : linear_f x) (h2 : ff_condition x)
  : f x = x + 1 :=
sorry

end find_linear_function_l638_638129


namespace integer_for_all_n_l638_638260

theorem integer_for_all_n (x y : ℝ) (h_xy : x ≠ y) 
  (h_consecutive : ∃ n, ∀ k ∈ {n, n+1, n+2, n+3}, 
                   (x^k - y^k) / (x - y) ∈ ℤ) : 
  ∀ n, (x^n - y^n) / (x - y) ∈ ℤ := 
sorry

end integer_for_all_n_l638_638260


namespace base7_calculation_result_l638_638887

-- Define the base 7 addition and multiplication
def base7_add (a b : ℕ) := (a + b)
def base7_mul (a b : ℕ) := (a * b)

-- Represent the given numbers in base 10 for calculations:
def num1 : ℕ := 2 * 7 + 5 -- 25 in base 7
def num2 : ℕ := 3 * 7^2 + 3 * 7 + 4 -- 334 in base 7
def mul_factor : ℕ := 2 -- 2 in base 7

-- Addition result
def sum : ℕ := base7_add num1 num2

-- Multiplication result
def result : ℕ := base7_mul sum mul_factor

-- Proving the result is equal to the final answer in base 7
theorem base7_calculation_result : result = 6 * 7^2 + 6 * 7 + 4 := 
by sorry

end base7_calculation_result_l638_638887


namespace smallest_marbles_l638_638420

theorem smallest_marbles
  : ∃ n : ℕ, ((n % 8 = 5) ∧ (n % 7 = 2) ∧ (n = 37) ∧ (37 % 9 = 1)) :=
by
  sorry

end smallest_marbles_l638_638420


namespace number_of_middle_managers_selected_l638_638029

-- Definitions based on conditions
def total_employees := 1000
def senior_managers := 50
def middle_managers := 150
def general_staff := 800
def survey_size := 200

-- Proposition to state the question and correct answer formally
theorem number_of_middle_managers_selected:
  200 * (150 / 1000) = 30 :=
by
  sorry

end number_of_middle_managers_selected_l638_638029


namespace complex_distance_l638_638712

open Complex

theorem complex_distance (z1 z2 : ℂ) (h1 : z1 = 2 + 3 * Complex.i) (h2 : z2 = -2 + 2 * Complex.i) :
  Complex.abs (z1 - z2) = Real.sqrt 17 := by
  sorry

end complex_distance_l638_638712


namespace s_parity_diff_l638_638546

-- Definitions based on the conditions given in the problem
def s (n : ℕ) : ℕ := (n.digits 10).sum

def P (n : ℕ) (a : Finₓ n.succ → ℕ) (k : ℕ) : ℕ :=
  Finₓ.toList (Finₓ.range n.succ).reverse.foldl (+) 0 (λ i, a i * k ^ i)

theorem s_parity_diff (n : ℕ) (a : Finₓ n.succ → ℕ) (hn : 2 ≤ n)
  (ha : ∀ i, 0 ≤ i → i ≤ n → 0 < a i) :
  ∃ k : ℕ, s k % 2 ≠ s (P n a k) % 2 :=
by
  sorry

end s_parity_diff_l638_638546


namespace glass_volume_correct_l638_638465

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638465


namespace pqrs_product_l638_638578

theorem pqrs_product :
  let P := sqrt 2010 + sqrt 2011
  let Q := - (sqrt 2010 + sqrt 2011)
  let R := sqrt 2010 - sqrt 2011
  let S := sqrt 2011 - sqrt 2010
  P * Q * R * S = 1 :=
by
  sorry

end pqrs_product_l638_638578


namespace productivity_comparison_l638_638373

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l638_638373


namespace count_11_digit_palindromes_l638_638183

-- Define the set of digits to work with
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 2, 4, 4, 5, 5, 5}

-- Non-computable theorem to state the number of 11-digit palindromes
noncomputable def eleven_digit_palindromes : ℕ :=
  Multiset.permutations digits.count 11

theorem count_11_digit_palindromes (h : eleven_digit_palindromes digits = 300) :
  eleven_digit_palindromes digits = 300 :=
sorry

end count_11_digit_palindromes_l638_638183


namespace coloring_satisfies_conditions_l638_638718

-- Define lattice point on the coordinate plane
structure LatticePoint where
  x : Int
  y : Int

-- Define colors
inductive Color
| white
| red
| black

-- Define the coloring function
def color (p : LatticePoint) : Color :=
  if (p.x + p.y) % 2 = 0 then Color.red
  else if p.x % 2 = 1 ∧ p.y % 2 = 0 then Color.white
  else Color.black

-- Define conditions for the problem
def condition1 : Prop :=
  (∃ k : Int, ∀ m n : Int, color ⟨2 * m, 2 * n⟩ = Color.red) ∧
  (∃ k : Int, ∀ m n : Int, color ⟨2 * m + 1, 2 * n⟩ = Color.white) ∧
  (∃ k : Int, ∀ m n : Int, color ⟨2 * m, 2 * n + 1⟩ = Color.black)

def forms_parallelogram (a b c d : LatticePoint) : Prop :=
  let midpoint (p q : LatticePoint) : LatticePoint :=
    ⟨(p.x + q.x) / 2, (p.y + q.y) / 2⟩ in
  midpoint a c = midpoint b d

def condition2 : Prop :=
  ∀ (a b c : LatticePoint), 
    color a = Color.white ∧ color b = Color.red ∧ color c = Color.black →
    ∃ d : LatticePoint, color d = Color.red ∧ forms_parallelogram a b c d

-- The main theorem statement
theorem coloring_satisfies_conditions : condition1 ∧ condition2 :=
  by
    sorry  -- Proof omitted

end coloring_satisfies_conditions_l638_638718


namespace range_of_a_min_value_a_plus_4_over_a_sq_l638_638159

noncomputable def f (x : ℝ) : ℝ :=
  |x - 10| + |x - 20|

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < 10 * a + 10) ↔ 0 < a :=
sorry

theorem min_value_a_plus_4_over_a_sq (a : ℝ) (h : 0 < a) :
  ∃ y : ℝ, a + 4 / a ^ 2 = y ∧ y = 3 :=
sorry

end range_of_a_min_value_a_plus_4_over_a_sq_l638_638159


namespace least_even_perimeter_l638_638771

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

theorem least_even_perimeter
  (a b : ℕ) (h1 : a = 24) (h2 : b = 37) (c : ℕ)
  (h3 : c > b) (h4 : a + b > c)
  (h5 : ∃ k : ℕ, k * 2 = triangle_perimeter a b c) :
  triangle_perimeter a b c = 100 :=
sorry

end least_even_perimeter_l638_638771


namespace systematic_sampling_example_l638_638739

noncomputable def systematic_sampling (start interval count : ℕ) : list ℕ :=
list.map (λ i, start + i * interval) (list.range count)

theorem systematic_sampling_example :
  ∃ start, start ∈ finset.range 11 ∧ systematic_sampling start 10 5 = [5, 15, 25, 35, 45] :=
begin
  sorry
end

end systematic_sampling_example_l638_638739


namespace cost_per_chicken_l638_638341

theorem cost_per_chicken (acres_cost : ℕ) (house_cost : ℕ) (cow_num : ℕ) (cow_cost : ℕ) 
  (install_hours : ℕ) (install_cost_per_hour : ℕ) (equipment_cost : ℕ) (total_cost : ℕ) 
  (chicken_num : ℕ) (chicken_cost_total : ℕ) : 
  (acres_cost = 30 * 20) →
  (house_cost = 120000) →
  (cow_num = 20) →
  (cow_cost = 1000) →
  (install_cost_per_hour = 100) →
  (install_hours = 6) →
  (equipment_cost = 6000) →
  (total_cost = 147700) →
  (chicken_num = 100) →
  acres_cost + house_cost + (cow_num * cow_cost) + (install_hours * install_cost_per_hour + equipment_cost) + chicken_cost_total = total_cost →
  (chicken_cost_total / chicken_num = 5) :=
begin
  intros,
  sorry
end

end cost_per_chicken_l638_638341


namespace P_2007_greater_P_2008_l638_638877

noncomputable def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2008) * ∑ k in finset.range 2008, P (n - k)

theorem P_2007_greater_P_2008 : P 2007 > P 2008 := 
sorry

end P_2007_greater_P_2008_l638_638877


namespace complex_number_coordinates_l638_638152

open Complex

-- Define the complex numbers and conditions
def z1 (a : ℝ) : ℂ := a + Complex.i
def z2 : ℂ := 1 - Complex.i

/-- Given that the fraction z1 / z2 is purely imaginary, prove that the coordinates of z1 in the complex plane are (1, 1) -/
theorem complex_number_coordinates (a : ℝ) (h : (z1 a) / z2).re = 0 :
  a = 1 ∧ (z1 a).re = 1 ∧ (z1 a).im = 1 := by
  sorry

end complex_number_coordinates_l638_638152


namespace P2007_gt_P2008_l638_638867

namespace ProbabilityProblem

def probability (k : ℕ) : ℝ := sorry  -- Placeholder for the probability function

axiom probability_rec :
  ∀ n, probability n = (1 / 2007) * (∑ k in finset.range 2007, probability (n - (k + 1)))

axiom P0 :
  probability 0 = 1

theorem P2007_gt_P2008 : probability 2007 > probability 2008 := sorry

end ProbabilityProblem

end P2007_gt_P2008_l638_638867


namespace glass_volume_l638_638425

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638425


namespace range_of_a_l638_638603

variable (x a : ℝ)

def p : Prop := |x - a| < 4
def q : Prop := (x - 2) * (3 - x) > 0

theorem range_of_a (ha : ¬p → ¬q) : -1 ≤ a ∧ a ≤ 6 := by
  sorry

end range_of_a_l638_638603


namespace mean_of_solutions_l638_638540

theorem mean_of_solutions :
  let p := (λ x : ℝ, x^3 + 6 * x^2 - 13 * x)
  let roots := {x | p x = 0} 
  ∃ a b c : ℝ, roots = {a, b, c} ∧ (a + b + c) / 3 = -2 :=
by
  sorry

end mean_of_solutions_l638_638540


namespace group_placement_l638_638647

theorem group_placement :
  let men := 3,
      women := 4,
      total_ways := 3 * 6 * 2
  in
  total_ways = 36 := by
  sorry

end group_placement_l638_638647


namespace solution_pairs_l638_638536

theorem solution_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
    (h_coprime: Nat.gcd (2 * a - 1) (2 * b + 1) = 1) 
    (h_divides : (a + b) ∣ (4 * a * b + 1)) :
    ∃ n : ℕ, a = n ∧ b = n + 1 :=
by
  -- statement
  sorry

end solution_pairs_l638_638536


namespace solve_for_x_l638_638110

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l638_638110


namespace angies_age_l638_638797

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l638_638797


namespace productivity_comparison_l638_638377

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l638_638377


namespace n_cubed_minus_n_plus_one_is_square_l638_638241

theorem n_cubed_minus_n_plus_one_is_square 
  (n : ℕ) (h : (∃ d : ℕ, nat.factors (n^5 + n^4 + 1) = d :: nil ∧ d = 6)) : 
  ∃ k : ℕ, n^3 - n + 1 = k^2 := 
by
  sorry

end n_cubed_minus_n_plus_one_is_square_l638_638241


namespace average_age_of_John_Mary_Tonya_is_35_l638_638232

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l638_638232


namespace angle_between_vectors_is_30_degrees_l638_638304

-- Define the conditions of the problem
variable (a : ℝ) (h : 0 < a)

-- Points in the coordinate system
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (a / 2, - (Real.sqrt 3) / 2 * a, 0)
def A1 : ℝ × ℝ × ℝ := (0, 0, Real.sqrt 2 * a)
def C1 : ℝ × ℝ × ℝ := (a, 0, Real.sqrt 2 * a)

-- Define the vector from A to D (midpoint of A1B1) and vector from A to C1
def vec_A_D : ℝ × ℝ × ℝ := (a / 4, - (Real.sqrt 3) / 4 * a, Real.sqrt 2 * a)
def vec_A_C1 : ℝ × ℝ × ℝ := (a, 0, Real.sqrt 2 * a)

-- Define the dot product of vectors
def dot_prod (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitudes of the vectors
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Calculate the angle between two vectors
noncomputable def angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_prod v1 v2) / (magnitude v1 * magnitude v2))

-- State the theorem
theorem angle_between_vectors_is_30_degrees :
  angle (vec_A_C1 a) (vec_A_D a) = π / 6 := sorry

end angle_between_vectors_is_30_degrees_l638_638304


namespace conic_section_definition_l638_638086

noncomputable def ellipse (focus : Point) (directrix : Line) (e : ℝ) : Set Point :=
{P | e * (distance P directrix) = (distance P focus)}

noncomputable def hyperbola (focus : Point) (directrix : Line) (e : ℝ) : Set Point :=
{P | e * (distance P directrix) = (distance P focus)}

noncomputable def parabola (focus : Point) (directrix : Line) : Set Point :=
{P | (distance P directrix) = (distance P focus)}

theorem conic_section_definition
  (P : Point) (focus : Point) (directrix : Line) (e : ℝ) :
  (P ∈ ellipse focus directrix e ∨ P ∈ hyperbola focus directrix e ∨ P ∈ parabola focus directrix) ↔
  (∃ r : ℝ, r > 0 ∧ distance P focus = r * distance P directrix) := sorry

end conic_section_definition_l638_638086


namespace symmetric_point_l638_638104

theorem symmetric_point (a b m n : ℝ) (h1 : (n - b) / (m - a) = 1)
  (h2 : (m + a) / 2 + (n + b) / 2 + 1 = 0) :
  m = -b - 1 ∧ n = -a - 1 :=
begin
  sorry
end

end symmetric_point_l638_638104


namespace absolute_value_equation_solution_l638_638929

theorem absolute_value_equation_solution (x : ℝ) : |x - 30| + |x - 24| = |3 * x - 72| ↔ x = 26 :=
by sorry

end absolute_value_equation_solution_l638_638929


namespace isosceles_if_angle_bisectors_eq_l638_638730

theorem isosceles_if_angle_bisectors_eq (A B C P Q : Type) 
  [is_triangle A B C] 
  [angle_bisector B A P] 
  [angle_bisector C A Q] 
  (h1 : (BP : ℝ) = (CQ : ℝ)) : 
  AB = AC :=
begin
  sorry
end

end isosceles_if_angle_bisectors_eq_l638_638730


namespace find_y_eq_54_div_23_l638_638544

open BigOperators

theorem find_y_eq_54_div_23 (y : ℚ) (h : (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2))) = 3) : y = 54 / 23 := 
by
  sorry

end find_y_eq_54_div_23_l638_638544


namespace problem_I_problem_II_l638_638168

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

noncomputable def g (x a : ℝ) : ℝ := |x + 1| - |x - a| + a

/-- Problem (I) -/
theorem problem_I (x : ℝ) (h : g x 1 + f x < 6) : x ∈ Ioo (-4 : ℝ) 1 :=
sorry

/-- Problem (II) -/
theorem problem_II (x₁ x₂ a : ℝ) (h : ∀ x₁ x₂, f x₁ ≥ g x₂ a) : a ∈ Iic 1 :=
sorry

end problem_I_problem_II_l638_638168


namespace product_is_solution_quotient_is_solution_l638_638255

-- Definitions and conditions from the problem statement
variable (a b c d : ℤ)

-- The conditions
axiom h1 : a^2 - 5 * b^2 = 1
axiom h2 : c^2 - 5 * d^2 = 1

-- Lean 4 statement for the first part: the product
theorem product_is_solution :
  ∃ (m n : ℤ), ((m + n * (5:ℚ)) = (a + b * (5:ℚ)) * (c + d * (5:ℚ))) ∧ (m^2 - 5 * n^2 = 1) :=
sorry

-- Lean 4 statement for the second part: the quotient
theorem quotient_is_solution :
  ∃ (p q : ℤ), ((p + q * (5:ℚ)) = (a + b * (5:ℚ)) / (c + d * (5:ℚ))) ∧ (p^2 - 5 * q^2 = 1) :=
sorry

end product_is_solution_quotient_is_solution_l638_638255


namespace function_graph_is_option_A_l638_638316

noncomputable def f (x : ℝ) : ℝ := -(cos x) * log (abs x)

theorem function_graph_is_option_A :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f(x)) ∧
  (∀ k : ℤ, f (π/2 + k * π) = 0) ∧
  (∀ x : ℝ, x ≠ 0 → f x ∈ ℝ) ∧
  (f 0 = undefined) → 
  graph_of_function_fits_description_A :=
sorry

end function_graph_is_option_A_l638_638316


namespace max_xy_l638_638956

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  xy ≤ 1 / 4 := 
sorry

end max_xy_l638_638956


namespace proof_problem_l638_638572

variables (a b c : ℝ)

theorem proof_problem (h : exp (a - c) + b * exp (c + 1) ≤ a + log b + 3) :
  a = c ∧ (∀ a b c, exp (a - c) + b * exp (c + 1) ≤ a + log b + 3 -> a + b + 2 * c = -3 * log 3) := 
sorry

end proof_problem_l638_638572


namespace intersection_M_N_l638_638172

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := U \ complement_U_N

theorem intersection_M_N : M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_M_N_l638_638172


namespace productivity_difference_l638_638379

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l638_638379


namespace integral_sin_over_cos_l638_638093

noncomputable def integralResult (x : Real) : Real :=
  (1 / (4 * (cos x)^4)) - (1 / (2 * (cos x)^2))

theorem integral_sin_over_cos :
  ∃ (C : Real), ∫ x in Real, (sin x)^3 / (cos x)^5 = integralResult + C :=
by
  sorry

end integral_sin_over_cos_l638_638093


namespace total_payment_for_combined_shopping_trip_l638_638040

noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else if amount ≤ 500 then amount * 0.9
  else 500 * 0.9 + (amount - 500) * 0.7

theorem total_payment_for_combined_shopping_trip :
  discount (168 + 423 / 0.9) = 546.6 :=
by
  sorry

end total_payment_for_combined_shopping_trip_l638_638040


namespace volume_of_cone_l638_638833

noncomputable def radius := 3 -- in cm
noncomputable def height := 4 -- in cm
noncomputable def pi_approx := 3.14159

theorem volume_of_cone :
  let r := (radius : ℝ) in
  let h := (height : ℝ) in
  let V := (1/3) * pi_approx * r^2 * h in
  V = 37.69908 :=
sorry

end volume_of_cone_l638_638833


namespace find_n_eq_l638_638935

theorem find_n_eq : 
  let a := 2^4
  let b := 3^3
  ∃ (n : ℤ), a - 7 = b + n :=
by
  let a := 2^4
  let b := 3^3
  use -18
  sorry

end find_n_eq_l638_638935


namespace correct_proposition_C_l638_638249

variables (a b c : Type) [line a] [line b] [line c] 
variables (α β γ : Type) [plane α] [plane β] [plane γ] 

-- Hypothesis: α is perpendicular to itself and not parallel to β
-- Conclusion: α is perpendicular to β
theorem correct_proposition_C (h1 : perp α α) (h2 : ¬parallel α β) : perp α β := 
sorry

end correct_proposition_C_l638_638249


namespace glass_volume_l638_638429

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638429


namespace max_value_f_l638_638944

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem max_value_f (h : ∀ ε > (0 : ℝ), ∃ x : ℝ, x < 1 ∧ ε < f x) : ∀ x : ℝ, x < 1 → f x ≤ -1 :=
by
  intros x hx
  dsimp [f]
  -- Proof steps are omitted.
  sorry

example (h: ∀ ε > 0, ∃ x : ℝ, x < 1 ∧ ε < f x) : ∃ x : ℝ, x < 1 ∧ f x = -1 :=
by
  use 0
  -- Proof steps are omitted.
  sorry

end max_value_f_l638_638944


namespace glass_volume_l638_638462

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l638_638462


namespace lambda_mu_sum_l638_638602

variable (λ μ : ℝ)
variable (a : ℝ × ℝ) (ha : a = (3, 4))
variable (h1 : (λ * a.1, λ * a.2) = (3 * λ, 2 * μ))
variable (h2 : (λ * 3)^2 + (λ * 4)^2 = 25)

theorem lambda_mu_sum (h3 : a = (3, 4)) (h4 : λ * a.2 = 2 * μ) (h5 : (λ * a.1)^2 + (λ * a.2)^2 = 25) :
  λ + μ = 3 ∨ λ + μ = -3 :=
sorry

end lambda_mu_sum_l638_638602


namespace collinear_L_E_F_l638_638472

-- Define the geometric setup and conditions
variables {A B C D E O1 O2 F L : Type*}
           [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] 
           [inhabited O1] [inhabited O2] [inhabited F] [inhabited L]

-- Definitions based on given conditions
variables 
  (triangle_ABC : triangle A B C)
  (altitude_BD : altitude triangle_ABC B D)
  (angle_AEC_90 : ∠AEC = 90)
  (circumcenter_O1 : circumcenter (triangle A E B) = O1)
  (circumcenter_O2 : circumcenter (triangle C E B) = O2)
  (midpoint_F : midpoint A C = F)
  (midpoint_L : midpoint O1 O2 = L)

-- Theorem to prove
theorem collinear_L_E_F 
  (Point_E_on_altitude : E ∈ altitude_BD)
  (angle_AEC_perp : angle_AEC_90) 
  (F_is_midpoint_AC : midpoint_F)
  (L_is_midpoint_O1O2 : midpoint_L) :
  collinear L E F :=
sorry

end collinear_L_E_F_l638_638472


namespace range_of_positive_integers_in_k_l638_638713

def is_mixed_number (x : ℝ) : Prop :=
  ∃ a : ℤ, ∃ b : ℚ, ∃ c : ℝ, ⟦c⟧ ∈ ℝ

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def fraction_with_prime_denominator (x : ℚ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ x.denom = p

theorem range_of_positive_integers_in_k (k : list ℝ)
  (h1 : k.length = 20)
  (h2 : ∃ l : ℤ, -3 ≤ l ∧ ∀ x ∈ k, x > l)
  (h3 : ∃ s ∈ k, irrational s)
  (h4 : ∀ x ∈ k, 0 ≤ x ∧ x ≤ real.sqrt 50)
  (h5 : ∀ x ∈ k, fraction_with_prime_denominator x) :
  ∃ (range : ℝ), range = 6 :=
by sorry

end range_of_positive_integers_in_k_l638_638713


namespace find_g2_l638_638704

theorem find_g2 (g : ℝ → ℝ)
  (h_poly : ∃ c d, g = λ x, c * x + d ∧ c ≠ 0)
  (h_eq : ∀ x ≠ 0, g (x - 1) + g (x + 1) = (g x)^2 / (504 * x)) :
  g 2 = 1008 :=
sorry

end find_g2_l638_638704


namespace initially_calculated_average_l638_638755

open List

theorem initially_calculated_average (numbers : List ℝ) (h_len : numbers.length = 10) 
  (h_wrong_reading : ∃ (n : ℝ), n ∈ numbers ∧ n ≠ 26 ∧ (numbers.erase n).sum + 26 = numbers.sum - 36 + 26) 
  (h_correct_avg : numbers.sum / 10 = 16) : 
  ((numbers.sum - 10) / 10 = 15) := 
sorry

end initially_calculated_average_l638_638755


namespace A_has_higher_probability_l638_638343

noncomputable def prob_pass_B : ℝ :=
  let prob_two_correct := ↑((3.choose 2) * (0.6 ^ 2) * (1 - 0.6)) in
  let prob_three_correct := (0.6 ^ 3) in
  prob_two_correct + prob_three_correct

noncomputable def prob_dist_A : ℕ → ℝ
| 0 := (4.choose 3) / (10.choose 3)
| 1 := (6.choose 1) * (4.choose 2) / (10.choose 3)
| 2 := (6.choose 2) * (4.choose 1) / (10.choose 3)
| 3 := (6.choose 3) / (10.choose 3)
| _ := 0

noncomputable def expected_X : ℝ :=
  0 * prob_dist_A 0 + 1 * prob_dist_A 1 + 2 * prob_dist_A 2 + 3 * prob_dist_A 3

noncomputable def prob_pass_A : ℝ :=
  prob_dist_A 2 + prob_dist_A 3

theorem A_has_higher_probability : prob_pass_A > prob_pass_B :=
  sorry

end A_has_higher_probability_l638_638343


namespace focal_length_of_ellipse_l638_638102

theorem focal_length_of_ellipse : 
  (∀ (x y : ℝ), (x^2 / 2 + y^2 / 4 = 2) → (√(4 - 8) * 2 = 4)) :=
by
  intros x y h
  sorry

end focal_length_of_ellipse_l638_638102


namespace jerry_one_way_trip_time_l638_638673

-- Definitions based on conditions
def distance_to_school : ℝ := 4   -- The school is 4 miles away
def carson_speed : ℝ := 8         -- Carson runs at a speed of 8 miles per hour
def round_trip_time := (distance_to_school / carson_speed) * 60  -- Time Carson takes in minutes

-- Theorem we want to prove
theorem jerry_one_way_trip_time : (round_trip_time / 2) = 15 := 
by
  -- skipping the actual proof  
  sorry

end jerry_one_way_trip_time_l638_638673


namespace OP_tangent_to_circumcircle_of_KLX_l638_638044

-- Definitions of geometric objects and properties
variables {A B C P X E F K L O : Type*}
variables [has_angle A B C] [has_angle A X P]
variables [lying_on_circle A] [lying_on_line P X] [lying_on_line O P]
variables [lies_on AB E] [lies_on AC F] [intersects EF K L]
variables [center_of_circle O A B C]

-- Conditions translated to Lean definitions
def acute_triangle (ABC : Triangle) (AC : > AB) : Prop := 
  ∀ x y z, is_acute x y z ∧ x = A ∧ y = B ∧ z = C

def perpendicular (AXP : Angle) := angle A X P = 90

def angles_equal_1 (EXP : Angle) (ACX : Angle) := angle E X P = angle A C X

def angles_equal_2 (FXO : Angle) (ABX : Angle) := angle F X O = angle A B X

def intersections (EF : Line) (circumcircle : Circle) := 
  intersects_line_circle EF circumcircle = {K, L}

-- The final statement to be proven
theorem OP_tangent_to_circumcircle_of_KLX 
  (O_center : circumcenter O A B C)
  (triangle_acute : acute_triangle A B C (AC > AB))
  (tangent_at_A : tangent_line P circumcircle ∧ lies_on_circle A)
  (perpendicular_1 : perpendicular angle AXP)
  (angles_1 : angles_equal_1 angle EXP angle ACX)
  (angles_2 : angles_equal_2 angle FXO angle ABX)
  (intersection_pts : intersections EF circumcircle)
  : tangent_line OP circumcircle_KLX := 
sorry

end OP_tangent_to_circumcircle_of_KLX_l638_638044


namespace disjoint_subsets_exist_l638_638258

theorem disjoint_subsets_exist (n : ℕ) (h : 0 < n) (A : fin (n+1) → finset (fin n)) (hA : ∀ i, A i ≠ ∅) :
  ∃ (I J : finset (fin (n+1))), I ≠ ∅ ∧ J ≠ ∅ ∧ disjoint I J ∧ (I.biUnion A) = (J.biUnion A) :=
sorry

end disjoint_subsets_exist_l638_638258


namespace glass_volume_230_l638_638438

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638438


namespace geom_series_sum_l638_638066

def a : ℚ := 1 / 3
def r : ℚ := 2 / 3
def n : ℕ := 9

def S_n (a r : ℚ) (n : ℕ) := a * (1 - r^n) / (1 - r)

theorem geom_series_sum :
  S_n a r n = 19171 / 19683 := by
    sorry

end geom_series_sum_l638_638066


namespace graph_in_fourth_quadrant_l638_638592

theorem graph_in_fourth_quadrant (t : ℝ) (ht : t > 0) : 
  let w := -sqrt (t^3) - (2 / t) 
  in w < 0 :=
by
  let w := -sqrt (t^3) - (2 / t)
  sorry

end graph_in_fourth_quadrant_l638_638592


namespace cards_given_to_each_friend_l638_638287

def total_initial_cards : ℕ := 130
def cards_kept_by_rick : ℕ := 15
def cards_given_to_miguel : ℕ := 13
def number_of_friends : ℕ := 8
def cards_given_to_each_sister : ℕ := 3

theorem cards_given_to_each_friend :
  let remaining_cards_after_rick_keeps := total_initial_cards - cards_kept_by_rick,
      remaining_cards_after_miguel := remaining_cards_after_rick_keeps - cards_given_to_miguel,
      total_cards_given_to_sisters := 2 * cards_given_to_each_sister,
      final_remaining_cards := remaining_cards_after_miguel - total_cards_given_to_sisters
  in final_remaining_cards / number_of_friends = 12 := by
  sorry

end cards_given_to_each_friend_l638_638287


namespace first_girl_productivity_higher_l638_638386

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l638_638386


namespace find_tangent_lines_passing_through_given_point_l638_638098

open Real

def is_tangent_line (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (P : ℝ × ℝ), C P.1 P.2 ∧ l = λ x y, (distance (0,0) (x,y) = 2)

def tangent_line_through_P : Prop :=
  ∀ P : ℝ × ℝ,
  P = (2, 3) →
  (∃ l : ℝ → ℝ → Prop, is_tangent_line l (λ x y, x^2 + y^2 = 4) ∧ 
   ( ∀ x y, (l x y ↔ (5 * x - 12 * y + 26 = 0)))
   ∨ (∀ x y, (l x y ↔ (x = 2)) ))

theorem find_tangent_lines_passing_through_given_point :
  tangent_line_through_P :=
sorry

end find_tangent_lines_passing_through_given_point_l638_638098


namespace geometric_first_term_l638_638763

theorem geometric_first_term (a r : ℝ) (h1 : a * r^3 = 720) (h2 : a * r^6 = 5040) : 
a = 720 / 7 :=
by
  sorry

end geometric_first_term_l638_638763


namespace line_through_point_l638_638013

-- Definitions for conditions
def point : (ℝ × ℝ) := (1, 2)

-- Function to check if a line equation holds for the given form 
def is_line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main Lean theorem statement
theorem line_through_point (a b c : ℝ) :
  (∃ a b c, (is_line_eq a b c 1 2) ∧ 
           ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 2 ∧ b = -1 ∧ c = 0))) :=
sorry

end line_through_point_l638_638013


namespace glass_volume_l638_638431

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l638_638431


namespace pyramid_cosine_l638_638758

variable {α : ℝ}

theorem pyramid_cosine (hb : is_equilateral_triangle base)
    (hf : is_perpendicular lateral_face₁ base)
    (ha : forms_angle lateral_face₂ base α)
    (hb' : forms_angle lateral_face₃ base α) :
    cos (angle_between lateral_face₂ lateral_face₃) = - (1 + 3 * (cos α)^2) / 4 :=
by sorry

end pyramid_cosine_l638_638758


namespace max_sum_of_products_l638_638601

theorem max_sum_of_products (a b c d : ℕ) 
(ha : a ∈ {2, 4, 6, 8}) (hb : b ∈ {2, 4, 6, 8}) (hc : c ∈ {2, 4, 6, 8}) (hd : d ∈ {2, 4, 6, 8}) 
(hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ab + bd + dc + ca ≤ 40 :=
sorry

end max_sum_of_products_l638_638601


namespace chord_length_cut_by_circle_l638_638928

theorem chord_length_cut_by_circle (x y : ℝ) :
  (x - 2)^2 + (y - 2)^2 = 4 → x = 0 → ∃ l : ℝ, l = 2 * real.sqrt 2 := by
  intros h_circle h_line
  sorry

end chord_length_cut_by_circle_l638_638928


namespace average_weight_calculation_l638_638788

noncomputable def new_average_weight (initial_people : ℕ) (initial_avg_weight : ℝ) 
                                     (new_person_weight : ℝ) (total_people : ℕ) : ℝ :=
  (initial_people * initial_avg_weight + new_person_weight) / total_people

theorem average_weight_calculation :
  new_average_weight 6 160 97 7 = 151 := by
  sorry

end average_weight_calculation_l638_638788


namespace baseball_card_decrease_l638_638021

theorem baseball_card_decrease (V : ℝ) (hV : V > 0) (x : ℝ) :
  (1 - x / 100) * (1 - 0.30) = 1 - 0.44 -> x = 20 :=
by {
  -- proof omitted 
  sorry
}

end baseball_card_decrease_l638_638021


namespace day_of_week_dec_27_l638_638619

theorem day_of_week_dec_27 (h : ∀ d, d = 15 → day_of_week (12, 27) = day_of_week (11, d)) : day_of_week (12, 27) = "Wednesday" :=
begin
  sorry
end

end day_of_week_dec_27_l638_638619


namespace line_properties_l638_638600

theorem line_properties (l l1 l2 : ℝ)
    (A B : ℝ × ℝ)
    (a b : ℝ)
    (hp : l = 3/4 * π)
    (hA : A = (3, 2))
    (hB : B = (a, -1))
    (hperp : ∀ x, l1 = 1 / l)
    (hparallel : ∀ x, l2 = (2*x + b*y + 1 = 0) ∧ ∀ x, l1 = l2) :
    a + b = -2 := by            
    have slope_l: l = -1 := 
        calc l = 3/4 * π : by rw [hp]
             ... = -1 : by sorry,
    have slope_l1: l1 = 1 := 
        calc l1 = 1 / l : by rw [hperp]
             ... = 1 / -1 : by rw [slope_l]
             ... = 1 : by norm_num,
    have value_a: a = 0 := 
        calc a = 3 * 2 - 3 : by sorry,
    have value_b: b = -2 := 
        calc b = -2 := by sorry,
    show a + b = -2 := calc a + b = 0 + -2 : by rw [value_a, value_b]
        ... = -2 : by norm_num

end line_properties_l638_638600


namespace factorize_expression_l638_638091

theorem factorize_expression (a x : ℝ) : a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
  sorry

end factorize_expression_l638_638091


namespace total_children_estimate_l638_638549

theorem total_children_estimate (k m n : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) 
(h4 : n ≤ m) (h5 : n ≤ k) (h6 : m ≤ k) :
  (∃ (total : ℕ), total = k * m / n) :=
sorry

end total_children_estimate_l638_638549


namespace david_initial_amount_l638_638910

noncomputable def initial_amount (remaining : ℕ) (diff : ℕ) : ℕ := remaining + diff + remaining

theorem david_initial_amount (remaining : ℕ) (diff : ℕ) (initial : ℕ) (h₁ : remaining = 500) 
  (h₂ : diff = 500) (h₃ : initial = initial_amount remaining diff) : initial = 1500 :=
by
  rw [h₁, h₂, initial_amount]
  simp
  sorry

end david_initial_amount_l638_638910


namespace interval_of_a_l638_638153

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  f a n.succ  -- since ℕ in Lean includes 0, use n.succ to start from 1

-- The main theorem to prove
theorem interval_of_a (a : ℝ) : (∀ n : ℕ, n ≠ 0 → a_n a n < a_n a (n + 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end interval_of_a_l638_638153


namespace mary_needs_more_flour_than_sugar_l638_638268

theorem mary_needs_more_flour_than_sugar (sugar_needed flour_needed flour_added : ℕ) (hs : sugar_needed = 9) (hf : flour_needed = 14) (fa : flour_added = 4) : flour_needed - flour_added - sugar_needed = 1 :=
by
  rw [hs, hf, fa]
  norm_num
  sorry

end mary_needs_more_flour_than_sugar_l638_638268


namespace number_of_x_values_l638_638609

noncomputable def count_x_values : ℕ :=
  (λ (n : ℕ), ∃ (x : ℝ), -10 < x ∧ x < 50 ∧ (∀ (x : ℝ), -10 < x ∧ x < 50 → (cos x)^2 + 3 * (sin x)^2 = 1 → x ∈ ((finset.range n).map (λ k : ℤ, k * real.pi))).to_finset.card = n) 19

theorem number_of_x_values :
  ∃ (n : ℕ), n = count_x_values :=
by {
  use 19,
  sorry
}

end number_of_x_values_l638_638609


namespace smallest_integer_y_l638_638359

theorem smallest_integer_y : ∃ y : ℤ, (∀ z : ℤ, (7 - 3*z ≥ 22) → (y ≤ z)) ∧ (7 - 3*y ≥ 22) ∧ y = -5 :=
by
  use -5
  split
  · intros z hz
    linarith
  · split
    · linarith
    · rfl

end smallest_integer_y_l638_638359


namespace problem_statement_l638_638970

variable {x y m : ℝ}

theorem problem_statement 
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : Real.log10 x = m)
  (h4 : y = 10^(m-1)) :
  x / y = 10 := 
sorry

end problem_statement_l638_638970


namespace perimeter_of_large_square_l638_638497

theorem perimeter_of_large_square (squares : List ℕ) (h : squares = [1, 1, 2, 3, 5, 8, 13]) : 2 * (21 + 13) = 68 := by
  sorry

end perimeter_of_large_square_l638_638497


namespace average_score_l638_638027

variable (T : ℝ) -- Total number of students
variable (M : ℝ) -- Number of male students
variable (F : ℝ) -- Number of female students

variable (avgM : ℝ) -- Average score for male students
variable (avgF : ℝ) -- Average score for female students

-- Conditions
def M_condition : Prop := M = 0.4 * T
def F_condition : Prop := F = 0.6 * T
def avgM_condition : Prop := avgM = 75
def avgF_condition : Prop := avgF = 80

theorem average_score (h1 : M_condition T M) (h2 : F_condition T F) 
    (h3 : avgM_condition avgM) (h4 : avgF_condition avgF) :
    (75 * M + 80 * F) / T = 78 := by
  sorry

end average_score_l638_638027


namespace sqrt3_minus_m_div_n_gt_half_mn_l638_638735

variable (n m : ℕ)
variable h₁ : (real.sqrt 3 - (m / n : ℝ)) > 0

theorem sqrt3_minus_m_div_n_gt_half_mn (n m : ℕ) (h₁ : (real.sqrt 3 - (m / n : ℝ)) > 0) :
  (real.sqrt 3 - (m / n : ℝ)) > (1 / (2 * m * n : ℝ)) :=
sorry

end sqrt3_minus_m_div_n_gt_half_mn_l638_638735


namespace cost_of_dozen_pens_l638_638001

variable (x : ℝ) (pen_cost pencil_cost : ℝ)
variable (h1 : 3 * pen_cost + 5 * pencil_cost = 260)
variable (h2 : pen_cost / pencil_cost = 5)

theorem cost_of_dozen_pens (x_pos : 0 < x) 
    (pen_cost_def : pen_cost = 5 * x) 
    (pencil_cost_def : pencil_cost = x) :
    12 * pen_cost = 780 := by
  sorry

end cost_of_dozen_pens_l638_638001


namespace glass_volume_correct_l638_638470

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l638_638470


namespace angle_FAE_deg_l638_638054

-- Definitions based on the conditions
variables {A B C D E F : Type}
variables (triangle_ABC : A ≠ B ∧ B ≠ C ∧ A ≠ C)
variables (eq_triangle : ∀ {A B C : Type}, equilateral (triangle A B C))
variables (square_BCDE : B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ B ∧ B ≠ D ∧ C ≠ E)
variables (semi_circle_BC : semicircle B C)
variables (mid_F : midpoint arc B C = F)

-- The goal to prove
theorem angle_FAE_deg : angle ∠ F A E = 30 :=
sorry

end angle_FAE_deg_l638_638054


namespace incorrect_statements_l638_638162

def f (x : ℝ) : ℝ := (2 - x) * Real.exp x
def f_prime (x : ℝ) : ℝ := (1 - x) * Real.exp x

theorem incorrect_statements :
  ¬ (f_prime 2 > 0) ∧
  ¬ (∀ x, x > 1 → ((1 - x) * Real.exp x > 0)) ∧
  ¬ (∀ a, f 1 = a → a < Real.exp 1) :=
by
  -- proving the statements
  sorry

end incorrect_statements_l638_638162


namespace union_intersection_l638_638707

variable (a : ℝ)

def setA (a : ℝ) : Set ℝ := { x | (x - 3) * (x - a) = 0 }
def setB : Set ℝ := {1, 4}

theorem union_intersection (a : ℝ) :
  (if a = 3 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = ∅ else 
   if a = 1 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {1} else
   if a = 4 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {4} else
   setA a ∪ setB = {1, 3, 4, a} ∧ setA a ∩ setB = ∅) := sorry

end union_intersection_l638_638707


namespace triangle_inequality_check_l638_638361

theorem triangle_inequality_check
  (A : ℕ × ℕ × ℕ := (3, 4, 8))
  (B : ℕ × ℕ × ℕ := (5, 6, 11))
  (C : ℕ × ℕ × ℕ := (4, 4, 8))
  (D : ℕ × ℕ × ℕ := (8, 8, 8)) :
  (A.1 + A.2 > A.3 ∧ A.1 + A.3 > A.2 ∧ A.2 + A.3 > A.1) = false ∧
  (B.1 + B.2 > B.3 ∧ B.1 + B.3 > B.2 ∧ B.2 + B.3 > B.1) = false ∧
  (C.1 + C.2 > C.3 ∧ C.1 + C.3 > C.2 ∧ C.2 + C.3 > C.1) = false ∧
  (D.1 + D.2 > D.3 ∧ D.1 + D.3 > D.2 ∧ D.2 + D.3 > D.1) = true :=
by
  sorry

end triangle_inequality_check_l638_638361


namespace exists_tangent_line_l638_638566

-- Definitions of the circle and the parabola
def circle (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + y ^ 2 = 2

def parabola (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

-- Definition of the given lines
def line1 (x y : ℝ) : Prop := y = real.sqrt 2
def line2 (x y : ℝ) : Prop := y = -real.sqrt 2
def line3 (x y : ℝ) : Prop := x - y + 1 = 0
def line4 (x y : ℝ) : Prop := x + y + 1 = 0
def line5 (x y : ℝ) : Prop := x - real.sqrt 7 * y + 7 = 0
def line6 (x y : ℝ) : Prop := x + real.sqrt 7 * y + 7 = 0

-- Theorem: There exists one line that has exactly one common point with both the circle and the parabola
theorem exists_tangent_line :
  ∃ l, (l = line1 ∨ l = line2 ∨ l = line3 ∨ l = line4 ∨ l = line5 ∨ l = line6) ∧
       ∃! p, circle p.1 p.2 ∧ parabola p.1 p.2 ∧ l p.1 p.2 :=
by
  sorry

end exists_tangent_line_l638_638566


namespace lattice_points_in_bounded_region_l638_638419

def isLatticePoint (p : ℤ × ℤ) : Prop :=
  true  -- All (n, m) ∈ ℤ × ℤ are lattice points

def boundedRegion (x y : ℤ) : Prop :=
  y = x ^ 2 ∨ y = 8 - x ^ 2
  
theorem lattice_points_in_bounded_region :
  ∃ S : Finset (ℤ × ℤ), 
    (∀ p ∈ S, isLatticePoint p ∧ boundedRegion p.1 p.2) ∧ S.card = 17 :=
by
  sorry

end lattice_points_in_bounded_region_l638_638419


namespace average_distance_to_sides_l638_638034

def rectangle := (length width : ℝ)
def rabbit_initial_position := (x y : ℝ)

noncomputable def rabbit_final_position 
  (r_l r_w : ℝ) (d : ℝ) (turn_distance : ℝ) : ℝ × ℝ :=
  let diag_length := Real.sqrt (r_l^2 + r_w^2) in
  let diag_fraction := d / diag_length in
  let diag_x := diag_fraction * r_l in
  let diag_y := diag_fraction * r_w in
  (diag_x + turn_distance, diag_y)

theorem average_distance_to_sides 
  (r_l r_w d turn_distance : ℝ) :
  r_l = 12 → r_w = 8 → d = 8 → turn_distance = 3 → 
  let (rabbit_x, rabbit_y) := rabbit_final_position r_l r_w d turn_distance in
  (rabbit_x + (r_l - rabbit_x) + rabbit_y + (r_w - rabbit_y)) / 4 = 5 :=
by
  intro r_l_eq r_w_eq d_eq turn_distance_eq
  rw [r_l_eq, r_w_eq, d_eq, turn_distance_eq]
  let (x, y) := rabbit_final_position 12 8 8 3
  sorry

end average_distance_to_sides_l638_638034


namespace m_times_t_eq_zero_l638_638256

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x y : ℝ, g (x * g y - 2 * x) = x * y + g x

def m : ℝ := if g 3 = 3 ∧ g 3 = -3 then 2 else sorry

def t : ℝ := 3 + (-3)

theorem m_times_t_eq_zero : m * t = 0 :=
by
  unfold m t
  simp
  sorry

end m_times_t_eq_zero_l638_638256


namespace build_time_40_workers_l638_638221

theorem build_time_40_workers (r : ℝ) : 
  (60 * r) * 5 = 1 → (40 * r) * t = 1 → t = 7.5 :=
by
  intros h1 h2
  sorry

end build_time_40_workers_l638_638221


namespace calculate_arithmetic_expression_l638_638504

noncomputable def arithmetic_sum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem calculate_arithmetic_expression :
  3 * (arithmetic_sum 71 2 99) = 3825 :=
by
  sorry

end calculate_arithmetic_expression_l638_638504


namespace tangents_and_area_of_quadrilateral_l638_638945

theorem tangents_and_area_of_quadrilateral (Q : ℝ × ℝ) (hQ : Q.2 = 0) :
  let M : set (ℝ × ℝ) := { p | p.1^2 + (p.2 - 2)^2 = 1 } in
  (Q = (-1, 0) → 
    (∃ t1 t2 : ℝ, (∀ p ∈ M, p.1 = t1 ∨ 3 * p.1 - 4 * p.2 + 3 = t2)) ∧ 
    (min_area : ℝ, min_area = sqrt 3)) :=
sorry

end tangents_and_area_of_quadrilateral_l638_638945


namespace glass_volume_is_230_l638_638454

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l638_638454


namespace problem_statement_l638_638240

-- Define the conditions and the statement of the theorem
theorem problem_statement (k : ℤ) (hk : 0 < k) (a : ℤ)
    (h1 : (a - 2) % 7 = 0)
    (h2 : (a^6 - 1) % (7^k) = 0) :
  ((a + 1)^6 - 1) % (7^k) = 0 := 
begin
  sorry
end

end problem_statement_l638_638240


namespace chair_height_proof_l638_638856

noncomputable def light_bulb_height (ceiling_height : ℕ) (distance_below_ceiling : ℕ) : ℕ :=
  ceiling_height - distance_below_ceiling

noncomputable def bobs_reach (bob_height : ℕ) (reach_extra : ℕ) : ℕ :=
  bob_height + reach_extra

theorem chair_height_proof :
  let ceiling_height := 280
  let distance_below_ceiling := 15
  let bob_height := 165
  let reach_extra := 55
  let bulb_height := light_bulb_height ceiling_height distance_below_ceiling
  let total_reach := bobs_reach bob_height reach_extra
  total_reach + 45 = bulb_height :=
by
  let ceiling_height := 280
  let distance_below_ceiling := 15
  let bob_height := 165
  let reach_extra := 55
  let bulb_height := light_bulb_height ceiling_height distance_below_ceiling
  let total_reach := bobs_reach bob_height reach_extra
  show total_reach + 45 = bulb_height, by sorry

end chair_height_proof_l638_638856


namespace find_prices_l638_638495

noncomputable def price_of_pastries : ℕ × ℕ :=
  let d := 12 in
  let v := 16 in
  (d, v)

theorem find_prices (d v k : ℕ) (h1 : v = d + 4)
  (h2 : (4 * k) / 3 * d = 96) : price_of_pastries = (12, 16) :=
by
  dsimp [price_of_pastries]
  have : d * k = 72, sorry
  have h3 : k = 72 / d, sorry
  have h4 : k * (d + 4) = 96, sorry
  have h5 : 72 * (d + 4) / d = 96, sorry
  have h6 : d = 12, sorry
  have h7 : v = 16, sorry
  simp [h6, h7]

end find_prices_l638_638495


namespace consecutive_lucky_tickets_l638_638711

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

def is_lucky_ticket (n : Nat) : Prop :=
  sum_of_digits n % 7 = 0

theorem consecutive_lucky_tickets :
  ∃ n : Nat, is_lucky_ticket n ∧ is_lucky_ticket (n + 1) :=
by
  sorry

end consecutive_lucky_tickets_l638_638711


namespace number_of_satisfying_polynomials_l638_638912

noncomputable def cube_root_unity_1 : Complex := (-1 + Complex.i * Real.sqrt 3) / 2
noncomputable def cube_root_unity_2 : Complex := (-1 - Complex.i * Real.sqrt 3) / 2

def poly_form (a b c d e f : ℝ) : Polynomial ℝ :=
  Polynomial.monomial 7 1 + Polynomial.monomial 6 a + Polynomial.monomial 5 b + 
  Polynomial.monomial 4 c + Polynomial.monomial 3 d + 
  Polynomial.monomial 2 e + Polynomial.monomial 1 f - 
  Polynomial.C 4040

def satisfies_root_conditions (P : Polynomial ℝ) : Prop :=
  ∀ s : ℂ, P.eval s = 0 → 
    P.eval (cube_root_unity_1 * s) = 0 ∧ 
    P.eval (cube_root_unity_2 * s) = 0

theorem number_of_satisfying_polynomials :
  {P : Polynomial ℝ // ∃ a b c d e f : ℝ, P = poly_form a b c d e f ∧ satisfies_root_conditions P}.card = 2 :=
sorry

end number_of_satisfying_polynomials_l638_638912


namespace Ferris_break_length_l638_638895

noncomputable def Audrey_rate_per_hour := (1:ℝ) / 4
noncomputable def Ferris_rate_per_hour := (1:ℝ) / 3
noncomputable def total_completion_time := (2:ℝ)
noncomputable def number_of_breaks := (6:ℝ)
noncomputable def job_completion_audrey := total_completion_time * Audrey_rate_per_hour
noncomputable def job_completion_ferris := 1 - job_completion_audrey
noncomputable def working_time_ferris := job_completion_ferris / Ferris_rate_per_hour
noncomputable def total_break_time := total_completion_time - working_time_ferris
noncomputable def break_length := total_break_time / number_of_breaks

theorem Ferris_break_length :
  break_length = (5:ℝ) / 60 := 
sorry

end Ferris_break_length_l638_638895


namespace probability_le_neg2_from_normal_distribution_and_interval_probability_l638_638247

noncomputable def normal_distribution_p_le_neg2 (ξ : ℝ) (δ : ℝ) : Prop :=
∀ (ξ ∈ ℝ), (ξ ≈ μ) → 0.4 → P(ξ <= -2) = 0.1

theorem probability_le_neg2_from_normal_distribution_and_interval_probability
  (ξ : ℝ) (μ : ℝ := 0) (δ : ℝ) (hξ : ξ ≈ μ ⟹ δ ^ 2) 
  (h2 : P(-2 ≤ ξ ⟹ 0) = 0.4) : P(ξ ≤ -2) = 0.1 :=
sorry

end probability_le_neg2_from_normal_distribution_and_interval_probability_l638_638247


namespace eq_one_solution_in_interval_l638_638193

theorem eq_one_solution_in_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (2 * a * x^2 - x - 1 = 0) ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 1 ∧ y ≠ x → (2 * a * y^2 - y - 1 ≠ 0))) → (1 < a) :=
by
  sorry

end eq_one_solution_in_interval_l638_638193


namespace product_of_variables_l638_638131

theorem product_of_variables :
  ∀ (a b c d : ℚ), 
  a + 3 = 3a → 
  b + 4 = 4b → 
  c + 5 = 5c → 
  d + 6 = 6d → 
  a * b * c * d = 3 := 
by 
  intros a b c d h1 h2 h3 h4 
  have ha : a = 3 / 2 := by sorry 
  have hb : b = 4 / 3 := by sorry 
  have hc : c = 5 / 4 := by sorry 
  have hd : d = 6 / 5 := by sorry 
  rw [ha, hb, hc, hd] 
  norm_num

end product_of_variables_l638_638131


namespace angie_age_l638_638803

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l638_638803


namespace largest_constant_inequality_l638_638914

noncomputable def largestConstant (x : Fin 100 → ℝ) :=
  let M := (x 49 + x 50) / 2
  ∀ (h1 : (∑ i, x i) = 0),
  (∑ i, (x i)^2) >= (5050/49) * M^2

theorem largest_constant_inequality
  (x : Fin 100 → ℝ)
  (h_sum_zero : (∑ i, x i) = 0) :
  let M := (x 49 + x 50) / 2
  in (∑ i, (x i)^2) >= (5050/49) * M^2 :=
by
  sorry

end largest_constant_inequality_l638_638914


namespace probability_even_distinct_digits_l638_638892

theorem probability_even_distinct_digits :
  let count_even_distinct := 1960
  let total_numbers := 8000
  count_even_distinct / total_numbers = 49 / 200 :=
by
  sorry

end probability_even_distinct_digits_l638_638892


namespace four_c_plus_d_l638_638299

theorem four_c_plus_d (c d : ℝ) (h1 : 2 * c = -6) (h2 : c^2 - d = 1) : 4 * c + d = -4 :=
by
  sorry

end four_c_plus_d_l638_638299


namespace exists_two_participants_with_matching_scores_l638_638643

-- Define the range of possible scores for each task
def score_range : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7]

-- Represent the scores for the three tasks as a triple
def score_triplet := (ℕ × ℕ × ℕ)

-- Condition that ensures scores are within the valid range
def valid_score (s : score_triplet) : Prop :=
  s.1 ∈ score_range ∧ s.2 ∈ score_range ∧ s.3 ∈ score_range

-- Define a list representing the 49 participants
def participants_scores : List score_triplet := List.replicate 49 (0, 0, 0) -- Example, to be populated

-- Predicate that one score triplet is at least as good as another
def at_least_as_good (s1 s2 : score_triplet) : Prop :=
  s1.1 ≥ s2.1 ∧ s1.2 ≥ s2.2 ∧ s1.3 ≥ s2.3

theorem exists_two_participants_with_matching_scores :
  ∃ (p1 p2 : score_triplet), p1 ≠ p2 ∧ valid_score p1 ∧ valid_score p2 ∧ at_least_as_good p1 p2 :=
sorry

end exists_two_participants_with_matching_scores_l638_638643


namespace polynomial_solution_l638_638705

theorem polynomial_solution (m : ℤ) (h : m ≠ 0) (P : ℝ[X]) :
  (∀ x : ℝ, (x^3 - (m : ℝ) * x^2 + 1) * P.eval (x+1) + (x^3 + (m : ℝ) * x^2 + 1) * P.eval (x-1) = 2 * (x^3 - (m : ℝ) * x + 1) * P.eval x) →
  ∃ t : ℝ, P = polynomial.C t * polynomial.X :=
by
  sorry

end polynomial_solution_l638_638705


namespace probability_product_multiple_of_4_l638_638121

/--
Geoff and Trevor each roll a fair eight-sided die (numbered 1 through 8).
Prove that the probability that the product of the numbers they roll is a multiple of 4 is 15/16.
-/
theorem probability_product_multiple_of_4 :
  let outcomes := { (d1, d2) | d1 ∈ Finset.range 1 9 ∧ d2 ∈ Finset.range 1 9 },
      multiples_of_4 := { (d1, d2) | d1 * d2 % 4 = 0 },
      favorable := Finset.card multiples_of_4
  let total := Finset.card outcomes
  in (favorable.toRat / total.toRat) = 15/16 :=
by
  sorry

end probability_product_multiple_of_4_l638_638121


namespace part_i_part_ii_l638_638573

section
variable (A : Set ℝ) (B : Set ℝ) (C : Set ℝ)
variable (a : ℝ) (m : ℝ)

-- Define A
def A_def := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

-- Define B
def B_def := {a^2 + 4*a + 8, a + 3, 3 * Real.log2 (abs a)}

-- Problem (i)
theorem part_i (B_sub_A : B ⊆ A) : a = -2 := 
by sorry

-- Define C
def C_def := {y : ℝ | ∃ x ∈ A, ∃ m ∈ Real, y = x ^ m}

-- Problem (ii)
theorem part_ii (A_union_C_eq_A : A ∪ C = A) : 0 ≤ m ∧ m ≤ 1 := 
by sorry

end

end part_i_part_ii_l638_638573


namespace sum_of_unpicked_cards_is_28_l638_638780

theorem sum_of_unpicked_cards_is_28 :
  ∃ (picked_cards : set ℕ), 
    picked_cards ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13} ∧
    picked_cards.card = 9 ∧
    (∃ (A B C : ℕ), A ∈ picked_cards ∧ B ∈ picked_cards ∧ C ∈ picked_cards ∧
       (∃ (A_known : ℕ → Prop), A_known A) ∧
       (B ≠ B ∧ B % 2 = B % 2) ∧
       (C = B - 2) ∧ (C = A + 1)) →
    (∑ x in ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13} \ picked_cards), x) = 28 :=
begin
  sorry
end

end sum_of_unpicked_cards_is_28_l638_638780


namespace glass_volume_230_l638_638434

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l638_638434


namespace find_x_l638_638545

theorem find_x (x : ℝ) (h₁ : 0 ≤ x ∧ x ≤ 180) (h₂ : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 120 := 
by
  -- The proof would go here
  sorry

end find_x_l638_638545


namespace max_sphere_surface_area_l638_638149

theorem max_sphere_surface_area (a : ℝ) (h : a = 2) : 
  let radius := a / 2 in
  4 * Real.pi * radius^2 = 4 * Real.pi :=
by
  sorry

end max_sphere_surface_area_l638_638149


namespace connie_initial_marbles_l638_638512

theorem connie_initial_marbles (marbles_given : ℕ) (marbles_left : ℕ) (initial_marbles : ℕ) 
    (h1 : marbles_given = 183) (h2 : marbles_left = 593) : initial_marbles = 776 :=
by
  sorry

end connie_initial_marbles_l638_638512


namespace P2007_gt_P2008_l638_638870

namespace ProbabilityProblem

def probability (k : ℕ) : ℝ := sorry  -- Placeholder for the probability function

axiom probability_rec :
  ∀ n, probability n = (1 / 2007) * (∑ k in finset.range 2007, probability (n - (k + 1)))

axiom P0 :
  probability 0 = 1

theorem P2007_gt_P2008 : probability 2007 > probability 2008 := sorry

end ProbabilityProblem

end P2007_gt_P2008_l638_638870


namespace solve_determinant_l638_638126

theorem solve_determinant (a b x : ℝ) (h₀: a ≠ 0) :
  (|matrix.det ![![x + a, x - b, x + b], ![x - b, x + a, x + b], ![x + b, x + b, x + a]] = 0) ↔
  (x = (-(a^2 - 2*b^2) + real.sqrt((a^2 - 2*b^2)^2 + 16*a^3*b)) / (4*b) ∨
   x = (-(a^2 - 2*b^2) - real.sqrt((a^2 - 2*b^2)^2 + 16*a^3*b)) / (4*b)) := 
sorry

end solve_determinant_l638_638126


namespace convex_polygon_from_non_overlapping_rectangles_is_rectangle_l638_638282

def isConvexPolygon (P : Set Point) : Prop := sorry
def canBeFormedByNonOverlappingRectangles (P : Set Point) (rects: List (Set Point)) : Prop := sorry
def isRectangle (P : Set Point) : Prop := sorry

theorem convex_polygon_from_non_overlapping_rectangles_is_rectangle
  (P : Set Point)
  (rects : List (Set Point))
  (h_convex : isConvexPolygon P)
  (h_form : canBeFormedByNonOverlappingRectangles P rects) :
  isRectangle P :=
sorry

end convex_polygon_from_non_overlapping_rectangles_is_rectangle_l638_638282


namespace trajectory_length_midpoint_l638_638585

theorem trajectory_length_midpoint (m : ℝ) :
  let line_eq := λ x : ℝ, x - m,
      ellipse_eq := λ x y : ℝ, x^2 + y^2 / 2 = 1,
      intersection_points := λ A B : ℝ × ℝ, 
        let A := (x_A, y_A), B := (x_B, y_B) in 
        ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 = A.2 ∧ line_eq B.1 = B.2 ∧ (A ≠ B) in
  let midpoint_eq := λ P : ℝ × ℝ, 
        let P := ((x_A + x_B) / 2, (y_A + y_B) / 2) in 
        ∀ A B : ℝ × ℝ, intersection_points A B → 
        P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  let trajectory_length := 
        ∃ P₁ P₂ : ℝ × ℝ, midpoint_eq P₁ ∧ midpoint_eq P₂ ∧ 
        (P₁ ≠ P₂) ∧ dist P₁ P₂ = (2 * real.sqrt 15) / 3 in
  trajectory_length
:= sorry

end trajectory_length_midpoint_l638_638585


namespace tangent_line_at_t0_normal_line_at_t0_l638_638100

noncomputable def tangent_line_equation (t : ℝ) : ℝ × ℝ :=
(((sin t), (cos t)))

theorem tangent_line_at_t0 :
  let x0 := sin (π/6)
  let y0 := cos (π/6)
  let dy_dx := -tan (π/6)
  ∃ (a b : ℝ), (∀ x : ℝ, y = a * x + b) ∧ a = dy_dx ∧ b = y0 - dy_dx * x0 :=
sorry

theorem normal_line_at_t0 :
  let x0 := sin (π/6)
  let y0 := cos (π/6)
  let dy_dx := -tan (π/6)
  let slope_normal := - (1 / dy_dx)
  ∃ (a b : ℝ), (∀ x : ℝ, y = a * x + b) ∧ a = slope_normal ∧ b = y0 - slope_normal * x0 :=
sorry

end tangent_line_at_t0_normal_line_at_t0_l638_638100


namespace two_digit_numbers_condition_l638_638925

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end two_digit_numbers_condition_l638_638925


namespace find_x_eq_neg15_l638_638113

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l638_638113


namespace player_A_can_ensure_less_than_5_player_A_cannot_ensure_less_than_4_l638_638724

theorem player_A_can_ensure_less_than_5 (positions : Finset ℕ) 
  (h₁ : ∀ n ∈ positions, n > 0 ∧ n < 9)
  (h₂ : ∀ (c : ℕ), c ∈ positions → c < 5):  (* This models that all positions taken by A can lead to a number < 5 *)
  ∃ (position : ℕ), position ∈ positions ∧ position < 5 :=
by sorry

theorem player_A_cannot_ensure_less_than_4 (positions : Finset ℕ)
  (h₁ : ∀ n ∈ positions, n > 0 ∧ n < 9)
  (h₂ : ∀ (c : ℕ), c ∈ positions ∨ c > 3):  (* Model B's counter-strategy ensuring pawn may move to a position marked 4 or higher *)
  ¬ ∃ (position : ℕ), position ∈ positions ∧ position < 4 :=
by sorry

end player_A_can_ensure_less_than_5_player_A_cannot_ensure_less_than_4_l638_638724


namespace sum_leq_sum_l638_638948

theorem sum_leq_sum (n : ℕ) (a b : Fin n → ℝ) 
  (hn : 2 ≤ n)
  (ha : ∀ i j, i ≤ j → a i ≥ a j)
  (hb : ∀ i j, i ≤ j → b i ≥ b j)
  (ha_prod : ∏ i, a i = ∏ i, b i)
  (ha_sum : ∑ i in Finset.range (n - 1), (a 0 - a (i + 1)) ≤ ∑ i in Finset.range (n - 1), (b 0 - b (i + 1))) :
  ∑ i, a i ≤ (n - 1) * ∑ i, b i :=
by
  sorry

end sum_leq_sum_l638_638948


namespace partial_fraction_product_is_correct_l638_638324

-- Given conditions
def fraction_decomposition (x A B C : ℝ) :=
  ( (x^2 + 5 * x - 14) / (x^3 - 3 * x^2 - x + 3) = A / (x - 1) + B / (x - 3) + C / (x + 1) )

-- Statement we want to prove
theorem partial_fraction_product_is_correct (A B C : ℝ) (h : ∀ x : ℝ, fraction_decomposition x A B C) :
  A * B * C = -25 / 2 :=
sorry

end partial_fraction_product_is_correct_l638_638324


namespace find_Y_l638_638951

theorem find_Y (Y : ℕ) 
  (h_top : 2 + 1 + Y + 3 = 6 + Y)
  (h_bottom : 4 + 3 + 1 + 5 = 13)
  (h_equal : 6 + Y = 13) : 
  Y = 7 := 
by
  sorry

end find_Y_l638_638951


namespace productivity_difference_l638_638378

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l638_638378


namespace diana_shops_for_newborns_l638_638080

theorem diana_shops_for_newborns (total_children : ℕ) (num_toddlers : ℕ) (teenager_ratio : ℕ) (num_teens : ℕ) (num_newborns : ℕ)
    (h1 : total_children = 40) (h2 : num_toddlers = 6) (h3 : teenager_ratio = 5) (h4 : num_teens = teenager_ratio * num_toddlers) 
    (h5 : num_newborns = total_children - num_teens - num_toddlers) : 
    num_newborns = 4 := sorry

end diana_shops_for_newborns_l638_638080


namespace first_girl_productivity_higher_l638_638384

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l638_638384


namespace find_x_l638_638178

def vector_a := (-1, 1)
def vector_b (x : ℝ) := (2, x)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := dot_product v1 v2 = 0

theorem find_x (x : ℝ) (h : orthogonal vector_a (vector_a.1 + vector_b(x).1, vector_a.2 + vector_b(x).2)) : x = 0 :=
by
  sorry

end find_x_l638_638178


namespace impossible_to_use_up_components_l638_638342

theorem impossible_to_use_up_components 
  (p q r x y z : ℕ) 
  (condition1 : 2 * x + 2 * z = 2 * p + 2 * r + 2)
  (condition2 : 2 * x + y = 2 * p + q + 1)
  (condition3 : y + z = q + r) : 
  False :=
by sorry

end impossible_to_use_up_components_l638_638342


namespace tangent_square_sum_eq_side_square_sum_l638_638253

theorem tangent_square_sum_eq_side_square_sum
  (A B C D : Point)
  (hABC : acute_triangle A B C)
  (hD : internal_point D (triangle A B C))
  (hTangentA : ∀ k, is_tangent (circle_diameter A D) k → k ≠ A)
  (hTangentB : ∀ k, is_tangent (circle_diameter B D) k → k ≠ B)
  (hTangentC : ∀ k, is_tangent (circle_diameter C D) k → k ≠ C) :
  let AE := length_tangent_segment A (circle_diameter B D)
  let AL := length_tangent_segment A (circle_diameter C D)
  let BF := length_tangent_segment B (circle_diameter A D)
  let BJ := length_tangent_segment B (circle_diameter C D)
  let CG := length_tangent_segment C (circle_diameter A D)
  let CK := length_tangent_segment C (circle_diameter B D)
  let AB := side_length A B
  let BC := side_length B C
  let AC := side_length A C in
  AE^2 + AL^2 + BF^2 + BJ^2 + CG^2 + CK^2 = AB^2 + BC^2 + AC^2 := 
sorry

end tangent_square_sum_eq_side_square_sum_l638_638253


namespace tax_calculation_l638_638883

theorem tax_calculation
  (total_value : ℝ)
  (tax_free_limit : ℝ)
  (tax_rate : ℝ)
  (h1 : total_value = 1720)
  (h2 : tax_free_limit = 800)
  (h3 : tax_rate = 0.10) :
  let taxable_amount := total_value - tax_free_limit in
  let tax := taxable_amount * tax_rate in
  tax = 92 :=
by
  sorry

end tax_calculation_l638_638883


namespace quadratic_roots_greater_than_2_l638_638933

theorem quadratic_roots_greater_than_2 (m : ℝ) :
  (∀ x : ℝ, x^2 + (m - 2) * x + 5 - m = 0 → x > 2) ↔ m ∈ Ioo (-5) (-4) ∨ m = -4 := by
  sorry

end quadratic_roots_greater_than_2_l638_638933


namespace is_function_1_is_not_function_2_is_function_3_l638_638052

-- Definitions
def A1 := {1, 4, 9}
def B1 := {-3, -2, -1, 1, 2, 3}
def f1 (x : ℝ) := Real.sqrt x

def A2 := Set.univ : Set ℝ
def B2 := Set.univ : Set ℝ
def f2 (x : ℝ) := 1 / x

def A3 := Set.univ : Set ℝ
def B3 := Set.univ : Set ℝ
def f3 (x : ℝ) := x^2 - 2

-- Proof theorems
theorem is_function_1 : ∀ x ∈ A1, ∃! y ∈ B1, y = f1 x := by sorry

theorem is_not_function_2 : ¬(∀ x ∈ A2, ∃! y ∈ B2, y = f2 x) := by sorry

theorem is_function_3 : ∀ x ∈ A3, ∃! y ∈ B3, y = f3 x := by sorry

end is_function_1_is_not_function_2_is_function_3_l638_638052


namespace distance_from_A_to_line_CD_l638_638789

noncomputable def distance_A_to_line_CD : ℝ :=
  let A := (0 : ℝ × ℝ)
  let B := (8, 0)
  let C := (12, 0)
  let D := (5, some y_coord_D)
  let E := (some x_coord_E, some y_coord_E)
  let y_coord_D := 0     -- To make D on the line y = 0
  let x_coord_E := 6 · cos_angle
  let y_coord_E := 6 · sin_angle
  let cos_angle := 0.8
  let sin_angle := 0.6
  (65 / 12)  -- The answer discovered

theorem distance_from_A_to_line_CD : distance_A_to_line_CD = 65 / 12 := by
  sorry

end distance_from_A_to_line_CD_l638_638789


namespace exists_triangle_vert_on_rays_sides_pass_points_l638_638272

-- Definitions bringing given rays, points and the construction problem into Lean framework.
variables {O X Y Z : Point}
variables {A B C : Point}
variables {OX OY OZ : Line}
variables [CommonOrigin : IncidenceStructure O OX ∧ IncidenceStructure O OY ∧ IncidenceStructure O OZ]
variables [PointsInAngles : (A ∈ Angle XOY) ∧ (B ∈ Angle XOZ) ∧ (C ∈ Angle YOZ)]

-- Statement to prove the construction possibility.
theorem exists_triangle_vert_on_rays_sides_pass_points (hOXOY : Line OX ∧ Line OY) 
(hOXOZ : Line OX ∧ Line OZ) (hOYOZ : Line OY ∧ Line OZ) (hPointsInAngles : PointsInAngles) :
  ∃ P Q R, (P ∈ OX) ∧ (Q ∈ OY) ∧ (R ∈ OZ) ∧ (SideOfTriangle P Q R A ∧ SideOfTriangle P Q R B ∧ SideOfTriangle P Q R C) :=
begin
  sorry
end

end exists_triangle_vert_on_rays_sides_pass_points_l638_638272


namespace cade_marbles_l638_638005

theorem cade_marbles :
  let initial_marbles := 87 in
  let marbles_given := 8 in
  initial_marbles - marbles_given = 79 :=
by
  intros
  dunfold initial_marbles marbles_given
  sorry

end cade_marbles_l638_638005


namespace line_equation_l638_638968

noncomputable def line_at_point (a b c x y : ℝ) : Prop := a * x + b * y + c = 0
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

theorem line_equation :
  let l := line_at_point in
  let perp := perpendicular in
  ∃ (m : ℝ), l 3 2 m (-1) 2 ∧ perp 3 2 2 (-3) ∧ l 3 2 m (-1) 2 = l 3 2 (-1) (-1) 2 :=
begin
  sorry
end

end line_equation_l638_638968


namespace inverse_34_mod_47_l638_638141

theorem inverse_34_mod_47 (h: 13⁻¹ ≡ 29 [MOD 47]) : 34⁻¹ ≡ 18 [MOD 47] :=
sorry

end inverse_34_mod_47_l638_638141


namespace sqrt_inequality_l638_638278

theorem sqrt_inequality : sqrt 10 - sqrt 5 < sqrt 7 - sqrt 2 := by
  sorry

end sqrt_inequality_l638_638278


namespace arithmetic_seq_nth_term_l638_638890

theorem arithmetic_seq_nth_term :
  ∃ (a₁ d : ℝ), (5 * a₁ + (5 * 4 / 2) * d = 5) ∧
                (a₁ + (a₁ + d) = (a₁ + 2 * d) + (a₁ + 3 * d) + (a₁ + 4 * d)) ∧
                (∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → (a₁ + (n - 1) * d = -⅙ * n + 3 / 2)) :=
begin
  sorry
end

end arithmetic_seq_nth_term_l638_638890


namespace find_b_l638_638094

theorem find_b (b : ℝ) (h : log b 256 = -4/3) : b = 1/64 := 
sorry

end find_b_l638_638094
