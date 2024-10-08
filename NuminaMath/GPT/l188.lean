import Mathlib

namespace notebook_cost_proof_l188_188353

-- Let n be the cost of the notebook and p be the cost of the pen.
variable (n p : ℝ)

-- Conditions:
def total_cost : Prop := n + p = 2.50
def notebook_more_pen : Prop := n = 2 + p

-- Theorem: Prove that the cost of the notebook is $2.25
theorem notebook_cost_proof (h1 : total_cost n p) (h2 : notebook_more_pen n p) : n = 2.25 := 
by 
  sorry

end notebook_cost_proof_l188_188353


namespace geometric_progression_fifth_term_sum_l188_188506

def gp_sum_fifth_term
    (p q : ℝ)
    (hpq_sum : p + q = 3)
    (hpq_6th : p^5 + q^5 = 573) : ℝ :=
p^4 + q^4

theorem geometric_progression_fifth_term_sum :
    ∃ p q : ℝ, p + q = 3 ∧ p^5 + q^5 = 573 ∧ gp_sum_fifth_term p q (by sorry) (by sorry) = 161 :=
by
  sorry

end geometric_progression_fifth_term_sum_l188_188506


namespace arithmetic_geometric_means_l188_188835

theorem arithmetic_geometric_means (a b : ℝ) (h1 : 2 * a = 1 + 2) (h2 : b^2 = (-1) * (-16)) : a * b = 6 ∨ a * b = -6 :=
by
  sorry

end arithmetic_geometric_means_l188_188835


namespace max_volume_of_pyramid_PABC_l188_188155

noncomputable def max_pyramid_volume (PA PB AB BC CA : ℝ) (hPA : PA = 3) (hPB : PB = 3) 
(hAB : AB = 2) (hBC : BC = 2) (hCA : CA = 2) : ℝ :=
  let D := 1 -- Midpoint of segment AB
  let PD : ℝ := Real.sqrt (PA ^ 2 - D ^ 2) -- Distance PD using Pythagorean theorem
  let S_ABC : ℝ := (Real.sqrt 3 / 4) * (AB ^ 2) -- Area of triangle ABC
  let V_PABC : ℝ := (1 / 3) * S_ABC * PD -- Volume of the pyramid
  V_PABC -- Return the volume

theorem max_volume_of_pyramid_PABC : 
  max_pyramid_volume 3 3 2 2 2  (rfl) (rfl) (rfl) (rfl) (rfl) = (2 * Real.sqrt 6) / 3 :=
by
  sorry

end max_volume_of_pyramid_PABC_l188_188155


namespace compound_interest_principal_amount_l188_188220

theorem compound_interest_principal_amount :
  ∀ (r : ℝ) (n : ℕ) (t : ℕ) (CI : ℝ) (P : ℝ),
    r = 0.04 ∧ n = 1 ∧ t = 2 ∧ CI = 612 →
    (CI = P * (1 + r / n) ^ (n * t) - P) →
    P = 7500 :=
by
  intros r n t CI P h_conditions h_CI
  -- Proof not needed
  sorry

end compound_interest_principal_amount_l188_188220


namespace seating_arrangements_l188_188234

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l188_188234


namespace sum_of_three_numbers_l188_188217

theorem sum_of_three_numbers
  (a b c : ℕ) (h_prime : Prime c)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + a * c = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l188_188217


namespace fractions_equiv_l188_188058

theorem fractions_equiv:
  (8 : ℝ) / (7 * 67) = (0.8 : ℝ) / (0.7 * 67) :=
by
  sorry

end fractions_equiv_l188_188058


namespace smallest_n_l188_188701

theorem smallest_n (m l n : ℕ) :
  (∃ m : ℕ, 2 * n = m ^ 4) ∧ (∃ l : ℕ, 3 * n = l ^ 6) → n = 1944 :=
by
  sorry

end smallest_n_l188_188701


namespace find_digit_D_l188_188026

theorem find_digit_D (A B C D : ℕ) (h1 : A + B = A + 10 * (B / 10)) (h2 : D + 10 * (A / 10) = A + C)
  (h3 : A + 10 * (B / 10) - C = A) (h4 : 0 ≤ A) (h5 : A ≤ 9) (h6 : 0 ≤ B) (h7 : B ≤ 9)
  (h8 : 0 ≤ C) (h9 : C ≤ 9) (h10 : 0 ≤ D) (h11 : D ≤ 9) : D = 9 := 
sorry

end find_digit_D_l188_188026


namespace square_field_area_l188_188366

theorem square_field_area (s A : ℝ) (h1 : 10 * 4 * s = 9280) (h2 : A = s^2) : A = 53824 :=
by {
  sorry -- The proof goes here
}

end square_field_area_l188_188366


namespace three_numbers_equal_l188_188275

theorem three_numbers_equal {a b c d : ℕ} 
  (h : ∀ {x y z w : ℕ}, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
                  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) → x * y + z * w = x * z + y * w) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end three_numbers_equal_l188_188275


namespace sum_a10_a11_l188_188312

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)

theorem sum_a10_a11 {a : ℕ → ℝ} (h_seq : geometric_sequence a)
  (h1 : a 1 + a 2 = 2)
  (h4 : a 4 + a 5 = 4) :
  a 10 + a 11 = 16 :=
by {
  sorry
}

end sum_a10_a11_l188_188312


namespace factorize_expression_l188_188231

theorem factorize_expression (a : ℝ) : 
  (2 * a + 1) * a - 4 * a - 2 = (2 * a + 1) * (a - 2) :=
by 
  -- proof is skipped with sorry
  sorry

end factorize_expression_l188_188231


namespace percents_multiplication_l188_188726

theorem percents_multiplication :
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  (p1 * p2 * p3 * p4) * 100 = 5.88 := 
by
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  sorry

end percents_multiplication_l188_188726


namespace circle_standard_equation_l188_188335

theorem circle_standard_equation {a : ℝ} :
  (∃ a : ℝ, a ≠ 0 ∧ (a = 2 * a - 3 ∨ a = 3 - 2 * a) ∧ 
  (((x - a)^2 + (y - (2 * a - 3))^2 = a^2) ∧ 
   ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1))) :=
sorry

end circle_standard_equation_l188_188335


namespace inverse_function_fixed_point_l188_188126

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the condition that graph of y = f(x-1) passes through the point (1, 2)
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

-- State the main theorem to prove
theorem inverse_function_fixed_point {f : ℝ → ℝ} (h : passes_through f 1 2) :
  ∃ x, x = 2 ∧ f x = 0 :=
sorry

end inverse_function_fixed_point_l188_188126


namespace paul_earns_from_license_plates_l188_188216

theorem paul_earns_from_license_plates
  (plates_from_40_states : ℕ)
  (total_50_states : ℕ)
  (reward_per_percentage_point : ℕ)
  (h1 : plates_from_40_states = 40)
  (h2 : total_50_states = 50)
  (h3 : reward_per_percentage_point = 2) :
  (40 / 50) * 100 * 2 = 160 := 
sorry

end paul_earns_from_license_plates_l188_188216


namespace find_roots_of_polynomial_l188_188182

noncomputable def polynomial := Polynomial ℝ

theorem find_roots_of_polynomial :
  (∃ (x : ℝ), x^3 + 3 * x^2 - 6 * x - 8 = 0) ↔ (x = -1 ∨ x = 2 ∨ x = -4) :=
sorry

end find_roots_of_polynomial_l188_188182


namespace positive_difference_median_mode_l188_188055

-- Definition of the data set
def data : List ℕ := [12, 13, 14, 15, 15, 22, 22, 22, 31, 31, 38, 39, 40, 41, 42, 43, 52, 58, 59]

-- Definition of the mode
def mode (l : List ℕ) : ℕ := 22  -- Specific to the data set provided

-- Definition of the median
def median (l : List ℕ) : ℕ := 31  -- Specific to the data set provided

-- Proof statement
theorem positive_difference_median_mode : 
  (median data - mode data) = 9 := by 
  sorry

end positive_difference_median_mode_l188_188055


namespace solve_x_plus_y_l188_188631

variable {x y : ℚ} -- Declare x and y as rational numbers

theorem solve_x_plus_y
  (h1: (1 / x) + (1 / y) = 1)
  (h2: (1 / x) - (1 / y) = 5) :
  x + y = -1 / 6 :=
sorry

end solve_x_plus_y_l188_188631


namespace compute_Q3_Qneg3_l188_188175

noncomputable def Q (x : ℝ) (a b c m : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + m

theorem compute_Q3_Qneg3 (a b c m : ℝ)
  (h1 : Q 1 a b c m = 3 * m)
  (h2 : Q (-1) a b c m = 4 * m)
  (h3 : Q 0 a b c m = m) :
  Q 3 a b c m + Q (-3) a b c m = 47 * m :=
by
  sorry

end compute_Q3_Qneg3_l188_188175


namespace quadratic_intersects_once_l188_188849

theorem quadratic_intersects_once (c : ℝ) : (∀ x : ℝ, x^2 - 6 * x + c = 0 → x = 3 ) ↔ c = 9 :=
by
  sorry

end quadratic_intersects_once_l188_188849


namespace blue_sequins_per_row_l188_188770

theorem blue_sequins_per_row : 
  ∀ (B : ℕ),
  (6 * B) + (5 * 12) + (9 * 6) = 162 → B = 8 :=
by
  intro B
  sorry

end blue_sequins_per_row_l188_188770


namespace remainder_123456789012_div_252_l188_188584

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l188_188584


namespace least_n_satisfies_condition_l188_188325

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l188_188325


namespace combined_gross_profit_correct_l188_188793

def calculate_final_selling_price (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  let marked_up_price := initial_price * (1 + markup)
  let final_price := List.foldl (λ price discount => price * (1 - discount)) marked_up_price discounts
  final_price

def calculate_gross_profit (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  calculate_final_selling_price initial_price markup discounts - initial_price

noncomputable def combined_gross_profit : ℝ :=
  let earrings_gross_profit := calculate_gross_profit 240 0.25 [0.15]
  let bracelet_gross_profit := calculate_gross_profit 360 0.30 [0.10, 0.05]
  let necklace_gross_profit := calculate_gross_profit 480 0.40 [0.20, 0.05]
  let ring_gross_profit := calculate_gross_profit 600 0.35 [0.10, 0.05, 0.02]
  let pendant_gross_profit := calculate_gross_profit 720 0.50 [0.20, 0.03, 0.07]
  earrings_gross_profit + bracelet_gross_profit + necklace_gross_profit + ring_gross_profit + pendant_gross_profit

theorem combined_gross_profit_correct : combined_gross_profit = 224.97 :=
  by
  sorry

end combined_gross_profit_correct_l188_188793


namespace rectangle_area_increase_l188_188271

theorem rectangle_area_increase (b : ℕ) (h1 : 2 * b = 40) (h2 : b = 20) : 
  let l := 2 * b
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 5
  let A_new := l_new * b_new
  A_new - A_original = 75 := 
by
  sorry

end rectangle_area_increase_l188_188271


namespace domain_of_f_l188_188766

theorem domain_of_f (c : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 5 * x + c ≠ 0) ↔ c < -25 / 28 :=
by
  sorry

end domain_of_f_l188_188766


namespace fraction_product_eq_l188_188418
-- Import the necessary library

-- Define the fractions and the product
def fraction_product : ℚ :=
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8)

-- State the theorem we want to prove
theorem fraction_product_eq : fraction_product = 3 / 8 := 
sorry

end fraction_product_eq_l188_188418


namespace smallest_class_size_l188_188644

theorem smallest_class_size (N : ℕ) (G : ℕ) (h1: 0.25 < (G : ℝ) / N) (h2: (G : ℝ) / N < 0.30) : N = 7 := 
sorry

end smallest_class_size_l188_188644


namespace probability_blue_point_l188_188460

-- Definitions of the random points
def is_random_point (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2

-- Definition of the condition for the probability problem
def condition (x y : ℝ) : Prop :=
  x < y ∧ y < 3 * x

-- Statement of the theorem
theorem probability_blue_point (x y : ℝ) (h1 : is_random_point x) (h2 : is_random_point y) :
  ∃ p : ℝ, (p = 1 / 3) ∧ (∃ (hx : x < y) (hy : y < 3 * x), x ≤ 2 ∧ 0 ≤ x ∧ y ≤ 2 ∧ 0 ≤ y) :=
by
  sorry

end probability_blue_point_l188_188460


namespace possible_values_of_m_l188_188978

theorem possible_values_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 1}) (hB : B = {x | m * x = 1}) (hUnion : A ∪ B = A) : m = 0 ∨ m = 1 ∨ m = -1 :=
sorry

end possible_values_of_m_l188_188978


namespace count_four_digit_numbers_with_repeated_digits_l188_188843

def countDistinctFourDigitNumbersWithRepeatedDigits : Nat :=
  let totalNumbers := 4 ^ 4
  let uniqueNumbers := 4 * 3 * 2 * 1
  totalNumbers - uniqueNumbers

theorem count_four_digit_numbers_with_repeated_digits :
  countDistinctFourDigitNumbersWithRepeatedDigits = 232 := by
  sorry

end count_four_digit_numbers_with_repeated_digits_l188_188843


namespace find_A_l188_188152

variable (U A CU_A : Set ℕ)

axiom U_is_universal : U = {1, 3, 5, 7, 9}
axiom CU_A_is_complement : CU_A = {5, 7}

theorem find_A (h1 : U = {1, 3, 5, 7, 9}) (h2 : CU_A = {5, 7}) : 
  A = {1, 3, 9} :=
by
  sorry

end find_A_l188_188152


namespace triangle_is_isosceles_l188_188459

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_sum_angles : A + B + C = π)
  (h_condition : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end triangle_is_isosceles_l188_188459


namespace sandy_friend_puppies_l188_188066

theorem sandy_friend_puppies (original_puppies friend_puppies final_puppies : ℕ)
    (h1 : original_puppies = 8) (h2 : final_puppies = 12) :
    friend_puppies = final_puppies - original_puppies := by
    sorry

end sandy_friend_puppies_l188_188066


namespace fractions_inequality_l188_188398

variable {a b c d : ℝ}
variable (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0)

theorem fractions_inequality : 
  (a > b) → (b > 0) → (c < d) → (d < 0) → (a / d < b / c) :=
by
  intros h1 h2 h3 h4
  sorry

end fractions_inequality_l188_188398


namespace james_baked_multiple_l188_188333

theorem james_baked_multiple (x : ℕ) (h1 : 115 ≠ 0) (h2 : 1380 = 115 * x) : x = 12 :=
sorry

end james_baked_multiple_l188_188333


namespace eq_determines_ratio_l188_188473

theorem eq_determines_ratio (a b x y : ℝ) (h : a * x^3 + b * x^2 * y + b * x * y^2 + a * y^3 = 0) :
  ∃ t : ℝ, t = x / y ∧ (a * t^3 + b * t^2 + b * t + a = 0) :=
sorry

end eq_determines_ratio_l188_188473


namespace complement_union_l188_188379

variable (U : Set ℤ) (A : Set ℤ) (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := 
by 
  -- Proof is omitted
  sorry

end complement_union_l188_188379


namespace division_of_field_l188_188628

theorem division_of_field :
  (∀ (hectares : ℕ) (parts : ℕ), hectares = 5 ∧ parts = 8 →
  (1 / parts = 1 / 8) ∧ (hectares / parts = 5 / 8)) :=
by
  sorry


end division_of_field_l188_188628


namespace element_of_M_l188_188720

def M : Set (ℕ × ℕ) := { (2, 3) }

theorem element_of_M : (2, 3) ∈ M :=
by
  sorry

end element_of_M_l188_188720


namespace linear_function_no_fourth_quadrant_l188_188546

theorem linear_function_no_fourth_quadrant (k : ℝ) (hk : k > 2) : 
  ∀ x (hx : x > 0), (k-2) * x + k ≥ 0 :=
by
  sorry

end linear_function_no_fourth_quadrant_l188_188546


namespace minimum_boxes_to_eliminate_50_percent_chance_l188_188400

def total_boxes : Nat := 30
def high_value_boxes : Nat := 6
def minimum_boxes_to_eliminate (total_boxes high_value_boxes : Nat) : Nat :=
  total_boxes - high_value_boxes - high_value_boxes

theorem minimum_boxes_to_eliminate_50_percent_chance :
  minimum_boxes_to_eliminate total_boxes high_value_boxes = 18 :=
by
  sorry

end minimum_boxes_to_eliminate_50_percent_chance_l188_188400


namespace factorize_polynomial_l188_188619

theorem factorize_polynomial (m : ℤ) : 4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end factorize_polynomial_l188_188619


namespace upgraded_video_card_multiple_l188_188328

noncomputable def multiple_of_video_card_cost (computer_cost monitor_cost_peripheral_cost base_video_card_cost total_spent upgraded_video_card_cost : ℝ) : ℝ :=
  upgraded_video_card_cost / base_video_card_cost

theorem upgraded_video_card_multiple
  (computer_cost : ℝ)
  (monitor_cost_ratio : ℝ)
  (base_video_card_cost : ℝ)
  (total_spent : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : monitor_cost_ratio = 1/5)
  (h3 : base_video_card_cost = 300)
  (h4 : total_spent = 2100) :
  multiple_of_video_card_cost computer_cost (computer_cost * monitor_cost_ratio) base_video_card_cost total_spent (total_spent - (computer_cost + computer_cost * monitor_cost_ratio)) = 1 :=
by
  sorry

end upgraded_video_card_multiple_l188_188328


namespace max_ratio_lemma_l188_188094

theorem max_ratio_lemma (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn : ∀ n, S n = (n + 1) / 2 * a n)
  (hSn_minus_one : ∀ n, S (n - 1) = n / 2 * a (n - 1)) :
  ∀ n > 1, (a n / a (n - 1) ≤ 2) ∧ (a 2 / a 1 = 2) := sorry

end max_ratio_lemma_l188_188094


namespace math_club_team_selection_l188_188638

open scoped BigOperators

def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem math_club_team_selection :
  (comb 7 2 * comb 9 4) + 
  (comb 7 3 * comb 9 3) +
  (comb 7 4 * comb 9 2) +
  (comb 7 5 * comb 9 1) +
  (comb 7 6 * comb 9 0) = 7042 := 
sorry

end math_club_team_selection_l188_188638


namespace impossible_to_arrange_circle_l188_188887

theorem impossible_to_arrange_circle : 
  ¬∃ (f : Fin 10 → Fin 10), 
    (∀ i : Fin 10, (abs ((f i).val - (f (i + 1)).val : Int) = 3 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 4 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 5)) :=
sorry

end impossible_to_arrange_circle_l188_188887


namespace ivy_covering_the_tree_l188_188757

def ivy_stripped_per_day := 6
def ivy_grows_per_night := 2
def days_to_strip := 10
def net_ivy_stripped_per_day := ivy_stripped_per_day - ivy_grows_per_night

theorem ivy_covering_the_tree : net_ivy_stripped_per_day * days_to_strip = 40 := by
  have h1 : net_ivy_stripped_per_day = 4 := by
    unfold net_ivy_stripped_per_day
    rfl
  rw [h1]
  show 4 * 10 = 40
  rfl

end ivy_covering_the_tree_l188_188757


namespace ratio_Mandy_to_Pamela_l188_188603

-- Definitions based on conditions in the problem
def exam_items : ℕ := 100
def Lowella_correct : ℕ := (35 * exam_items) / 100  -- 35% of 100
def Pamela_correct : ℕ := Lowella_correct + (20 * Lowella_correct) / 100 -- 20% more than Lowella
def Mandy_score : ℕ := 84

-- The proof problem statement
theorem ratio_Mandy_to_Pamela : Mandy_score / Pamela_correct = 2 := by
  sorry

end ratio_Mandy_to_Pamela_l188_188603


namespace find_n_l188_188308

-- Given Variables
variables (n x y : ℝ)

-- Given Conditions
axiom h1 : n * x = 6 * y
axiom h2 : x * y ≠ 0
axiom h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998

-- Conclusion
theorem find_n : n = 5 := sorry

end find_n_l188_188308


namespace sport_formulation_water_content_l188_188237

theorem sport_formulation_water_content :
  ∀ (f_s c_s w_s : ℕ) (f_p c_p w_p : ℕ),
    f_s / c_s = 1 / 12 →
    f_s / w_s = 1 / 30 →
    f_p / c_p = 1 / 4 →
    f_p / w_p = 1 / 60 →
    c_p = 4 →
    w_p = 60 := by
  sorry

end sport_formulation_water_content_l188_188237


namespace find_g2_l188_188437

variable {R : Type*} [Nonempty R] [Field R]

-- Define the function g
def g (x : R) : R := sorry

-- Given conditions
axiom condition1 : ∀ x y : R, x * g y = 2 * y * g x
axiom condition2 : g 10 = 5

-- The statement to be proved
theorem find_g2 : g 2 = 2 :=
by
  sorry

end find_g2_l188_188437


namespace union_of_A_and_B_l188_188579

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 5} := 
by
  sorry

end union_of_A_and_B_l188_188579


namespace range_of_x_l188_188107

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_x (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
                   (f_at_one_third : f (1/3) = 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | (0 < x ∧ x < 1/2) ∨ 2 < x} :=
sorry

end range_of_x_l188_188107


namespace evaluate_at_two_l188_188796

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluate_at_two : f (g 2) + g (f 2) = 38 / 7 := by
  sorry

end evaluate_at_two_l188_188796


namespace length_ST_l188_188834

theorem length_ST (PQ QR RS SP SQ PT RT : ℝ) 
  (h1 : PQ = 6) (h2 : QR = 6)
  (h3 : RS = 6) (h4 : SP = 6)
  (h5 : SQ = 6) (h6 : PT = 14)
  (h7 : RT = 14) : 
  ∃ ST : ℝ, ST = 10 := 
by
  -- sorry is used to complete the theorem without a proof
  sorry

end length_ST_l188_188834


namespace range_of_a_l188_188527

open Function

theorem range_of_a (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ > f x₂) (a : ℝ) (h_gt : f a > f 2) : a < -2 ∨ a > 2 :=
  sorry

end range_of_a_l188_188527


namespace find_d_values_l188_188183

theorem find_d_values (u v : ℝ) (c d : ℝ)
  (hpu : u^3 + c * u + d = 0)
  (hpv : v^3 + c * v + d = 0)
  (hqu : (u + 2)^3 + c * (u + 2) + d - 120 = 0)
  (hqv : (v - 5)^3 + c * (v - 5) + d - 120 = 0) :
  d = 396 ∨ d = 8 :=
by
  -- placeholder for the actual proof
  sorry

end find_d_values_l188_188183


namespace inequality_holds_for_all_x_l188_188232

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3 / 5 < a ∧ a ≤ 1) :=
by
  sorry

end inequality_holds_for_all_x_l188_188232


namespace find_positive_integer_n_l188_188410

theorem find_positive_integer_n (n : ℕ) (hpos : 0 < n) : 
  (n + 1) ∣ (2 * n^2 + 5 * n) ↔ n = 2 :=
by
  sorry

end find_positive_integer_n_l188_188410


namespace sum_of_numbers_l188_188212

theorem sum_of_numbers (x : ℝ) 
  (h_ratio : ∃ x, (2 * x) / x = 2 ∧ (3 * x) / x = 3)
  (h_squares : x^2 + (2 * x)^2 + (3 * x)^2 = 2744) :
  x + 2 * x + 3 * x = 84 :=
by
  sorry

end sum_of_numbers_l188_188212


namespace evaluate_expression_l188_188615

-- Define the given numbers as real numbers
def x : ℝ := 175.56
def y : ℝ := 54321
def z : ℝ := 36947
def w : ℝ := 1521

-- State the theorem to be proved
theorem evaluate_expression : (x / y) * (z / w) = 0.07845 :=
by 
  -- We skip the proof here
  sorry

end evaluate_expression_l188_188615


namespace intersection_M_N_l188_188148

open Set

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | x^2 - 2*x - 3 < 0}
def intersection_sets := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = intersection_sets :=
  sorry

end intersection_M_N_l188_188148


namespace find_number_l188_188084

theorem find_number (N : ℝ) (h : (5/4 : ℝ) * N = (4/5 : ℝ) * N + 27) : N = 60 :=
by
  sorry

end find_number_l188_188084


namespace union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l188_188288

def setA (a : ℝ) : Set ℝ := { x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3 }
def setB : Set ℝ := { x | -1 / 2 < x ∧ x < 2 }
def complementB : Set ℝ := { x | x ≤ -1 / 2 ∨ x ≥ 2 }

theorem union_complement_A_when_a_eq_1 :
  (complementB ∪ setA 1) = { x | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

theorem A_cap_B_eq_A_range_of_a (a : ℝ) :
  (setA a ∩ setB = setA a) → (-1 < a ∧ a ≤ 1) :=
by
  sorry

end union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l188_188288


namespace elevator_max_weight_l188_188851

theorem elevator_max_weight :
  let avg_weight_adult := 150
  let num_adults := 7
  let avg_weight_child := 70
  let num_children := 5
  let orig_max_weight := 1500
  let weight_adults := num_adults * avg_weight_adult
  let weight_children := num_children * avg_weight_child
  let current_weight := weight_adults + weight_children
  let upgrade_percentage := 0.10
  let new_max_weight := orig_max_weight * (1 + upgrade_percentage)
  new_max_weight - current_weight = 250 := 
  by
    sorry

end elevator_max_weight_l188_188851


namespace inequality_proof_l188_188639

variable (a b c d : ℝ)

theorem inequality_proof
  (h_pos: 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1)
  (h_product: a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 :=
by
  sorry

end inequality_proof_l188_188639


namespace breakfast_time_correct_l188_188868

noncomputable def breakfast_time_calc (x : ℚ) : ℚ :=
  (7 * 60) + (300 / 13)

noncomputable def coffee_time_calc (y : ℚ) : ℚ :=
  (7 * 60) + (420 / 11)

noncomputable def total_breakfast_time : ℚ :=
  coffee_time_calc ((420 : ℚ) / 11) - breakfast_time_calc ((300 : ℚ) / 13)

theorem breakfast_time_correct :
  total_breakfast_time = 15 + (6 / 60) :=
by
  sorry

end breakfast_time_correct_l188_188868


namespace mark_performance_length_l188_188813

theorem mark_performance_length :
  ∃ (x : ℕ), (x > 0) ∧ (6 * 5 * x = 90) ∧ (x = 3) :=
by
  sorry

end mark_performance_length_l188_188813


namespace second_quadrant_y_value_l188_188315

theorem second_quadrant_y_value :
  ∀ (b : ℝ), (-3, b).2 > 0 → b = 2 :=
by
  sorry

end second_quadrant_y_value_l188_188315


namespace factorization_correct_l188_188981

noncomputable def original_poly (x : ℝ) : ℝ := 12 * x ^ 2 + 18 * x - 24
noncomputable def factored_poly (x : ℝ) : ℝ := 6 * (2 * x - 1) * (x + 4)

theorem factorization_correct (x : ℝ) : original_poly x = factored_poly x :=
by
  sorry

end factorization_correct_l188_188981


namespace craft_store_pricing_maximize_daily_profit_l188_188453

theorem craft_store_pricing (profit_per_item marked_price cost_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₂ : 8 * 0.85 * marked_price + 12 * (marked_price - 35) = 20 * cost_price)
  : cost_price = 155 ∧ marked_price = 200 := 
sorry

theorem maximize_daily_profit (profit_per_item cost_price marked_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₃ : ∀ p : ℝ, (100 + 4 * (200 - p)) * (p - cost_price) ≤ 4900)
  : p = 190 ∧ daily_profit = 4900 :=
sorry

end craft_store_pricing_maximize_daily_profit_l188_188453


namespace value_of_x_for_g_equals_g_inv_l188_188341

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l188_188341


namespace total_length_figure2_l188_188555

-- Define the initial lengths of each segment in Figure 1.
def initial_length_horizontal1 := 5
def initial_length_vertical1 := 10
def initial_length_horizontal2 := 4
def initial_length_vertical2 := 3
def initial_length_horizontal3 := 3
def initial_length_vertical3 := 5
def initial_length_horizontal4 := 4
def initial_length_vertical_sum := 10 + 3 + 5

-- Define the transformations.
def bottom_length := initial_length_horizontal1
def rightmost_vertical_length := initial_length_vertical1 - 2
def top_horizontal_length := initial_length_horizontal2 - 3
def leftmost_vertical_length := initial_length_vertical1

-- Define the total length in Figure 2 as a theorem to be proved.
theorem total_length_figure2:
  bottom_length + rightmost_vertical_length + top_horizontal_length + leftmost_vertical_length = 24 := by
  sorry

end total_length_figure2_l188_188555


namespace person_A_money_left_l188_188504

-- We define the conditions and question in terms of Lean types.
def initial_money_ratio : ℚ := 7 / 6
def money_spent_A : ℚ := 50
def money_spent_B : ℚ := 60
def final_money_ratio : ℚ := 3 / 2
def x : ℚ := 30

-- The theorem to prove the amount of money left by person A
theorem person_A_money_left 
  (init_ratio : initial_money_ratio = 7 / 6)
  (spend_A : money_spent_A = 50)
  (spend_B : money_spent_B = 60)
  (final_ratio : final_money_ratio = 3 / 2)
  (hx : x = 30) : 3 * x = 90 := by 
  sorry

end person_A_money_left_l188_188504


namespace smallest_positive_angle_same_terminal_side_l188_188830

theorem smallest_positive_angle_same_terminal_side : 
  ∃ k : ℤ, (∃ α : ℝ, α > 0 ∧ α = -660 + k * 360) ∧ (∀ β : ℝ, β > 0 ∧ β = -660 + k * 360 → β ≥ α) :=
sorry

end smallest_positive_angle_same_terminal_side_l188_188830


namespace complement_of_A_l188_188705

def A : Set ℝ := {y : ℝ | ∃ (x : ℝ), y = 2^x}

theorem complement_of_A : (Set.compl A) = {y : ℝ | y ≤ 0} :=
by
  sorry

end complement_of_A_l188_188705


namespace c_sub_a_eq_60_l188_188089

theorem c_sub_a_eq_60 (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := 
by 
  sorry

end c_sub_a_eq_60_l188_188089


namespace find_f_2015_l188_188590

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_periodic_2 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom f_at_1 : f 1 = 2

theorem find_f_2015 : f 2015 = 13 / 2 :=
by
  sorry

end find_f_2015_l188_188590


namespace workers_count_l188_188261

noncomputable def numberOfWorkers (W: ℕ) : Prop :=
  let old_supervisor_salary := 870
  let new_supervisor_salary := 690
  let avg_old := 430
  let avg_new := 410
  let total_after_old := (W + 1) * avg_old
  let total_after_new := 9 * avg_new
  total_after_old - old_supervisor_salary = total_after_new - new_supervisor_salary

theorem workers_count : numberOfWorkers 8 :=
by
  sorry

end workers_count_l188_188261


namespace James_total_area_l188_188760

theorem James_total_area :
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  total_area = 1800 :=
by
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  have h : total_area = 1800 := by sorry
  exact h

end James_total_area_l188_188760


namespace solution_set_inequality_l188_188700

theorem solution_set_inequality : 
  {x : ℝ | abs ((x - 3) / x) > ((x - 3) / x)} = {x : ℝ | 0 < x ∧ x < 3} :=
sorry

end solution_set_inequality_l188_188700


namespace find_number_l188_188993

theorem find_number (N : ℝ) (h : 0.1 * 0.3 * 0.5 * N = 90) : N = 6000 :=
by
  sorry

end find_number_l188_188993


namespace domain_of_f_l188_188841

theorem domain_of_f :
  (∀ x : ℝ, (0 < 1 - x) ∧ (0 < 3 * x + 1) ↔ ( - (1 / 3 : ℝ) < x ∧ x < 1)) :=
by
  sorry

end domain_of_f_l188_188841


namespace find_number_l188_188431

theorem find_number (x : ℤ) (h : 5 * (x - 12) = 40) : x = 20 := 
by
  sorry

end find_number_l188_188431


namespace cost_of_first_shirt_l188_188267

theorem cost_of_first_shirt (x : ℝ) (h1 : x + (x + 6) = 24) : x + 6 = 15 :=
by
  sorry

end cost_of_first_shirt_l188_188267


namespace compute_value_l188_188812

theorem compute_value {p q : ℝ} (h1 : 3 * p^2 - 5 * p - 8 = 0) (h2 : 3 * q^2 - 5 * q - 8 = 0) :
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 :=
by
  sorry

end compute_value_l188_188812


namespace inequality_solution_l188_188028

open Set Real

theorem inequality_solution (x : ℝ) :
  (1 / (x + 1) + 3 / (x + 7) ≥ 2 / 3) ↔ (x ∈ Ioo (-7 : ℝ) (-4) ∪ Ioo (-1) (2) ∪ {(-4 : ℝ), 2}) :=
by sorry

end inequality_solution_l188_188028


namespace first_term_geometric_sequence_l188_188929

theorem first_term_geometric_sequence (a r : ℕ) (h1 : r = 3) (h2 : a * r^4 = 81) : a = 1 :=
by
  sorry

end first_term_geometric_sequence_l188_188929


namespace part1_part2_l188_188651

noncomputable def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

theorem part1 (a : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → f x ≤ a) ↔ a ≥ 4 :=
sorry

theorem part2 : {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ -4 < x ∧ x < 1} :=
sorry

end part1_part2_l188_188651


namespace line_quadrant_relationship_l188_188442

theorem line_quadrant_relationship
  (a b c : ℝ)
  (passes_first_second_fourth : ∀ x y : ℝ, (a * x + b * y + c = 0) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) :
  (a * b > 0) ∧ (b * c < 0) :=
sorry

end line_quadrant_relationship_l188_188442


namespace justin_current_age_l188_188616

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end justin_current_age_l188_188616


namespace bug_meeting_point_l188_188135
-- Import the necessary library

-- Define the side lengths of the triangle
variables (DE EF FD : ℝ)
variables (bugs_meet : ℝ)

-- State the conditions and the result
theorem bug_meeting_point
  (h1 : DE = 6)
  (h2 : EF = 8)
  (h3 : FD = 10)
  (h4 : bugs_meet = 1 / 2 * (DE + EF + FD)) :
  bugs_meet - DE = 6 :=
by
  sorry

end bug_meeting_point_l188_188135


namespace points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l188_188587

-- Problem 1: Prove that if \(x^3 + y^3 + z^3 = (x + y + z)^3\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_cubic_eq (x y z : ℝ) (h : x^3 + y^3 + z^3 = (x + y + z)^3) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

-- Problem 2: Prove that if \(x^5 + y^5 + z^5 = (x + y + z)^5\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_quintic_eq (x y z : ℝ) (h : x^5 + y^5 + z^5 = (x + y + z)^5) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

end points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l188_188587


namespace tangent_line_min_slope_l188_188798

noncomputable def curve (x : ℝ) : ℝ := x^3 + 3*x - 1

noncomputable def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 3

theorem tangent_line_min_slope :
  ∃ k b : ℝ, (∀ x : ℝ, curve_derivative x ≥ 3) ∧ 
             k = 3 ∧ b = 1 ∧
             (∀ x y : ℝ, y = k * x + b ↔ 3 * x - y + 1 = 0) := 
by {
  sorry
}

end tangent_line_min_slope_l188_188798


namespace jonathan_daily_burn_l188_188387

-- Conditions
def daily_calories : ℕ := 2500
def extra_saturday_calories : ℕ := 1000
def weekly_deficit : ℕ := 2500

-- Question and Answer
theorem jonathan_daily_burn :
  let weekly_intake := 6 * daily_calories + (daily_calories + extra_saturday_calories)
  let total_weekly_burn := weekly_intake + weekly_deficit
  total_weekly_burn / 7 = 3000 :=
by
  sorry

end jonathan_daily_burn_l188_188387


namespace number_of_rabbits_is_38_l188_188966

-- Conditions: 
def ducks : ℕ := 52
def chickens : ℕ := 78
def condition (ducks rabbits chickens : ℕ) : Prop := 
  chickens = ducks + rabbits - 12

-- Statement: Prove that the number of rabbits is 38
theorem number_of_rabbits_is_38 : ∃ R : ℕ, condition ducks R chickens ∧ R = 38 := by
  sorry

end number_of_rabbits_is_38_l188_188966


namespace x_is_half_l188_188465

theorem x_is_half (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : x = 0.5 :=
sorry

end x_is_half_l188_188465


namespace sum_of_babies_ages_in_five_years_l188_188630

-- Given Definitions
def lioness_age := 12
def hyena_age := lioness_age / 2
def lioness_baby_age := lioness_age / 2
def hyena_baby_age := hyena_age / 2

-- The declaration of the statement to be proven
theorem sum_of_babies_ages_in_five_years : (lioness_baby_age + 5) + (hyena_baby_age + 5) = 19 :=
by 
  sorry 

end sum_of_babies_ages_in_five_years_l188_188630


namespace complex_number_pow_two_l188_188142

theorem complex_number_pow_two (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by sorry

end complex_number_pow_two_l188_188142


namespace work_increase_percentage_l188_188115

theorem work_increase_percentage (p : ℕ) (hp : p > 0) : 
  let absent_fraction := 1 / 6
  let work_per_person_original := 1 / p
  let present_people := p - p * absent_fraction
  let work_per_person_new := 1 / present_people
  let work_increase := work_per_person_new - work_per_person_original
  let percentage_increase := (work_increase / work_per_person_original) * 100
  percentage_increase = 20 :=
by
  sorry

end work_increase_percentage_l188_188115


namespace S_nine_l188_188193

noncomputable def S : ℕ → ℚ
| 3 => 8
| 6 => 10
| _ => 0  -- Placeholder for other values, as we're interested in these specific ones

theorem S_nine (S_3_eq : S 3 = 8) (S_6_eq : S 6 = 10) : S 9 = 21 / 2 :=
by
  -- Construct the proof here
  sorry

end S_nine_l188_188193


namespace find_integer_sets_l188_188204

noncomputable def satisfy_equation (A B C : ℤ) : Prop :=
  A ^ 2 - B ^ 2 - C ^ 2 = 1 ∧ B + C - A = 3

theorem find_integer_sets :
  { (A, B, C) : ℤ × ℤ × ℤ | satisfy_equation A B C } = {(9, 8, 4), (9, 4, 8), (-3, 2, -2), (-3, -2, 2)} :=
  sorry

end find_integer_sets_l188_188204


namespace sqrt_sum_equality_l188_188452

theorem sqrt_sum_equality :
  (Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6) :=
by
  sorry

end sqrt_sum_equality_l188_188452


namespace student_marks_problem_l188_188609

-- Define the variables
variables (M P C X : ℕ)

-- State the conditions
-- Condition 1: M + P = 70
def condition1 : Prop := M + P = 70

-- Condition 2: C = P + X
def condition2 : Prop := C = P + X

-- Condition 3: (M + C) / 2 = 45
def condition3 : Prop := (M + C) / 2 = 45

-- The theorem stating the problem
theorem student_marks_problem (h1 : condition1 M P) (h2 : condition2 C P X) (h3 : condition3 M C) : X = 20 :=
by sorry

end student_marks_problem_l188_188609


namespace compare_xyz_l188_188995

open Real

theorem compare_xyz (x y z : ℝ) : x = Real.log π → y = log 2 / log 5 → z = exp (-1 / 2) → y < z ∧ z < x := by
  intros h_x h_y h_z
  sorry

end compare_xyz_l188_188995


namespace ed_total_pets_l188_188954

theorem ed_total_pets (num_dogs num_cats : ℕ) (h_dogs : num_dogs = 2) (h_cats : num_cats = 3) :
  ∃ num_fish : ℕ, (num_fish = 2 * (num_dogs + num_cats)) ∧ (num_dogs + num_cats + num_fish) = 15 :=
by
  sorry

end ed_total_pets_l188_188954


namespace calculate_expression_l188_188667

theorem calculate_expression : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end calculate_expression_l188_188667


namespace inequality_solution_l188_188810

theorem inequality_solution 
  (x : ℝ) 
  (h : 2*x^4 + x^2 - 4*x - 3*x^2 * |x - 2| + 4 ≥ 0) : 
  x ∈ Set.Iic (-2) ∪ Set.Icc ((-1 - Real.sqrt 17) / 4) ((-1 + Real.sqrt 17) / 4) ∪ Set.Ici 1 :=
sorry

end inequality_solution_l188_188810


namespace part_a_part_b_l188_188989

-- Let p_k represent the probability that at the moment of completing the first collection, the second collection is missing exactly k crocodiles.
def p (k : ℕ) : ℝ := sorry

-- The conditions 
def totalCrocodiles : ℕ := 10
def probabilityEachEgg : ℝ := 0.1

-- Problems to prove:

-- (a) Prove that p_1 = p_2
theorem part_a : p 1 = p 2 := sorry

-- (b) Prove that p_2 > p_3 > p_4 > ... > p_10
theorem part_b : ∀ k, 2 ≤ k ∧ k < totalCrocodiles → p k > p (k + 1) := sorry

end part_a_part_b_l188_188989


namespace line_equation_midpoint_ellipse_l188_188268

theorem line_equation_midpoint_ellipse (x1 y1 x2 y2 : ℝ) 
  (h_midpoint_x : x1 + x2 = 4) (h_midpoint_y : y1 + y2 = 2)
  (h_ellipse_x1_y1 : (x1^2) / 12 + (y1^2) / 4 = 1) (h_ellipse_x2_y2 : (x2^2) / 12 + (y2^2) / 4 = 1) :
  2 * (x1 - x2) + 3 * (y1 - y2) = 0 :=
sorry

end line_equation_midpoint_ellipse_l188_188268


namespace min_AB_dot_CD_l188_188450

theorem min_AB_dot_CD (a b : ℝ) (h1 : 0 <= (a - 1)^2 + (b - 3 / 2)^2 - 13/4) :
  ∃ (a b : ℝ), (a-1)^2 + (b - 3 / 2)^2 - 13/4 = 0 :=
by
  sorry

end min_AB_dot_CD_l188_188450


namespace problem1_problem2_l188_188665

-- Define the total number of balls for clarity
def total_red_balls : ℕ := 4
def total_white_balls : ℕ := 6
def total_balls_drawn : ℕ := 4

-- Define binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := n.choose k

-- Problem 1: Prove that the number of ways to draw 4 balls that include both colors is 194
theorem problem1 :
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) +
  (binom total_red_balls 1 * binom total_white_balls 3) = 194 :=
  sorry

-- Problem 2: Prove that the number of ways to draw 4 balls where the number of red balls is at least the number of white balls is 115
theorem problem2 :
  (binom total_red_balls 4 * binom total_white_balls 0) +
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) = 115 :=
  sorry

end problem1_problem2_l188_188665


namespace alice_outfits_l188_188201

theorem alice_outfits :
  let trousers := 5
  let shirts := 8
  let jackets := 4
  let shoes := 2
  trousers * shirts * jackets * shoes = 320 :=
by
  sorry

end alice_outfits_l188_188201


namespace fraction_of_recipe_l188_188432

theorem fraction_of_recipe 
  (recipe_sugar recipe_milk recipe_flour : ℚ)
  (have_sugar have_milk have_flour : ℚ)
  (h1 : recipe_sugar = 3/4) (h2 : recipe_milk = 2/3) (h3 : recipe_flour = 3/8)
  (h4 : have_sugar = 2/4) (h5 : have_milk = 1/2) (h6 : have_flour = 1/4) : 
  (min ((have_sugar / recipe_sugar)) (min ((have_milk / recipe_milk)) (have_flour / recipe_flour)) = 2/3) := 
by sorry

end fraction_of_recipe_l188_188432


namespace linda_five_dollar_bills_l188_188573

theorem linda_five_dollar_bills :
  ∃ (x y : ℕ), x + y = 15 ∧ 5 * x + 10 * y = 100 ∧ x = 10 :=
by
  sorry

end linda_five_dollar_bills_l188_188573


namespace count_complex_numbers_l188_188282

theorem count_complex_numbers (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + b ≤ 5) : 
  ∃ n : ℕ, n = 10 :=
by
  sorry

end count_complex_numbers_l188_188282


namespace probability_at_least_two_boys_one_girl_l188_188585

-- Define what constitutes a family of four children
def family := {s : Fin 4 → Bool // ∃ (b g : Fin 4), b ≠ g}

-- Define the probability equation
noncomputable def probability_of_boy_or_girl : ℚ := 1 / 2

-- Define what it means to have at least two boys and one girl
def at_least_two_boys_one_girl (f : family) : Prop :=
  ∃ (count_boys count_girls : ℕ), count_boys + count_girls = 4 
  ∧ count_boys ≥ 2 
  ∧ count_girls ≥ 1

-- Calculate the probability
theorem probability_at_least_two_boys_one_girl : 
  (∃ (f : family), at_least_two_boys_one_girl f) →
  probability_of_boy_or_girl ^ 4 * ( (6 / 16 : ℚ) + (4 / 16 : ℚ) + (1 / 16 : ℚ) ) = 11 / 16 :=
by
  sorry

end probability_at_least_two_boys_one_girl_l188_188585


namespace corrected_mean_is_36_74_l188_188820

noncomputable def corrected_mean (incorrect_mean : ℝ) 
(number_of_observations : ℕ) 
(correct_value wrong_value : ℝ) : ℝ :=
(incorrect_mean * number_of_observations - wrong_value + correct_value) / number_of_observations

theorem corrected_mean_is_36_74 :
  corrected_mean 36 50 60 23 = 36.74 :=
by
  sorry

end corrected_mean_is_36_74_l188_188820


namespace right_angles_in_2_days_l188_188015

-- Definitions
def hands_right_angle_twice_a_day (n : ℕ) : Prop :=
  n = 22

def right_angle_12_hour_frequency : Nat := 22
def hours_per_day : Nat := 24
def days : Nat := 2

-- Theorem to prove
theorem right_angles_in_2_days :
  hands_right_angle_twice_a_day right_angle_12_hour_frequency →
  right_angle_12_hour_frequency * (hours_per_day / 12) * days = 88 :=
by
  unfold hands_right_angle_twice_a_day
  intros 
  sorry

end right_angles_in_2_days_l188_188015


namespace Jason_reroll_exactly_two_dice_probability_l188_188962

noncomputable def probability_reroll_two_dice : ℚ :=
  let favorable_outcomes := 5 * 3 + 1 * 3 + 5 * 3
  let total_possibilities := 6^3
  favorable_outcomes / total_possibilities

theorem Jason_reroll_exactly_two_dice_probability : probability_reroll_two_dice = 5 / 9 := 
  sorry

end Jason_reroll_exactly_two_dice_probability_l188_188962


namespace compute_g_x_h_l188_188909

def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

theorem compute_g_x_h (x h : ℝ) : 
  g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end compute_g_x_h_l188_188909


namespace aeroplane_distance_l188_188430

theorem aeroplane_distance
  (speed : ℝ) (time : ℝ) (distance : ℝ)
  (h1 : speed = 590)
  (h2 : time = 8)
  (h3 : distance = speed * time) :
  distance = 4720 :=
by {
  -- The proof will contain the steps to show that distance = 4720
  sorry
}

end aeroplane_distance_l188_188430


namespace rahul_task_days_l188_188850

theorem rahul_task_days (R : ℕ) (h1 : ∀ x : ℤ, x > 0 → 1 / R + 1 / 84 = 1 / 35) : R = 70 := 
by
  -- placeholder for the proof
  sorry

end rahul_task_days_l188_188850


namespace distance_between_M_and_focus_l188_188964

theorem distance_between_M_and_focus
  (θ : ℝ)
  (x y : ℝ)
  (M : ℝ × ℝ := (1/2, 0))
  (F : ℝ × ℝ := (0, 1/2))
  (hx : x = 2 * Real.cos θ)
  (hy : y = 1 + Real.cos (2 * θ)) :
  Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = Real.sqrt 2 / 2 :=
by
  sorry

end distance_between_M_and_focus_l188_188964


namespace simplify_and_rationalize_l188_188005

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l188_188005


namespace sequence_n_value_l188_188257

theorem sequence_n_value (n : ℤ) : (2 * n^2 - 3 = 125) → (n = 8) := 
by {
    sorry
}

end sequence_n_value_l188_188257


namespace max_expression_value_l188_188299

theorem max_expression_value (a b c d e f g h k : ℤ) 
  (ha : a = 1 ∨ a = -1)
  (hb : b = 1 ∨ b = -1)
  (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1)
  (he : e = 1 ∨ e = -1)
  (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1)
  (hh : h = 1 ∨ h = -1)
  (hk : k = 1 ∨ k = -1) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 :=
sorry

end max_expression_value_l188_188299


namespace t50_mod_7_l188_188023

def T (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | n + 1 => 3 ^ T n

theorem t50_mod_7 : T 50 % 7 = 6 := sorry

end t50_mod_7_l188_188023


namespace negation_of_proposition_l188_188361

variable (x y : ℝ)

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, (x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) ↔ 
  (∃ x y : ℝ, (x^2 + y^2 ≠ 0) ∧ (x ≠ 0 ∨ y ≠ 0)) :=
sorry

end negation_of_proposition_l188_188361


namespace product_is_correct_l188_188323

def number : ℕ := 3460
def multiplier : ℕ := 12
def correct_product : ℕ := 41520

theorem product_is_correct : multiplier * number = correct_product := by
  sorry

end product_is_correct_l188_188323


namespace fraction_meaningful_if_not_neg_two_l188_188617

theorem fraction_meaningful_if_not_neg_two {a : ℝ} : (a + 2 ≠ 0) ↔ (a ≠ -2) :=
by sorry

end fraction_meaningful_if_not_neg_two_l188_188617


namespace fill_parentheses_correct_l188_188822

theorem fill_parentheses_correct (a b : ℝ) :
  (3 * b + a) * (3 * b - a) = 9 * b^2 - a^2 :=
by 
  sorry

end fill_parentheses_correct_l188_188822


namespace profit_percent_is_26_l188_188840

variables (P C : ℝ)
variables (h1 : (2/3) * P = 0.84 * C)

theorem profit_percent_is_26 :
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_is_26_l188_188840


namespace cab_speed_fraction_l188_188521

def usual_time := 30 -- The usual time of the journey in minutes
def delay_time := 6   -- The delay time in minutes
def usual_speed : ℝ := sorry -- Placeholder for the usual speed
def reduced_speed : ℝ := sorry -- Placeholder for the reduced speed

-- Given the conditions:
-- 1. The usual time for the cab to cover the journey is 30 minutes.
-- 2. The cab is 6 minutes late when walking at a reduced speed.
-- Prove that the fraction of the cab's usual speed it is walking at is 5/6

theorem cab_speed_fraction : (reduced_speed / usual_speed) = (5 / 6) :=
sorry

end cab_speed_fraction_l188_188521


namespace original_ribbon_length_l188_188610

theorem original_ribbon_length :
  ∃ x : ℝ, 
    (∀ a b : ℝ, 
       a = x - 18 ∧ 
       b = x - 12 ∧ 
       b = 2 * a → x = 24) :=
by
  sorry

end original_ribbon_length_l188_188610


namespace remainder_div_357_l188_188944

theorem remainder_div_357 (N : ℤ) (h : N % 17 = 2) : N % 357 = 2 :=
sorry

end remainder_div_357_l188_188944


namespace integer_pair_solution_l188_188836

theorem integer_pair_solution (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pair_solution_l188_188836


namespace units_digit_of_product_of_first_four_composites_l188_188515

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l188_188515


namespace root_expression_value_l188_188109

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end root_expression_value_l188_188109


namespace total_houses_in_lincoln_county_l188_188123

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (built_houses : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : built_houses = 97741) : 
  original_houses + built_houses = 118558 := 
by
  -- Sorry is used to skip the proof.
  sorry

end total_houses_in_lincoln_county_l188_188123


namespace simplified_expression_num_terms_l188_188847

noncomputable def num_terms_polynomial (n: ℕ) : ℕ :=
  (n/2) * (1 + (n+1))

theorem simplified_expression_num_terms :
  num_terms_polynomial 2012 = 1012608 :=
by
  sorry

end simplified_expression_num_terms_l188_188847


namespace positive_solution_is_perfect_square_l188_188039

theorem positive_solution_is_perfect_square
  (t : ℤ)
  (n : ℕ)
  (h : n > 0)
  (root_cond : (n : ℤ)^2 + (4 * t - 1) * n + 4 * t^2 = 0) :
  ∃ k : ℕ, n = k^2 :=
sorry

end positive_solution_is_perfect_square_l188_188039


namespace maximum_value_of_function_l188_188772

theorem maximum_value_of_function : ∃ x, x > (1 : ℝ) ∧ (∀ y, y > 1 → (x + 1 / (x - 1) ≥ y + 1 / (y - 1))) ∧ (x = 2 ∧ (x + 1 / (x - 1) = 3)) :=
sorry

end maximum_value_of_function_l188_188772


namespace arithmetic_sequence_sum_l188_188675

variable {α : Type*} [LinearOrderedField α]

def sum_n_terms (a₁ d : α) (n : ℕ) : α :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a₁ : α) (h : sum_n_terms a₁ 1 4 = 1) :
  sum_n_terms a₁ 1 8 = 18 := by
  sorry

end arithmetic_sequence_sum_l188_188675


namespace triangle_angle_type_l188_188763

theorem triangle_angle_type (a b c R : ℝ) (hc_max : c ≥ a ∧ c ≥ b) :
  (a^2 + b^2 + c^2 - 8 * R^2 > 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 = 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2)) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 < 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2)) :=
sorry

end triangle_angle_type_l188_188763


namespace rows_seating_8_people_l188_188143

theorem rows_seating_8_people (x : ℕ) (h₁ : x ≡ 4 [MOD 7]) (h₂ : x ≤ 6) :
  x = 4 := by
  sorry

end rows_seating_8_people_l188_188143


namespace geometric_sequence_sum_l188_188461

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a_n 1 + a_n 3 = 5) :
  a_n 3 + a_n 5 = 20 :=
by
  -- The proof would go here, but it is not required for this task.
  sorry

end geometric_sequence_sum_l188_188461


namespace range_omega_l188_188807

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def f' (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

theorem range_omega (t ω φ : ℝ) (hω_pos : ω > 0) (hf_t_zero : f t ω φ = 0) (hf'_t_pos : f' t ω φ > 0) (no_min_value : ∀ x, t ≤ x ∧ x < t + 1 → ∃ y, y ≠ x ∧ f y ω φ < f x ω φ) : π < ω ∧ ω ≤ (3 * π / 2) :=
sorry

end range_omega_l188_188807


namespace smallest_x_value_l188_188020

theorem smallest_x_value (x : ℝ) (h : |4 * x + 9| = 37) : x = -11.5 :=
sorry

end smallest_x_value_l188_188020


namespace value_of_x_squared_plus_reciprocal_squared_l188_188038

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l188_188038


namespace total_pages_read_correct_l188_188888

-- Definition of the problem conditions
def first_week_books := 5
def first_week_book_pages := 300
def first_week_magazines := 3
def first_week_magazine_pages := 120
def first_week_newspapers := 2
def first_week_newspaper_pages := 50

def second_week_books := 2 * first_week_books
def second_week_book_pages := 350
def second_week_magazines := 4
def second_week_magazine_pages := 150
def second_week_newspapers := 1
def second_week_newspaper_pages := 60

def third_week_books := 3 * first_week_books
def third_week_book_pages := 400
def third_week_magazines := 5
def third_week_magazine_pages := 125
def third_week_newspapers := 1
def third_week_newspaper_pages := 70

-- Total pages read in each week
def first_week_total_pages : Nat :=
  (first_week_books * first_week_book_pages) +
  (first_week_magazines * first_week_magazine_pages) +
  (first_week_newspapers * first_week_newspaper_pages)

def second_week_total_pages : Nat :=
  (second_week_books * second_week_book_pages) +
  (second_week_magazines * second_week_magazine_pages) +
  (second_week_newspapers * second_week_newspaper_pages)

def third_week_total_pages : Nat :=
  (third_week_books * third_week_book_pages) +
  (third_week_magazines * third_week_magazine_pages) +
  (third_week_newspapers * third_week_newspaper_pages)

-- Grand total pages read over three weeks
def total_pages_read : Nat :=
  first_week_total_pages + second_week_total_pages + third_week_total_pages

-- Theorem statement to be proven
theorem total_pages_read_correct :
  total_pages_read = 12815 :=
by
  -- Proof will be provided here
  sorry

end total_pages_read_correct_l188_188888


namespace power_difference_mod_7_l188_188699

theorem power_difference_mod_7 :
  (45^2011 - 23^2011) % 7 = 5 := by
  have h45 : 45 % 7 = 3 := by norm_num
  have h23 : 23 % 7 = 2 := by norm_num
  sorry

end power_difference_mod_7_l188_188699


namespace greatest_price_drop_is_april_l188_188319

-- Define the price changes for each month
def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => 1.00
  | 2 => -1.50
  | 3 => -0.50
  | 4 => -3.75 -- including the -1.25 adjustment
  | 5 => 0.50
  | 6 => -2.25
  | _ => 0 -- default case, although we only deal with months 1-6

-- Define a predicate for the month with the greatest drop
def greatest_drop_month (m : ℕ) : Prop :=
  m = 4

-- Main theorem: Prove that the month with the greatest price drop is April
theorem greatest_price_drop_is_april : greatest_drop_month 4 :=
by
  -- Use Lean tactics to prove the statement
  sorry

end greatest_price_drop_is_april_l188_188319


namespace geom_seq_a11_l188_188890

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_a11
  (a : ℕ → α)
  (q : α)
  (ha3 : a 3 = 3)
  (ha7 : a 7 = 6)
  (hgeom : geom_seq a q) :
  a 11 = 12 :=
by
  sorry

end geom_seq_a11_l188_188890


namespace find_S5_l188_188564

-- Assuming the sequence is geometric and defining the conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Definitions of the conditions based on the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def condition_1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 5 = 3 * a 3

def condition_2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 9 * a 7) / 2 = 2

-- Sum of the first n terms of a geometric sequence
noncomputable def S_n (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

-- The theorem stating the final goal
theorem find_S5 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) 
    (h1 : condition_1 a q) (h2 : condition_2 a q) : S_n a q 5 = 121 :=
by
  -- This adds "sorry" to bypass the actual proof
  sorry

end find_S5_l188_188564


namespace sum_of_three_consecutive_odds_is_69_l188_188980

-- Definition for the smallest of three consecutive odd numbers
def smallest_consecutive_odd := 21

-- Define the three consecutive odd numbers based on the smallest one
def first_consecutive_odd := smallest_consecutive_odd
def second_consecutive_odd := smallest_consecutive_odd + 2
def third_consecutive_odd := smallest_consecutive_odd + 4

-- Calculate the sum of these three consecutive odd numbers
def sum_consecutive_odds := first_consecutive_odd + second_consecutive_odd + third_consecutive_odd

-- Theorem statement that the sum of these three consecutive odd numbers is 69
theorem sum_of_three_consecutive_odds_is_69 : 
  sum_consecutive_odds = 69 := by
    sorry

end sum_of_three_consecutive_odds_is_69_l188_188980


namespace x_intercepts_of_parabola_l188_188795

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l188_188795


namespace prove_R36_div_R6_minus_R3_l188_188754

noncomputable def R (k : ℕ) : ℤ := (10^k - 1) / 9

theorem prove_R36_div_R6_minus_R3 :
  (R 36 / R 6) - R 3 = 100000100000100000100000100000099989 := sorry

end prove_R36_div_R6_minus_R3_l188_188754


namespace factor_expression_l188_188012

theorem factor_expression (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = 
    ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) :=
by
  sorry

end factor_expression_l188_188012


namespace finite_operations_invariant_final_set_l188_188664

theorem finite_operations (n : ℕ) (a : Fin n → ℕ) :
  ∃ N : ℕ, ∀ k, k > N → ((∃ i j, i ≠ j ∧ ¬ (a i ∣ a j ∨ a j ∣ a i)) → False) :=
sorry

theorem invariant_final_set (n : ℕ) (a : Fin n → ℕ) :
  ∃ b : Fin n → ℕ, (∀ i, ∃ j, b i = a j) ∧ ∀ (c : Fin n → ℕ), (∀ i, ∃ j, c i = a j) → c = b :=
sorry

end finite_operations_invariant_final_set_l188_188664


namespace intersection_line_canonical_equation_l188_188676

def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0
def canonical_equation (x y z : ℝ) : Prop := 
  (x - 1) / 35 = (y - 4 / 7) / 23 ∧ (y - 4 / 7) / 23 = z / 49

theorem intersection_line_canonical_equation (x y z : ℝ) :
  plane1 x y z → plane2 x y z → canonical_equation x y z :=
by
  intros h1 h2
  unfold plane1 at h1
  unfold plane2 at h2
  unfold canonical_equation
  sorry

end intersection_line_canonical_equation_l188_188676


namespace sufficient_but_not_necessary_l188_188411

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end sufficient_but_not_necessary_l188_188411


namespace interest_difference_l188_188247

theorem interest_difference (P R T: ℝ) (hP: P = 2500) (hR: R = 8) (hT: T = 8) :
  let I := P * R * T / 100
  (P - I = 900) :=
by
  -- definition of I
  let I := P * R * T / 100
  -- proof goal
  sorry

end interest_difference_l188_188247


namespace proof_statements_imply_negation_l188_188870

-- Define propositions p, q, and r
variables (p q r : Prop)

-- Statement (1): p, q, and r are all true.
def statement_1 : Prop := p ∧ q ∧ r

-- Statement (2): p is true, q is false, and r is true.
def statement_2 : Prop := p ∧ ¬ q ∧ r

-- Statement (3): p is false, q is true, and r is false.
def statement_3 : Prop := ¬ p ∧ q ∧ ¬ r

-- Statement (4): p and r are false, q is true.
def statement_4 : Prop := ¬ p ∧ q ∧ ¬ r

-- The negation of "p and q are true, and r is false" is "¬(p ∧ q) ∨ r"
def negation : Prop := ¬(p ∧ q) ∨ r

-- Proof statement that each of the 4 statements implies the negation
theorem proof_statements_imply_negation :
  (statement_1 p q r → negation p q r) ∧
  (statement_2 p q r → negation p q r) ∧
  (statement_3 p q r → negation p q r) ∧
  (statement_4 p q r → negation p q r) :=
by
  sorry

end proof_statements_imply_negation_l188_188870


namespace one_equation_does_not_pass_origin_l188_188727

def passes_through_origin (eq : ℝ → ℝ) : Prop := eq 0 = 0

def equation1 (x : ℝ) : ℝ := x^4 + 1
def equation2 (x : ℝ) : ℝ := x^4 + x
def equation3 (x : ℝ) : ℝ := x^4 + x^2
def equation4 (x : ℝ) : ℝ := x^4 + x^3

theorem one_equation_does_not_pass_origin :
  (¬ passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  ¬ passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  ¬ passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  ¬ passes_through_origin equation4) :=
sorry

end one_equation_does_not_pass_origin_l188_188727


namespace additional_boxes_needed_l188_188716

theorem additional_boxes_needed
  (total_chocolates : ℕ)
  (chocolates_not_in_box : ℕ)
  (boxes_filled : ℕ)
  (friend_brought_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 50)
  (h2 : chocolates_not_in_box = 5)
  (h3 : boxes_filled = 3)
  (h4 : friend_brought_chocolates = 25)
  (h5 : chocolates_per_box = 15) :
  (chocolates_not_in_box + friend_brought_chocolates) / chocolates_per_box = 2 :=
by
  sorry
  
end additional_boxes_needed_l188_188716


namespace team_total_points_l188_188660

theorem team_total_points : 
  ∀ (Tobee Jay Sean : ℕ),
  (Tobee = 4) →
  (Jay = Tobee + 6) →
  (Sean = Tobee + Jay - 2) →
  (Tobee + Jay + Sean = 26) :=
by
  intros Tobee Jay Sean h1 h2 h3
  rw [h1, h2, h3]
  sorry

end team_total_points_l188_188660


namespace value_of_x_l188_188056

theorem value_of_x (x : ℝ) : (1 / 8) * (2 : ℝ) ^ 32 = (4 : ℝ) ^ x → x = 29 / 2 :=
by
  sorry

end value_of_x_l188_188056


namespace consecutive_integers_sum_and_difference_l188_188710

theorem consecutive_integers_sum_and_difference (x y : ℕ) 
(h1 : y = x + 1) 
(h2 : x * y = 552) 
: x + y = 47 ∧ y - x = 1 :=
by {
  sorry
}

end consecutive_integers_sum_and_difference_l188_188710


namespace angles_with_same_terminal_side_l188_188386

theorem angles_with_same_terminal_side (k : ℤ) : 
  (∃ (α : ℝ), α = -437 + k * 360) ↔ (∃ (β : ℝ), β = 283 + k * 360) := 
by
  sorry

end angles_with_same_terminal_side_l188_188386


namespace chess_game_problem_l188_188554

-- Mathematical definitions based on the conditions
def petr_wins : ℕ := 6
def petr_draws : ℕ := 2
def karel_points : ℤ := 9
def points_for_win : ℕ := 3
def points_for_loss : ℕ := 2
def points_for_draw : ℕ := 0

-- Defining the final statement to prove
theorem chess_game_problem :
    ∃ (total_games : ℕ) (leader : String), total_games = 15 ∧ leader = "Karel" := 
by
  -- Placeholder for proof
  sorry

end chess_game_problem_l188_188554


namespace sister_sandcastle_height_l188_188759

theorem sister_sandcastle_height (miki_height : ℝ)
                                (height_diff : ℝ)
                                (h_miki : miki_height = 0.8333333333333334)
                                (h_diff : height_diff = 0.3333333333333333) :
  miki_height - height_diff = 0.5 :=
by
  sorry

end sister_sandcastle_height_l188_188759


namespace weekly_goal_cans_l188_188113

theorem weekly_goal_cans (c₁ c₂ c₃ c₄ c₅ : ℕ) (h₁ : c₁ = 20) (h₂ : c₂ = c₁ + 5) (h₃ : c₃ = c₂ + 5) 
  (h₄ : c₄ = c₃ + 5) (h₅ : c₅ = c₄ + 5) : 
  c₁ + c₂ + c₃ + c₄ + c₅ = 150 :=
by
  sorry

end weekly_goal_cans_l188_188113


namespace triangle_third_side_range_l188_188994

variable (a b c : ℝ)

theorem triangle_third_side_range 
  (h₁ : |a + b - 4| + (a - b + 2)^2 = 0)
  (h₂ : a + b > c)
  (h₃ : a + c > b)
  (h₄ : b + c > a) : 2 < c ∧ c < 4 := 
sorry

end triangle_third_side_range_l188_188994


namespace miss_davis_items_left_l188_188738

theorem miss_davis_items_left 
  (popsicle_sticks_per_group : ℕ := 15) 
  (straws_per_group : ℕ := 20) 
  (num_groups : ℕ := 10) 
  (total_items_initial : ℕ := 500) : 
  total_items_initial - (num_groups * (popsicle_sticks_per_group + straws_per_group)) = 150 :=
by 
  sorry

end miss_davis_items_left_l188_188738


namespace ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l188_188178

theorem ellipse_foci_on_x_axis_major_axis_twice_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m * y^2 = 1) → (∃ a b : ℝ, a = 1 ∧ b = Real.sqrt (1 / m) ∧ a = 2 * b) → m = 4 :=
by
  sorry

end ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l188_188178


namespace sqrt_40000_eq_200_l188_188385

theorem sqrt_40000_eq_200 : Real.sqrt 40000 = 200 := 
sorry

end sqrt_40000_eq_200_l188_188385


namespace angle_of_isosceles_trapezoid_in_monument_l188_188035

-- Define the larger interior angle x of an isosceles trapezoid in the monument
def larger_interior_angle_of_trapezoid (x : ℝ) : Prop :=
  ∃ n : ℕ, 
    n = 12 ∧
    ∃ α : ℝ, 
      α = 360 / (2 * n) ∧
      ∃ θ : ℝ, 
        θ = (180 - α) / 2 ∧
        x = 180 - θ

-- The theorem stating the larger interior angle x is 97.5 degrees
theorem angle_of_isosceles_trapezoid_in_monument : larger_interior_angle_of_trapezoid 97.5 :=
by 
  sorry

end angle_of_isosceles_trapezoid_in_monument_l188_188035


namespace edric_hours_per_day_l188_188607

/--
Edric's monthly salary is $576. He works 6 days a week for 4 weeks in a month and 
his hourly rate is $3. Prove that Edric works 8 hours in a day.
-/
theorem edric_hours_per_day (m : ℕ) (r : ℕ) (d : ℕ) (w : ℕ)
  (h_m : m = 576) (h_r : r = 3) (h_d : d = 6) (h_w : w = 4) :
  (m / r) / (d * w) = 8 := by
    sorry

end edric_hours_per_day_l188_188607


namespace seq_sum_eq_314_l188_188289

theorem seq_sum_eq_314 (d r : ℕ) (k : ℕ) (a_n b_n c_n : ℕ → ℕ)
  (h1 : ∀ n, a_n n = 1 + (n - 1) * d)
  (h2 : ∀ n, b_n n = r ^ (n - 1))
  (h3 : ∀ n, c_n n = a_n n + b_n n)
  (hk1 : c_n (k - 1) = 150)
  (hk2 : c_n (k + 1) = 900) :
  c_n k = 314 := by
  sorry

end seq_sum_eq_314_l188_188289


namespace unique_even_odd_decomposition_l188_188486

def is_symmetric (s : Set ℝ) : Prop := ∀ x ∈ s, -x ∈ s

def is_even (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = f x

def is_odd (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = -f x

theorem unique_even_odd_decomposition (s : Set ℝ) (hs : is_symmetric s) (f : ℝ → ℝ) (hf : ∀ x ∈ s, True) :
  ∃! g h : ℝ → ℝ, (is_even g s) ∧ (is_odd h s) ∧ (∀ x ∈ s, f x = g x + h x) :=
sorry

end unique_even_odd_decomposition_l188_188486


namespace exponent_equality_l188_188687

theorem exponent_equality (y : ℕ) (z : ℕ) (h1 : 16 ^ y = 4 ^ z) (h2 : y = 8) : z = 16 := by
  sorry

end exponent_equality_l188_188687


namespace large_pizza_cost_l188_188495

theorem large_pizza_cost
  (small_side : ℕ) (small_cost : ℝ) (large_side : ℕ) (friend_money : ℝ) (extra_square_inches : ℝ)
  (A_small : small_side * small_side = 196)
  (A_large : large_side * large_side = 441)
  (small_cost_per_sq_in : 196 / small_cost = 19.6)
  (individual_area : (30 / small_cost) * 196 = 588)
  (total_individual_area : 2 * 588 = 1176)
  (pool_area_eq : (60 / (441 / x)) = 1225)
  : (x = 21.6) := 
by
  sorry

end large_pizza_cost_l188_188495


namespace total_votes_cast_l188_188456

theorem total_votes_cast (F A T : ℕ) (h1 : F = A + 70) (h2 : A = 2 * T / 5) (h3 : T = F + A) : T = 350 :=
by
  sorry

end total_votes_cast_l188_188456


namespace percentage_increase_l188_188938

theorem percentage_increase 
  (distance : ℝ) (time_q : ℝ) (time_y : ℝ) 
  (speed_q : ℝ) (speed_y : ℝ) 
  (percentage_increase : ℝ) 
  (h_distance : distance = 80)
  (h_time_q : time_q = 2)
  (h_time_y : time_y = 1.3333333333333333)
  (h_speed_q : speed_q = distance / time_q)
  (h_speed_y : speed_y = distance / time_y)
  (h_faster : speed_y > speed_q)
  : percentage_increase = ((speed_y - speed_q) / speed_q) * 100 :=
by
  sorry

end percentage_increase_l188_188938


namespace simplify_sqrt_of_square_l188_188124

-- The given condition
def x : ℤ := -9

-- The theorem stating the simplified form
theorem simplify_sqrt_of_square : (Real.sqrt ((x : ℝ) ^ 2) = 9) := by    
    sorry

end simplify_sqrt_of_square_l188_188124


namespace find_y_l188_188574

noncomputable def inverse_proportion_y_value (x y k : ℝ) : Prop :=
  (x * y = k) ∧ (x + y = 52) ∧ (x = 3 * y) ∧ (x = -10) → (y = -50.7)

theorem find_y (x y k : ℝ) (h : inverse_proportion_y_value x y k) : y = -50.7 :=
  sorry

end find_y_l188_188574


namespace spent_on_board_game_l188_188622

theorem spent_on_board_game (b : ℕ)
  (h1 : 4 * 7 = 28)
  (h2 : b + 28 = 30) : 
  b = 2 := 
sorry

end spent_on_board_game_l188_188622


namespace find_b_when_a_is_1600_l188_188203

variable (a b : ℝ)

def inversely_vary (a b : ℝ) : Prop := a * b = 400

theorem find_b_when_a_is_1600 
  (h1 : inversely_vary 800 0.5)
  (h2 : inversely_vary a b)
  (h3 : a = 1600) :
  b = 0.25 := by
  sorry

end find_b_when_a_is_1600_l188_188203


namespace find_n_l188_188728

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - imaginary_unit) = 1 + n * imaginary_unit) : n = 1 :=
sorry

end find_n_l188_188728


namespace find_A_l188_188286

def is_divisible (n : ℕ) (d : ℕ) : Prop := d ∣ n

noncomputable def valid_digit (A : ℕ) : Prop :=
  A < 10

noncomputable def digit_7_number := 653802 * 10

theorem find_A (A : ℕ) (h : valid_digit A) :
  is_divisible (digit_7_number + A) 2 ∧
  is_divisible (digit_7_number + A) 3 ∧
  is_divisible (digit_7_number + A) 4 ∧
  is_divisible (digit_7_number + A) 6 ∧
  is_divisible (digit_7_number + A) 8 ∧
  is_divisible (digit_7_number + A) 9 ∧
  is_divisible (digit_7_number + A) 25 →
  A = 0 :=
sorry

end find_A_l188_188286


namespace smallest_m_integral_roots_l188_188365

theorem smallest_m_integral_roots (m : ℕ) : 
  (∃ p q : ℤ, (10 * p * p - ↑m * p + 360 = 0) ∧ (p + q = m / 10) ∧ (p * q = 36) ∧ (p % q = 0 ∨ q % p = 0)) → 
  m = 120 :=
by
sorry

end smallest_m_integral_roots_l188_188365


namespace common_pasture_area_l188_188305

variable (Area_Ivanov Area_Petrov Area_Sidorov Area_Vasilev Area_Ermolaev : ℝ)
variable (Common_Pasture : ℝ)

theorem common_pasture_area :
  Area_Ivanov = 24 ∧
  Area_Petrov = 28 ∧
  Area_Sidorov = 10 ∧
  Area_Vasilev = 20 ∧
  Area_Ermolaev = 30 →
  Common_Pasture = 17.5 :=
sorry

end common_pasture_area_l188_188305


namespace laboratory_painting_area_laboratory_paint_needed_l188_188445

section
variable (l w h excluded_area : ℝ)
variable (paint_per_sqm : ℝ)

def painting_area (l w h excluded_area : ℝ) : ℝ :=
  let total_area := (l * w + w * h + h * l) * 2 - (l * w)
  total_area - excluded_area

def paint_needed (painting_area paint_per_sqm : ℝ) : ℝ :=
  painting_area * paint_per_sqm

theorem laboratory_painting_area :
  painting_area 12 8 6 28.4 = 307.6 :=
by
  simp [painting_area, *]
  norm_num

theorem laboratory_paint_needed :
  paint_needed 307.6 0.2 = 61.52 :=
by
  simp [paint_needed, *]
  norm_num

end

end laboratory_painting_area_laboratory_paint_needed_l188_188445


namespace problem_proof_l188_188197

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + (1 / Real.sqrt (2 - x))
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {y | y ≥ 1}
def CU_B : Set ℝ := {y | y < 1}
def U : Set ℝ := Set.univ

theorem problem_proof :
  (∀ x, x ∈ A ↔ -1 ≤ x ∧ x < 2) ∧
  (∀ y, y ∈ B ↔ y ≥ 1) ∧
  (A ∩ CU_B = {x | -1 ≤ x ∧ x < 1}) :=
by
  sorry

end problem_proof_l188_188197


namespace cube_volume_increase_l188_188797

theorem cube_volume_increase (s : ℝ) (surface_area : ℝ) 
  (h1 : surface_area = 6 * s^2) (h2 : surface_area = 864) : 
  (1.5 * s)^3 = 5832 :=
by
  sorry

end cube_volume_increase_l188_188797


namespace cos_angle_B_bounds_l188_188959

theorem cos_angle_B_bounds {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ)
  (angle_ADC : ℝ) (angle_B : ℝ)
  (h1 : AB = 2) (h2 : BC = 3) (h3 : CD = 2) (h4 : angle_ADC = 180 - angle_B) :
  (1 / 4) < Real.cos angle_B ∧ Real.cos angle_B < (3 / 4) := 
sorry -- Proof to be provided

end cos_angle_B_bounds_l188_188959


namespace value_of_expression_l188_188704

theorem value_of_expression (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := 
by 
  sorry

end value_of_expression_l188_188704


namespace christmas_bonus_remainder_l188_188331

theorem christmas_bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l188_188331


namespace payment_to_C_l188_188036

/-- 
If A can complete a work in 6 days, B can complete the same work in 8 days, 
they signed to do the work for Rs. 2400 and completed the work in 3 days with 
the help of C, then the payment to C should be Rs. 300.
-/
theorem payment_to_C (total_payment : ℝ) (days_A : ℝ) (days_B : ℝ) (days_worked : ℝ) (portion_C : ℝ) :
   total_payment = 2400 ∧ days_A = 6 ∧ days_B = 8 ∧ days_worked = 3 ∧ portion_C = 1 / 8 →
   (portion_C * total_payment) = 300 := 
by 
  intros h
  cases h
  sorry

end payment_to_C_l188_188036


namespace shortest_chord_line_l188_188194

theorem shortest_chord_line (x y : ℝ) (P : (ℝ × ℝ)) (C : ℝ → ℝ → Prop) (h₁ : C x y) (hx : P = (1, 1)) (hC : ∀ x y, C x y ↔ x^2 + y^2 = 4) : 
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -2 ∧ a * x + b * y + c = 0 :=
by
  sorry

end shortest_chord_line_l188_188194


namespace number_of_possible_schedules_l188_188522

-- Define the six teams
inductive Team : Type
| A | B | C | D | E | F

open Team

-- Define the function to get the number of different schedules possible
noncomputable def number_of_schedules : ℕ := 70

-- Define the theorem statement
theorem number_of_possible_schedules (teams : Finset Team) (play_games : Team → Finset Team) (h : teams.card = 6) 
  (h2 : ∀ t ∈ teams, (play_games t).card = 3 ∧ ∀ t' ∈ (play_games t), t ≠ t') : 
  number_of_schedules = 70 :=
by sorry

end number_of_possible_schedules_l188_188522


namespace sum_of_three_squares_l188_188161

theorem sum_of_three_squares (a b : ℝ)
  (h1 : 3 * a + 2 * b = 18)
  (h2 : 2 * a + 3 * b = 22) :
  3 * b = 18 :=
sorry

end sum_of_three_squares_l188_188161


namespace somu_age_relation_l188_188884

-- Somu’s present age (S) is 20 years
def somu_present_age : ℕ := 20

-- Somu’s age is one-third of his father’s age (F)
def father_present_age : ℕ := 3 * somu_present_age

-- Proof statement: Y years ago, Somu's age was one-fifth of his father's age
theorem somu_age_relation : ∃ (Y : ℕ), somu_present_age - Y = (1 : ℕ) / 5 * (father_present_age - Y) ∧ Y = 10 :=
by
  have h := "" -- Placeholder for the proof steps
  sorry

end somu_age_relation_l188_188884


namespace find_g_inverse_sum_l188_188949

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * x + 2 else 3 - x

theorem find_g_inverse_sum :
  (∃ x, g x = -2 ∧ x = 5) ∧
  (∃ x, g x = 0 ∧ x = 3) ∧
  (∃ x, g x = 2 ∧ x = 0) ∧
  (5 + 3 + 0 = 8) := by
  sorry

end find_g_inverse_sum_l188_188949


namespace seventh_observation_value_l188_188723

def average_initial_observations (S : ℝ) (n : ℕ) : Prop :=
  S / n = 13

def total_observations (n : ℕ) : Prop :=
  n + 1 = 7

def new_average (S : ℝ) (x : ℝ) (n : ℕ) : Prop :=
  (S + x) / (n + 1) = 12

theorem seventh_observation_value (S : ℝ) (n : ℕ) (x : ℝ) :
  average_initial_observations S n →
  total_observations n →
  new_average S x n →
  x = 6 :=
by
  intros h1 h2 h3
  sorry

end seventh_observation_value_l188_188723


namespace complex_problem_l188_188029

theorem complex_problem (a b : ℝ) (i : ℂ) (hi : i^2 = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b = 1 :=
by
  sorry

end complex_problem_l188_188029


namespace sum_of_mnp_l188_188014

theorem sum_of_mnp (m n p : ℕ) (h_gcd : gcd m (gcd n p) = 1)
  (h : ∀ x : ℝ, 5 * x^2 - 11 * x + 6 = 0 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 22 :=
by
  sorry

end sum_of_mnp_l188_188014


namespace sqrt_31_estimate_l188_188344

theorem sqrt_31_estimate : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := 
by
  sorry

end sqrt_31_estimate_l188_188344


namespace riverton_soccer_physics_l188_188758

theorem riverton_soccer_physics : 
  let total_players := 15
  let math_players := 9
  let both_subjects := 3
  let only_physics := total_players - math_players
  let physics_players := only_physics + both_subjects
  physics_players = 9 :=
by
  sorry

end riverton_soccer_physics_l188_188758


namespace find_b_l188_188285

def f (x : ℝ) : ℝ := 5 * x + 3

theorem find_b : ∃ b : ℝ, f b = -2 ∧ b = -1 := by
  have h : 5 * (-1 : ℝ) + 3 = -2 := by norm_num
  use -1
  simp [f, h]
  sorry

end find_b_l188_188285


namespace max_value_x_plus_y_l188_188627

theorem max_value_x_plus_y :
  ∃ x y : ℝ, 5 * x + 3 * y ≤ 10 ∧ 3 * x + 5 * y = 15 ∧ x + y = 47 / 16 :=
by
  sorry

end max_value_x_plus_y_l188_188627


namespace equilateral_triangle_side_length_l188_188958

theorem equilateral_triangle_side_length 
  (x1 y1 : ℝ) 
  (hx1y1 : y1 = - (1 / 4) * x1^2)
  (h_eq_tri: ∃ (x2 y2 : ℝ), x2 = -x1 ∧ y2 = y1 ∧ (x2, y2) ≠ (x1, y1) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = x1^2 + y1^2 ∧ (x1 - 0)^2 + (y1 - 0)^2 = (x1 - x2)^2 + (y1 - y2)^2)):
  2 * x1 = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l188_188958


namespace area_of_triangle_l188_188190

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l188_188190


namespace gcd_840_1764_l188_188034

theorem gcd_840_1764 : Int.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l188_188034


namespace power_function_is_odd_l188_188496

open Function

noncomputable def power_function (a : ℝ) (b : ℝ) : ℝ → ℝ := λ x => (a - 1) * x^b

theorem power_function_is_odd (a b : ℝ) (h : power_function a b a = 1 / 8)
  :  a = 2 ∧ b = -3 → (∀ x : ℝ, power_function a b (-x) = -power_function a b x) :=
by
  intro ha hb
  -- proofs can be filled later with details
  sorry

end power_function_is_odd_l188_188496


namespace perfect_cubes_in_range_l188_188864

theorem perfect_cubes_in_range :
  ∃ (n : ℕ), (∀ (k : ℕ), (50 < k^3 ∧ k^3 ≤ 1000) → (k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10)) ∧
    (∃ m, (m = 7)) :=
by
  sorry

end perfect_cubes_in_range_l188_188864


namespace constant_sequence_if_and_only_if_arith_geo_progression_l188_188374

/-- A sequence a_n is both an arithmetic and geometric progression if and only if it is constant --/
theorem constant_sequence_if_and_only_if_arith_geo_progression (a : ℕ → ℝ) :
  (∃ q d : ℝ, (∀ n : ℕ, a (n+1) - a n = d) ∧ (∀ n : ℕ, a n = a 0 * q ^ n)) ↔ (∃ c : ℝ, ∀ n : ℕ, a n = c) := 
sorry

end constant_sequence_if_and_only_if_arith_geo_progression_l188_188374


namespace not_p_and_q_equiv_not_p_or_not_q_l188_188642

variable (p q : Prop)

theorem not_p_and_q_equiv_not_p_or_not_q (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end not_p_and_q_equiv_not_p_or_not_q_l188_188642


namespace mats_weaved_by_mat_weavers_l188_188896

variable (M : ℕ)

theorem mats_weaved_by_mat_weavers :
  -- 10 mat-weavers can weave 25 mats in 10 days
  (10 * 10) * M / (4 * 4) = 25 / (10 / 4)  →
  -- number of mats woven by 4 mat-weavers in 4 days
  M = 4 :=
sorry

end mats_weaved_by_mat_weavers_l188_188896


namespace number_of_results_l188_188547

theorem number_of_results (n : ℕ)
  (avg_all : (summation : ℤ) → summation / n = 42)
  (avg_first_5 : (sum_first_5 : ℤ) → sum_first_5 / 5 = 49)
  (avg_last_7 : (sum_last_7 : ℤ) → sum_last_7 / 7 = 52)
  (fifth_result : (r5 : ℤ) → r5 = 147) :
  n = 11 :=
by
  -- Conditions
  let sum_first_5 := 5 * 49
  let sum_last_7 := 7 * 52
  let summed_results := sum_first_5 + sum_last_7 - 147
  let sum_all := 42 * n 
  -- Since sum of all results = 42n
  exact sorry

end number_of_results_l188_188547


namespace arithmetic_sequence_8th_term_l188_188355

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l188_188355


namespace necessary_but_not_sufficient_condition_l188_188768

variable (a b : ℝ) (lna lnb : ℝ)

theorem necessary_but_not_sufficient_condition (h1 : lna < lnb) (h2 : lna = Real.log a) (h3 : lnb = Real.log b) :
  (a > 0 ∧ b > 0 ∧ a < b ∧ a ^ 3 < b ^ 3) ∧ ¬(a ^ 3 < b ^ 3 → 0 < a ∧ a < b ∧ 0 < b) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l188_188768


namespace intersection_eq_l188_188077

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℝ := {x : ℝ | x > 2 ∨ x < -1}

theorem intersection_eq : (setA ∩ setB) = {x : ℝ | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l188_188077


namespace stones_in_courtyard_l188_188390

theorem stones_in_courtyard (S T B : ℕ) (h1 : T = S + 3 * S) (h2 : B = 2 * (T + S)) (h3 : B = 400) : S = 40 :=
by
  sorry

end stones_in_courtyard_l188_188390


namespace probability_same_group_l188_188694

noncomputable def num_students : ℕ := 800
noncomputable def num_groups : ℕ := 4
noncomputable def group_size : ℕ := num_students / num_groups
noncomputable def amy := 0
noncomputable def ben := 1
noncomputable def clara := 2

theorem probability_same_group : ∃ p : ℝ, p = 1 / 16 :=
by
  let P_ben_with_amy : ℝ := group_size / num_students
  let P_clara_with_amy : ℝ := group_size / num_students
  let P_all_same := P_ben_with_amy * P_clara_with_amy
  use P_all_same
  sorry

end probability_same_group_l188_188694


namespace speed_in_still_water_l188_188174

/--
A man can row upstream at 55 kmph and downstream at 65 kmph.
Prove that his speed in still water is 60 kmph.
-/
theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_upstream : upstream_speed = 55) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 60 := by
  sorry

end speed_in_still_water_l188_188174


namespace game_result_l188_188773

theorem game_result (a : ℤ) : ((2 * a + 6) / 2 - a = 3) :=
by
  sorry

end game_result_l188_188773


namespace GCF_LCM_15_21_14_20_l188_188133

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_15_21_14_20 :
  GCF (LCM 15 21) (LCM 14 20) = 35 :=
by
  sorry

end GCF_LCM_15_21_14_20_l188_188133


namespace circle_radius_l188_188891

theorem circle_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ (∀ x y, x^2 - 8*x + y^2 - 4*y + 16 = 0 → r = 2)) :=
sorry

end circle_radius_l188_188891


namespace tangent_line_eqn_l188_188565

theorem tangent_line_eqn (r x0 y0 : ℝ) (h : x0^2 + y0^2 = r^2) : 
  ∃ a b c : ℝ, a = x0 ∧ b = y0 ∧ c = r^2 ∧ (a*x + b*y = c) :=
sorry

end tangent_line_eqn_l188_188565


namespace kim_money_l188_188592

theorem kim_money (S P K A : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : A = 1.25 * (S + K)) (h4 : S + P + A = 3.60) : K = 0.96 :=
by
  sorry

end kim_money_l188_188592


namespace proof_problem_l188_188623

theorem proof_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := 
by 
  sorry

end proof_problem_l188_188623


namespace multiplication_24_12_l188_188205

theorem multiplication_24_12 :
  let a := 24
  let b := 12
  let b1 := 10
  let b2 := 2
  let p1 := a * b2
  let p2 := a * b1
  let sum := p1 + p2
  b = b1 + b2 →
  p1 = a * b2 →
  p2 = a * b1 →
  sum = p1 + p2 →
  a * b = sum :=
by
  intros
  sorry

end multiplication_24_12_l188_188205


namespace ben_bonus_amount_l188_188854

variables (B : ℝ)

-- Conditions
def condition1 := B - (1/22) * B - (1/4) * B - (1/8) * B = 867

-- Theorem statement
theorem ben_bonus_amount (h : condition1 B) : B = 1496.50 := 
sorry

end ben_bonus_amount_l188_188854


namespace line_through_P0_perpendicular_to_plane_l188_188346

-- Definitions of the given conditions
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P0 : Point3D := { x := 3, y := 4, z := 2 }

def plane (x y z : ℝ) : Prop := 8 * x - 4 * y + 5 * z - 4 = 0

-- The proof problem statement
theorem line_through_P0_perpendicular_to_plane :
  ∃ t : ℝ, (P0.x + 8 * t = x ∧ P0.y - 4 * t = y ∧ P0.z + 5 * t = z) ↔
    (∃ t : ℝ, x = 3 + 8 * t ∧ y = 4 - 4 * t ∧ z = 2 + 5 * t) → 
    (∃ t : ℝ, (x - 3) / 8 = t ∧ (y - 4) / -4 = t ∧ (z - 2) / 5 = t) := sorry

end line_through_P0_perpendicular_to_plane_l188_188346


namespace range_of_m_l188_188030

theorem range_of_m (p_false : ¬ (∀ x : ℝ, ∃ m : ℝ, 2 * x + 1 + m = 0)) : ∀ m : ℝ, m ≤ 1 :=
sorry

end range_of_m_l188_188030


namespace triangle_equilateral_of_angles_and_intersecting_segments_l188_188970

theorem triangle_equilateral_of_angles_and_intersecting_segments
    (A B C : Type) (angle_A : ℝ) (intersect_at_one_point : Prop)
    (angle_M_bisects : Prop) (N_is_median : Prop) (L_is_altitude : Prop) :
  angle_A = 60 ∧ angle_M_bisects ∧ N_is_median ∧ L_is_altitude ∧ intersect_at_one_point → 
  ∀ (angle_B angle_C : ℝ), angle_B = 60 ∧ angle_C = 60 := 
by
  intro h
  sorry

end triangle_equilateral_of_angles_and_intersecting_segments_l188_188970


namespace moles_of_CaCO3_formed_l188_188240

theorem moles_of_CaCO3_formed (m n : ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : ∀ m n : ℕ, (m = n) → (m = 3) → (n = 3) → moles_of_CaCO3 = m) : 
  moles_of_CaCO3 = 3 := by
  sorry

end moles_of_CaCO3_formed_l188_188240


namespace percentage_error_in_area_l188_188150

theorem percentage_error_in_area (s : ℝ) (h : s ≠ 0) :
  let s' := 1.02 * s
  let A := s^2
  let A' := s'^2
  ((A' - A) / A) * 100 = 4.04 := by
  sorry

end percentage_error_in_area_l188_188150


namespace teams_working_together_l188_188032

theorem teams_working_together
    (m n : ℕ) 
    (hA : ∀ t : ℕ, t = m → (t ≥ 0)) 
    (hB : ∀ t : ℕ, t = n → (t ≥ 0)) : 
  ∃ t : ℕ, t = (m * n) / (m + n) :=
by
  sorry

end teams_working_together_l188_188032


namespace smallest_a_b_sum_l188_188078

theorem smallest_a_b_sum :
∀ (a b : ℕ), 
  (5 * a + 6 = 6 * b + 5) ∧ 
  (∀ d : ℕ, d < 10 → d < a) ∧ 
  (∀ d : ℕ, d < 10 → d < b) ∧ 
  (0 < a) ∧ 
  (0 < b) 
  → a + b = 13 :=
by
  sorry

end smallest_a_b_sum_l188_188078


namespace square_area_increase_l188_188879

variable (s : ℝ)

theorem square_area_increase (h : s > 0) : 
  let s_new := 1.30 * s
  let A_original := s^2
  let A_new := s_new^2
  let percentage_increase := ((A_new - A_original) / A_original) * 100
  percentage_increase = 69 := by
sorry

end square_area_increase_l188_188879


namespace radius_of_circle_l188_188575

noncomputable def circle_radius {k : ℝ} (hk : k > -6) : ℝ := 6 * Real.sqrt 2 + 6

theorem radius_of_circle (k : ℝ) (hk : k > -6)
  (tangent_y_eq_x : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_negx : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_neg6 : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -6) = 6 * Real.sqrt 2 + 6) :
  circle_radius hk = 6 * Real.sqrt 2 + 6 :=
by
  sorry

end radius_of_circle_l188_188575


namespace total_money_spent_l188_188090

def time_in_minutes_at_arcade : ℕ := 3 * 60
def cost_per_interval : ℕ := 50 -- in cents
def interval_duration : ℕ := 6 -- in minutes
def total_intervals : ℕ := time_in_minutes_at_arcade / interval_duration

theorem total_money_spent :
  ((total_intervals * cost_per_interval) = 1500) := 
by
  sorry

end total_money_spent_l188_188090


namespace typeA_cloth_typeB_cloth_typeC_cloth_l188_188878

section ClothPrices

variables (CPA CPB CPC : ℝ)

theorem typeA_cloth :
  (300 * CPA * 0.90 = 9000) → CPA = 33.33 :=
by
  intro hCPA
  sorry

theorem typeB_cloth :
  (250 * CPB * 1.05 = 7000) → CPB = 26.67 :=
by
  intro hCPB
  sorry

theorem typeC_cloth :
  (400 * (CPC + 8) = 12000) → CPC = 22 :=
by
  intro hCPC
  sorry

end ClothPrices

end typeA_cloth_typeB_cloth_typeC_cloth_l188_188878


namespace maria_min_score_fifth_term_l188_188329

theorem maria_min_score_fifth_term (score1 score2 score3 score4 : ℕ) (avg_required : ℕ) 
  (h1 : score1 = 84) (h2 : score2 = 80) (h3 : score3 = 82) (h4 : score4 = 78)
  (h_avg_required : avg_required = 85) :
  ∃ x : ℕ, x ≥ 101 :=
by
  sorry

end maria_min_score_fifth_term_l188_188329


namespace discriminant_nonnegative_l188_188855

theorem discriminant_nonnegative (x : ℤ) (h : x^2 * (25 - 24 * x^2) ≥ 0) : x = 0 ∨ x = 1 ∨ x = -1 :=
by sorry

end discriminant_nonnegative_l188_188855


namespace gift_items_l188_188815

theorem gift_items (x y z : ℕ) : 
  x + y + z = 20 ∧ 60 * x + 50 * y + 10 * z = 720 ↔ 
  ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) :=
by sorry

end gift_items_l188_188815


namespace Tim_age_l188_188885

theorem Tim_age (T t : ℕ) (h1 : T = 22) (h2 : T = 2 * t + 6) : t = 8 := by
  sorry

end Tim_age_l188_188885


namespace find_n_lcm_l188_188688

theorem find_n_lcm (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : n ≥ 100) (h3 : n < 1000) (h4 : ¬ (3 ∣ n)) (h5 : ¬ (2 ∣ m)) : n = 230 :=
sorry

end find_n_lcm_l188_188688


namespace find_x_l188_188655

def f (x: ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * (f x) - 10 = f (x - 2)) : x = 3 :=
by
  sorry

end find_x_l188_188655


namespace shirt_cost_l188_188105

theorem shirt_cost (J S : ℕ) 
  (h₁ : 3 * J + 2 * S = 69) 
  (h₂ : 2 * J + 3 * S = 61) :
  S = 9 :=
by 
  sorry

end shirt_cost_l188_188105


namespace skylar_total_donations_l188_188006

-- Define the conditions
def start_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the statement to be proven
theorem skylar_total_donations : 
  (current_age - start_age) * annual_donation = 432000 := by
    sorry

end skylar_total_donations_l188_188006


namespace exists_equal_sum_disjoint_subsets_l188_188509

-- Define the set and conditions
def is_valid_set (S : Finset ℕ) : Prop :=
  S.card = 15 ∧ ∀ x ∈ S, x ≤ 2020

-- Define the problem statement
theorem exists_equal_sum_disjoint_subsets (S : Finset ℕ) (h : is_valid_set S) :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end exists_equal_sum_disjoint_subsets_l188_188509


namespace carter_lucy_ratio_l188_188844

-- Define the number of pages Oliver can read in 1 hour
def oliver_pages : ℕ := 40

-- Define the number of additional pages Lucy can read compared to Oliver
def additional_pages : ℕ := 20

-- Define the number of pages Carter can read in 1 hour
def carter_pages : ℕ := 30

-- Calculate the number of pages Lucy can read in 1 hour
def lucy_pages : ℕ := oliver_pages + additional_pages

-- Prove the ratio of the number of pages Carter can read to the number of pages Lucy can read is 1/2
theorem carter_lucy_ratio : (carter_pages : ℚ) / (lucy_pages : ℚ) = 1 / 2 := by
  sorry

end carter_lucy_ratio_l188_188844


namespace secant_line_slope_positive_l188_188802

theorem secant_line_slope_positive (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, 0 < (deriv f x)) :
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → 0 < (f x1 - f x2) / (x1 - x2) :=
by
  intros x1 x2 h_ne
  sorry

end secant_line_slope_positive_l188_188802


namespace average_last_12_results_l188_188809

theorem average_last_12_results (S25 S12 S_last12 : ℕ) (A : ℕ) 
  (h1 : S25 = 25 * 24) 
  (h2: S12 = 12 * 14) 
  (h3: 12 * A = S_last12)
  (h4: S25 = S12 + 228 + S_last12) : A = 17 := 
by
  sorry

end average_last_12_results_l188_188809


namespace marias_workday_end_time_l188_188168

theorem marias_workday_end_time :
  ∀ (start_time : ℕ) (lunch_time : ℕ) (work_duration : ℕ) (lunch_break : ℕ) (total_work_time : ℕ),
  start_time = 8 ∧ lunch_time = 13 ∧ work_duration = 8 ∧ lunch_break = 1 →
  (total_work_time = work_duration - (lunch_time - start_time - lunch_break)) →
  lunch_time + 1 + (work_duration - (lunch_time - start_time)) = 17 :=
by
  sorry

end marias_workday_end_time_l188_188168


namespace width_of_smaller_cuboids_is_4_l188_188415

def length_smaller_cuboid := 5
def height_smaller_cuboid := 3
def length_larger_cuboid := 16
def width_larger_cuboid := 10
def height_larger_cuboid := 12
def num_smaller_cuboids := 32

theorem width_of_smaller_cuboids_is_4 :
  ∃ W : ℝ, W = 4 ∧ (length_smaller_cuboid * W * height_smaller_cuboid) * num_smaller_cuboids = 
            length_larger_cuboid * width_larger_cuboid * height_larger_cuboid :=
by
  sorry

end width_of_smaller_cuboids_is_4_l188_188415


namespace num_pairs_with_math_book_l188_188097

theorem num_pairs_with_math_book (books : Finset String) (h : books = {"Chinese", "Mathematics", "English", "Biology", "History"}):
  (∃ pairs : Finset (Finset String), pairs.card = 4 ∧ ∀ pair ∈ pairs, "Mathematics" ∈ pair) :=
by
  sorry

end num_pairs_with_math_book_l188_188097


namespace problem_l188_188571

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1 else S n - S (n - 1)

def sum_abs_a_10 : ℤ :=
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|)

theorem problem : sum_abs_a_10 = 67 := by
  sorry

end problem_l188_188571


namespace selection_schemes_correct_l188_188424

-- Define the problem parameters
def number_of_selection_schemes (persons : ℕ) (cities : ℕ) (persons_cannot_visit : ℕ) : ℕ :=
  let choices_for_paris := persons - persons_cannot_visit
  let remaining_people := persons - 1
  choices_for_paris * remaining_people * (remaining_people - 1) * (remaining_people - 2)

-- Define the example constants
def total_people : ℕ := 6
def total_cities : ℕ := 4
def cannot_visit_paris : ℕ := 2

-- The statement to be proved
theorem selection_schemes_correct : 
  number_of_selection_schemes total_people total_cities cannot_visit_paris = 240 := by
  sorry

end selection_schemes_correct_l188_188424


namespace patricia_earns_more_than_jose_l188_188153

noncomputable def jose_final_amount : ℝ :=
  50000 * (1 + 0.04)^2

noncomputable def patricia_final_amount : ℝ :=
  50000 * (1 + 0.01)^8

theorem patricia_earns_more_than_jose :
  patricia_final_amount - jose_final_amount = 63 :=
by
  -- from solution steps
  /-
  jose_final_amount = 50000 * (1 + 0.04)^2 = 54080
  patricia_final_amount = 50000 * (1 + 0.01)^8 ≈ 54143
  patricia_final_amount - jose_final_amount ≈ 63
  -/
  sorry

end patricia_earns_more_than_jose_l188_188153


namespace eval_expr_ceil_floor_l188_188001

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l188_188001


namespace average_tree_height_is_800_l188_188915

def first_tree_height : ℕ := 1000
def other_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200
def total_height : ℕ := first_tree_height + other_tree_height + other_tree_height + last_tree_height
def average_height : ℕ := total_height / 4

theorem average_tree_height_is_800 :
  average_height = 800 := by
  sorry

end average_tree_height_is_800_l188_188915


namespace number_of_boys_l188_188618

theorem number_of_boys (n : ℕ) (handshakes : ℕ) (h_handshakes : handshakes = n * (n - 1) / 2) (h_total : handshakes = 55) : n = 11 := by
  sorry

end number_of_boys_l188_188618


namespace cake_pieces_in_pan_l188_188358

theorem cake_pieces_in_pan :
  (24 * 30) / (3 * 2) = 120 := by
  sorry

end cake_pieces_in_pan_l188_188358


namespace proof_problem_l188_188443

theorem proof_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
sorry

end proof_problem_l188_188443


namespace train_speed_45_kmph_l188_188910

variable (length_train length_bridge time_passed : ℕ)

def total_distance (length_train length_bridge : ℕ) : ℕ :=
  length_train + length_bridge

def speed_m_per_s (length_train length_bridge time_passed : ℕ) : ℚ :=
  (total_distance length_train length_bridge) / time_passed

def speed_km_per_h (length_train length_bridge time_passed : ℕ) : ℚ :=
  (speed_m_per_s length_train length_bridge time_passed) * 3.6

theorem train_speed_45_kmph :
  length_train = 360 → length_bridge = 140 → time_passed = 40 → speed_km_per_h length_train length_bridge time_passed = 45 := 
by
  sorry

end train_speed_45_kmph_l188_188910


namespace length_of_ab_l188_188368

variable (a b c d e : ℝ)
variable (bc cd de ac ae ab : ℝ)

axiom bc_eq_3cd : bc = 3 * cd
axiom de_eq_7 : de = 7
axiom ac_eq_11 : ac = 11
axiom ae_eq_20 : ae = 20
axiom ac_def : ac = ab + bc -- Definition of ac
axiom ae_def : ae = ab + bc + cd + de -- Definition of ae

theorem length_of_ab : ab = 5 := by
  sorry

end length_of_ab_l188_188368


namespace ratio_distance_l188_188518

-- Definitions based on conditions
def speed_ferry_P : ℕ := 6 -- speed of ferry P in km/h
def time_ferry_P : ℕ := 3 -- travel time of ferry P in hours
def speed_ferry_Q : ℕ := speed_ferry_P + 3 -- speed of ferry Q in km/h
def time_ferry_Q : ℕ := time_ferry_P + 1 -- travel time of ferry Q in hours

-- Calculating the distances
def distance_ferry_P : ℕ := speed_ferry_P * time_ferry_P -- distance covered by ferry P
def distance_ferry_Q : ℕ := speed_ferry_Q * time_ferry_Q -- distance covered by ferry Q

-- Main theorem to prove
theorem ratio_distance (d_P d_Q : ℕ) (h_dP : d_P = distance_ferry_P) (h_dQ : d_Q = distance_ferry_Q) : d_Q / d_P = 2 :=
by
  sorry

end ratio_distance_l188_188518


namespace prove_seq_properties_l188_188591

theorem prove_seq_properties (a b : ℕ → ℕ) (S T : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_sum : ∀ n, 2 * S n = a n ^ 2 + n)
  (h_b : ∀ n, b n = a (n + 1) * 2 ^ n)
  : (∀ n, a n = n) ∧ (∀ n, T n = n * 2 ^ (n + 1)) :=
sorry

end prove_seq_properties_l188_188591


namespace f_zero_eq_one_f_positive_f_increasing_f_range_x_l188_188543

noncomputable def f : ℝ → ℝ := sorry
axiom f_condition1 : f 0 ≠ 0
axiom f_condition2 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_condition3 : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_positive : ∀ x : ℝ, f x > 0 :=
sorry

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
sorry

theorem f_range_x (x : ℝ) (h : f x * f (2 * x - x^2) > 1) : x ∈ { x : ℝ | f x > 1 ∧ f (2 * x - x^2) > 1 } :=
sorry

end f_zero_eq_one_f_positive_f_increasing_f_range_x_l188_188543


namespace probability_2_1_to_2_5_l188_188416

noncomputable def F (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then (x - 2)^2
else 1

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then 2 * (x - 2)
else 0

theorem probability_2_1_to_2_5 : 
  (F 2.5 - F 2.1 = 0.24) := 
by
  -- calculations and proof go here, but we skip it with sorry
  sorry

end probability_2_1_to_2_5_l188_188416


namespace unique_t_digit_l188_188076

theorem unique_t_digit (t : ℕ) (ht : t < 100) (ht2 : 10 ≤ t) (h : 13 * t ≡ 42 [MOD 100]) : t = 34 := 
by
-- Proof is omitted
sorry

end unique_t_digit_l188_188076


namespace diagonals_in_nine_sided_polygon_l188_188530

-- Define the conditions
def sides : ℕ := 9
def right_angles : ℕ := 2

-- The function to calculate the number of diagonals for a polygon
def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The theorem to prove
theorem diagonals_in_nine_sided_polygon : number_of_diagonals sides = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l188_188530


namespace password_count_l188_188278

theorem password_count : ∃ s : Finset ℕ, s.card = 4 ∧ s.sum id = 27 ∧ 
  (s = {9, 8, 7, 3} ∨ s = {9, 8, 6, 4} ∨ s = {9, 7, 6, 5}) ∧ 
  (s.toList.permutations.length = 72) := sorry

end password_count_l188_188278


namespace water_pressure_on_dam_l188_188488

theorem water_pressure_on_dam :
  let a := 10 -- length of upper base in meters
  let b := 20 -- length of lower base in meters
  let h := 3 -- height in meters
  let ρg := 9810 -- natural constant for water pressure in N/m^3
  let P := ρg * ((a + 2 * b) * h^2 / 6)
  P = 735750 :=
by
  sorry

end water_pressure_on_dam_l188_188488


namespace ratio_surfer_malibu_santa_monica_l188_188908

theorem ratio_surfer_malibu_santa_monica (M S : ℕ) (hS : S = 20) (hTotal : M + S = 60) : M / S = 2 :=
by 
  sorry

end ratio_surfer_malibu_santa_monica_l188_188908


namespace grandfather_older_than_grandmother_l188_188041

noncomputable def Milena_age : ℕ := 7

noncomputable def Grandmother_age : ℕ := Milena_age * 9

noncomputable def Grandfather_age : ℕ := Milena_age + 58

theorem grandfather_older_than_grandmother :
  Grandfather_age - Grandmother_age = 2 := by
  sorry

end grandfather_older_than_grandmother_l188_188041


namespace T_n_formula_l188_188479

def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ n
def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a_n (k + 1) * b_n (k + 1))

theorem T_n_formula (n : ℕ) : T_n n = 8 - 8 * 2 ^ n + 3 * n * 2 ^ (n + 1) :=
by 
  sorry

end T_n_formula_l188_188479


namespace least_n_div_mod_l188_188111

theorem least_n_div_mod (n : ℕ) (h_pos : n > 1) (h_mod25 : n % 25 = 1) (h_mod7 : n % 7 = 1) : n = 176 :=
by
  sorry

end least_n_div_mod_l188_188111


namespace teachers_per_grade_correct_l188_188470

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def parents_per_grade : ℕ := 2
def number_of_grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Total number of students
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders

-- Total number of parents
def total_parents : ℕ := parents_per_grade * number_of_grades

-- Total number of seats available on the buses
def total_seats : ℕ := buses * seats_per_bus

-- Seats left for teachers
def seats_for_teachers : ℕ := total_seats - total_students - total_parents

-- Teachers per grade
def teachers_per_grade : ℕ := seats_for_teachers / number_of_grades

theorem teachers_per_grade_correct : teachers_per_grade = 4 := sorry

end teachers_per_grade_correct_l188_188470


namespace ratio_area_triangles_to_square_l188_188120

theorem ratio_area_triangles_to_square (x : ℝ) :
  let A := (0, x)
  let B := (x, x)
  let C := (x, 0)
  let D := (0, 0)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let area_AMN := 1/2 * ((M.1 - A.1) * (N.2 - A.2) - (M.2 - A.2) * (N.1 - A.1))
  let area_MNP := 1/2 * ((N.1 - M.1) * (P.2 - M.2) - (N.2 - M.2) * (P.1 - M.1))
  let total_area_triangles := area_AMN + area_MNP
  let area_square := x * x
  total_area_triangles / area_square = 1/4 := 
by
  sorry

end ratio_area_triangles_to_square_l188_188120


namespace cylinder_heights_relation_l188_188302

variables {r1 r2 h1 h2 : ℝ}

theorem cylinder_heights_relation 
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = (6 / 5) * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_heights_relation_l188_188302


namespace sufficient_but_not_necessary_condition_l188_188976

open Real

theorem sufficient_but_not_necessary_condition :
  ∀ (m : ℝ),
  (∀ x, (x^2 - 3*x - 4 ≤ 0) → (x^2 - 6*x + 9 - m^2 ≤ 0)) ∧
  (∃ x, ¬(x^2 - 3*x - 4 ≤ 0) ∧ (x^2 - 6*x + 9 - m^2 ≤ 0)) ↔
  m ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
by
  sorry

end sufficient_but_not_necessary_condition_l188_188976


namespace paper_plate_cup_cost_l188_188307

variables (P C : ℝ)

theorem paper_plate_cup_cost (h : 100 * P + 200 * C = 6) : 20 * P + 40 * C = 1.20 :=
by sorry

end paper_plate_cup_cost_l188_188307


namespace find_sum_of_integers_l188_188248

theorem find_sum_of_integers (x y : ℕ) (h_diff : x - y = 8) (h_prod : x * y = 180) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : x + y = 28 :=
by
  sorry

end find_sum_of_integers_l188_188248


namespace factor_expression_l188_188652

variable (x : ℝ)

-- Mathematically define the expression e
def e : ℝ := 4 * x * (x + 2) + 10 * (x + 2) + 2 * (x + 2)

-- State that e is equivalent to the factored form
theorem factor_expression : e x = (x + 2) * (4 * x + 12) :=
by
  sorry

end factor_expression_l188_188652


namespace segment_length_of_absolute_value_l188_188095

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l188_188095


namespace hyperbola_eccentricity_l188_188138

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_eq1 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : c = Real.sqrt (a^2 + b^2))
  (h_dist : ∀ x, x = b * c / Real.sqrt (a^2 + b^2))
  (h_eq3 : a = b) :
  e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l188_188138


namespace power_function_value_at_4_l188_188125

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_value_at_4 :
  ∃ a : ℝ, power_function a 2 = (Real.sqrt 2) / 2 → power_function a 4 = 1 / 2 :=
by
  sorry

end power_function_value_at_4_l188_188125


namespace find_a_l188_188060

theorem find_a (a : ℝ) (h : ∃ (b : ℝ), (16 * (x : ℝ) * x) + 40 * x + a = (4 * x + b) ^ 2) : a = 25 := sorry

end find_a_l188_188060


namespace cookies_per_bag_l188_188000

theorem cookies_per_bag (n_bags : ℕ) (total_cookies : ℕ) (n_candies : ℕ) (h_bags : n_bags = 26) (h_cookies : total_cookies = 52) (h_candies : n_candies = 15) : (total_cookies / n_bags) = 2 :=
by sorry

end cookies_per_bag_l188_188000


namespace inscribed_circle_radius_integer_l188_188227

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l188_188227


namespace initial_miles_correct_l188_188303

-- Definitions and conditions
def miles_per_gallon : ℕ := 30
def gallons_per_tank : ℕ := 20
def current_miles : ℕ := 2928
def tanks_filled : ℕ := 2

-- Question: How many miles were on the car before the road trip?
def initial_miles : ℕ := current_miles - (miles_per_gallon * gallons_per_tank * tanks_filled)

-- Proof problem statement
theorem initial_miles_correct : initial_miles = 1728 :=
by
  -- Here we expect the proof, but are skipping it with 'sorry'
  sorry

end initial_miles_correct_l188_188303


namespace rectangle_side_ratio_l188_188925

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end rectangle_side_ratio_l188_188925


namespace sum_lent_l188_188301

theorem sum_lent (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ)
  (hR: R = 4) 
  (hT: T = 8) 
  (hI1 : I = P - 306) 
  (hI2 : I = P * R * T / 100) :
  P = 450 :=
by
  sorry

end sum_lent_l188_188301


namespace lucas_150_mod_9_l188_188536

-- Define the Lucas sequence recursively
def lucas (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- Since L_1 in the sequence provided is actually the first Lucas number (index starts from 1)
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

-- Define the theorem for the remainder when the 150th term is divided by 9
theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by
  sorry

end lucas_150_mod_9_l188_188536


namespace problem1_problem2_problem3_problem4_l188_188037

-- Problem 1
theorem problem1 (x : ℝ) : 0.75 * x = (1 / 2) * 12 → x = 8 := 
by 
  intro h
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (0.7 / x) = (14 / 5) → x = 0.25 := 
by 
  intro h
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (1 / 6) * x = (2 / 15) * (2 / 3) → x = (8 / 15) := 
by 
  intro h
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : 4.5 * x = 4 * 27 → x = 24 := 
by 
  intro h
  sorry

end problem1_problem2_problem3_problem4_l188_188037


namespace initial_money_amount_l188_188650

theorem initial_money_amount (M : ℝ)
  (h_clothes : M * (1 / 3) = c)
  (h_food : (M - c) * (1 / 5) = f)
  (h_travel : (M - c - f) * (1 / 4) = t)
  (h_remaining : M - c - f - t = 600) : M = 1500 := by
  sorry

end initial_money_amount_l188_188650


namespace elodie_rats_l188_188448

-- Define the problem conditions as hypotheses
def E (H : ℕ) : ℕ := H + 10
def K (H : ℕ) : ℕ := 3 * (E H + H)

-- The goal is to prove E = 30 given the conditions
theorem elodie_rats (H : ℕ) (h1 : E (H := H) + H + K (H := H) = 200) : E H = 30 :=
by
  sorry

end elodie_rats_l188_188448


namespace total_students_in_college_l188_188127

theorem total_students_in_college (B G : ℕ) (h_ratio: 8 * G = 5 * B) (h_girls: G = 175) :
  B + G = 455 := 
  sorry

end total_students_in_college_l188_188127


namespace weight_difference_l188_188221

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h_avg_ABC : (W_A + W_B + W_C) / 3 = 80)
  (h_WA : W_A = 95)
  (h_avg_ABCD : (W_A + W_B + W_C + W_D) / 4 = 82)
  (h_avg_BCDE : (W_B + W_C + W_D + W_E) / 4 = 81) :
  W_E - W_D = 3 :=
by
  sorry

end weight_difference_l188_188221


namespace combined_sleep_hours_l188_188381

theorem combined_sleep_hours :
  let connor_sleep_hours := 6
  let luke_sleep_hours := connor_sleep_hours + 2
  let emma_sleep_hours := connor_sleep_hours - 1
  let ava_sleep_hours :=
    2 * 5 + 
    2 * (5 + 1) + 
    2 * (5 + 2) + 
    (5 + 3)
  let puppy_sleep_hours := 2 * luke_sleep_hours
  let cat_sleep_hours := 4 + 7
  7 * connor_sleep_hours +
  7 * luke_sleep_hours +
  7 * emma_sleep_hours +
  ava_sleep_hours +
  7 * puppy_sleep_hours +
  7 * cat_sleep_hours = 366 :=
by
  sorry

end combined_sleep_hours_l188_188381


namespace marching_band_total_weight_l188_188213

def weight_trumpets := 5
def weight_clarinets := 5
def weight_trombones := 10
def weight_tubas := 20
def weight_drums := 15

def count_trumpets := 6
def count_clarinets := 9
def count_trombones := 8
def count_tubas := 3
def count_drums := 2

theorem marching_band_total_weight :
  (count_trumpets * weight_trumpets) + (count_clarinets * weight_clarinets) + (count_trombones * weight_trombones) + 
  (count_tubas * weight_tubas) + (count_drums * weight_drums) = 245 :=
by
  sorry

end marching_band_total_weight_l188_188213


namespace brie_clothes_washer_l188_188360

theorem brie_clothes_washer (total_blouses total_skirts total_slacks : ℕ)
  (blouses_pct skirts_pct slacks_pct : ℝ)
  (h_blouses : total_blouses = 12)
  (h_skirts : total_skirts = 6)
  (h_slacks : total_slacks = 8)
  (h_blouses_pct : blouses_pct = 0.75)
  (h_skirts_pct : skirts_pct = 0.5)
  (h_slacks_pct : slacks_pct = 0.25) :
  let blouses_in_hamper := total_blouses * blouses_pct
  let skirts_in_hamper := total_skirts * skirts_pct
  let slacks_in_hamper := total_slacks * slacks_pct
  blouses_in_hamper + skirts_in_hamper + slacks_in_hamper = 14 := 
by
  sorry

end brie_clothes_washer_l188_188360


namespace khali_total_snow_volume_l188_188852

def length1 : ℝ := 25
def width1 : ℝ := 3
def depth1 : ℝ := 0.75

def length2 : ℝ := 15
def width2 : ℝ := 3
def depth2 : ℝ := 1

def volume1 : ℝ := length1 * width1 * depth1
def volume2 : ℝ := length2 * width2 * depth2
def total_volume : ℝ := volume1 + volume2

theorem khali_total_snow_volume : total_volume = 101.25 := by
  sorry

end khali_total_snow_volume_l188_188852


namespace range_of_a_l188_188075

variable (a : ℝ) (x : ℝ) (x₀ : ℝ)

def p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ (x₀ : ℝ), ∃ (x : ℝ), x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l188_188075


namespace logarithmic_inequality_l188_188434

theorem logarithmic_inequality (a : ℝ) (h : a > 1) : 
  1 / 2 + 1 / Real.log a ≥ 1 := 
sorry

end logarithmic_inequality_l188_188434


namespace solve_for_a_l188_188779

theorem solve_for_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end solve_for_a_l188_188779


namespace probability_white_marble_l188_188359

theorem probability_white_marble :
  ∀ (p_blue p_green p_white : ℝ),
    p_blue = 0.25 →
    p_green = 0.4 →
    p_blue + p_green + p_white = 1 →
    p_white = 0.35 :=
by
  intros p_blue p_green p_white h_blue h_green h_total
  sorry

end probability_white_marble_l188_188359


namespace tangent_line_to_parabola_k_value_l188_188816

theorem tangent_line_to_parabola_k_value (k : ℝ) :
  (∀ x y : ℝ, 4 * x - 3 * y + k = 0 → y^2 = 16 * x → (4 * x - 3 * y + k = 0 ∧ y^2 = 16 * x) ∧ (144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_k_value_l188_188816


namespace gcd_equivalence_l188_188986

theorem gcd_equivalence : 
  let m := 2^2100 - 1
  let n := 2^2091 + 31
  gcd m n = gcd (2^2091 + 31) 511 :=
by
  sorry

end gcd_equivalence_l188_188986


namespace chess_tournament_games_l188_188017

theorem chess_tournament_games (n : ℕ) (h : n = 16) :
  (n * (n - 1) * 2) / 2 = 480 :=
by
  rw [h]
  simp
  norm_num
  sorry

end chess_tournament_games_l188_188017


namespace martin_speed_l188_188707

theorem martin_speed (distance time : ℝ) (h_distance : distance = 12) (h_time : time = 6) :
  distance / time = 2 :=
by
  rw [h_distance, h_time]
  norm_num

end martin_speed_l188_188707


namespace area_of_region_l188_188440

-- Definitions from the problem's conditions.
def equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 10*y = -9

-- Statement of the theorem.
theorem area_of_region : 
  ∃ (area : ℝ), (∀ x y : ℝ, equation x y → True) ∧ area = 32 * Real.pi :=
by
  sorry

end area_of_region_l188_188440


namespace radii_touching_circles_l188_188280

noncomputable def radius_of_circles_touching_unit_circles 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (centerA centerB centerC : A) 
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius) 
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius) 
  : Prop :=
  ∃ r₁ r₂ : ℝ, r₁ = 1/3 ∧ r₂ = 7/3

theorem radii_touching_circles (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (centerA centerB centerC : A)
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius)
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius)
  : radius_of_circles_touching_unit_circles A B C centerA centerB centerC unit_radius h1 h2 h3 :=
sorry

end radii_touching_circles_l188_188280


namespace tan_x_over_tan_y_plus_tan_y_over_tan_x_l188_188500

open Real

theorem tan_x_over_tan_y_plus_tan_y_over_tan_x (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 5) :
  tan x / tan y + tan y / tan x = 10 := 
by
  sorry

end tan_x_over_tan_y_plus_tan_y_over_tan_x_l188_188500


namespace medians_form_right_triangle_medians_inequality_l188_188892

variable {α : Type*}
variables {a b c : ℝ}
variables {m_a m_b m_c : ℝ}
variable (orthogonal_medians : m_a * m_b = 0)

-- Part (a)
theorem medians_form_right_triangle
  (orthogonal_medians : m_a * m_b = 0) :
  m_a^2 + m_b^2 = m_c^2 :=
sorry

-- Part (b)
theorem medians_inequality
  (orthogonal_medians : m_a * m_b = 0)
  (triangle_sides : a^2 + b^2 = 5 * c^2): 
  5 * (a^2 + b^2 - c^2) ≥ 8 * a * b :=
sorry

end medians_form_right_triangle_medians_inequality_l188_188892


namespace largest_constant_l188_188596

def equation_constant (c d : ℝ) : ℝ :=
  5 * c + (d - 12)^2

theorem largest_constant : ∃ constant : ℝ, (∀ c, c ≤ 47) → (∀ d, equation_constant 47 d = constant) → constant = 235 := 
by
  sorry

end largest_constant_l188_188596


namespace dice_probability_l188_188497

theorem dice_probability (p : ℚ) (h : p = (1 / 42)) : 
  p = 0.023809523809523808 := 
sorry

end dice_probability_l188_188497


namespace point_on_y_axis_l188_188087

theorem point_on_y_axis (a : ℝ) 
  (h : (a - 2) = 0) : a = 2 := 
  by 
    sorry

end point_on_y_axis_l188_188087


namespace angle_bisector_ratio_l188_188990

theorem angle_bisector_ratio (XY XZ YZ : ℝ) (hXY : XY = 8) (hXZ : XZ = 6) (hYZ : YZ = 4) :
  ∃ (Q : Point) (YQ QV : ℝ), YQ / QV = 2 :=
by
  sorry

end angle_bisector_ratio_l188_188990


namespace radius_of_inner_tangent_circle_l188_188899

theorem radius_of_inner_tangent_circle (side_length : ℝ) (num_semicircles_per_side : ℝ) (semicircle_radius : ℝ)
  (h_side_length : side_length = 4) (h_num_semicircles_per_side : num_semicircles_per_side = 3) 
  (h_semicircle_radius : semicircle_radius = side_length / (2 * num_semicircles_per_side)) :
  ∃ (inner_circle_radius : ℝ), inner_circle_radius = 7 / 6 :=
by
  sorry

end radius_of_inner_tangent_circle_l188_188899


namespace grade_assignment_ways_l188_188790

-- Define the number of students and the number of grade choices
def students : ℕ := 12
def grade_choices : ℕ := 4

-- Define the number of ways to assign grades
def num_ways_to_assign_grades : ℕ := grade_choices ^ students

-- Prove that the number of ways to assign grades is 16777216
theorem grade_assignment_ways :
  num_ways_to_assign_grades = 16777216 :=
by
  -- Calculation validation omitted (proof step)
  sorry

end grade_assignment_ways_l188_188790


namespace distance_between_towns_l188_188514

theorem distance_between_towns (D S : ℝ) (h1 : D = S * 3) (h2 : 200 = S * 5) : D = 120 :=
by
  sorry

end distance_between_towns_l188_188514


namespace gordon_total_cost_l188_188988

noncomputable def DiscountA (price : ℝ) : ℝ :=
if price > 22.00 then price * 0.70 else price

noncomputable def DiscountB (price : ℝ) : ℝ :=
if 10.00 < price ∧ price <= 20.00 then price * 0.80 else price

noncomputable def DiscountC (price : ℝ) : ℝ :=
if price < 10.00 then price * 0.85 else price

noncomputable def apply_discount (price : ℝ) : ℝ :=
if price > 22.00 then DiscountA price
else if price > 10.00 then DiscountB price
else DiscountC price

noncomputable def total_price (prices : List ℝ) : ℝ :=
(prices.map apply_discount).sum

noncomputable def total_with_tax_and_fee (prices : List ℝ) (tax_rate extra_fee : ℝ) : ℝ :=
let total := total_price prices
let tax := total * tax_rate
total + tax + extra_fee

theorem gordon_total_cost :
  total_with_tax_and_fee
    [25.00, 18.00, 21.00, 35.00, 12.00, 10.00, 8.50, 23.00, 6.00, 15.50, 30.00, 9.50]
    0.05 2.00
  = 171.27 :=
  sorry

end gordon_total_cost_l188_188988


namespace solution_set_of_f_inequality_l188_188033

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_deriv : ∀ x, f' x < f x)
variable (h_even : ∀ x, f (x + 2) = f (-x + 2))
variable (h_initial : f 0 = Real.exp 4)

theorem solution_set_of_f_inequality :
  {x : ℝ | f x < Real.exp x} = {x : ℝ | x > 4} := 
sorry

end solution_set_of_f_inequality_l188_188033


namespace simplify_expression_l188_188079

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 2))) = (Real.sqrt 3 - 2 * Real.sqrt 5 - 3) :=
by
  sorry

end simplify_expression_l188_188079


namespace sum_two_numbers_in_AP_and_GP_equals_20_l188_188544

theorem sum_two_numbers_in_AP_and_GP_equals_20 :
  ∃ a b : ℝ, 
    (a > 0) ∧ (b > 0) ∧ 
    (4 < a) ∧ (a < b) ∧ 
    (4 + (a - 4) = a) ∧ (4 + 2 * (a - 4) = b) ∧
    (a * (b / a) = b) ∧ (b * (b / a) = 16) ∧ 
    a + b = 20 :=
by
  sorry

end sum_two_numbers_in_AP_and_GP_equals_20_l188_188544


namespace trapezoid_area_l188_188176

-- Definitions of the problem's conditions
def a : ℕ := 4
def b : ℕ := 8
def h : ℕ := 3

-- Lean statement to prove the area of the trapezoid is 18 square centimeters
theorem trapezoid_area : (a + b) * h / 2 = 18 := by
  sorry

end trapezoid_area_l188_188176


namespace interest_calculation_years_l188_188128

theorem interest_calculation_years (P r : ℝ) (diff : ℝ) (n : ℕ) 
  (hP : P = 3600) (hr : r = 0.10) (hdiff : diff = 36) 
  (h_eq : P * (1 + r)^n - P - (P * r * n) = diff) : n = 2 :=
sorry

end interest_calculation_years_l188_188128


namespace ellipse_eccentricity_range_l188_188670

theorem ellipse_eccentricity_range (a b : ℝ) (h : a > b) (h_b : b > 0) : 
  ∃ e : ℝ, (e = (Real.sqrt (a^2 - b^2)) / a) ∧ (e > 1/2 ∧ e < 1) :=
by
  sorry

end ellipse_eccentricity_range_l188_188670


namespace expected_value_in_classroom_l188_188295

noncomputable def expected_pairs_next_to_each_other (boys girls : ℕ) : ℕ :=
  if boys = 9 ∧ girls = 14 ∧ boys + girls = 23 then
    10 -- Based on provided conditions and conclusion
  else
    0

theorem expected_value_in_classroom :
  expected_pairs_next_to_each_other 9 14 = 10 :=
by
  sorry

end expected_value_in_classroom_l188_188295


namespace range_of_a_l188_188714

noncomputable def f (a x : ℝ) := Real.logb (1 / 2) (x^2 - a * x - a)

theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f a x ∈ Set.univ) ∧ 
            (∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 1 - Real.sqrt 3 → f a x1 < f a x2)) → 
  (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l188_188714


namespace part_a_l188_188265

theorem part_a (cities : Finset (ℝ × ℝ)) (h_cities : cities.card = 100) 
  (distances : Finset (ℝ × ℝ → ℝ)) (h_distances : distances.card = 4950) :
  ∃ (erased_distance : ℝ × ℝ → ℝ), ¬ ∃ (restored_distance : ℝ × ℝ → ℝ), 
    restored_distance = erased_distance :=
sorry

end part_a_l188_188265


namespace factor_expression_l188_188911

theorem factor_expression (x : ℝ) : 72 * x ^ 5 - 162 * x ^ 9 = -18 * x ^ 5 * (9 * x ^ 4 - 4) :=
by
  sorry

end factor_expression_l188_188911


namespace f_correct_l188_188260

noncomputable def f (n : ℕ) : ℕ :=
  if h : n ≥ 15 then (n - 1) / 2
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n = 6 then 4
  else if 7 ≤ n ∧ n ≤ 15 then 7
  else 0

theorem f_correct (n : ℕ) (hn : n ≥ 3) : 
  f n = if n ≥ 15 then (n - 1) / 2
        else if n = 3 then 1
        else if n = 4 then 1
        else if n = 5 then 2
        else if n = 6 then 4
        else if 7 ≤ n ∧ n ≤ 15 then 7
        else 0 := sorry

end f_correct_l188_188260


namespace factor_polynomial_l188_188291

theorem factor_polynomial (x : ℝ) :
  3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) :=
by
  sorry

end factor_polynomial_l188_188291


namespace solve_equation_l188_188602

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  -x^2 = (4 * x + 2) / (x^2 + 3 * x + 2) ↔ x = -1 :=
by
  sorry

end solve_equation_l188_188602


namespace find_m_of_quadratic_function_l188_188010

theorem find_m_of_quadratic_function :
  ∀ (m : ℝ), (m + 1 ≠ 0) → ((m + 1) * x ^ (m^2 + 1) + 5 = a * x^2 + b * x + c) → m = 1 :=
by
  intro m h h_quad
  -- Proof Here
  sorry

end find_m_of_quadratic_function_l188_188010


namespace max_value_OP_OQ_l188_188897

def circle_1_polar_eq (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

def circle_2_polar_eq (rho theta : ℝ) : Prop :=
  rho = 2 * Real.sin theta

theorem max_value_OP_OQ (alpha : ℝ) :
  (∃ rho1 rho2 : ℝ, circle_1_polar_eq rho1 alpha ∧ circle_2_polar_eq rho2 alpha) ∧
  (∃ max_OP_OQ : ℝ, max_OP_OQ = 4) :=
sorry

end max_value_OP_OQ_l188_188897


namespace mixture_concentration_l188_188136

-- Definitions reflecting the given conditions
def sol1_concentration : ℝ := 0.30
def sol1_volume : ℝ := 8

def sol2_concentration : ℝ := 0.50
def sol2_volume : ℝ := 5

def sol3_concentration : ℝ := 0.70
def sol3_volume : ℝ := 7

-- The proof problem stating that the resulting concentration is 49%
theorem mixture_concentration :
  (sol1_concentration * sol1_volume + sol2_concentration * sol2_volume + sol3_concentration * sol3_volume) /
  (sol1_volume + sol2_volume + sol3_volume) * 100 = 49 :=
by
  sorry

end mixture_concentration_l188_188136


namespace sequence_mod_100_repeats_l188_188198

theorem sequence_mod_100_repeats (a0 : ℕ) : ∃ k l, k ≠ l ∧ (∃ seq : ℕ → ℕ, seq 0 = a0 ∧ (∀ n, seq (n + 1) = seq n + 54 ∨ seq (n + 1) = seq n + 77) ∧ (seq k % 100 = seq l % 100)) :=
by 
  sorry

end sequence_mod_100_repeats_l188_188198


namespace cost_of_adult_ticket_l188_188462

theorem cost_of_adult_ticket
    (child_ticket_cost : ℝ)
    (total_tickets : ℕ)
    (total_receipts : ℝ)
    (adult_tickets_sold : ℕ)
    (A : ℝ)
    (child_tickets_sold : ℕ := total_tickets - adult_tickets_sold)
    (total_revenue_adult : ℝ := adult_tickets_sold * A)
    (total_revenue_child : ℝ := child_tickets_sold * child_ticket_cost) :
    child_ticket_cost = 4 →
    total_tickets = 130 →
    total_receipts = 840 →
    adult_tickets_sold = 90 →
    total_revenue_adult + total_revenue_child = total_receipts →
    A = 7.56 :=
by
  intros
  sorry

end cost_of_adult_ticket_l188_188462


namespace number_of_solutions_l188_188081

theorem number_of_solutions :
  ∃ (x y z : ℝ), 
    (x = 4036 - 4037 * Real.sign (y - z)) ∧ 
    (y = 4036 - 4037 * Real.sign (z - x)) ∧ 
    (z = 4036 - 4037 * Real.sign (x - y)) :=
sorry

end number_of_solutions_l188_188081


namespace cp_of_apple_l188_188785

theorem cp_of_apple (SP : ℝ) (hSP : SP = 17) (loss_fraction : ℝ) (h_loss_fraction : loss_fraction = 1 / 6) : 
  ∃ CP : ℝ, CP = 20.4 ∧ SP = CP - loss_fraction * CP :=
by
  -- Placeholder for proof
  sorry

end cp_of_apple_l188_188785


namespace total_swimming_hours_over_4_weeks_l188_188751

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l188_188751


namespace repeating_decimal_product_l188_188402

-- Define the repeating decimal 0.\overline{137} as a fraction
def repeating_decimal_137 : ℚ := 137 / 999

-- Define the repeating decimal 0.\overline{6} as a fraction
def repeating_decimal_6 : ℚ := 2 / 3

-- The problem is to prove that the product of these fractions is 274 / 2997
theorem repeating_decimal_product : repeating_decimal_137 * repeating_decimal_6 = 274 / 2997 := by
  sorry

end repeating_decimal_product_l188_188402


namespace calculate_amount_left_l188_188214

def base_income : ℝ := 2000
def bonus_percentage : ℝ := 0.15
def public_transport_percentage : ℝ := 0.05
def rent : ℝ := 500
def utilities : ℝ := 100
def food : ℝ := 300
def miscellaneous_percentage : ℝ := 0.10
def savings_percentage : ℝ := 0.07
def investment_percentage : ℝ := 0.05
def medical_expense : ℝ := 250
def tax_percentage : ℝ := 0.15

def total_income (base_income : ℝ) (bonus_percentage : ℝ) : ℝ :=
  base_income + (bonus_percentage * base_income)

def taxes (base_income : ℝ) (tax_percentage : ℝ) : ℝ :=
  tax_percentage * base_income

def total_fixed_expenses (rent : ℝ) (utilities : ℝ) (food : ℝ) : ℝ :=
  rent + utilities + food

def public_transport_expense (total_income : ℝ) (public_transport_percentage : ℝ) : ℝ :=
  public_transport_percentage * total_income

def miscellaneous_expense (total_income : ℝ) (miscellaneous_percentage : ℝ) : ℝ :=
  miscellaneous_percentage * total_income

def variable_expenses (public_transport_expense : ℝ) (miscellaneous_expense : ℝ) : ℝ :=
  public_transport_expense + miscellaneous_expense

def savings (total_income : ℝ) (savings_percentage : ℝ) : ℝ :=
  savings_percentage * total_income

def investment (total_income : ℝ) (investment_percentage : ℝ) : ℝ :=
  investment_percentage * total_income

def total_savings_investments (savings : ℝ) (investment : ℝ) : ℝ :=
  savings + investment

def total_expenses_contributions 
  (fixed_expenses : ℝ) 
  (variable_expenses : ℝ) 
  (medical_expense : ℝ) 
  (total_savings_investments : ℝ) : ℝ :=
  fixed_expenses + variable_expenses + medical_expense + total_savings_investments

def amount_left (income_after_taxes : ℝ) (total_expenses_contributions : ℝ) : ℝ :=
  income_after_taxes - total_expenses_contributions

theorem calculate_amount_left 
  (base_income : ℝ)
  (bonus_percentage : ℝ)
  (public_transport_percentage : ℝ)
  (rent : ℝ)
  (utilities : ℝ)
  (food : ℝ)
  (miscellaneous_percentage : ℝ)
  (savings_percentage : ℝ)
  (investment_percentage : ℝ)
  (medical_expense : ℝ)
  (tax_percentage : ℝ)
  (total_income : ℝ := total_income base_income bonus_percentage)
  (taxes : ℝ := taxes base_income tax_percentage)
  (income_after_taxes : ℝ := total_income - taxes)
  (fixed_expenses : ℝ := total_fixed_expenses rent utilities food)
  (public_transport_expense : ℝ := public_transport_expense total_income public_transport_percentage)
  (miscellaneous_expense : ℝ := miscellaneous_expense total_income miscellaneous_percentage)
  (variable_expenses : ℝ := variable_expenses public_transport_expense miscellaneous_expense)
  (savings : ℝ := savings total_income savings_percentage)
  (investment : ℝ := investment total_income investment_percentage)
  (total_savings_investments : ℝ := total_savings_investments savings investment)
  (total_expenses_contributions : ℝ := total_expenses_contributions fixed_expenses variable_expenses medical_expense total_savings_investments)
  : amount_left income_after_taxes total_expenses_contributions = 229 := 
sorry

end calculate_amount_left_l188_188214


namespace downstream_distance_l188_188717

theorem downstream_distance (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ) (distance : ℝ) :
  speed_boat = 20 ∧ speed_current = 5 ∧ time_minutes = 24 ∧ distance = 10 →
  (speed_boat + speed_current) * (time_minutes / 60) = distance :=
by
  sorry

end downstream_distance_l188_188717


namespace parabolas_intersect_at_points_l188_188313

theorem parabolas_intersect_at_points :
  ∀ (x y : ℝ), (y = 3 * x^2 - 12 * x - 9) ↔ (y = 2 * x^2 - 8 * x + 5) →
  (x, y) = (2 + 3 * Real.sqrt 2, 66 - 36 * Real.sqrt 2) ∨ (x, y) = (2 - 3 * Real.sqrt 2, 66 + 36 * Real.sqrt 2) :=
by
  sorry

end parabolas_intersect_at_points_l188_188313


namespace order_of_abc_l188_188137

noncomputable def a : ℝ := (0.3)^3
noncomputable def b : ℝ := (3)^3
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

theorem order_of_abc : b > a ∧ a > c :=
by
  have ha : a = (0.3)^3 := rfl
  have hb : b = (3)^3 := rfl
  have hc : c = Real.log 0.3 / Real.log 3 := rfl
  sorry

end order_of_abc_l188_188137


namespace no_valid_n_l188_188052

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n.minFac

theorem no_valid_n (n : ℕ) (h1 : n > 1)
  (h2 : is_prime (greatest_prime_factor n))
  (h3 : greatest_prime_factor n = Nat.sqrt n)
  (h4 : is_prime (greatest_prime_factor (n + 36)))
  (h5 : greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) :
  false :=
sorry

end no_valid_n_l188_188052


namespace solve_quadratic_substitution_l188_188130

theorem solve_quadratic_substitution (x : ℝ) : 
  (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 := 
by sorry

end solve_quadratic_substitution_l188_188130


namespace savings_relationship_l188_188235

def combined_salary : ℝ := 3000
def salary_A : ℝ := 2250
def salary_B : ℝ := combined_salary - salary_A
def savings_A : ℝ := 0.05 * salary_A
def savings_B : ℝ := 0.15 * salary_B

theorem savings_relationship : savings_A = 112.5 ∧ savings_B = 112.5 := by
  have h1 : salary_B = 750 := by sorry
  have h2 : savings_A = 0.05 * 2250 := by sorry
  have h3 : savings_B = 0.15 * 750 := by sorry
  have h4 : savings_A = 112.5 := by sorry
  have h5 : savings_B = 112.5 := by sorry
  exact And.intro h4 h5

end savings_relationship_l188_188235


namespace no_pos_int_squares_l188_188225

open Nat

theorem no_pos_int_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬(∃ k m : ℕ, k ^ 2 = a ^ 2 + b ∧ m ^ 2 = b ^ 2 + a) :=
sorry

end no_pos_int_squares_l188_188225


namespace division_simplification_l188_188842

theorem division_simplification :
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18 / 7 :=
by
  sorry

end division_simplification_l188_188842


namespace proof_equivalent_expression_l188_188202

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2

theorem proof_equivalent_expression (x y : ℝ) :
  (dollar ((x + y) ^ 2) (dollar y x)) - (dollar (dollar x y) (dollar x y)) = 
  4 * (x + y) ^ 2 * ((x + y) ^ 2 - 1) :=
by
  sorry

end proof_equivalent_expression_l188_188202


namespace solve_for_y_l188_188276

theorem solve_for_y (y : ℝ) (hy : y ≠ -2) : 
  (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ↔ y = 7 / 6 :=
by sorry

end solve_for_y_l188_188276


namespace completing_the_square_l188_188466

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end completing_the_square_l188_188466


namespace simplify_sqrt_l188_188426

-- Define the domain and main trigonometric properties
open Real

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  sqrt (1 - 2 * sin x * cos x)

-- Define the main theorem with given conditions
theorem simplify_sqrt {x : ℝ} (h1 : (5 / 4) * π < x) (h2 : x < (3 / 2) * π) (h3 : cos x > sin x) :
  simplify_expression x = cos x - sin x :=
  sorry

end simplify_sqrt_l188_188426


namespace symmetric_about_line_l188_188463

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)
noncomputable def g (x a : ℝ) : ℝ := f (x + a)

theorem symmetric_about_line (a : ℝ) : (∀ x, g x a = x + 1) ↔ a = 0 :=
by sorry

end symmetric_about_line_l188_188463


namespace residents_ticket_price_l188_188449

theorem residents_ticket_price
  (total_attendees : ℕ)
  (resident_count : ℕ)
  (non_resident_price : ℝ)
  (total_revenue : ℝ)
  (R : ℝ)
  (h1 : total_attendees = 586)
  (h2 : resident_count = 219)
  (h3 : non_resident_price = 17.95)
  (h4 : total_revenue = 9423.70)
  (total_residents_pay : ℝ := resident_count * R)
  (total_non_residents_pay : ℝ := (total_attendees - resident_count) * non_resident_price)
  (h5 : total_revenue = total_residents_pay + total_non_residents_pay) :
  R = 12.95 := by
  sorry

end residents_ticket_price_l188_188449


namespace sum_of_percentages_l188_188941

theorem sum_of_percentages :
  let percent1 := 7.35 / 100
  let percent2 := 13.6 / 100
  let percent3 := 21.29 / 100
  let num1 := 12658
  let num2 := 18472
  let num3 := 29345
  let result := percent1 * num1 + percent2 * num2 + percent3 * num3
  result = 9689.9355 :=
by
  sorry

end sum_of_percentages_l188_188941


namespace recipe_flour_requirement_l188_188961

def sugar_cups : ℕ := 9
def salt_cups : ℕ := 40
def flour_initial_cups : ℕ := 4
def additional_flour : ℕ := sugar_cups + 1
def total_flour_cups : ℕ := additional_flour

theorem recipe_flour_requirement : total_flour_cups = 10 := by
  sorry

end recipe_flour_requirement_l188_188961


namespace function_classification_l188_188709

theorem function_classification {f : ℝ → ℝ} 
    (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
    ∀ x : ℝ, f x = 0 ∨ f x = 1 :=
by
  sorry

end function_classification_l188_188709


namespace exists_a_l188_188775

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end exists_a_l188_188775


namespace fraction_ordering_l188_188223

theorem fraction_ordering:
  (6 / 22) < (5 / 17) ∧ (5 / 17) < (8 / 24) :=
by
  sorry

end fraction_ordering_l188_188223


namespace girls_in_class_l188_188392

theorem girls_in_class (B G : ℕ) 
  (h1 : G = B + 3) 
  (h2 : G + B = 41) : 
  G = 22 := 
sorry

end girls_in_class_l188_188392


namespace commentator_mistake_l188_188457

def round_robin_tournament : Prop :=
  ∀ (x y : ℝ),
    x + 2 * x + 13 * y = 105 ∧ x < y ∧ y < 2 * x → False

theorem commentator_mistake : round_robin_tournament :=
  by {
    sorry
  }

end commentator_mistake_l188_188457


namespace soccer_substitutions_mod_2000_l188_188780

theorem soccer_substitutions_mod_2000 :
  let a_0 := 1
  let a_1 := 11 * 11
  let a_2 := 11 * 10 * a_1
  let a_3 := 11 * 9 * a_2
  let a_4 := 11 * 8 * a_3
  let n := a_0 + a_1 + a_2 + a_3 + a_4
  n % 2000 = 942 :=
by
  sorry

end soccer_substitutions_mod_2000_l188_188780


namespace four_digit_num_exists_l188_188612

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem four_digit_num_exists :
  ∃ (n : ℕ), (is_two_digit (n / 100)) ∧ (is_two_digit (n % 100)) ∧
  ((n / 100) + (n % 100))^2 = 100 * (n / 100) + (n % 100) :=
by
  sorry

end four_digit_num_exists_l188_188612


namespace line_slope_intercept_product_l188_188886

theorem line_slope_intercept_product :
  ∃ (m b : ℝ), (b = -1) ∧ ((1 - (m * -1 + b) = 0) ∧ (mb = m * b)) ∧ (mb = 2) :=
by sorry

end line_slope_intercept_product_l188_188886


namespace all_initial_rectangles_are_squares_l188_188258

theorem all_initial_rectangles_are_squares (n : ℕ) (total_squares : ℕ) (h_prime : Nat.Prime total_squares) 
  (cut_rect_into_squares : ℕ → ℕ → ℕ → Prop) :
  ∀ (a b : ℕ), (∀ i, i < n → cut_rect_into_squares a b total_squares) → a = b :=
by 
  sorry

end all_initial_rectangles_are_squares_l188_188258


namespace axis_of_symmetry_l188_188062

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) : 
  ∀ x : ℝ, f x = f (4 - x) := 
  by sorry

end axis_of_symmetry_l188_188062


namespace smallest_N_for_percentages_l188_188429

theorem smallest_N_for_percentages 
  (N : ℕ) 
  (h1 : ∃ N, ∀ f ∈ [1/10, 2/5, 1/5, 3/10], ∃ k : ℕ, N * f = k) :
  N = 10 := 
by
  sorry

end smallest_N_for_percentages_l188_188429


namespace expression_evaluation_l188_188753

theorem expression_evaluation :
  (3 : ℝ) + 3 * Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (3 - Real.sqrt 3)) = 4 + 3 * Real.sqrt 3 :=
sorry

end expression_evaluation_l188_188753


namespace joy_remaining_tape_l188_188376

theorem joy_remaining_tape (total_tape length width : ℕ) (h_total_tape : total_tape = 250) (h_length : length = 60) (h_width : width = 20) :
  total_tape - 2 * (length + width) = 90 :=
by
  sorry

end joy_remaining_tape_l188_188376


namespace ratio_of_w_y_l188_188792

variable (w x y z : ℚ)

theorem ratio_of_w_y (h1 : w / x = 4 / 3)
                     (h2 : y / z = 3 / 2)
                     (h3 : z / x = 1 / 3) :
                     w / y = 8 / 3 := by
  sorry

end ratio_of_w_y_l188_188792


namespace perpendicular_vectors_implies_k_eq_2_l188_188739

variable (k : ℝ)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, k)

theorem perpendicular_vectors_implies_k_eq_2 (h : (2 : ℝ) * (-1 : ℝ) + (1 : ℝ) * k = 0) : k = 2 := by
  sorry

end perpendicular_vectors_implies_k_eq_2_l188_188739


namespace wrapping_paper_area_correct_l188_188524

structure Box :=
  (l : ℝ)  -- length of the box
  (w : ℝ)  -- width of the box
  (h : ℝ)  -- height of the box
  (h_lw : l > w)  -- condition that length is greater than width

def wrapping_paper_area (b : Box) : ℝ :=
  3 * (b.l + b.w) * b.h

theorem wrapping_paper_area_correct (b : Box) : 
  wrapping_paper_area b = 3 * (b.l + b.w) * b.h :=
sorry

end wrapping_paper_area_correct_l188_188524


namespace spring_membership_decrease_l188_188901

theorem spring_membership_decrease (init_members : ℝ) (increase_percent : ℝ) (total_change_percent : ℝ) 
  (fall_members := init_members * (1 + increase_percent / 100)) 
  (spring_members := init_members * (1 + total_change_percent / 100)) :
  increase_percent = 8 → total_change_percent = -12.52 → 
  (fall_members - spring_members) / fall_members * 100 = 19 :=
by
  intros h1 h2
  -- The complicated proof goes here.
  sorry

end spring_membership_decrease_l188_188901


namespace hyperbola_focus_coordinates_l188_188277

theorem hyperbola_focus_coordinates:
  ∀ (x y : ℝ), 
    (x - 5)^2 / 7^2 - (y - 12)^2 / 10^2 = 1 → 
      ∃ (c : ℝ), c = 5 + Real.sqrt 149 ∧ (x, y) = (c, 12) :=
by
  intros x y h
  -- prove the coordinates of the focus with the larger x-coordinate are (5 + sqrt 149, 12)
  sorry

end hyperbola_focus_coordinates_l188_188277


namespace travelers_cross_river_l188_188166

variables (traveler1 traveler2 traveler3 : ℕ)  -- weights of travelers
variable (raft_capacity : ℕ)  -- maximum carrying capacity of the raft

-- Given conditions
def conditions :=
  traveler1 = 3 ∧ traveler2 = 3 ∧ traveler3 = 5 ∧ raft_capacity = 7

-- Prove that the travelers can all cross the river successfully
theorem travelers_cross_river :
  conditions traveler1 traveler2 traveler3 raft_capacity →
  (traveler1 + traveler2 ≤ raft_capacity) ∧
  (traveler1 ≤ raft_capacity) ∧
  (traveler3 ≤ raft_capacity) ∧
  (traveler1 + traveler2 ≤ raft_capacity) →
  true :=
by
  intros h_conditions h_validity
  sorry

end travelers_cross_river_l188_188166


namespace find_x_l188_188939

noncomputable def x : ℝ :=
  0.49

theorem find_x (h : (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt x) = 2.507936507936508) : 
  x = 0.49 :=
sorry

end find_x_l188_188939


namespace afternoon_sales_l188_188918

theorem afternoon_sales (x : ℕ) (H1 : 2 * x + x = 390) : 2 * x = 260 :=
by
  sorry

end afternoon_sales_l188_188918


namespace value_of_c_l188_188293

variables (a b c : ℝ)

theorem value_of_c :
  a + b = 3 ∧
  a * c + b = 18 ∧
  b * c + a = 6 →
  c = 7 :=
by
  intro h
  sorry

end value_of_c_l188_188293


namespace find_common_ratio_l188_188419

variable (a₃ a₂ : ℝ)
variable (S₁ S₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * S₂ = a₃ - 2
def condition2 : Prop := 3 * S₁ = a₂ - 2

-- Theorem statement
theorem find_common_ratio (h1 : condition1 a₃ S₂)
                          (h2 : condition2 a₂ S₁) : 
                          (a₃ / a₂ = 4) :=
by 
  sorry

end find_common_ratio_l188_188419


namespace functional_eq_solution_l188_188273

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solution_l188_188273


namespace sum_of_first_39_natural_numbers_l188_188975

theorem sum_of_first_39_natural_numbers : (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_of_first_39_natural_numbers_l188_188975


namespace remainder_3012_div_96_l188_188557

theorem remainder_3012_div_96 : 3012 % 96 = 36 :=
by 
  sorry

end remainder_3012_div_96_l188_188557


namespace simplify_and_evaluate_expr_evaluate_at_zero_l188_188969

theorem simplify_and_evaluate_expr (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1)) = (2 + x) / (2 - x) :=
by
  sorry

theorem evaluate_at_zero :
  (2 + 0 : ℝ) / (2 - 0) = 1 :=
by
  norm_num

end simplify_and_evaluate_expr_evaluate_at_zero_l188_188969


namespace initial_percentage_of_water_l188_188211

variable (V : ℝ) (W : ℝ) (P : ℝ)

theorem initial_percentage_of_water 
  (h1 : V = 120) 
  (h2 : W = 8)
  (h3 : (V + W) * 0.25 = ((P / 100) * V) + W) : 
  P = 20 :=
by
  sorry

end initial_percentage_of_water_l188_188211


namespace number_of_correct_statements_l188_188955

-- Definitions of the conditions from the problem
def seq_is_graphical_points := true  -- Statement 1
def seq_is_finite (s : ℕ → ℝ) := ∀ n, s n = 0 -- Statement 2
def seq_decreasing_implies_finite (s : ℕ → ℝ) := (∀ n, s (n + 1) ≤ s n) → seq_is_finite s -- Statement 3

-- Prove the number of correct statements is 1
theorem number_of_correct_statements : (seq_is_graphical_points = true ∧ ¬(∃ s: ℕ → ℝ, ¬seq_is_finite s) ∧ ∃ s : ℕ → ℝ, ¬seq_decreasing_implies_finite s) → 1 = 1 :=
by
  sorry

end number_of_correct_statements_l188_188955


namespace heads_count_l188_188894

theorem heads_count (H T : ℕ) (h1 : H + T = 128) (h2 : H = T + 12) : H = 70 := by
  sorry

end heads_count_l188_188894


namespace arithmetic_seq_sum_l188_188185

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 3 + a 4 + a 5 + a 6 + a 7 = 250) : a 2 + a 8 = 100 :=
sorry

end arithmetic_seq_sum_l188_188185


namespace distance_between_foci_l188_188859

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end distance_between_foci_l188_188859


namespace set_star_result_l188_188101

-- Define the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Define the operation ∗ between sets A and B
def set_star (A B : Set ℕ) : Set ℕ := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}

-- Rewrite the main theorem to be proven
theorem set_star_result : set_star A B = {2, 3, 4, 5} :=
  sorry

end set_star_result_l188_188101


namespace width_of_jordan_rectangle_l188_188968

def carol_length := 5
def carol_width := 24
def jordan_length := 2
def jordan_area := carol_length * carol_width

theorem width_of_jordan_rectangle : ∃ (w : ℝ), jordan_length * w = jordan_area ∧ w = 60 :=
by
  use 60
  simp [carol_length, carol_width, jordan_length, jordan_area]
  sorry

end width_of_jordan_rectangle_l188_188968


namespace value_of_expression_l188_188144

theorem value_of_expression (x y : ℝ) (h₁ : x * y = -3) (h₂ : x + y = -4) :
  x^2 + 3 * x * y + y^2 = 13 :=
by
  sorry

end value_of_expression_l188_188144


namespace derivative_of_sin_squared_minus_cos_squared_l188_188067

noncomputable def func (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem derivative_of_sin_squared_minus_cos_squared (x : ℝ) :
  deriv func x = 2 * Real.sin (2 * x) :=
sorry

end derivative_of_sin_squared_minus_cos_squared_l188_188067


namespace solve_fractional_equation_l188_188624

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end solve_fractional_equation_l188_188624


namespace little_john_spent_on_sweets_l188_188552

theorem little_john_spent_on_sweets:
  let initial_amount := 10.10
  let amount_given_to_each_friend := 2.20
  let amount_left := 2.45
  let total_given_to_friends := 2 * amount_given_to_each_friend
  let amount_before_sweets := initial_amount - total_given_to_friends
  let amount_spent_on_sweets := amount_before_sweets - amount_left
  amount_spent_on_sweets = 3.25 :=
by
  sorry

end little_john_spent_on_sweets_l188_188552


namespace oak_taller_than_shortest_l188_188388

noncomputable def pine_tree_height : ℚ := 14 + 1 / 2
noncomputable def elm_tree_height : ℚ := 13 + 1 / 3
noncomputable def oak_tree_height : ℚ := 19 + 1 / 2

theorem oak_taller_than_shortest : 
  oak_tree_height - elm_tree_height = 6 + 1 / 6 := 
  sorry

end oak_taller_than_shortest_l188_188388


namespace positive_difference_between_loans_l188_188102

noncomputable def loan_amount : ℝ := 12000

noncomputable def option1_interest_rate : ℝ := 0.08
noncomputable def option1_years_1 : ℕ := 3
noncomputable def option1_years_2 : ℕ := 9

noncomputable def option2_interest_rate : ℝ := 0.09
noncomputable def option2_years : ℕ := 12

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate)^years

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal + principal * rate * years

noncomputable def payment_at_year_3 : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 / 3

noncomputable def remaining_balance_after_3_years : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 - payment_at_year_3

noncomputable def total_payment_option1 : ℝ :=
  payment_at_year_3 + compound_interest remaining_balance_after_3_years option1_interest_rate option1_years_2

noncomputable def total_payment_option2 : ℝ :=
  simple_interest loan_amount option2_interest_rate option2_years

noncomputable def positive_difference : ℝ :=
  abs (total_payment_option1 - total_payment_option2)

theorem positive_difference_between_loans : positive_difference = 1731 := by
  sorry

end positive_difference_between_loans_l188_188102


namespace find_x_l188_188478

/-
If two minus the reciprocal of (3 - x) equals the reciprocal of (2 + x), 
then x equals (1 + sqrt(15)) / 2 or (1 - sqrt(15)) / 2.
-/
theorem find_x (x : ℝ) :
  (2 - (1 / (3 - x)) = (1 / (2 + x))) → 
  (x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2) :=
by 
  sorry

end find_x_l188_188478


namespace total_cube_volume_l188_188181

theorem total_cube_volume 
  (carl_cubes : ℕ)
  (carl_cube_side : ℕ)
  (kate_cubes : ℕ)
  (kate_cube_side : ℕ)
  (hcarl : carl_cubes = 4)
  (hcarl_side : carl_cube_side = 3)
  (hkate : kate_cubes = 6)
  (hkate_side : kate_cube_side = 4) :
  (carl_cubes * carl_cube_side ^ 3) + (kate_cubes * kate_cube_side ^ 3) = 492 :=
by
  sorry

end total_cube_volume_l188_188181


namespace arithmetic_sequence_sum_zero_l188_188375

theorem arithmetic_sequence_sum_zero {a1 d n : ℤ} 
(h1 : a1 = 35) 
(h2 : d = -2) 
(h3 : (n * (2 * a1 + (n - 1) * d)) / 2 = 0) : 
n = 36 :=
by sorry

end arithmetic_sequence_sum_zero_l188_188375


namespace jack_keeps_deers_weight_is_correct_l188_188924

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end jack_keeps_deers_weight_is_correct_l188_188924


namespace probability_standard_weight_l188_188129

noncomputable def total_students : ℕ := 500
noncomputable def standard_students : ℕ := 350

theorem probability_standard_weight : (standard_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by {
  sorry
}

end probability_standard_weight_l188_188129


namespace bananas_oranges_equivalence_l188_188937

theorem bananas_oranges_equivalence :
  (3 / 4) * 12 * banana_value = 9 * orange_value →
  (2 / 3) * 6 * banana_value = 4 * orange_value :=
by
  intros h
  sorry

end bananas_oranges_equivalence_l188_188937


namespace sum_first_five_terms_l188_188825

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

theorem sum_first_five_terms (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 6) : S_5 a = 15 :=
by
  -- skipping actual proof
  sorry

end sum_first_five_terms_l188_188825


namespace primes_between_30_and_60_l188_188403

theorem primes_between_30_and_60 (list_of_primes : List ℕ) 
  (H1 : list_of_primes = [31, 37, 41, 43, 47, 53, 59]) :
  (list_of_primes.headI * list_of_primes.reverse.headI) = 1829 := by
  sorry

end primes_between_30_and_60_l188_188403


namespace angle_between_clock_hands_at_7_oclock_l188_188730

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l188_188730


namespace odd_cube_difference_divisible_by_power_of_two_l188_188837

theorem odd_cube_difference_divisible_by_power_of_two {a b n : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) :
  (2^n ∣ (a^3 - b^3)) ↔ (2^n ∣ (a - b)) :=
by
  sorry

end odd_cube_difference_divisible_by_power_of_two_l188_188837


namespace probability_red_chips_drawn_first_l188_188620

def probability_all_red_drawn (total_chips : Nat) (red_chips : Nat) (green_chips : Nat) : ℚ :=
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose (total_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem probability_red_chips_drawn_first :
  probability_all_red_drawn 9 5 4 = 4 / 9 :=
by
  sorry

end probability_red_chips_drawn_first_l188_188620


namespace longest_side_of_triangle_l188_188767

theorem longest_side_of_triangle :
  ∃ y : ℚ, 6 + (y + 3) + (3 * y - 2) = 40 ∧ max (6 : ℚ) (max (y + 3) (3 * y - 2)) = 91 / 4 :=
by
  sorry

end longest_side_of_triangle_l188_188767


namespace kramer_vote_percentage_l188_188399

def percentage_of_votes_cast (K : ℕ) (V : ℕ) : ℕ :=
  (K * 100) / V

theorem kramer_vote_percentage (K : ℕ) (V : ℕ) (h1 : K = 942568) 
  (h2 : V = 4 * K) : percentage_of_votes_cast K V = 25 := 
by 
  rw [h1, h2, percentage_of_votes_cast]
  sorry

end kramer_vote_percentage_l188_188399


namespace smallest_common_multiple_five_digit_l188_188179

def is_multiple (a b : ℕ) : Prop := ∃ k, a = k * b

def smallest_five_digit_multiple_of_3_and_5 (x : ℕ) : Prop :=
  is_multiple x 3 ∧ is_multiple x 5 ∧ 10000 ≤ x ∧ x ≤ 99999 ∧ (∀ y, (10000 ≤ y ∧ y ≤ 99999 ∧ is_multiple y 3 ∧ is_multiple y 5) → x ≤ y)

theorem smallest_common_multiple_five_digit : smallest_five_digit_multiple_of_3_and_5 10005 :=
sorry

end smallest_common_multiple_five_digit_l188_188179


namespace stuffed_animal_cost_l188_188996

variable (S : ℝ)  -- Cost of the stuffed animal
variable (total_cost_after_discount_gave_30_dollars : S * 0.10 = 3.6) 
-- Condition: cost of stuffed animal = $4.44
theorem stuffed_animal_cost :
  S = 4.44 :=
by
  sorry

end stuffed_animal_cost_l188_188996


namespace smallest_prime_sum_l188_188541

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_distinct_primes (n k : ℕ) (s : List ℕ) : Prop :=
  s.length = k ∧ (∀ x ∈ s, is_prime x) ∧ (∀ (x y : ℕ), x ≠ y → x ∈ s → y ∈ s → x ≠ y) ∧ s.sum = n

theorem smallest_prime_sum :
  (is_prime 61) ∧ 
  (∃ s2, is_sum_of_distinct_primes 61 2 s2) ∧ 
  (∃ s3, is_sum_of_distinct_primes 61 3 s3) ∧ 
  (∃ s4, is_sum_of_distinct_primes 61 4 s4) ∧ 
  (∃ s5, is_sum_of_distinct_primes 61 5 s5) ∧ 
  (∃ s6, is_sum_of_distinct_primes 61 6 s6) :=
by
  sorry

end smallest_prime_sum_l188_188541


namespace initial_bags_count_l188_188913

theorem initial_bags_count
  (points_per_bag : ℕ)
  (non_recycled_bags : ℕ)
  (total_possible_points : ℕ)
  (points_earned : ℕ)
  (B : ℕ)
  (h1 : points_per_bag = 5)
  (h2 : non_recycled_bags = 2)
  (h3 : total_possible_points = 45)
  (h4 : points_earned = 5 * (B - non_recycled_bags))
  : B = 11 :=
by {
  sorry
}

end initial_bags_count_l188_188913


namespace pipes_fill_cistern_together_time_l188_188690

theorem pipes_fill_cistern_together_time
  (t : ℝ)
  (h1 : t * (1 / 12 + 1 / 15) + 6 * (1 / 15) = 1) : 
  t = 4 := 
by
  -- Proof is omitted here as instructed
  sorry

end pipes_fill_cistern_together_time_l188_188690


namespace particular_solution_satisfies_l188_188121

noncomputable def particular_solution (x : ℝ) : ℝ :=
  (1/3) * Real.exp (-4 * x) - (1/3) * Real.exp (2 * x) + (x ^ 2 + 3 * x) * Real.exp (2 * x)

def initial_conditions (f df : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ df 0 = 1

def differential_equation (f df ddf : ℝ → ℝ) : Prop :=
  ∀ x, ddf x + 2 * df x - 8 * f x = (12 * x + 20) * Real.exp (2 * x)

theorem particular_solution_satisfies :
  ∃ C1 C2 : ℝ, initial_conditions (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
              (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) ∧ 
              differential_equation (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
                                  (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) 
                                  (λ x => 16 * C1 * Real.exp (-4 * x) + 4 * C2 * Real.exp (2 * x) + (4 * x^2 + 12 * x + 1) * Real.exp (2 * x)) :=
sorry

end particular_solution_satisfies_l188_188121


namespace geometric_sequence_sum_l188_188957

theorem geometric_sequence_sum (a_1 q n S : ℕ) (h1 : a_1 = 2) (h2 : q = 2) (h3 : S = 126) 
    (h4 : S = (a_1 * (1 - q^n)) / (1 - q)) : 
    n = 6 :=
by
  sorry

end geometric_sequence_sum_l188_188957


namespace symmetric_points_on_ellipse_are_m_in_range_l188_188629

open Real

theorem symmetric_points_on_ellipse_are_m_in_range (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1 ∧ 
                   (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1 ∧ 
                   ∃ x0 y0 : ℝ, y0 = 4 * x0 + m ∧ x0 = (A.1 + B.1) / 2 ∧ y0 = (A.2 + B.2) / 2) 
  ↔ -2 * sqrt 13 / 13 < m ∧ m < 2 * sqrt 13 / 13 := 
 sorry

end symmetric_points_on_ellipse_are_m_in_range_l188_188629


namespace jimmy_paid_total_l188_188572

-- Data for the problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100
def park_pizzas : ℕ := 3
def building_distance : ℕ := 2000
def building_pizzas : ℕ := 2
def house_distance : ℕ := 800
def house_pizzas : ℕ := 4
def community_center_distance : ℕ := 1500
def community_center_pizzas : ℕ := 5
def office_distance : ℕ := 300
def office_pizzas : ℕ := 1
def bus_stop_distance : ℕ := 1200
def bus_stop_pizzas : ℕ := 3

def cost (distance pizzas : ℕ) : ℕ := 
  let base_cost := pizzas * pizza_cost
  if distance > 1000 then base_cost + delivery_charge else base_cost

def total_cost : ℕ :=
  cost park_distance park_pizzas +
  cost building_distance building_pizzas +
  cost house_distance house_pizzas +
  cost community_center_distance community_center_pizzas +
  cost office_distance office_pizzas +
  cost bus_stop_distance bus_stop_pizzas

theorem jimmy_paid_total : total_cost = 222 :=
  by
    -- Proof omitted
    sorry

end jimmy_paid_total_l188_188572


namespace thabo_hardcover_books_l188_188233

theorem thabo_hardcover_books:
  ∃ (H P F : ℕ), H + P + F = 280 ∧ P = H + 20 ∧ F = 2 * P ∧ H = 55 := by
  sorry

end thabo_hardcover_books_l188_188233


namespace regular_polygon_sides_l188_188919

theorem regular_polygon_sides (n : ℕ) (h₁ : n > 2) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → True) (h₃ : (360 / n : ℝ) = 30) : n = 12 := by
  sorry

end regular_polygon_sides_l188_188919


namespace number_of_valid_pairs_l188_188412

theorem number_of_valid_pairs :
  (∃! S : ℕ, S = 1250 ∧ ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 1000) →
  (3^n < 4^m ∧ 4^m < 4^(m+1) ∧ 4^(m+1) < 3^(n+1))) :=
sorry

end number_of_valid_pairs_l188_188412


namespace min_value_expression_l188_188252

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    let a := 2
    let b := 3
    let term1 := 2*x + 1/(3*y)
    let term2 := 3*y + 1/(2*x)
    (term1 * (term1 - 2023) + term2 * (term2 - 2023)) = -2050529.5 :=
sorry

end min_value_expression_l188_188252


namespace total_students_l188_188831

theorem total_students (f1 f2 f3 total : ℕ)
  (h_ratio : f1 * 2 = f2)
  (h_ratio2 : f1 * 3 = f3)
  (h_f1 : f1 = 6)
  (h_total : total = f1 + f2 + f3) :
  total = 48 :=
by
  sorry

end total_students_l188_188831


namespace algorithm_find_GCD_Song_Yuan_l188_188363

theorem algorithm_find_GCD_Song_Yuan :
  (∀ method, method = "continuous subtraction" → method_finds_GCD_Song_Yuan) :=
sorry

end algorithm_find_GCD_Song_Yuan_l188_188363


namespace mechanic_charge_per_hour_l188_188354

/-- Definitions based on provided conditions -/
def total_amount_paid : ℝ := 300
def part_cost : ℝ := 150
def hours : ℕ := 2

/-- Theorem stating the labor cost per hour is $75 -/
theorem mechanic_charge_per_hour (total_amount_paid part_cost hours : ℝ) : hours = 2 → part_cost = 150 → total_amount_paid = 300 → 
  (total_amount_paid - part_cost) / hours = 75 :=
by
  sorry

end mechanic_charge_per_hour_l188_188354


namespace train_speed_in_kmh_l188_188009

theorem train_speed_in_kmh (length_of_train : ℕ) (time_to_cross : ℕ) (speed_in_m_per_s : ℕ) (speed_in_km_per_h : ℕ) :
  length_of_train = 300 →
  time_to_cross = 12 →
  speed_in_m_per_s = length_of_train / time_to_cross →
  speed_in_km_per_h = speed_in_m_per_s * 3600 / 1000 →
  speed_in_km_per_h = 90 :=
by
  sorry

end train_speed_in_kmh_l188_188009


namespace student_arrangement_count_l188_188098

theorem student_arrangement_count :
  let males := 4
  let females := 5
  let select_males := 2
  let select_females := 3
  let total_selected := select_males + select_females
  (Nat.choose males select_males) * (Nat.choose females select_females) * (Nat.factorial total_selected) = 7200 := 
by
  sorry

end student_arrangement_count_l188_188098


namespace find_a_l188_188818

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l188_188818


namespace singer_arrangements_l188_188510

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end singer_arrangements_l188_188510


namespace pandas_and_bamboo_l188_188877

-- Definitions for the conditions
def number_of_pandas (x : ℕ) :=
  (∃ y : ℕ, y = 5 * x + 11 ∧ y = 2 * (3 * x - 5) - 8)

-- Theorem stating the solution
theorem pandas_and_bamboo (x y : ℕ) (h1 : y = 5 * x + 11) (h2 : y = 2 * (3 * x - 5) - 8) : x = 29 ∧ y = 156 :=
by {
  sorry
}

end pandas_and_bamboo_l188_188877


namespace tomato_plant_relationship_l188_188409

theorem tomato_plant_relationship :
  ∃ (T1 T2 T3 : ℕ), T1 = 24 ∧ T3 = T2 + 2 ∧ T1 + T2 + T3 = 60 ∧ T1 - T2 = 7 :=
by
  sorry

end tomato_plant_relationship_l188_188409


namespace eddies_sister_pies_per_day_l188_188608

theorem eddies_sister_pies_per_day 
  (Eddie_daily : ℕ := 3) 
  (Mother_daily : ℕ := 8) 
  (total_days : ℕ := 7)
  (total_pies : ℕ := 119) :
  ∃ (S : ℕ), S = 6 ∧ (Eddie_daily * total_days + Mother_daily * total_days + S * total_days = total_pies) :=
by
  sorry

end eddies_sister_pies_per_day_l188_188608


namespace A_equals_4_of_rounded_to_tens_9430_l188_188904

variable (A B : ℕ)

theorem A_equals_4_of_rounded_to_tens_9430
  (h1 : 9430 = 9000 + 100 * A + 10 * 3 + B)
  (h2 : B < 5)
  (h3 : 0 ≤ A ∧ A ≤ 9)
  (h4 : 0 ≤ B ∧ B ≤ 9) :
  A = 4 :=
by
  sorry

end A_equals_4_of_rounded_to_tens_9430_l188_188904


namespace range_of_m_l188_188405

open Real

noncomputable def complex_modulus_log_condition (m : ℝ) : Prop :=
  Complex.abs (Complex.log (m : ℂ) / Complex.log 2 + Complex.I * 4) ≤ 5

theorem range_of_m (m : ℝ) (h : complex_modulus_log_condition m) : 
  (1 / 8 : ℝ) ≤ m ∧ m ≤ (8 : ℝ) :=
sorry

end range_of_m_l188_188405


namespace find_DP_l188_188713

theorem find_DP (AP BP CP DP : ℚ) (h1 : AP = 4) (h2 : BP = 6) (h3 : CP = 9) (h4 : AP * BP = CP * DP) :
  DP = 8 / 3 :=
by
  rw [h1, h2, h3] at h4
  sorry

end find_DP_l188_188713


namespace percent_profit_is_25_percent_l188_188507

theorem percent_profit_is_25_percent
  (CP SP : ℝ)
  (h : 75 * (CP - 0.05 * CP) = 60 * SP) :
  let profit := SP - (0.95 * CP)
  let percent_profit := (profit / (0.95 * CP)) * 100
  percent_profit = 25 :=
by
  sorry

end percent_profit_is_25_percent_l188_188507


namespace demokhar_lifespan_l188_188647

-- Definitions based on the conditions
def boy_fraction := 1 / 4
def young_man_fraction := 1 / 5
def adult_man_fraction := 1 / 3
def old_man_years := 13

-- Statement without proof
theorem demokhar_lifespan :
  ∀ (x : ℕ), (boy_fraction * x) + (young_man_fraction * x) + (adult_man_fraction * x) + old_man_years = x → x = 60 :=
by
  sorry

end demokhar_lifespan_l188_188647


namespace cone_volume_divided_by_pi_l188_188604

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def sector_to_cone_radius (arc_len : ℝ) : ℝ := arc_len / (2 * Real.pi)

noncomputable def cone_height (r_base : ℝ) (slant_height : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - r_base ^ 2)

noncomputable def cone_volume (r_base : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r_base ^ 2 * height

theorem cone_volume_divided_by_pi (r slant_height θ : ℝ) (h : slant_height = 15 ∧ θ = 270):
  cone_volume (sector_to_cone_radius (arc_length r θ)) (cone_height (sector_to_cone_radius (arc_length r θ)) slant_height) / Real.pi = (453.515625 * Real.sqrt 10.9375) :=
by
  sorry

end cone_volume_divided_by_pi_l188_188604


namespace line_bisects_circle_area_l188_188337

theorem line_bisects_circle_area (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + b ↔ x^2 + y^2 - 2 * x - 4 * y + 4 = 0) → b = 0 :=
by
  sorry

end line_bisects_circle_area_l188_188337


namespace geometric_series_sum_l188_188435

theorem geometric_series_sum :
  let a := 6
  let r := - (2 / 5)
  s = ∑' n, (a * r ^ n) ↔ s = 30 / 7 :=
sorry

end geometric_series_sum_l188_188435


namespace num_of_B_sets_l188_188294

def A : Set ℕ := {1, 2}

theorem num_of_B_sets (S : Set ℕ) (A : Set ℕ) (h : A = {1, 2}) (h1 : ∀ B : Set ℕ, A ∪ B = S) : 
  ∃ n : ℕ, n = 4 ∧ (∀ B : Set ℕ, B ⊆ {1, 2} → S = {1, 2}) :=
by {
  sorry
}

end num_of_B_sets_l188_188294


namespace ratio_of_boys_to_girls_l188_188589

theorem ratio_of_boys_to_girls {T G B : ℕ} (h1 : (2/3 : ℚ) * G = (1/4 : ℚ) * T) (h2 : T = G + B) : (B : ℚ) / G = 5 / 3 :=
by
  sorry

end ratio_of_boys_to_girls_l188_188589


namespace zookeeper_fish_total_l188_188550

def fish_given : ℕ := 19
def fish_needed : ℕ := 17

theorem zookeeper_fish_total : fish_given + fish_needed = 36 :=
by
  sorry

end zookeeper_fish_total_l188_188550


namespace seq_equality_iff_initial_equality_l188_188490

variable {α : Type*} [AddGroup α]

-- Definition of sequences and their differences
def sequence_diff (u : ℕ → α) (v : ℕ → α) : Prop := ∀ n, (u (n+1) - u n) = (v (n+1) - v n)

-- Main theorem statement
theorem seq_equality_iff_initial_equality (u v : ℕ → α) :
  sequence_diff u v → (∀ n, u n = v n) ↔ (u 1 = v 1) :=
by
  sorry

end seq_equality_iff_initial_equality_l188_188490


namespace find_z_l188_188562

-- Definitions from the problem statement
variables {x y z : ℤ}
axiom consecutive (h1: x = z + 2) (h2: y = z + 1) : true
axiom ordered (h3: x > y) (h4: y > z) : true
axiom equation (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : true

-- The proof goal
theorem find_z (h1: x = z + 2) (h2: y = z + 1) (h3: x > y) (h4: y > z) (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : z = 2 :=
by 
  sorry

end find_z_l188_188562


namespace find_number_l188_188545

theorem find_number (N : ℝ) (h : 0.15 * 0.30 * 0.50 * N = 108) : N = 4800 :=
by
  sorry

end find_number_l188_188545


namespace find_q_l188_188659

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end find_q_l188_188659


namespace compute_f_1986_l188_188160

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_nonneg_integers : ∀ x : ℕ, ∃ y : ℤ, f x = y
axiom f_one : f 1 = 1
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b)

theorem compute_f_1986 : f 1986 = 0 :=
  sorry

end compute_f_1986_l188_188160


namespace inequality_solution_l188_188686

theorem inequality_solution (x : ℝ) (h : x ≠ 1) : (x + 1) * (x + 3) / (x - 1)^2 ≤ 0 ↔ (-3 ≤ x ∧ x ≤ -1) :=
by
  sorry

end inequality_solution_l188_188686


namespace triangle_area_l188_188250

theorem triangle_area (P : ℝ) (r : ℝ) (s : ℝ) (A : ℝ) :
  P = 42 → r = 5 → s = P / 2 → A = r * s → A = 105 :=
by
  intro hP hr hs hA
  sorry

end triangle_area_l188_188250


namespace weekly_tax_percentage_is_zero_l188_188747

variables (daily_expense : ℕ) (daily_revenue_fries : ℕ) (daily_revenue_poutine : ℕ) (weekly_net_income : ℕ)

def weekly_expense := daily_expense * 7
def weekly_revenue := daily_revenue_fries * 7 + daily_revenue_poutine * 7
def weekly_total_income := weekly_net_income + weekly_expense
def weekly_tax := weekly_total_income - weekly_revenue

theorem weekly_tax_percentage_is_zero
  (h1 : daily_expense = 10)
  (h2 : daily_revenue_fries = 12)
  (h3 : daily_revenue_poutine = 8)
  (h4 : weekly_net_income = 56) :
  weekly_tax = 0 :=
by sorry

end weekly_tax_percentage_is_zero_l188_188747


namespace find_number_to_be_multiplied_l188_188927

def correct_multiplier := 43
def incorrect_multiplier := 34
def difference := 1224

theorem find_number_to_be_multiplied (x : ℕ) : correct_multiplier * x - incorrect_multiplier * x = difference → x = 136 :=
by
  sorry

end find_number_to_be_multiplied_l188_188927


namespace first_group_person_count_l188_188531

theorem first_group_person_count
  (P : ℕ)
  (h1 : P * 24 * 5 = 30 * 26 * 6) : 
  P = 39 :=
by
  sorry

end first_group_person_count_l188_188531


namespace compute_expression_l188_188210

noncomputable def roots_exist (P : Polynomial ℝ) (α β γ : ℝ) : Prop :=
  P = Polynomial.C (-13) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-7) + Polynomial.X))

theorem compute_expression (α β γ : ℝ) (h : roots_exist (Polynomial.X^3 - 7 * Polynomial.X^2 + 11 * Polynomial.X - 13) α β γ) :
  (α ≠ 0) → (β ≠ 0) → (γ ≠ 0) → (α^2 * β^2 + β^2 * γ^2 + γ^2 * α^2 = -61) :=
  sorry

end compute_expression_l188_188210


namespace relationship_among_a_b_c_l188_188310

noncomputable def a : ℝ := 2^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l188_188310


namespace range_of_m_l188_188985

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + |x - 1| ≥ (m + 2) * x - 1) ↔ (-3 - 2 * Real.sqrt 2) ≤ m ∧ m ≤ 0 := 
sorry

end range_of_m_l188_188985


namespace irwin_basketball_l188_188481

theorem irwin_basketball (A B C D : ℕ) (h1 : C = 2) (h2 : 2^A * 5^B * 11^C * 13^D = 2420) : A = 2 :=
by
  sorry

end irwin_basketball_l188_188481


namespace product_abc_l188_188972

theorem product_abc 
  (a b c : ℝ)
  (h1 : a + b + c = 1) 
  (h2 : 3 * (4 * a + 2 * b + c) = 15) 
  (h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a * b * c = -4 :=
by
  sorry

end product_abc_l188_188972


namespace net_increase_proof_l188_188072

def initial_cars := 50
def initial_motorcycles := 75
def initial_vans := 25

def car_arrival_rate : ℝ := 70
def car_departure_rate : ℝ := 40
def motorcycle_arrival_rate : ℝ := 120
def motorcycle_departure_rate : ℝ := 60
def van_arrival_rate : ℝ := 30
def van_departure_rate : ℝ := 20

def play_duration : ℝ := 2.5

def net_increase_car : ℝ := play_duration * (car_arrival_rate - car_departure_rate)
def net_increase_motorcycle : ℝ := play_duration * (motorcycle_arrival_rate - motorcycle_departure_rate)
def net_increase_van : ℝ := play_duration * (van_arrival_rate - van_departure_rate)

theorem net_increase_proof :
  net_increase_car = 75 ∧
  net_increase_motorcycle = 150 ∧
  net_increase_van = 25 :=
by
  -- Proof would go here.
  sorry

end net_increase_proof_l188_188072


namespace range_g_l188_188750

variable (x : ℝ)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g (y : ℝ) : 
  (∃ x, g x = y) ↔ y > 0 :=
by
  sorry

end range_g_l188_188750


namespace fg_eval_l188_188948

def f (x : ℤ) : ℤ := x^3
def g (x : ℤ) : ℤ := 4 * x + 5

theorem fg_eval : f (g (-2)) = -27 := by
  sorry

end fg_eval_l188_188948


namespace point_location_l188_188187

variables {A B C m n : ℝ}

theorem point_location (h1 : A > 0) (h2 : B < 0) (h3 : A * m + B * n + C < 0) : 
  -- Statement: the point P(m, n) is on the upper right side of the line Ax + By + C = 0
  true :=
sorry

end point_location_l188_188187


namespace distance_between_A_and_B_l188_188484

def scale : ℕ := 20000
def map_distance : ℕ := 6
def actual_distance_cm : ℕ := scale * map_distance
def actual_distance_m : ℕ := actual_distance_cm / 100

theorem distance_between_A_and_B : actual_distance_m = 1200 := by
  sorry

end distance_between_A_and_B_l188_188484


namespace order_of_logs_l188_188172

open Real

noncomputable def a := log 10 / log 5
noncomputable def b := log 12 / log 6
noncomputable def c := 1 + log 2 / log 7

theorem order_of_logs : a > b ∧ b > c :=
by
  sorry

end order_of_logs_l188_188172


namespace fisherman_gets_14_tunas_every_day_l188_188693

-- Define the conditions
def red_snappers_per_day := 8
def cost_per_red_snapper := 3
def cost_per_tuna := 2
def total_earnings_per_day := 52

-- Define the hypothesis
def total_earnings_from_red_snappers := red_snappers_per_day * cost_per_red_snapper  -- $24
def total_earnings_from_tunas := total_earnings_per_day - total_earnings_from_red_snappers -- $28
def number_of_tunas := total_earnings_from_tunas / cost_per_tuna -- 14

-- Lean statement to verify
theorem fisherman_gets_14_tunas_every_day : number_of_tunas = 14 :=
by 
  sorry

end fisherman_gets_14_tunas_every_day_l188_188693


namespace river_current_speed_l188_188783

/-- A man rows 18 miles upstream in three hours more time than it takes him to row 
the same distance downstream. If he halves his usual rowing rate, the time upstream 
becomes only two hours more than the time downstream. Prove that the speed of 
the river's current is 2 miles per hour. -/
theorem river_current_speed (r w : ℝ) 
    (h1 : 18 / (r - w) - 18 / (r + w) = 3)
    (h2 : 18 / (r / 2 - w) - 18 / (r / 2 + w) = 2) : 
    w = 2 := 
sorry

end river_current_speed_l188_188783


namespace general_formula_for_sequence_a_l188_188226

noncomputable def S (n : ℕ) : ℕ := 3^n + 1

def a (n : ℕ) : ℕ :=
if n = 1 then 4 else 2 * 3^(n-1)

theorem general_formula_for_sequence_a (n : ℕ) :
  a n = if n = 1 then 4 else 2 * 3^(n-1) :=
by {
  sorry
}

end general_formula_for_sequence_a_l188_188226


namespace min_sets_bound_l188_188340

theorem min_sets_bound (A : Type) (n k : ℕ) (S : Finset (Finset A))
  (h₁ : S.card = k)
  (h₂ : ∀ x y : A, x ≠ y → ∃ B ∈ S, (x ∈ B ∧ y ∉ B) ∨ (y ∈ B ∧ x ∉ B)) :
  2^k ≥ n :=
sorry

end min_sets_bound_l188_188340


namespace interval_between_segments_systematic_sampling_l188_188290

theorem interval_between_segments_systematic_sampling 
  (total_students : ℕ) (sample_size : ℕ) 
  (h_total_students : total_students = 1000) 
  (h_sample_size : sample_size = 40):
  total_students / sample_size = 25 :=
by
  sorry

end interval_between_segments_systematic_sampling_l188_188290


namespace count_edge_cubes_l188_188558

/-- 
A cube is painted red on all faces and then cut into 27 equal smaller cubes.
Prove that the number of smaller cubes that are painted on only 2 faces is 12. 
-/
theorem count_edge_cubes (c : ℕ) (inner : ℕ)  (edge : ℕ) (face : ℕ) :
  (c = 27 ∧ inner = 1 ∧ edge = 12 ∧ face = 6) → edge = 12 :=
by
  -- Given the conditions from the problem statement
  sorry

end count_edge_cubes_l188_188558


namespace dogs_neither_long_furred_nor_brown_l188_188491

theorem dogs_neither_long_furred_nor_brown :
  (∀ (total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown : ℕ),
     total_dogs = 45 →
     long_furred_dogs = 26 →
     brown_dogs = 22 →
     both_long_furred_and_brown = 11 →
     neither_long_furred_nor_brown = total_dogs - (long_furred_dogs + brown_dogs - both_long_furred_and_brown) → 
     neither_long_furred_nor_brown = 8) :=
by
  intros total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown
  sorry

end dogs_neither_long_furred_nor_brown_l188_188491


namespace smaller_rectangle_length_ratio_l188_188149

theorem smaller_rectangle_length_ratio 
  (s : ℝ)
  (h1 : 5 = 5)
  (h2 : ∃ r : ℝ, r = s)
  (h3 : ∀ x : ℝ, x = s)
  (h4 : ∀ y : ℝ, y / 2 = s / 2)
  (h5 : ∀ z : ℝ, z = 3 * s)
  (h6 : ∀ w : ℝ, w = s) :
  ∃ l : ℝ, l / s = 4 :=
sorry

end smaller_rectangle_length_ratio_l188_188149


namespace geometric_series_sum_l188_188997

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1/3) :
  (∑' n : ℕ, a * r^n) = 3/2 :=
by
  -- proof goes here
  sorry

end geometric_series_sum_l188_188997


namespace quad_side_difference_l188_188394

theorem quad_side_difference (a b c d s x y : ℝ)
  (h1 : a = 80) (h2 : b = 100) (h3 : c = 150) (h4 : d = 120)
  (semiperimeter : s = (a + b + c + d) / 2)
  (h5 : x + y = c) 
  (h6 : (|x - y| = 30)) : 
  |x - y| = 30 :=
sorry

end quad_side_difference_l188_188394


namespace pie_count_correct_l188_188046

structure Berries :=
  (strawberries : ℕ)
  (blueberries : ℕ)
  (raspberries : ℕ)

def christine_picking : Berries := {strawberries := 10, blueberries := 8, raspberries := 20}

def rachel_picking : Berries :=
  let c := christine_picking
  {strawberries := 2 * c.strawberries,
   blueberries := 2 * c.blueberries,
   raspberries := c.raspberries / 2}

def total_berries (b1 b2 : Berries) : Berries :=
  {strawberries := b1.strawberries + b2.strawberries,
   blueberries := b1.blueberries + b2.blueberries,
   raspberries := b1.raspberries + b2.raspberries}

def pie_requirements : Berries := {strawberries := 3, blueberries := 2, raspberries := 4}

def max_pies (total : Berries) (requirements : Berries) : Berries :=
  {strawberries := total.strawberries / requirements.strawberries,
   blueberries := total.blueberries / requirements.blueberries,
   raspberries := total.raspberries / requirements.raspberries}

def correct_pies : Berries := {strawberries := 10, blueberries := 12, raspberries := 7}

theorem pie_count_correct :
  let total := total_berries christine_picking rachel_picking;
  max_pies total pie_requirements = correct_pies :=
by {
  sorry
}

end pie_count_correct_l188_188046


namespace dress_designs_possible_l188_188254

theorem dress_designs_possible (colors patterns fabric_types : Nat) (color_choices : colors = 5) (pattern_choices : patterns = 6) (fabric_type_choices : fabric_types = 2) : 
  colors * patterns * fabric_types = 60 := by 
  sorry

end dress_designs_possible_l188_188254


namespace find_13_numbers_l188_188073

theorem find_13_numbers :
  ∃ (a : Fin 13 → ℕ),
    (∀ i, a i % 21 = 0) ∧
    (∀ i j, i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i)) ∧
    (∀ i j, i ≠ j → (a i ^ 5) % (a j ^ 4) = 0) :=
sorry

end find_13_numbers_l188_188073


namespace homogeneous_variances_l188_188334

noncomputable def sample_sizes : (ℕ × ℕ × ℕ) := (9, 13, 15)
noncomputable def sample_variances : (ℝ × ℝ × ℝ) := (3.2, 3.8, 6.3)
noncomputable def significance_level : ℝ := 0.05
noncomputable def degrees_of_freedom : ℕ := 2
noncomputable def V : ℝ := 1.43
noncomputable def critical_value : ℝ := 6.0

theorem homogeneous_variances :
  V < critical_value :=
by
  sorry

end homogeneous_variances_l188_188334


namespace imaginary_part_of_z_l188_188043

-- Define the imaginary unit i where i^2 = -1
def imaginary_unit : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (2 + imaginary_unit) * (1 - imaginary_unit)

-- State the theorem to prove the imaginary part of z
theorem imaginary_part_of_z : Complex.im z = -1 := by
  sorry

end imaginary_part_of_z_l188_188043


namespace max_vertex_sum_l188_188880

theorem max_vertex_sum
  (a U : ℤ)
  (hU : U ≠ 0)
  (hA : 0 = a * 0 * (0 - 3 * U))
  (hB : 0 = a * (3 * U) * ((3 * U) - 3 * U))
  (hC : 12 = a * (3 * U - 1) * ((3 * U - 1) - 3 * U))
  : ∃ N : ℝ, N = (3 * U) / 2 - (9 * a * U^2) / 4 ∧ N ≤ 17.75 :=
by sorry

end max_vertex_sum_l188_188880


namespace smallest_perfect_square_gt_100_has_odd_number_of_factors_l188_188748

theorem smallest_perfect_square_gt_100_has_odd_number_of_factors : 
  ∃ n : ℕ, (n > 100) ∧ (∃ k : ℕ, n = k * k) ∧ (∀ m > 100, ∃ t : ℕ, m = t * t → n ≤ m) := 
sorry

end smallest_perfect_square_gt_100_has_odd_number_of_factors_l188_188748


namespace solution_set_of_inequality_l188_188151

noncomputable def f : ℝ → ℝ := sorry

axiom ax1 : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → 
  (x1 * f x2 - x2 * f x1) / (x2 - x1) > 1

axiom ax2 : f 3 = 2

theorem solution_set_of_inequality :
  {x : ℝ | 0 < x ∧ f x < x - 1} = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end solution_set_of_inequality_l188_188151


namespace Martin_correct_answers_l188_188519

theorem Martin_correct_answers (C K M : ℕ) 
  (h1 : C = 35)
  (h2 : K = C + 8)
  (h3 : M = K - 3) : 
  M = 40 :=
by
  sorry

end Martin_correct_answers_l188_188519


namespace max_a_squared_b_squared_c_squared_l188_188215

theorem max_a_squared_b_squared_c_squared (a b c : ℤ)
  (h1 : a + b + c = 3)
  (h2 : a^3 + b^3 + c^3 = 3) :
  a^2 + b^2 + c^2 ≤ 57 :=
sorry

end max_a_squared_b_squared_c_squared_l188_188215


namespace cherry_trees_leaves_l188_188725

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l188_188725


namespace difference_of_squares_l188_188945

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := 
by
  sorry

end difference_of_squares_l188_188945


namespace smallest_m_l188_188900

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 10*(p:ℤ)^2 - m*(p:ℤ) + 360 = 0) (h_cond : q = 2 * p) :
  p * q = 36 → 3 * p + 3 * q = m → m = 90 :=
by sorry

end smallest_m_l188_188900


namespace transformation_correctness_l188_188712

variable (x x' y y' : ℝ)

-- Conditions
def original_curve : Prop := y^2 = 4
def transformed_curve : Prop := (x'^2)/1 + (y'^2)/4 = 1
def transformation_formula : Prop := (x = 2 * x') ∧ (y = y')

-- Proof Statement
theorem transformation_correctness (h1 : original_curve y) (h2 : transformed_curve x' y') :
  transformation_formula x x' y y' :=
  sorry

end transformation_correctness_l188_188712


namespace max_value_xyz_l188_188068

theorem max_value_xyz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : 2 * x + 3 * x * y^2 + 2 * z = 36) : 
  x^2 * y^2 * z ≤ 144 :=
sorry

end max_value_xyz_l188_188068


namespace total_number_of_toys_l188_188865

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l188_188865


namespace max_a_plus_b_l188_188542

theorem max_a_plus_b (a b c d e : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : a + 2*b + 3*c + 4*d + 5*e = 300) : a + b ≤ 35 :=
sorry

end max_a_plus_b_l188_188542


namespace total_fence_poles_needed_l188_188643

def number_of_poles_per_side := 27

theorem total_fence_poles_needed (n : ℕ) (h : n = number_of_poles_per_side) : 
  4 * n - 4 = 104 :=
by sorry

end total_fence_poles_needed_l188_188643


namespace standard_ellipse_eq_l188_188059

def ellipse_standard_eq (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_ellipse_eq (P: ℝ × ℝ) (Q: ℝ × ℝ) (a b : ℝ) (h1 : P = (-3, 0)) (h2 : Q = (0, -2)) :
  ellipse_standard_eq 3 2 x y :=
by
  sorry

end standard_ellipse_eq_l188_188059


namespace inequality_condition_l188_188367

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_condition (a b: ℝ) (h : ∀ x : ℝ, f a b x ≥ 0) : (b * (a + 1)) / 2 < 3 / 4 := 
sorry

end inequality_condition_l188_188367


namespace simplify_evaluate_expression_l188_188744

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * (a * b^2) - 2 = -2 :=
by
  sorry

end simplify_evaluate_expression_l188_188744


namespace initial_price_correct_l188_188298

-- Definitions based on the conditions
def initial_price : ℝ := 3  -- Rs. 3 per kg
def new_price : ℝ := 5      -- Rs. 5 per kg
def reduction_in_consumption : ℝ := 0.4  -- 40%

-- The main theorem we need to prove
theorem initial_price_correct :
  initial_price = 3 :=
sorry

end initial_price_correct_l188_188298


namespace max_perimeter_of_polygons_l188_188681

noncomputable def largest_possible_perimeter (sides1 sides2 sides3 : Nat) (len : Nat) : Nat :=
  (sides1 + sides2 + sides3) * len

theorem max_perimeter_of_polygons
  (a b c : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b % 2 = 0)
  (h3 : c % 2 = 0)
  (h4 : 180 * (a - 2) / a + 180 * (b - 2) / b + 180 * (c - 2) / c = 360)
  (h5 : ∃ (p : ℕ), ∃ q : ℕ, (a = p ∧ c = p ∧ a = q ∨ a = q ∧ b = p ∨ b = q ∧ c = p))
  : largest_possible_perimeter a b c 2 = 24 := 
sorry

end max_perimeter_of_polygons_l188_188681


namespace third_median_length_l188_188492

-- Proposition stating the problem with conditions and the conclusion
theorem third_median_length (m1 m2 : ℝ) (area : ℝ) (h1 : m1 = 4) (h2 : m2 = 5) (h_area : area = 10 * Real.sqrt 3) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry  -- proof is not included

end third_median_length_l188_188492


namespace arithmetic_geometric_mean_inequality_l188_188304

theorem arithmetic_geometric_mean_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := 
  sorry

end arithmetic_geometric_mean_inequality_l188_188304


namespace number_of_intersections_l188_188468

def line₁ (x y : ℝ) := 2 * x - 3 * y + 6 = 0
def line₂ (x y : ℝ) := 5 * x + 2 * y - 10 = 0
def line₃ (x y : ℝ) := x - 2 * y + 1 = 0
def line₄ (x y : ℝ) := 3 * x - 4 * y + 8 = 0

theorem number_of_intersections : 
  ∃! (p₁ p₂ p₃ : ℝ × ℝ),
    (line₁ p₁.1 p₁.2 ∨ line₂ p₁.1 p₁.2) ∧ (line₃ p₁.1 p₁.2 ∨ line₄ p₁.1 p₁.2) ∧
    (line₁ p₂.1 p₂.2 ∨ line₂ p₂.1 p₂.2) ∧ (line₃ p₂.1 p₂.2 ∨ line₄ p₂.1 p₂.2) ∧
    (line₁ p₃.1 p₃.2 ∨ line₂ p₃.1 p₃.2) ∧ (line₃ p₃.1 p₃.2 ∨ line₄ p₃.1 p₃.2) ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ := 
sorry

end number_of_intersections_l188_188468


namespace percentage_of_students_owning_only_cats_l188_188698

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ) (students_owning_dogs : ℕ) (students_owning_cats : ℕ) (students_owning_both : ℕ)
  (h1 : total_students = 500) (h2 : students_owning_dogs = 200) (h3 : students_owning_cats = 100) (h4 : students_owning_both = 50) :
  ((students_owning_cats - students_owning_both) * 100 / total_students) = 10 :=
by
  -- Placeholder for proof
  sorry

end percentage_of_students_owning_only_cats_l188_188698


namespace three_character_license_plates_l188_188019

theorem three_character_license_plates :
  let consonants := 20
  let vowels := 6
  (consonants * consonants * vowels = 2400) :=
by
  sorry

end three_character_license_plates_l188_188019


namespace gcd_38_23_is_1_l188_188646

theorem gcd_38_23_is_1 : Nat.gcd 38 23 = 1 := by
  sorry

end gcd_38_23_is_1_l188_188646


namespace point_and_sum_of_coordinates_l188_188921

-- Definitions
def point_on_graph_of_g_over_3 (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = (g p.1) / 3

def point_on_graph_of_inv_g_over_3 (g : ℝ → ℝ) (q : ℝ × ℝ) : Prop :=
  q.2 = (g⁻¹ q.1) / 3

-- Main statement
theorem point_and_sum_of_coordinates {g : ℝ → ℝ} (h : point_on_graph_of_g_over_3 g (2, 3)) :
  point_on_graph_of_inv_g_over_3 g (9, 2 / 3) ∧ (9 + 2 / 3 = 29 / 3) :=
by
  sorry

end point_and_sum_of_coordinates_l188_188921


namespace bananas_to_oranges_l188_188935

variables (banana apple orange : Type) 
variables (cost_banana : banana → ℕ) 
variables (cost_apple : apple → ℕ)
variables (cost_orange : orange → ℕ)

-- Conditions given in the problem
axiom cond1 : ∀ (b1 b2 b3 : banana) (a1 a2 : apple), cost_banana b1 = cost_banana b2 → cost_banana b2 = cost_banana b3 → 3 * cost_banana b1 = 2 * cost_apple a1
axiom cond2 : ∀ (a3 a4 a5 a6 : apple) (o1 o2 : orange), cost_apple a3 = cost_apple a4 → cost_apple a4 = cost_apple a5 → cost_apple a5 = cost_apple a6 → 6 * cost_apple a3 = 4 * cost_orange o1

-- Prove that 8 oranges cost as much as 18 bananas
theorem bananas_to_oranges (b1 b2 b3 : banana) (a1 a2 a3 a4 a5 a6 : apple) (o1 o2 : orange) :
    3 * cost_banana b1 = 2 * cost_apple a1 →
    6 * cost_apple a3 = 4 * cost_orange o1 →
    18 * cost_banana b1 = 8 * cost_orange o2 := 
sorry

end bananas_to_oranges_l188_188935


namespace units_digit_is_valid_l188_188601

theorem units_digit_is_valid (n : ℕ) : 
  (∃ k : ℕ, (k^3 % 10 = n)) → 
  (n = 2 ∨ n = 3 ∨ n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end units_digit_is_valid_l188_188601


namespace machine_C_works_in_6_hours_l188_188734

theorem machine_C_works_in_6_hours :
  ∃ C : ℝ, (0 < C ∧ (1/4 + 1/12 + 1/C = 1/2)) → C = 6 :=
by
  sorry

end machine_C_works_in_6_hours_l188_188734


namespace number_of_good_numbers_lt_1000_l188_188397

def is_good_number (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  sum % 10 < 10 ∧
  (sum / 10) % 10 < 10 ∧
  (sum / 100) % 10 < 10 ∧
  (sum < 1000)

theorem number_of_good_numbers_lt_1000 : ∃ n : ℕ, n = 48 ∧
  (forall k, k < 1000 → k < 1000 → is_good_number k → k = 48) := sorry

end number_of_good_numbers_lt_1000_l188_188397


namespace sum_of_squares_l188_188163

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + a * c + b * c = 131) (h2 : a + b + c = 22) : a^2 + b^2 + c^2 = 222 :=
by
  sorry

end sum_of_squares_l188_188163


namespace initial_water_amount_l188_188745

theorem initial_water_amount (x : ℝ) (h : x + 6.8 = 9.8) : x = 3 := 
by
  sorry

end initial_water_amount_l188_188745


namespace max_m_divides_f_l188_188370

noncomputable def f (n : ℕ) : ℤ :=
  (2 * n + 7) * 3^n + 9

theorem max_m_divides_f (m n : ℕ) (h1 : n > 0) (h2 : ∀ n : ℕ, n > 0 → m ∣ ((2 * n + 7) * 3^n + 9)) : m = 36 :=
sorry

end max_m_divides_f_l188_188370


namespace real_z9_count_l188_188680

theorem real_z9_count (z : ℂ) (hz : z^18 = 1) : 
  (∃! z : ℂ, z^18 = 1 ∧ (z^9).im = 0) :=
sorry

end real_z9_count_l188_188680


namespace stock_profit_percentage_l188_188165

theorem stock_profit_percentage 
  (total_stock : ℝ) (total_loss : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ)
  (percentage_sold_at_profit : ℝ) :
  total_stock = 12499.99 →
  total_loss = 500 →
  profit_percentage = 0.20 →
  loss_percentage = 0.10 →
  (0.10 * ((100 - percentage_sold_at_profit) / 100) * 12499.99) - (0.20 * (percentage_sold_at_profit / 100) * 12499.99) = 500 →
  percentage_sold_at_profit = 20 :=
sorry

end stock_profit_percentage_l188_188165


namespace solve_system_of_equations_l188_188523

theorem solve_system_of_equations
  (a b c d x y z u : ℝ)
  (h1 : a^3 * x + a^2 * y + a * z + u = 0)
  (h2 : b^3 * x + b^2 * y + b * z + u = 0)
  (h3 : c^3 * x + c^2 * y + c * z + u = 0)
  (h4 : d^3 * x + d^2 * y + d * z + u = 1) :
  x = 1 / ((d - a) * (d - b) * (d - c)) ∧
  y = -(a + b + c) / ((d - a) * (d - b) * (d - c)) ∧
  z = (a * b + b * c + c * a) / ((d - a) * (d - b) * (d - c)) ∧
  u = - (a * b * c) / ((d - a) * (d - b) * (d - c)) :=
sorry

end solve_system_of_equations_l188_188523


namespace eq_from_conditions_l188_188170

theorem eq_from_conditions (a b : ℂ) :
  (1 / (a + b)) ^ 2003 = 1 ∧ (-a + b) ^ 2005 = 1 → a ^ 2003 + b ^ 2004 = 1 := 
by
  sorry

end eq_from_conditions_l188_188170


namespace combined_yearly_return_percentage_l188_188246

-- Given conditions
def investment1 : ℝ := 500
def return_rate1 : ℝ := 0.07
def investment2 : ℝ := 1500
def return_rate2 : ℝ := 0.15

-- Question to prove
theorem combined_yearly_return_percentage :
  let yearly_return1 := investment1 * return_rate1
  let yearly_return2 := investment2 * return_rate2
  let total_yearly_return := yearly_return1 + yearly_return2
  let total_investment := investment1 + investment2
  ((total_yearly_return / total_investment) * 100) = 13 :=
by
  -- skipping the proof
  sorry

end combined_yearly_return_percentage_l188_188246


namespace birds_count_l188_188692

theorem birds_count (N B : ℕ) 
  (h1 : B = 5 * N)
  (h2 : B = N + 360) : 
  B = 450 := by
  sorry

end birds_count_l188_188692


namespace intersection_M_N_l188_188588

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end intersection_M_N_l188_188588


namespace solution_set_of_inequality_l188_188658

theorem solution_set_of_inequality (a b x : ℝ) (h1 : 0 < a) (h2 : b = 2 * a) : ax > b ↔ x > -2 :=
by sorry

end solution_set_of_inequality_l188_188658


namespace radius_of_sphere_is_approximately_correct_l188_188905

noncomputable def radius_of_sphere_in_cylinder_cone : ℝ :=
  let radius_cylinder := 12
  let height_cylinder := 30
  let radius_sphere := 21 - 0.5 * Real.sqrt (30^2 + 12^2)
  radius_sphere

theorem radius_of_sphere_is_approximately_correct : abs (radius_of_sphere_in_cylinder_cone - 4.84) < 0.01 :=
by
  sorry

end radius_of_sphere_is_approximately_correct_l188_188905


namespace evaluate_expression_l188_188674

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem evaluate_expression : ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = (7 / 49) :=
by
  sorry

end evaluate_expression_l188_188674


namespace joint_savings_account_total_l188_188752

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l188_188752


namespace chelsea_guaranteed_victory_l188_188472

noncomputable def minimum_bullseye_shots_to_win (k : ℕ) (n : ℕ) : ℕ :=
  if (k + 5 * n + 500 > k + 930) then n else sorry

theorem chelsea_guaranteed_victory (k : ℕ) :
  minimum_bullseye_shots_to_win k 87 = 87 :=
by
  sorry

end chelsea_guaranteed_victory_l188_188472


namespace multiplication_scaling_l188_188749

theorem multiplication_scaling (h : 28 * 15 = 420) : 
  (28 / 10) * (15 / 10) = 2.8 * 1.5 ∧ 
  (28 / 100) * 1.5 = 0.28 * 1.5 ∧ 
  (28 / 1000) * (15 / 100) = 0.028 * 0.15 :=
by 
  sorry

end multiplication_scaling_l188_188749


namespace Lakers_win_in_7_games_l188_188934

-- Variables for probabilities given in the problem
variable (p_Lakers_win : ℚ := 1 / 4) -- Lakers' probability of winning a single game
variable (p_Celtics_win : ℚ := 3 / 4) -- Celtics' probability of winning a single game

-- Probabilities and combinations
def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_Lakers_win_game7 : ℚ :=
  let first_6_games := binom 6 3 * (p_Lakers_win ^ 3) * (p_Celtics_win ^ 3)
  let seventh_game := p_Lakers_win
  first_6_games * seventh_game

theorem Lakers_win_in_7_games : probability_Lakers_win_game7 = 540 / 16384 := by
  sorry

end Lakers_win_in_7_games_l188_188934


namespace find_x_l188_188857

theorem find_x (h₁ : 2994 / 14.5 = 175) (h₂ : 29.94 / x = 17.5) : x = 29.94 / 17.5 :=
by
  -- skipping proofs
  sorry

end find_x_l188_188857


namespace smallest_five_digit_multiple_of_18_correct_l188_188189

def smallest_five_digit_multiple_of_18 : ℕ := 10008

theorem smallest_five_digit_multiple_of_18_correct :
  (smallest_five_digit_multiple_of_18 >= 10000) ∧ 
  (smallest_five_digit_multiple_of_18 < 100000) ∧ 
  (smallest_five_digit_multiple_of_18 % 18 = 0) :=
by
  sorry

end smallest_five_digit_multiple_of_18_correct_l188_188189


namespace fifth_plot_difference_l188_188441

-- Define the dimensions of the plots
def plot_width (n : Nat) : Nat := 3 + 2 * (n - 1)
def plot_length (n : Nat) : Nat := 4 + 3 * (n - 1)

-- Define the number of tiles in a plot
def tiles_in_plot (n : Nat) : Nat := plot_width n * plot_length n

-- The main theorem to prove the required difference
theorem fifth_plot_difference :
  tiles_in_plot 5 - tiles_in_plot 4 = 59 := sorry

end fifth_plot_difference_l188_188441


namespace min_n_for_triangle_pattern_l188_188134

/-- 
There are two types of isosceles triangles with a waist length of 1:
-  Type 1: An acute isosceles triangle with a vertex angle of 30 degrees.
-  Type 2: A right isosceles triangle with a vertex angle of 90 degrees.
They are placed around a point in a clockwise direction in a sequence such that:
- The 1st and 2nd are acute isosceles triangles (30 degrees),
- The 3rd is a right isosceles triangle (90 degrees),
- The 4th and 5th are acute isosceles triangles (30 degrees),
- The 6th is a right isosceles triangle (90 degrees), and so on.

Prove that the minimum value of n such that the nth triangle coincides exactly with
the 1st triangle is 23.
-/
theorem min_n_for_triangle_pattern : ∃ n : ℕ, n = 23 ∧ (∀ m < 23, m ≠ 23) :=
sorry

end min_n_for_triangle_pattern_l188_188134


namespace calculation_1500_increased_by_45_percent_l188_188171

theorem calculation_1500_increased_by_45_percent :
  1500 * (1 + 45 / 100) = 2175 := 
by
  sorry

end calculation_1500_increased_by_45_percent_l188_188171


namespace pairs_of_polygons_with_angle_ratio_l188_188498

theorem pairs_of_polygons_with_angle_ratio :
  ∃ n, n = 2 ∧ (∀ {k r : ℕ}, (k > 2 ∧ r > 2) → 
  (4 * (180 * r - 360) = 3 * (180 * k - 360) →
  ((k = 3 ∧ r = 18) ∨ (k = 2 ∧ r = 6)))) :=
by
  -- The proof should be provided here, but we skip it
  sorry

end pairs_of_polygons_with_angle_ratio_l188_188498


namespace scientific_notation_l188_188930

def billion : ℝ := 10^9
def fifteenPointSeventyFiveBillion : ℝ := 15.75 * billion

theorem scientific_notation :
  fifteenPointSeventyFiveBillion = 1.575 * 10^10 :=
  sorry

end scientific_notation_l188_188930


namespace common_non_integer_root_eq_l188_188191

theorem common_non_integer_root_eq (p1 p2 q1 q2 : ℤ) 
  (x : ℝ) (hx1 : x^2 + p1 * x + q1 = 0) (hx2 : x^2 + p2 * x + q2 = 0) 
  (hnint : ¬ ∃ (n : ℤ), x = n) : p1 = p2 ∧ q1 = q2 :=
sorry

end common_non_integer_root_eq_l188_188191


namespace max_value_PXQ_l188_188296

theorem max_value_PXQ :
  ∃ (X P Q : ℕ), (XX = 10 * X + X) ∧ (10 * X + X) * X = 100 * P + 10 * X + Q ∧ 
  (X = 1 ∨ X = 5 ∨ X = 6) ∧ 
  (100 * P + 10 * X + Q) = 396 :=
sorry

end max_value_PXQ_l188_188296


namespace day_of_week_2_2312_wednesday_l188_188672

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem day_of_week_2_2312_wednesday (birth_year : ℕ) (birth_day : String) 
  (h1 : birth_year = 2312 - 300)
  (h2 : birth_day = "Wednesday") :
  "Monday" = "Monday" :=
sorry

end day_of_week_2_2312_wednesday_l188_188672


namespace product_of_first_three_terms_of_arithmetic_sequence_l188_188219

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l188_188219


namespace largest_stamps_per_page_l188_188947

-- Definitions of the conditions
def stamps_book1 : ℕ := 1260
def stamps_book2 : ℕ := 1470

-- Statement to be proven: The largest number of stamps per page (gcd of 1260 and 1470)
theorem largest_stamps_per_page : Nat.gcd stamps_book1 stamps_book2 = 210 :=
by
  sorry

end largest_stamps_per_page_l188_188947


namespace tan_arithmetic_geometric_l188_188080

noncomputable def a_seq : ℕ → ℝ := sorry -- Define a_n as an arithmetic sequence (details abstracted)
noncomputable def b_seq : ℕ → ℝ := sorry -- Define b_n as a geometric sequence (details abstracted)

axiom a_seq_is_arithmetic : ∀ n m : ℕ, a_seq (n + 1) - a_seq n = a_seq (m + 1) - a_seq m
axiom b_seq_is_geometric : ∀ n : ℕ, ∃ r : ℝ, b_seq (n + 1) = b_seq n * r
axiom a_seq_sum : a_seq 2017 + a_seq 2018 = Real.pi
axiom b_seq_square : b_seq 20 ^ 2 = 4

theorem tan_arithmetic_geometric : 
  (Real.tan ((a_seq 2 + a_seq 4033) / (b_seq 1 * b_seq 39)) = 1) :=
sorry

end tan_arithmetic_geometric_l188_188080


namespace calculate_expression_l188_188873

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := 
by
  -- proof goes here
  sorry

end calculate_expression_l188_188873


namespace find_breadth_of_rectangle_l188_188169

noncomputable def breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) (breadth : ℝ) : Prop :=
  A = length_to_breadth_ratio * breadth * breadth → breadth = 20

-- Now we can state the theorem.
theorem find_breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) : breadth_of_rectangle A length_to_breadth_ratio 20 :=
by
  intros h
  sorry

end find_breadth_of_rectangle_l188_188169


namespace Tiffany_bags_l188_188159

theorem Tiffany_bags (x : ℕ) 
  (h1 : 8 = x + 1) : 
  x = 7 :=
by
  sorry

end Tiffany_bags_l188_188159


namespace P_at_6_l188_188871

noncomputable def P (x : ℕ) : ℚ := (720 * x) / (x^2 - 1)

theorem P_at_6 : P 6 = 48 :=
by
  -- Definitions and conditions derived from the problem.
  -- Establishing given condition and deriving P(6) value.
  sorry

end P_at_6_l188_188871


namespace lcm_36_90_eq_180_l188_188040

theorem lcm_36_90_eq_180 : Nat.lcm 36 90 = 180 := 
by 
  sorry

end lcm_36_90_eq_180_l188_188040


namespace triangle_is_isosceles_l188_188140

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop := ∃ (s : ℝ), a = s ∧ b = s

theorem triangle_is_isosceles 
  (A B C a b c : ℝ) 
  (h_sides_angles : a = c ∧ b = c) 
  (h_cos_eq : a * Real.cos B = b * Real.cos A) : 
  is_isosceles_triangle A B C a b c := 
by 
  sorry

end triangle_is_isosceles_l188_188140


namespace interest_earned_l188_188576

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (I : ℝ):
  P = 1500 → r = 0.12 → n = 4 →
  A = compound_interest P r n →
  I = A - P →
  I = 862.2 :=
by
  intros hP hr hn hA hI
  sorry

end interest_earned_l188_188576


namespace angles_relation_l188_188192

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end angles_relation_l188_188192


namespace speed_of_stream_l188_188025

variable (D : ℝ) -- The distance rowed in both directions
variable (vs : ℝ) -- The speed of the stream
variable (Vb : ℝ := 78) -- The speed of the boat in still water

theorem speed_of_stream (h : (D / (Vb - vs) = 2 * (D / (Vb + vs)))) : vs = 26 := by
    sorry

end speed_of_stream_l188_188025


namespace distance_home_to_school_l188_188998

def speed_walk := 5
def speed_car := 15
def time_difference := 2

variable (d : ℝ) -- distance from home to school
variable (T1 T2 : ℝ) -- T1: time to school, T2: time back home

-- Conditions
axiom h1 : T1 = d / speed_walk / 2 + d / speed_car / 2
axiom h2 : d = speed_car * T2 / 3 + speed_walk * 2 * T2 / 3
axiom h3 : T1 = T2 + time_difference

-- Theorem to prove
theorem distance_home_to_school : d = 150 :=
by
  sorry

end distance_home_to_school_l188_188998


namespace tv_price_increase_percentage_l188_188096

theorem tv_price_increase_percentage (P Q : ℝ) (x : ℝ) :
  (P * (1 + x / 100) * Q * 0.8 = P * Q * 1.28) → x = 60 :=
by sorry

end tv_price_increase_percentage_l188_188096


namespace solution_set_inequality_l188_188269

theorem solution_set_inequality :
  {x : ℝ | (x^2 + 4) / (x - 4)^2 ≥ 0} = {x | x < 4} ∪ {x | x > 4} :=
by
  sorry

end solution_set_inequality_l188_188269


namespace shape_of_triangle_l188_188493

-- Define the problem conditions
variable {a b : ℝ}
variable {A B C : ℝ}
variable (triangle_condition : (a^2 / b^2 = tan A / tan B))

-- Define the theorem to be proved
theorem shape_of_triangle ABC
  (h : triangle_condition):
  (A = B ∨ A + B = π / 2) :=
sorry

end shape_of_triangle_l188_188493


namespace linear_function_quadrants_l188_188156

theorem linear_function_quadrants (k : ℝ) :
  (k - 3 > 0) ∧ (-k + 2 < 0) → k > 3 :=
by
  intro h
  sorry

end linear_function_quadrants_l188_188156


namespace illegal_simplification_works_for_specific_values_l188_188224

-- Definitions for the variables
def a : ℕ := 43
def b : ℕ := 17
def c : ℕ := 26

-- Define the sum of cubes
def sum_of_cubes (x y : ℕ) : ℕ := x ^ 3 + y ^ 3

-- Define the illegal simplification fraction
def illegal_simplification_fraction_correct (a b c : ℕ) : Prop :=
  (a^3 + b^3) / (a^3 + c^3) = (a + b) / (a + c)

-- The theorem to prove
theorem illegal_simplification_works_for_specific_values :
  illegal_simplification_fraction_correct a b c :=
by
  -- Proof will reside here
  sorry

end illegal_simplification_works_for_specific_values_l188_188224


namespace marble_prob_l188_188774

theorem marble_prob (T : ℕ) (hT1 : T > 12) 
  (hP : ((T - 12) / T : ℚ) * ((T - 12) / T) = 36 / 49) : T = 84 :=
sorry

end marble_prob_l188_188774


namespace eighth_term_matchstick_count_l188_188336

def matchstick_sequence (n : ℕ) : ℕ := (n + 1) * 3

theorem eighth_term_matchstick_count : matchstick_sequence 8 = 27 :=
by
  -- the proof will go here
  sorry

end eighth_term_matchstick_count_l188_188336


namespace Sandy_marks_per_correct_sum_l188_188721

theorem Sandy_marks_per_correct_sum
  (x : ℝ)  -- number of marks Sandy gets for each correct sum
  (marks_lost_per_incorrect : ℝ := 2)  -- 2 marks lost for each incorrect sum, default value is 2
  (total_attempts : ℤ := 30)  -- Sandy attempts 30 sums, default value is 30
  (total_marks : ℝ := 60)  -- Sandy obtains 60 marks, default value is 60
  (correct_sums : ℤ := 24)  -- Sandy got 24 sums correct, default value is 24
  (incorrect_sums := total_attempts - correct_sums) -- incorrect sums are the remaining attempts
  (marks_from_correct := correct_sums * x) -- total marks from the correct sums
  (marks_lost_from_incorrect := incorrect_sums * marks_lost_per_incorrect) -- total marks lost from the incorrect sums
  (total_marks_obtained := marks_from_correct - marks_lost_from_incorrect) -- total marks obtained

  -- The theorem states that x must be 3 given the conditions above
  : total_marks_obtained = total_marks → x = 3 := by sorry

end Sandy_marks_per_correct_sum_l188_188721


namespace smallest_integer_to_make_y_perfect_square_l188_188965

-- Define y as given in the problem
def y : ℕ :=
  2^33 * 3^54 * 4^45 * 5^76 * 6^57 * 7^38 * 8^69 * 9^10

-- Smallest integer n such that (y * n) is a perfect square
theorem smallest_integer_to_make_y_perfect_square
  : ∃ n : ℕ, (∀ k : ℕ, y * n = k * k) ∧ (∀ m : ℕ, (∀ k : ℕ, y * m = k * k) → n ≤ m) := 
sorry

end smallest_integer_to_make_y_perfect_square_l188_188965


namespace number_of_participants_l188_188050

theorem number_of_participants (n : ℕ) (h : n - 1 = 25) : n = 26 := 
by sorry

end number_of_participants_l188_188050


namespace triangle_vertex_y_coordinate_l188_188869

theorem triangle_vertex_y_coordinate (h : ℝ) :
  let A := (0, 0)
  let C := (8, 0)
  let B := (4, h)
  (1/2) * (8) * h = 32 → h = 8 :=
by
  intro h
  intro H
  sorry

end triangle_vertex_y_coordinate_l188_188869


namespace negation_proof_l188_188729

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

theorem negation_proof : (¬(∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬(P x)) :=
by sorry

end negation_proof_l188_188729


namespace remaining_calories_proof_l188_188559

def volume_of_rectangular_block (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_cube (side : ℝ) : ℝ :=
  side * side * side

def remaining_volume (initial_volume eaten_volume : ℝ) : ℝ :=
  initial_volume - eaten_volume

def remaining_calories (remaining_volume calorie_density : ℝ) : ℝ :=
  remaining_volume * calorie_density

theorem remaining_calories_proof :
  let calorie_density := 110
  let original_length := 4
  let original_width := 8
  let original_height := 2
  let cube_side := 2
  let original_volume := volume_of_rectangular_block original_length original_width original_height
  let eaten_volume := volume_of_cube cube_side
  let remaining_vol := remaining_volume original_volume eaten_volume
  let resulting_calories := remaining_calories remaining_vol calorie_density
  resulting_calories = 6160 := by
  repeat { sorry }

end remaining_calories_proof_l188_188559


namespace no_roots_ge_two_l188_188684

theorem no_roots_ge_two (x : ℝ) (h : x ≥ 2) : 4 * x^3 - 5 * x^2 - 6 * x + 3 ≠ 0 := by
  sorry

end no_roots_ge_two_l188_188684


namespace remove_one_and_average_l188_188139

theorem remove_one_and_average (l : List ℕ) (n : ℕ) (avg : ℚ) :
  l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] →
  avg = 8.5 →
  (l.sum - n : ℚ) = 14 * avg →
  n = 1 :=
by
  intros hlist havg hsum
  sorry

end remove_one_and_average_l188_188139


namespace kyle_money_left_l188_188613

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end kyle_money_left_l188_188613


namespace sum_of_ages_is_18_l188_188292

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end sum_of_ages_is_18_l188_188292


namespace find_f_neg_one_l188_188002

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then x^2 + 2 * x else - ( (x^2) + (2 * x))

theorem find_f_neg_one : 
  f (-1) = -3 :=
by 
  sorry

end find_f_neg_one_l188_188002


namespace bulb_cheaper_than_lamp_by_4_l188_188270

/-- Jim bought a $7 lamp and a bulb. The bulb cost a certain amount less than the lamp. 
    He bought 2 lamps and 6 bulbs and paid $32 in all. 
    The amount by which the bulb is cheaper than the lamp is $4. -/
theorem bulb_cheaper_than_lamp_by_4
  (lamp_price bulb_price : ℝ)
  (h1 : lamp_price = 7)
  (h2 : bulb_price = 7 - 4)
  (h3 : 2 * lamp_price + 6 * bulb_price = 32) :
  (7 - bulb_price = 4) :=
by {
  sorry
}

end bulb_cheaper_than_lamp_by_4_l188_188270


namespace problem_statement_l188_188505

-- Given conditions
variables {p q r t n : ℕ}

axiom prime_p : Nat.Prime p
axiom prime_q : Nat.Prime q
axiom prime_r : Nat.Prime r

axiom nat_n : n ≥ 1
axiom nat_t : t ≥ 1

axiom eqn1 : p^2 + q * t = (p + t)^n
axiom eqn2 : p^2 + q * r = t^4

-- Statement to prove
theorem problem_statement : n < 3 ∧ (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) :=
by
  sorry

end problem_statement_l188_188505


namespace line_through_intersection_points_of_circles_l188_188241

theorem line_through_intersection_points_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
    (x - 2*y + 6 = 0) :=
by
  intro x y h
  -- Condition of circle 1
  have circle1 : x^2 + y^2 + 4*x - 4*y - 1 = 0 := h.left
  -- Condition of circle 2
  have circle2 : x^2 + y^2 + 2*x - 13 = 0 := h.right
  sorry

end line_through_intersection_points_of_circles_l188_188241


namespace weight_of_tin_of_cookies_l188_188626

def weight_of_bag_of_chips := 20 -- in ounces
def weight_jasmine_carries := 336 -- converting 21 pounds to ounces
def bags_jasmine_buys := 6
def tins_multiplier := 4

theorem weight_of_tin_of_cookies 
  (weight_of_bag_of_chips : ℕ := weight_of_bag_of_chips)
  (weight_jasmine_carries : ℕ := weight_jasmine_carries)
  (bags_jasmine_buys : ℕ := bags_jasmine_buys)
  (tins_multiplier : ℕ := tins_multiplier) : 
  ℕ :=
  let total_weight_bags := bags_jasmine_buys * weight_of_bag_of_chips
  let total_weight_cookies := weight_jasmine_carries - total_weight_bags
  let num_of_tins := bags_jasmine_buys * tins_multiplier
  total_weight_cookies / num_of_tins

example : weight_of_tin_of_cookies weight_of_bag_of_chips weight_jasmine_carries bags_jasmine_buys tins_multiplier = 9 :=
by sorry

end weight_of_tin_of_cookies_l188_188626


namespace factor_expression_eq_l188_188833

theorem factor_expression_eq (x : ℤ) : 75 * x + 50 = 25 * (3 * x + 2) :=
by
  -- The actual proof is omitted
  sorry

end factor_expression_eq_l188_188833


namespace cost_price_one_meter_l188_188633

theorem cost_price_one_meter (selling_price : ℤ) (total_meters : ℤ) (profit_per_meter : ℤ) 
  (h1 : selling_price = 6788) (h2 : total_meters = 78) (h3 : profit_per_meter = 29) : 
  (selling_price - (profit_per_meter * total_meters)) / total_meters = 58 := 
by 
  sorry

end cost_price_one_meter_l188_188633


namespace ryan_fish_count_l188_188423

theorem ryan_fish_count
  (R : ℕ)
  (J : ℕ)
  (Jeffery_fish : ℕ)
  (h1 : Jeffery_fish = 60)
  (h2 : Jeffery_fish = 2 * R)
  (h3 : J + R + Jeffery_fish = 100)
  : R = 30 :=
by
  sorry

end ryan_fish_count_l188_188423


namespace muffins_equation_l188_188916

def remaining_muffins : ℕ := 48
def total_muffins : ℕ := 83
def initially_baked_muffins : ℕ := 35

theorem muffins_equation : initially_baked_muffins + remaining_muffins = total_muffins :=
  by
    -- Skipping the proof here
    sorry

end muffins_equation_l188_188916


namespace elementary_schools_in_Lansing_l188_188177

theorem elementary_schools_in_Lansing (total_students : ℕ) (students_per_school : ℕ) (h1 : total_students = 6175) (h2 : students_per_school = 247) : total_students / students_per_school = 25 := 
by sorry

end elementary_schools_in_Lansing_l188_188177


namespace george_earnings_l188_188940

theorem george_earnings (cars_sold : ℕ) (price_per_car : ℕ) (lego_set_price : ℕ) (h1 : cars_sold = 3) (h2 : price_per_car = 5) (h3 : lego_set_price = 30) :
  cars_sold * price_per_car + lego_set_price = 45 :=
by
  sorry

end george_earnings_l188_188940


namespace factor_difference_of_squares_l188_188347

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end factor_difference_of_squares_l188_188347


namespace pizza_area_increase_l188_188952

theorem pizza_area_increase 
  (r : ℝ) 
  (A_medium A_large : ℝ) 
  (h_medium_area : A_medium = Real.pi * r^2)
  (h_large_area : A_large = Real.pi * (1.40 * r)^2) : 
  ((A_large - A_medium) / A_medium) * 100 = 96 := 
by 
  sorry

end pizza_area_increase_l188_188952


namespace boat_speed_in_still_water_l188_188963

theorem boat_speed_in_still_water:
  ∀ (V_b : ℝ) (V_s : ℝ) (D : ℝ),
    V_s = 3 → 
    (D = (V_b + V_s) * 1) → 
    (D = (V_b - V_s) * 1.5) → 
    V_b = 15 :=
by
  intros V_b V_s D V_s_eq H_downstream H_upstream
  sorry

end boat_speed_in_still_water_l188_188963


namespace smallest_number_property_l188_188866

theorem smallest_number_property : 
  ∃ n, ((n - 7) % 12 = 0) ∧ ((n - 7) % 16 = 0) ∧ ((n - 7) % 18 = 0) ∧ ((n - 7) % 21 = 0) ∧ ((n - 7) % 28 = 0) ∧ n = 1015 :=
by
  sorry  -- Proof is omitted

end smallest_number_property_l188_188866


namespace euler_conjecture_disproof_l188_188458

theorem euler_conjecture_disproof :
    ∃ (n : ℕ), 133^4 + 110^4 + 56^4 = n^4 ∧ n = 143 :=
by {
  use 143,
  sorry
}

end euler_conjecture_disproof_l188_188458


namespace mass_percentage_H_in_C4H8O2_l188_188984

theorem mass_percentage_H_in_C4H8O2 (molar_mass_C : Real := 12.01) 
                                    (molar_mass_H : Real := 1.008) 
                                    (molar_mass_O : Real := 16.00) 
                                    (num_C_atoms : Nat := 4)
                                    (num_H_atoms : Nat := 8)
                                    (num_O_atoms : Nat := 2) :
    (num_H_atoms * molar_mass_H) / ((num_C_atoms * molar_mass_C) + (num_H_atoms * molar_mass_H) + (num_O_atoms * molar_mass_O)) * 100 = 9.15 :=
by
  sorry

end mass_percentage_H_in_C4H8O2_l188_188984


namespace dogwood_trees_total_is_100_l188_188380

def initial_dogwood_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20
def total_dogwood_trees : ℕ := initial_dogwood_trees + trees_planted_today + trees_planted_tomorrow

theorem dogwood_trees_total_is_100 : total_dogwood_trees = 100 := by
  sorry  -- Proof goes here

end dogwood_trees_total_is_100_l188_188380


namespace units_digit_of_6_to_the_6_l188_188244

theorem units_digit_of_6_to_the_6 : (6^6) % 10 = 6 := by
  sorry

end units_digit_of_6_to_the_6_l188_188244


namespace find_third_side_l188_188164

theorem find_third_side (a b : ℝ) (c : ℕ) 
  (h1 : a = 3.14)
  (h2 : b = 0.67)
  (h_triangle_ineq : a + b > ↑c ∧ a + ↑c > b ∧ b + ↑c > a) : 
  c = 3 := 
by
  -- Proof goes here
  sorry

end find_third_side_l188_188164


namespace expression_nonnegative_l188_188746

theorem expression_nonnegative (x : ℝ) :
  0 <= x ∧ x < 3 → (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 := 
by
  sorry

end expression_nonnegative_l188_188746


namespace problem_l188_188489

def p (x y : Int) : Int :=
  if x ≥ 0 ∧ y ≥ 0 then x * y
  else if x < 0 ∧ y < 0 then x - 2 * y
  else if x ≥ 0 ∧ y < 0 then 2 * x + 3 * y
  else if x < 0 ∧ y ≥ 0 then x + 3 * y
  else 3 * x + y

theorem problem : p (p 2 (-3)) (p (-1) 4) = 28 := by
  sorry

end problem_l188_188489


namespace part_I_part_II_l188_188401

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x - 1|

theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ 2 - |x - 1|) : 0 ≤ a ∧ a ≤ 4 := 
sorry

theorem part_II (a : ℝ) (h₁ : a < 2) (h₂ : ∀ x : ℝ, f x a ≥ 3) : a = -4 := 
sorry

end part_I_part_II_l188_188401


namespace surface_area_increase_96_percent_l188_188259

variable (s : ℝ)

def original_surface_area : ℝ := 6 * s^2
def new_edge_length : ℝ := 1.4 * s
def new_surface_area : ℝ := 6 * (new_edge_length s)^2

theorem surface_area_increase_96_percent :
  (new_surface_area s - original_surface_area s) / (original_surface_area s) * 100 = 96 :=
by
  simp [original_surface_area, new_edge_length, new_surface_area]
  sorry

end surface_area_increase_96_percent_l188_188259


namespace evaluate_g_at_3_l188_188771

def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 200 := by
  sorry

end evaluate_g_at_3_l188_188771


namespace angle_same_terminal_side_l188_188311

theorem angle_same_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 → α = 330 :=
by
  sorry

end angle_same_terminal_side_l188_188311


namespace range_of_a_l188_188874

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then -x^2 - 1 else Real.log (x + 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≤ a * x) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l188_188874


namespace smallest_n_leq_l188_188982

theorem smallest_n_leq (n : ℤ) : (n ^ 2 - 13 * n + 40 ≤ 0) → (n = 5) :=
sorry

end smallest_n_leq_l188_188982


namespace average_of_remaining_two_numbers_l188_188104

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95) 
  (h_avg_ab : (a + b) / 2 = 3.8) 
  (h_avg_cd : (c + d) / 2 = 3.85) :
  ((e + f) / 2) = 4.2 := 
by 
  sorry

end average_of_remaining_two_numbers_l188_188104


namespace gumballs_multiple_purchased_l188_188378

-- Definitions
def joanna_initial : ℕ := 40
def jacques_initial : ℕ := 60
def final_each : ℕ := 250

-- Proof statement
theorem gumballs_multiple_purchased (m : ℕ) :
  (joanna_initial + joanna_initial * m) + (jacques_initial + jacques_initial * m) = 2 * final_each →
  m = 4 :=
by 
  sorry

end gumballs_multiple_purchased_l188_188378


namespace emily_seeds_start_with_l188_188516

-- Define the conditions as hypotheses
variables (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)

-- Conditions: Emily planted 29 seeds in the big garden and 4 seeds in each of her 3 small gardens.
def emily_conditions := big_garden_seeds = 29 ∧ small_gardens = 3 ∧ seeds_per_small_garden = 4

-- Define the statement to prove the total number of seeds Emily started with
theorem emily_seeds_start_with (h : emily_conditions big_garden_seeds small_gardens seeds_per_small_garden) : 
(big_garden_seeds + small_gardens * seeds_per_small_garden) = 41 :=
by
  -- Assuming the proof follows logically from conditions
  sorry

end emily_seeds_start_with_l188_188516


namespace complex_expression_evaluation_l188_188666

-- Conditions
def i : ℂ := Complex.I -- Representing the imaginary unit i

-- Defining the inverse of a complex number
noncomputable def complex_inv (z : ℂ) := 1 / z

-- Proof statement
theorem complex_expression_evaluation :
  (i - complex_inv i + 3)⁻¹ = (3 - 2 * i) / 13 := by
sorry

end complex_expression_evaluation_l188_188666


namespace total_time_of_four_sets_of_stairs_l188_188011

def time_first : ℕ := 15
def time_increment : ℕ := 10
def num_sets : ℕ := 4

theorem total_time_of_four_sets_of_stairs :
  let a := time_first
  let d := time_increment
  let n := num_sets
  let l := a + (n - 1) * d
  let S := n / 2 * (a + l)
  S = 120 :=
by
  sorry

end total_time_of_four_sets_of_stairs_l188_188011


namespace desired_cost_of_mixture_l188_188320

theorem desired_cost_of_mixture 
  (w₈ : ℝ) (c₈ : ℝ) -- weight and cost per pound of the $8 candy
  (w₅ : ℝ) (c₅ : ℝ) -- weight and cost per pound of the $5 candy
  (total_w : ℝ) (desired_cost : ℝ) -- total weight and desired cost per pound of the mixture
  (h₁ : w₈ = 30) (h₂ : c₈ = 8) 
  (h₃ : w₅ = 60) (h₄ : c₅ = 5)
  (h₅ : total_w = w₈ + w₅)
  (h₆ : desired_cost = (w₈ * c₈ + w₅ * c₅) / total_w) :
  desired_cost = 6 := 
by
  sorry

end desired_cost_of_mixture_l188_188320


namespace equation_equivalence_and_rst_l188_188606

theorem equation_equivalence_and_rst 
  (a x y c : ℝ) 
  (r s t : ℤ) 
  (h1 : r = 3) 
  (h2 : s = 1) 
  (h3 : t = 5)
  (h_eq1 : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^3) = a^5 * c^5 ∧ r * s * t = 15 :=
by
  sorry

end equation_equivalence_and_rst_l188_188606


namespace geometric_sequence_s4_l188_188021

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0, a1, q => 0
| (n+1), a1, q => a1 * (1 - q^(n+1)) / (1 - q)

variable (a1 q : ℝ) (n : ℕ)

theorem geometric_sequence_s4  (h1 : a1 * (q^1) * (q^3) = 16) (h2 : geometric_sequence_sum 2 a1 q + a1 * (q^2) = 7) :
  geometric_sequence_sum 3 a1 q = 15 :=
sorry

end geometric_sequence_s4_l188_188021


namespace find_x_l188_188832

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 - q.2)

theorem find_x : ∃ x : ℤ, ∃ y : ℤ, star (4, 5) (1, 3) = star (x, y) (2, 1) ∧ x = 3 :=
by 
  sorry

end find_x_l188_188832


namespace initial_apps_count_l188_188251

theorem initial_apps_count (x A : ℕ) 
  (h₁ : A - 18 + x = 5) : A = 23 - x :=
by
  sorry

end initial_apps_count_l188_188251


namespace solve_fractional_equation_l188_188895

theorem solve_fractional_equation (x : ℝ) (h : (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : 
  x = 1 :=
sorry

end solve_fractional_equation_l188_188895


namespace shiela_paintings_l188_188867

theorem shiela_paintings (h1 : 18 % 2 = 0) : 18 / 2 = 9 := 
by sorry

end shiela_paintings_l188_188867


namespace four_point_questions_l188_188022

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 := 
sorry

end four_point_questions_l188_188022


namespace change_factor_l188_188300

theorem change_factor (avg1 avg2 : ℝ) (n : ℕ) (h_avg1 : avg1 = 40) (h_n : n = 10) (h_avg2 : avg2 = 80) : avg2 * (n : ℝ) / (avg1 * (n : ℝ)) = 2 :=
by
  sorry

end change_factor_l188_188300


namespace management_sampled_count_l188_188321

variable (total_employees salespeople management_personnel logistical_support staff_sample_size : ℕ)
variable (proportional_sampling : Prop)
variable (n_management_sampled : ℕ)

axiom h1 : total_employees = 160
axiom h2 : salespeople = 104
axiom h3 : management_personnel = 32
axiom h4 : logistical_support = 24
axiom h5 : proportional_sampling
axiom h6 : staff_sample_size = 20

theorem management_sampled_count : n_management_sampled = 4 :=
by
  -- The proof is omitted as per instructions
  sorry

end management_sampled_count_l188_188321


namespace sequence_a3_l188_188377

theorem sequence_a3 (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (recursion : ∀ n, a (n + 1) = a n / (1 + a n)) : 
  a 3 = 1 / 3 :=
by 
  sorry

end sequence_a3_l188_188377


namespace contrapositive_of_happy_people_possess_it_l188_188794

variable (P Q : Prop)

theorem contrapositive_of_happy_people_possess_it
  (h : P → Q) : ¬ Q → ¬ P := by
  intro hq
  intro p
  apply hq
  apply h
  exact p

#check contrapositive_of_happy_people_possess_it

end contrapositive_of_happy_people_possess_it_l188_188794


namespace average_score_bounds_l188_188561

/-- Problem data definitions -/
def n_100 : ℕ := 2
def n_90_99 : ℕ := 9
def n_80_89 : ℕ := 17
def n_70_79 : ℕ := 28
def n_60_69 : ℕ := 36
def n_50_59 : ℕ := 7
def n_48 : ℕ := 1

def sum_scores_min : ℕ := (100 * n_100 + 90 * n_90_99 + 80 * n_80_89 + 70 * n_70_79 + 60 * n_60_69 + 50 * n_50_59 + 48)
def sum_scores_max : ℕ := (100 * n_100 + 99 * n_90_99 + 89 * n_80_89 + 79 * n_70_79 + 69 * n_60_69 + 59 * n_50_59 + 48)
def total_people : ℕ := n_100 + n_90_99 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_48

/-- Prove the minimum and maximum average scores. -/
theorem average_score_bounds :
  (sum_scores_min / total_people : ℚ) = 68.88 ∧
  (sum_scores_max / total_people : ℚ) = 77.61 :=
by
  sorry

end average_score_bounds_l188_188561


namespace paint_grid_condition_l188_188228

variables {a b c d e A B C D E : ℕ}

def is_valid (n : ℕ) : Prop := n = 2 ∨ n = 3

theorem paint_grid_condition 
  (ha : is_valid a) (hb : is_valid b) (hc : is_valid c) 
  (hd : is_valid d) (he : is_valid e) (hA : is_valid A) 
  (hB : is_valid B) (hC : is_valid C) (hD : is_valid D) 
  (hE : is_valid E) :
  a + b + c + d + e = A + B + C + D + E :=
sorry

end paint_grid_condition_l188_188228


namespace turnip_pulled_by_mice_l188_188928

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end turnip_pulled_by_mice_l188_188928


namespace company_salary_decrease_l188_188512

variables {E S : ℝ} -- Let the initial number of employees be E and the initial average salary be S

theorem company_salary_decrease :
  (0.8 * E * (1.15 * S)) / (E * S) = 0.92 := 
by
  -- The proof will go here, but we use sorry to skip it for now
  sorry

end company_salary_decrease_l188_188512


namespace triangle_third_side_possibilities_l188_188946

theorem triangle_third_side_possibilities (x : ℕ) : 
  (6 + 8 > x) ∧ (x + 6 > 8) ∧ (x + 8 > 6) → 
  3 ≤ x ∧ x < 14 → 
  ∃ n, n = 11 :=
by
  sorry

end triangle_third_side_possibilities_l188_188946


namespace men_in_first_group_l188_188685

theorem men_in_first_group (M : ℕ) (h1 : M * 18 * 6 = 15 * 12 * 6) : M = 10 :=
by
  sorry

end men_in_first_group_l188_188685


namespace abs_inequality_solution_l188_188784

theorem abs_inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| < 7} = {x : ℝ | -4 < x ∧ x < 3} :=
sorry

end abs_inequality_solution_l188_188784


namespace tank_weight_when_full_l188_188356

theorem tank_weight_when_full (p q : ℝ) (x y : ℝ)
  (h1 : x + (3/4) * y = p)
  (h2 : x + (1/3) * y = q) :
  x + y = (8/5) * p - (8/5) * q :=
by
  sorry

end tank_weight_when_full_l188_188356


namespace largest_angle_in_pentagon_l188_188207

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
  (hA : A = 70) 
  (hB : B = 120) 
  (hCD : C = D) 
  (hE : E = 3 * C - 30) 
  (sum_angles : A + B + C + D + E = 540) :
  E = 198 := 
by 
  sorry

end largest_angle_in_pentagon_l188_188207


namespace unit_price_of_first_batch_minimum_selling_price_l188_188326

-- Proof Problem 1
theorem unit_price_of_first_batch :
  (∃ x : ℝ, (3200 / x) * 2 = 7200 / (x + 10) ∧ x = 80) := 
  sorry

-- Proof Problem 2
theorem minimum_selling_price (x : ℝ) (hx : x = 80) :
  (40 * x + 80 * (x + 10) - 3200 - 7200 + 20 * 0.8 * x ≥ 3520) → 
  (∃ y : ℝ, y ≥ 120) :=
  sorry

end unit_price_of_first_batch_minimum_selling_price_l188_188326


namespace right_triangle_min_area_l188_188853

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l188_188853


namespace find_e_l188_188743

-- Define the conditions and state the theorem.
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) 
  (h1: ∃ a b c : ℝ, (a + b + c)/3 = -3 ∧ a * b * c = -3 ∧ 3 + d + e + f = -3)
  (h2: Q 0 d e f = 9) : e = -42 :=
by
  sorry

end find_e_l188_188743


namespace find_y_in_similar_triangles_l188_188907

-- Define the variables and conditions of the problem
def is_similar (a1 b1 a2 b2 : ℚ) : Prop :=
  a1 / b1 = a2 / b2

-- Problem statement
theorem find_y_in_similar_triangles
  (a1 b1 a2 b2 : ℚ)
  (h1 : a1 = 15)
  (h2 : b1 = 12)
  (h3 : b2 = 10)
  (similarity_condition : is_similar a1 b1 a2 b2) :
  a2 = 25 / 2 :=
by
  rw [h1, h2, h3, is_similar] at similarity_condition
  sorry

end find_y_in_similar_triangles_l188_188907


namespace find_a_plus_b_l188_188264

variables {a b : ℝ}

theorem find_a_plus_b (h1 : a - b = -3) (h2 : a * b = 2) : a + b = Real.sqrt 17 ∨ a + b = -Real.sqrt 17 := by
  -- Proof can be filled in here
  sorry

end find_a_plus_b_l188_188264


namespace second_machine_time_l188_188520

/-- Given:
1. A first machine can address 600 envelopes in 10 minutes.
2. Both machines together can address 600 envelopes in 4 minutes.
We aim to prove that the second machine alone would take 20/3 minutes to address 600 envelopes. -/
theorem second_machine_time (x : ℝ) 
  (first_machine_rate : ℝ := 600 / 10)
  (combined_rate_needed : ℝ := 600 / 4)
  (second_machine_rate : ℝ := combined_rate_needed - first_machine_rate) 
  (secs_envelope_rate : ℝ := second_machine_rate) 
  (envelopes : ℝ := 600) : 
  x = envelopes / secs_envelope_rate :=
sorry

end second_machine_time_l188_188520


namespace cupcakes_left_l188_188801

def pack_count := 3
def cupcakes_per_pack := 4
def cupcakes_eaten := 5

theorem cupcakes_left : (pack_count * cupcakes_per_pack - cupcakes_eaten) = 7 := 
by 
  sorry

end cupcakes_left_l188_188801


namespace fraction_eq_zero_has_solution_l188_188117

theorem fraction_eq_zero_has_solution :
  ∀ (x : ℝ), x^2 - x - 2 = 0 ∧ x + 1 ≠ 0 → x = 2 :=
by
  sorry

end fraction_eq_zero_has_solution_l188_188117


namespace find_x_for_g_inv_eq_3_l188_188875

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem find_x_for_g_inv_eq_3 : ∃ x : ℝ, g x = 113 :=
by
  exists 3
  unfold g
  norm_num

end find_x_for_g_inv_eq_3_l188_188875


namespace parabola_hyperbola_tangent_l188_188502

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5
noncomputable def hyperbola (x y : ℝ) (m : ℝ) : ℝ := y^2 - m * x^2 - 1

theorem parabola_hyperbola_tangent (m : ℝ) :
(∃ x y : ℝ, y = parabola x ∧ hyperbola x y m = 0) ↔ 
m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6 := by
  sorry

end parabola_hyperbola_tangent_l188_188502


namespace diff_set_Q_minus_P_l188_188433

def P (x : ℝ) : Prop := 1 - (2 / x) < 0
def Q (x : ℝ) : Prop := |x - 2| < 1
def diff_set (P Q : ℝ → Prop) (x : ℝ) : Prop := Q x ∧ ¬ P x

theorem diff_set_Q_minus_P :
  ∀ x : ℝ, diff_set Q P x ↔ (2 ≤ x ∧ x < 3) :=
by
  sorry

end diff_set_Q_minus_P_l188_188433


namespace correct_calculation_l188_188540

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end correct_calculation_l188_188540


namespace remainder_of_sum_mod_11_l188_188024

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l188_188024


namespace meat_pie_cost_l188_188474

variable (total_farthings : ℕ) (farthings_per_pfennig : ℕ) (remaining_pfennigs : ℕ)

def total_pfennigs (total_farthings farthings_per_pfennig : ℕ) : ℕ :=
  total_farthings / farthings_per_pfennig

def pie_cost (total_farthings farthings_per_pfennig remaining_pfennigs : ℕ) : ℕ :=
  total_pfennigs total_farthings farthings_per_pfennig - remaining_pfennigs

theorem meat_pie_cost
  (h1 : total_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7) :
  pie_cost total_farthings farthings_per_pfennig remaining_pfennigs = 2 :=
by
  sorry

end meat_pie_cost_l188_188474


namespace maximum_value_expression_l188_188371

theorem maximum_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ p, p = x * y ∧ (4 * p^3 - 92 * p^2 + 754 * p) = 441 / 2 :=
by {
  sorry
}

end maximum_value_expression_l188_188371


namespace complement_of_A_l188_188663

def A : Set ℝ := { x | x^2 - x ≥ 0 }
def R_complement_A : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem complement_of_A :
  ∀ x : ℝ, x ∈ R_complement_A ↔ x ∉ A :=
sorry

end complement_of_A_l188_188663


namespace tank_empty_time_l188_188088

theorem tank_empty_time (V : ℝ) (r_inlet r_outlet1 r_outlet2 : ℝ) (I : V = 20 * 12^3)
  (r_inlet_val : r_inlet = 5) (r_outlet1_val : r_outlet1 = 9) 
  (r_outlet2_val : r_outlet2 = 8) : 
  (V / ((r_outlet1 + r_outlet2) - r_inlet) = 2880) :=
by
  sorry

end tank_empty_time_l188_188088


namespace train_length_l188_188906

theorem train_length (speed_km_per_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_per_hr = 80) (h_time : time_sec = 9) :
  ∃ length_m : ℕ, length_m = 200 :=
by
  sorry

end train_length_l188_188906


namespace library_table_count_l188_188085

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 36 + d1 * 6 + d0 

theorem library_table_count (chairs people_per_table : Nat) (h1 : chairs = 231) (h2 : people_per_table = 3) :
    Nat.ceil ((base6_to_base10 chairs) / people_per_table) = 31 :=
by
  sorry

end library_table_count_l188_188085


namespace center_of_circle_param_eq_l188_188600

theorem center_of_circle_param_eq (θ : ℝ) : 
  (∃ c : ℝ × ℝ, ∀ θ, 
    ∃ (x y : ℝ), 
      (x = 2 + 2 * Real.cos θ) ∧ 
      (y = 2 * Real.sin θ) ∧ 
      (x - c.1)^2 + y^2 = 4) 
  ↔ 
  c = (2, 0) :=
by
  sorry

end center_of_circle_param_eq_l188_188600


namespace find_parabola_constant_l188_188057

theorem find_parabola_constant (a b c : ℝ) (h_vertex : ∀ y, (4:ℝ) = -5 / 4 * y * y + 5 / 2 * y + c)
  (h_point : (-1:ℝ) = -5 / 4 * (3:ℝ) ^ 2 + 5 / 2 * (3:ℝ) + c ) :
  c = 11 / 4 :=
sorry

end find_parabola_constant_l188_188057


namespace liliane_has_44_44_more_cookies_l188_188053

variables (J : ℕ) (L O : ℕ) (totalCookies : ℕ)

def liliane_has_more_30_percent (J L : ℕ) : Prop :=
  L = J + (3 * J / 10)

def oliver_has_less_10_percent (J O : ℕ) : Prop :=
  O = J - (J / 10)

def total_cookies (J L O totalCookies : ℕ) : Prop :=
  J + L + O = totalCookies

theorem liliane_has_44_44_more_cookies
  (h1 : liliane_has_more_30_percent J L)
  (h2 : oliver_has_less_10_percent J O)
  (h3 : total_cookies J L O totalCookies)
  (h4 : totalCookies = 120) :
  (L - O) * 100 / O = 4444 / 100 := sorry

end liliane_has_44_44_more_cookies_l188_188053


namespace number_of_footballs_is_3_l188_188438

-- Define the variables and conditions directly from the problem

-- Let F be the cost of one football and S be the cost of one soccer ball
variable (F S : ℝ)

-- Condition 1: Some footballs and 1 soccer ball cost 155 dollars
variable (number_of_footballs : ℝ)
variable (H1 : F * number_of_footballs + S = 155)

-- Condition 2: 2 footballs and 3 soccer balls cost 220 dollars
variable (H2 : 2 * F + 3 * S = 220)

-- Condition 3: The cost of one soccer ball is 50 dollars
variable (H3 : S = 50)

-- Theorem: Prove that the number of footballs in the first set is 3
theorem number_of_footballs_is_3 (H1 H2 H3 : Prop) :
  number_of_footballs = 3 := by
  sorry

end number_of_footballs_is_3_l188_188438


namespace sets_equal_l188_188501

def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, abs (-(Real.sqrt 3))}

theorem sets_equal : A = B :=
by 
  sorry

end sets_equal_l188_188501


namespace a8_eq_128_l188_188103

-- Definitions of conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions
axiom a2_eq_2 : a 2 = 2
axiom a3_mul_a4_eq_32 : a 3 * a 4 = 32
axiom is_geometric : is_geometric_sequence a q

-- Statement to prove
theorem a8_eq_128 : a 8 = 128 :=
sorry

end a8_eq_128_l188_188103


namespace smallest_difference_l188_188451

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_difference_l188_188451


namespace problem1_problem2_l188_188196

-- Define Set A
def SetA : Set ℝ := { y | ∃ x, (2 ≤ x ∧ x ≤ 3) ∧ y = -2^x }

-- Define Set B parameterized by a
def SetB (a : ℝ) : Set ℝ := { x | x^2 + 3 * x - a^2 - 3 * a > 0 }

-- Problem 1: Prove that when a = 4, A ∩ B = {-8 < x < -7}
theorem problem1 : A ∩ SetB 4 = { x | -8 < x ∧ x < -7 } :=
sorry

-- Problem 2: Prove the range of a for which "x ∈ A" is a sufficient but not necessary condition for "x ∈ B"
theorem problem2 : ∀ a : ℝ, (∀ x, x ∈ SetA → x ∈ SetB a) → -4 < a ∧ a < 1 :=
sorry

end problem1_problem2_l188_188196


namespace solution_for_x_l188_188454

theorem solution_for_x (x : ℝ) : x^2 - x - 1 = (x + 1)^0 → x = 2 :=
by
  intro h
  have h_simp : x^2 - x - 1 = 1 := by simp [h]
  sorry

end solution_for_x_l188_188454


namespace M_inter_N_is_5_l188_188708

/-- Define the sets M and N. -/
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {2, 5, 8}

/-- Prove the intersection of M and N is {5}. -/
theorem M_inter_N_is_5 : M ∩ N = {5} :=
by
  sorry

end M_inter_N_is_5_l188_188708


namespace inscribed_square_sum_c_d_eq_200689_l188_188781

theorem inscribed_square_sum_c_d_eq_200689 :
  ∃ (c d : ℕ), Nat.gcd c d = 1 ∧ (∃ x : ℚ, x = (c : ℚ) / (d : ℚ) ∧ 
    let a := 48
    let b := 55
    let longest_side := 73
    let s := (a + b + longest_side) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - longest_side))
    area = 1320 ∧ x = 192720 / 7969 ∧ c + d = 200689) :=
sorry

end inscribed_square_sum_c_d_eq_200689_l188_188781


namespace remaining_bananas_l188_188594

def original_bananas : ℕ := 46
def removed_bananas : ℕ := 5

theorem remaining_bananas : original_bananas - removed_bananas = 41 := by
  sorry

end remaining_bananas_l188_188594


namespace roger_earned_correct_amount_l188_188648

def small_lawn_rate : ℕ := 9
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def initial_small_lawns : ℕ := 5
def initial_medium_lawns : ℕ := 4
def initial_large_lawns : ℕ := 5

def forgot_small_lawns : ℕ := 2
def forgot_medium_lawns : ℕ := 3
def forgot_large_lawns : ℕ := 3

def actual_small_lawns := initial_small_lawns - forgot_small_lawns
def actual_medium_lawns := initial_medium_lawns - forgot_medium_lawns
def actual_large_lawns := initial_large_lawns - forgot_large_lawns

def money_earned_small := actual_small_lawns * small_lawn_rate
def money_earned_medium := actual_medium_lawns * medium_lawn_rate
def money_earned_large := actual_large_lawns * large_lawn_rate

def total_money_earned := money_earned_small + money_earned_medium + money_earned_large

theorem roger_earned_correct_amount : total_money_earned = 69 := by
  sorry

end roger_earned_correct_amount_l188_188648


namespace parking_lot_problem_l188_188487

theorem parking_lot_problem :
  let total_spaces := 50
  let cars := 2
  let total_ways := total_spaces * (total_spaces - 1)
  let adjacent_ways := (total_spaces - 1) * 2
  let valid_ways := total_ways - adjacent_ways
  valid_ways = 2352 :=
by
  sorry

end parking_lot_problem_l188_188487


namespace polygon_angle_pairs_l188_188414

theorem polygon_angle_pairs
  {r k : ℕ}
  (h_ratio : (180 * r - 360) / r = (4 / 3) * (180 * k - 360) / k)
  (h_k_lt_15 : k < 15)
  (h_r_ge_3 : r ≥ 3) :
  (k = 7 ∧ r = 42) ∨ (k = 6 ∧ r = 18) ∨ (k = 5 ∧ r = 10) ∨ (k = 4 ∧ r = 6) :=
sorry

end polygon_angle_pairs_l188_188414


namespace remainder_when_doubling_l188_188563

theorem remainder_when_doubling:
  ∀ (n k : ℤ), n = 30 * k + 16 → (2 * n) % 15 = 2 :=
by
  intros n k h
  sorry

end remainder_when_doubling_l188_188563


namespace complex_transformation_l188_188471

open Complex

theorem complex_transformation :
  let z := -1 + (7 : ℂ) * I
  let rotation := (1 / 2 + (Real.sqrt 3) / 2 * I)
  let dilation := 2
  (z * rotation * dilation = -22 - ((Real.sqrt 3) - 7) * I) :=
by
  sorry

end complex_transformation_l188_188471


namespace hash_op_correct_l188_188004

-- Definition of the custom operation #
def hash_op (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- The theorem to prove that 3 # 8 = 80
theorem hash_op_correct : hash_op 3 8 = 80 :=
by
  sorry

end hash_op_correct_l188_188004


namespace rectangle_length_to_width_ratio_l188_188595

variables (s : ℝ)

-- Given conditions
def small_square_side := s
def large_square_side := 3 * s
def rectangle_length := large_square_side
def rectangle_width := large_square_side - 2 * small_square_side

-- Theorem to prove the ratio of the length to the width of the rectangle
theorem rectangle_length_to_width_ratio : 
  ∃ (r : ℝ), r = rectangle_length s / rectangle_width s ∧ r = 3 := 
by
  sorry

end rectangle_length_to_width_ratio_l188_188595


namespace negation_equiv_l188_188173

theorem negation_equiv (x : ℝ) : ¬ (x^2 - 1 < 0) ↔ (x^2 - 1 ≥ 0) :=
by
  sorry

end negation_equiv_l188_188173


namespace arnold_danny_age_l188_188805

theorem arnold_danny_age:
  ∃ x : ℝ, (x + 1) * (x + 1) = x * x + 11 ∧ x = 5 :=
by
  sorry

end arnold_danny_age_l188_188805


namespace smallest_n_mod_equiv_l188_188031

theorem smallest_n_mod_equiv (n : ℕ) (h : 5 * n ≡ 4960 [MOD 31]) : n = 31 := by 
  sorry

end smallest_n_mod_equiv_l188_188031


namespace z_in_fourth_quadrant_l188_188483

def complex_quadrant (re im : ℤ) : String :=
  if re > 0 ∧ im > 0 then "First Quadrant"
  else if re < 0 ∧ im > 0 then "Second Quadrant"
  else if re < 0 ∧ im < 0 then "Third Quadrant"
  else if re > 0 ∧ im < 0 then "Fourth Quadrant"
  else "Axis"

theorem z_in_fourth_quadrant : complex_quadrant 2 (-3) = "Fourth Quadrant" :=
by
  sorry

end z_in_fourth_quadrant_l188_188483


namespace average_age_of_dance_group_l188_188806

theorem average_age_of_dance_group (S_f S_m : ℕ) (avg_females avg_males : ℕ) 
(hf : avg_females = S_f / 12) (hm : avg_males = S_m / 18) 
(h1 : avg_females = 25) (h2 : avg_males = 40) : 
  (S_f + S_m) / 30 = 34 :=
by
  sorry

end average_age_of_dance_group_l188_188806


namespace find_N_l188_188635

theorem find_N (x y : ℕ) (N : ℕ) (h1 : N = x * (x + 9)) (h2 : N = y * (y + 6)) : 
  N = 112 :=
  sorry

end find_N_l188_188635


namespace starting_elevation_l188_188485

variable (rate time final_elevation : ℝ)
variable (h_rate : rate = 10)
variable (h_time : time = 5)
variable (h_final_elevation : final_elevation = 350)

theorem starting_elevation (start_elevation : ℝ) :
  start_elevation = 400 :=
  by
    sorry

end starting_elevation_l188_188485


namespace division_quotient_remainder_l188_188824

theorem division_quotient_remainder (A : ℕ) (h1 : A / 9 = 2) (h2 : A % 9 = 6) : A = 24 := 
by
  sorry

end division_quotient_remainder_l188_188824


namespace parallel_lines_same_slope_l188_188673

theorem parallel_lines_same_slope (k : ℝ) : 
  (2*x + y + 1 = 0) ∧ (y = k*x + 3) → (k = -2) := 
by
  sorry

end parallel_lines_same_slope_l188_188673


namespace intersection_with_x_axis_l188_188828

theorem intersection_with_x_axis :
  (∃ x, ∃ y, y = 0 ∧ y = -3 * x + 3 ∧ (x = 1 ∧ y = 0)) :=
by
  -- proof will go here
  sorry

end intersection_with_x_axis_l188_188828


namespace fujian_provincial_games_distribution_count_l188_188145

theorem fujian_provincial_games_distribution_count 
  (staff_members : Finset String)
  (locations : Finset String)
  (A B C D E F : String)
  (A_in_B : A ∈ staff_members)
  (B_in_B : B ∈ staff_members)
  (C_in_B : C ∈ staff_members)
  (D_in_B : D ∈ staff_members)
  (E_in_B : E ∈ staff_members)
  (F_in_B : F ∈ staff_members)
  (locations_count : locations.card = 2)
  (staff_count : staff_members.card = 6)
  (must_same_group : ∀ g₁ g₂ : Finset String, A ∈ g₁ → B ∈ g₁ → g₁ ∪ g₂ = staff_members)
  (min_two_people : ∀ g : Finset String, 2 ≤ g.card) :
  ∃ distrib_methods : ℕ, distrib_methods = 22 := 
by
  sorry

end fujian_provincial_games_distribution_count_l188_188145


namespace problem_l188_188447

noncomputable def f(x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1
noncomputable def f_prime(x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + b

theorem problem (a b : ℝ) 
  (h₁ : f_prime 1 a b = 4) 
  (h₂ : f 1 a b = 3) : 
  a + b = 2 :=
sorry

end problem_l188_188447


namespace profit_percentage_l188_188045

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 78) :
  ((selling_price - cost_price) / cost_price) * 100 = 30 :=
by
  sorry

end profit_percentage_l188_188045


namespace find_f3_l188_188314

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 - c * x + 2

theorem find_f3 (a b c : ℝ)
  (h1 : f a b c (-3) = 9) :
  f a b c 3 = -5 :=
by
  sorry

end find_f3_l188_188314


namespace valid_three_digit_numbers_count_l188_188656

def is_prime_or_even (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def count_valid_numbers : ℕ :=
  (4 * 4) -- number of valid combinations for hundreds and tens digits

theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by 
  -- outline the structure of the proof here, but we use sorry to indicate the proof is not complete
  sorry

end valid_three_digit_numbers_count_l188_188656


namespace complex_number_solution_l188_188086

theorem complex_number_solution (a b : ℝ) (z : ℂ) :
  z = a + b * I →
  (a - 2) ^ 2 + b ^ 2 = 25 →
  (a + 4) ^ 2 + b ^ 2 = 25 →
  a ^ 2 + (b - 2) ^ 2 = 25 →
  z = -1 - 4 * I :=
sorry

end complex_number_solution_l188_188086


namespace proof_x_y_l188_188570

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end proof_x_y_l188_188570


namespace no_neighboring_beads_same_color_probability_l188_188715

theorem no_neighboring_beads_same_color_probability : 
  let total_beads := 9
  let count_red := 4
  let count_white := 3
  let count_blue := 2
  let total_permutations := Nat.factorial total_beads / (Nat.factorial count_red * Nat.factorial count_white * Nat.factorial count_blue)
  ∃ valid_permutations : ℕ,
  valid_permutations = 100 ∧
  valid_permutations / total_permutations = 5 / 63 := by
  sorry

end no_neighboring_beads_same_color_probability_l188_188715


namespace total_ways_to_buy_l188_188477

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l188_188477


namespace log_diff_lt_one_l188_188383

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_diff_lt_one
  (b c x : ℝ)
  (h_eq_sym : ∀ (t : ℝ), (t - 2)^2 + b * (t - 2) + c = (t + 2)^2 + b * (t + 2) + c)
  (h_f_zero_pos : (0)^2 + b * (0) + c > 0)
  (m n : ℝ)
  (h_fm_0 : m^2 + b * m + c = 0)
  (h_fn_0 : n^2 + b * n + c = 0)
  (h_m_ne_n : m ≠ n)
  : log_base 4 m - log_base (1/4) n < 1 :=
  sorry

end log_diff_lt_one_l188_188383


namespace max_value_of_quadratic_l188_188902

theorem max_value_of_quadratic : ∃ x : ℝ, (∀ y : ℝ, (-3 * y^2 + 9 * y - 1) ≤ (-3 * (3/2)^2 + 9 * (3/2) - 1)) ∧ x = 3/2 :=
by
  sorry

end max_value_of_quadratic_l188_188902


namespace who_is_who_l188_188114

-- Define the types for inhabitants
inductive Inhabitant
| A : Inhabitant
| B : Inhabitant

-- Define the property of being a liar
def is_liar (x : Inhabitant) : Prop := 
  match x with
  | Inhabitant.A  => false -- Initial assumption, to be refined
  | Inhabitant.B  => false -- Initial assumption, to be refined

-- Define the statement made by A
def statement_by_A : Prop :=
  (is_liar Inhabitant.A ∧ ¬ is_liar Inhabitant.B)

-- The main theorem to prove
theorem who_is_who (h : ¬statement_by_A) :
  is_liar Inhabitant.A ∧ is_liar Inhabitant.B :=
by
  -- Proof goes here
  sorry

end who_is_who_l188_188114


namespace circle_equation_l188_188679

-- Define the given conditions
def point_P : ℝ × ℝ := (-1, 0)
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def center_C : ℝ × ℝ := (1, 2)

-- Define the required equation of the circle and the claim
def required_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- The Lean theorem statement
theorem circle_equation :
  ∃ (x y : ℝ), required_circle x y :=
sorry

end circle_equation_l188_188679


namespace solve_equation_l188_188625

theorem solve_equation :
  ∀ x : ℝ, (-x^2 = (2*x + 4) / (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intro x
  -- the proof steps would go here
  sorry

end solve_equation_l188_188625


namespace correct_operation_l188_188476

theorem correct_operation (a : ℝ) : 2 * (a^2) * a = 2 * (a^3) := by sorry

end correct_operation_l188_188476


namespace simplify_expression_l188_188577

open Real

-- Define the given expression as a function of x
noncomputable def given_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  sqrt (2 * (1 + sqrt (1 + ( (x^4 - 1) / (2 * x^2) )^2)))

-- Define the expected simplified expression
noncomputable def expected_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  (x^2 + 1) / x

-- Proof statement to verify the simplification
theorem simplify_expression (x : ℝ) (hx : 0 < x) :
  given_expression x hx = expected_expression x hx :=
sorry

end simplify_expression_l188_188577


namespace probability_of_condition1_before_condition2_l188_188069

-- Definitions for conditions
def condition1 (draw_counts : List ℕ) : Prop :=
  ∃ count ∈ draw_counts, count ≥ 3

def condition2 (draw_counts : List ℕ) : Prop :=
  ∀ count ∈ draw_counts, count ≥ 1

-- Probability function
def probability_condition1_before_condition2 : ℚ :=
  13 / 27

-- The proof statement
theorem probability_of_condition1_before_condition2 :
  (∃ draw_counts : List ℕ, (condition1 draw_counts) ∧  ¬(condition2 draw_counts)) →
  probability_condition1_before_condition2 = 13 / 27 :=
sorry

end probability_of_condition1_before_condition2_l188_188069


namespace solution_set_of_inequality_l188_188083

theorem solution_set_of_inequality :
  {x : ℝ | (x-1)*(2-x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l188_188083


namespace diamond_and_face_card_probability_l188_188632

noncomputable def probability_first_diamond_second_face_card : ℚ :=
  let total_cards := 52
  let total_faces := 12
  let diamond_faces := 3
  let diamond_non_faces := 10
  (9/52) * (12/51) + (3/52) * (11/51)

theorem diamond_and_face_card_probability :
  probability_first_diamond_second_face_card = 47 / 884 := 
by {
  sorry
}

end diamond_and_face_card_probability_l188_188632


namespace lesser_fraction_l188_188856

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 17 / 24) (h_prod : x * y = 1 / 8) : min x y = 1 / 3 := by
  sorry

end lesser_fraction_l188_188856


namespace sum_of_digits_is_11_l188_188755

def digits_satisfy_conditions (A B C : ℕ) : Prop :=
  (C = 0 ∨ C = 5) ∧
  (A = 2 * B) ∧
  (A * B * C = 40)

theorem sum_of_digits_is_11 (A B C : ℕ) (h : digits_satisfy_conditions A B C) : A + B + C = 11 :=
by
  sorry

end sum_of_digits_is_11_l188_188755


namespace quadratic_eq_coeff_l188_188979

theorem quadratic_eq_coeff (x : ℝ) : 
  (x^2 + 2 = 3 * x) = (∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 2 ∧ (a * x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_eq_coeff_l188_188979


namespace find_four_digit_number_l188_188677

noncomputable def reverse_num (n : ℕ) : ℕ := -- assume definition to reverse digits
  sorry

theorem find_four_digit_number :
  ∃ (A : ℕ), 1000 ≤ A ∧ A ≤ 9999 ∧ reverse_num (9 * A) = A ∧ 9 * A = reverse_num A ∧ A = 1089 :=
sorry

end find_four_digit_number_l188_188677


namespace min_moves_to_reassemble_l188_188662

theorem min_moves_to_reassemble (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, (∀ pieces, pieces = n - 1) ∧ pieces = 1 → move_count = n - 1 :=
by
  sorry

end min_moves_to_reassemble_l188_188662


namespace jane_rejected_percentage_l188_188007

theorem jane_rejected_percentage (P : ℕ) (John_rejected : ℤ) (Jane_inspected_rejected : ℤ) :
  John_rejected = 7 * P ∧
  Jane_inspected_rejected = 5 * P ∧
  (John_rejected + Jane_inspected_rejected) = 75 * P → 
  Jane_inspected_rejected = P  :=
by sorry

end jane_rejected_percentage_l188_188007


namespace three_digit_numbers_l188_188064

theorem three_digit_numbers (n : ℕ) (a b c : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n = 100 * a + 10 * b + c)
  (h3 : b^2 = a * c)
  (h4 : (10 * b + c) % 4 = 0) :
  n = 124 ∨ n = 248 ∨ n = 444 ∨ n = 964 ∨ n = 888 :=
sorry

end three_digit_numbers_l188_188064


namespace intersection_of_P_and_Q_l188_188188

def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}
def R : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = R := by
  sorry

end intersection_of_P_and_Q_l188_188188


namespace solve_for_x_l188_188761

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l188_188761


namespace yanna_baked_butter_cookies_in_morning_l188_188162

-- Define the conditions
def biscuits_morning : ℕ := 40
def biscuits_afternoon : ℕ := 20
def cookies_afternoon : ℕ := 10
def total_more_biscuits : ℕ := 30

-- Define the statement to be proved
theorem yanna_baked_butter_cookies_in_morning (B : ℕ) : 
  (biscuits_morning + biscuits_afternoon = (B + cookies_afternoon) + total_more_biscuits) → B = 20 :=
by
  sorry

end yanna_baked_butter_cookies_in_morning_l188_188162


namespace cricket_team_members_l188_188243

theorem cricket_team_members (avg_whole_team: ℕ) (captain_age: ℕ) (wicket_keeper_age: ℕ) 
(remaining_avg_age: ℕ) (n: ℕ):
avg_whole_team = 23 →
captain_age = 25 →
wicket_keeper_age = 30 →
remaining_avg_age = 22 →
(n * avg_whole_team - captain_age - wicket_keeper_age = (n - 2) * remaining_avg_age) →
n = 11 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cricket_team_members_l188_188243


namespace store_loses_out_l188_188695

theorem store_loses_out (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (x y : ℝ)
    (h1 : a = b * x) (h2 : b = a * y) : x + y > 2 :=
by
  sorry

end store_loses_out_l188_188695


namespace vector_k_range_l188_188318

noncomputable def vector_length (v : (ℝ × ℝ)) : ℝ := (v.1 ^ 2 + v.2 ^ 2).sqrt

theorem vector_k_range :
  let a := (-2, 2)
  let b := (5, k)
  vector_length (a.1 + b.1, a.2 + b.2) ≤ 5 → -6 ≤ k ∧ k ≤ 2 := by
  sorry

end vector_k_range_l188_188318


namespace ratio_of_andy_age_in_5_years_to_rahim_age_l188_188719

def rahim_age_now : ℕ := 6
def andy_age_now : ℕ := rahim_age_now + 1
def andy_age_in_5_years : ℕ := andy_age_now + 5
def ratio (a b : ℕ) : ℕ := a / b

theorem ratio_of_andy_age_in_5_years_to_rahim_age : ratio andy_age_in_5_years rahim_age_now = 2 := by
  sorry

end ratio_of_andy_age_in_5_years_to_rahim_age_l188_188719


namespace correct_calculation_l188_188119

-- Definitions of calculations based on conditions
def calc_A (a : ℝ) := a^2 + a^2 = a^4
def calc_B (a : ℝ) := (a^2)^3 = a^5
def calc_C (a : ℝ) := a + 2 = 2 * a
def calc_D (a b : ℝ) := (a * b)^3 = a^3 * b^3

-- Theorem stating that only the fourth calculation is correct
theorem correct_calculation (a b : ℝ) :
  ¬(calc_A a) ∧ ¬(calc_B a) ∧ ¬(calc_C a) ∧ calc_D a b :=
by sorry

end correct_calculation_l188_188119


namespace ben_gave_18_fish_l188_188322

variable (initial_fish : ℕ) (total_fish : ℕ) (given_fish : ℕ)

theorem ben_gave_18_fish
    (h1 : initial_fish = 31)
    (h2 : total_fish = 49)
    (h3 : total_fish = initial_fish + given_fish) :
    given_fish = 18 :=
by
  sorry

end ben_gave_18_fish_l188_188322


namespace tan_double_angle_l188_188549

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 1 / 3) : Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l188_188549


namespace problems_on_each_worksheet_l188_188860

-- Define the conditions
def worksheets_total : Nat := 9
def worksheets_graded : Nat := 5
def problems_left : Nat := 16

-- Define the number of remaining worksheets and the problems per worksheet
def remaining_worksheets : Nat := worksheets_total - worksheets_graded
def problems_per_worksheet : Nat := problems_left / remaining_worksheets

-- Prove the number of problems on each worksheet
theorem problems_on_each_worksheet : problems_per_worksheet = 4 :=
by
  sorry

end problems_on_each_worksheet_l188_188860


namespace div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l188_188845

theorem div_by_3_9_then_mul_by_5_6_eq_div_by_5_2 :
  (∀ (x : ℚ), (x / (3/9)) * (5/6) = x / (5/2)) :=
by
  sorry

end div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l188_188845


namespace cost_price_per_metre_l188_188364

theorem cost_price_per_metre (total_metres total_sale total_loss_per_metre total_sell_price : ℕ) (h1: total_metres = 500) (h2: total_sell_price = 15000) (h3: total_loss_per_metre = 10) : total_sell_price + (total_loss_per_metre * total_metres) / total_metres = 40 :=
by
  sorry

end cost_price_per_metre_l188_188364


namespace range_of_m_l188_188689

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 2) + 2 * m / (2 - x) = 3) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l188_188689


namespace arcsin_one_half_eq_pi_six_l188_188583

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = π / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l188_188583


namespace students_failed_exam_l188_188499

def total_students : ℕ := 740
def percent_passed : ℝ := 0.35
def percent_failed : ℝ := 1 - percent_passed
def failed_students : ℝ := percent_failed * total_students

theorem students_failed_exam : failed_students = 481 := 
by sorry

end students_failed_exam_l188_188499


namespace avg_of_multiples_of_10_eq_305_l188_188953

theorem avg_of_multiples_of_10_eq_305 (N : ℕ) (h : N % 10 = 0) (h_avg : (10 + N) / 2 = 305) : N = 600 :=
sorry

end avg_of_multiples_of_10_eq_305_l188_188953


namespace part1_part2_l188_188092

variables {A B C : ℝ} {a b c : ℝ}

-- conditions of the problem
def condition_1 (a b c : ℝ) (C : ℝ) : Prop :=
  a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0

def condition_2 (C : ℝ) : Prop :=
  0 < C ∧ C < Real.pi

-- Part 1: Proving the value of angle A
theorem part1 (a b c C : ℝ) (h1 : condition_1 a b c C) (h2 : condition_2 C) : 
  A = Real.pi / 3 :=
sorry

-- Part 2: Range of possible values for the perimeter, given c = 3
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2

theorem part2 (a b A B C : ℝ) (h1 : condition_1 a b 3 C) (h2 : condition_2 C) 
           (h3 : A = Real.pi / 3) (h4 : is_acute_triangle A B C) :
  ∃ p, p ∈ Set.Ioo ((3 * Real.sqrt 3 + 9) / 2) (9 + 3 * Real.sqrt 3) :=
sorry

end part1_part2_l188_188092


namespace infinite_3_stratum_numbers_l188_188788

-- Condition for 3-stratum number
def is_3_stratum_number (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = (Finset.range (n + 1)).filter (λ x => n % x = 0) ∧
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Part (a): Find a 3-stratum number
example : is_3_stratum_number 120 := sorry

-- Part (b): Prove there are infinitely many 3-stratum numbers
theorem infinite_3_stratum_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_3_stratum_number (f n) := sorry

end infinite_3_stratum_numbers_l188_188788


namespace winnie_lollipops_remainder_l188_188735

theorem winnie_lollipops_remainder :
  ∃ (k : ℕ), k = 505 % 14 ∧ k = 1 :=
by
  sorry

end winnie_lollipops_remainder_l188_188735


namespace alvin_age_l188_188016

theorem alvin_age (A S : ℕ) (h_s : S = 10) (h_cond : S = 1/2 * A - 5) : A = 30 := by
  sorry

end alvin_age_l188_188016


namespace inequality_bounds_l188_188324

theorem inequality_bounds (x y : ℝ) : |y - 3 * x| < 2 * x ↔ x > 0 ∧ x < y ∧ y < 5 * x := by
  sorry

end inequality_bounds_l188_188324


namespace harmonic_mean_is_54_div_11_l188_188404

-- Define lengths of sides
def a : ℕ := 3
def b : ℕ := 6
def c : ℕ := 9

-- Define the harmonic mean calculation function
def harmonic_mean (x y z : ℕ) : ℚ :=
  let reciprocals_sum : ℚ := (1 / x + 1 / y + 1 / z)
  let average_reciprocal : ℚ := reciprocals_sum / 3
  1 / average_reciprocal

-- Prove that the harmonic mean of the given lengths is 54/11
theorem harmonic_mean_is_54_div_11 : harmonic_mean a b c = 54 / 11 := by
  sorry

end harmonic_mean_is_54_div_11_l188_188404


namespace shipping_cost_l188_188218

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

/-- Lizzy's total shipping cost for 540 pounds of fish packed in 30-pound crates at $1.5 per crate is $27. -/
theorem shipping_cost : (total_weight / weight_per_crate) * cost_per_crate = 27 := by
  sorry

end shipping_cost_l188_188218


namespace fraction_cubed_equality_l188_188598

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end fraction_cubed_equality_l188_188598


namespace abs_inequality_solution_l188_188893

theorem abs_inequality_solution (x : ℝ) (h : |x - 4| ≤ 6) : -2 ≤ x ∧ x ≤ 10 := 
sorry

end abs_inequality_solution_l188_188893


namespace smallest_k_for_a_l188_188229

theorem smallest_k_for_a (a n : ℕ) (h : 10 ^ 2013 ≤ a^n ∧ a^n < 10 ^ 2014) : ∀ k : ℕ, k < 46 → ∃ n : ℕ, (10 ^ (k - 1)) ≤ a ∧ a < 10 ^ k :=
by sorry

end smallest_k_for_a_l188_188229


namespace train_speed_kmh_l188_188343

theorem train_speed_kmh (T P: ℝ) (L: ℝ):
  (T = L + 320) ∧ (L = 18 * P) ->
  P = 20 -> 
  P * 3.6 = 72 := 
by
  sorry

end train_speed_kmh_l188_188343


namespace f_2015_value_l188_188206

noncomputable def f : ℝ → ℝ := sorry -- Define f with appropriate conditions

theorem f_2015_value :
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) →
  f 2015 = -2 :=
by
  sorry -- Proof to be provided

end f_2015_value_l188_188206


namespace maximum_value_l188_188421

-- Define the variables as positive real numbers
variables (a b c : ℝ)

-- Define the conditions
def condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 2*a*b*c + 1

-- Define the expression
def expr (a b c : ℝ) : ℝ := (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b)

-- The theorem stating that under the given conditions, the expression has a maximum value of 1/8
theorem maximum_value : ∀ (a b c : ℝ), condition a b c → expr a b c ≤ 1/8 :=
by
  sorry

end maximum_value_l188_188421


namespace min_value_of_fraction_sum_l188_188811

theorem min_value_of_fraction_sum (a b : ℤ) (h1 : a = b + 1) : 
  (a > b) -> (∃ x, x > 0 ∧ ((a + b) / (a - b) + (a - b) / (a + b)) = 2) :=
by
  sorry

end min_value_of_fraction_sum_l188_188811


namespace total_students_suggestion_l188_188703

theorem total_students_suggestion :
  let m := 324
  let b := 374
  let t := 128
  m + b + t = 826 := by
  sorry

end total_students_suggestion_l188_188703


namespace remainder_div_9_l188_188099

theorem remainder_div_9 (x y : ℤ) (h : 9 ∣ (x + 2 * y)) : (2 * (5 * x - 8 * y - 4)) % 9 = -8 ∨ (2 * (5 * x - 8 * y - 4)) % 9 = 1 :=
by
  sorry

end remainder_div_9_l188_188099


namespace soccer_ball_cost_l188_188071

theorem soccer_ball_cost (x : ℕ) (h : 5 * x + 4 * 65 = 980) : x = 144 :=
by
  sorry

end soccer_ball_cost_l188_188071


namespace deriv_prob1_deriv_prob2_l188_188967

noncomputable def prob1 (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem deriv_prob1 : ∀ x, deriv prob1 x = -x * Real.sin x :=
by 
  sorry

noncomputable def prob2 (x : ℝ) : ℝ := x / (Real.exp x - 1)

theorem deriv_prob2 : ∀ x, x ≠ 0 → deriv prob2 x = (Real.exp x * (1 - x) - 1) / (Real.exp x - 1)^2 :=
by
  sorry

end deriv_prob1_deriv_prob2_l188_188967


namespace cost_to_make_each_pop_l188_188827

-- Define the conditions as given in step a)
def selling_price : ℝ := 1.50
def pops_sold : ℝ := 300
def pencil_cost : ℝ := 1.80
def pencils_to_buy : ℝ := 100

-- Define the total revenue from selling the ice-pops
def total_revenue : ℝ := pops_sold * selling_price

-- Define the total cost to buy the pencils
def total_pencil_cost : ℝ := pencils_to_buy * pencil_cost

-- Define the total profit
def total_profit : ℝ := total_revenue - total_pencil_cost

-- Define the cost to make each ice-pop
theorem cost_to_make_each_pop : total_profit / pops_sold = 0.90 :=
by
  sorry

end cost_to_make_each_pop_l188_188827


namespace find_n_for_2013_in_expansion_l188_188420

/-- Define the pattern for the last term of the expansion of n^3 -/
def last_term (n : ℕ) : ℕ :=
  n^2 + n - 1

/-- The main problem statement -/
theorem find_n_for_2013_in_expansion :
  ∃ n : ℕ, last_term (n - 1) ≤ 2013 ∧ 2013 < last_term n ∧ n = 45 :=
by
  sorry

end find_n_for_2013_in_expansion_l188_188420


namespace steve_fraction_of_skylar_l188_188736

variables (S : ℤ) (Stacy Skylar Steve : ℤ)

-- Given conditions
axiom h1 : 32 = 3 * Steve + 2 -- Stacy's berries = 2 + 3 * Steve's berries
axiom h2 : Skylar = 20        -- Skylar has 20 berries
axiom h3 : Stacy = 32         -- Stacy has 32 berries

-- Final goal
theorem steve_fraction_of_skylar (h1: 32 = 3 * Steve + 2) (h2: 20 = Skylar) (h3: Stacy = 32) :
  Steve = Skylar / 2 := 
sorry

end steve_fraction_of_skylar_l188_188736


namespace min_correct_answers_l188_188614

theorem min_correct_answers (x : ℕ) (hx : 10 * x - 5 * (30 - x) > 90) : x ≥ 17 :=
by {
  -- calculations and solution steps go here.
  sorry
}

end min_correct_answers_l188_188614


namespace distance_Bella_Galya_l188_188180

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end distance_Bella_Galya_l188_188180


namespace find_m_when_power_function_decreasing_l188_188436

theorem find_m_when_power_function_decreasing :
  ∃ m : ℝ, (m^2 - 2 * m - 2 = 1) ∧ (-4 * m - 2 < 0) ∧ (m = 3) :=
by
  sorry

end find_m_when_power_function_decreasing_l188_188436


namespace frog_arrangement_count_l188_188373

theorem frog_arrangement_count :
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let frogs := green_frogs + red_frogs + blue_frogs
  -- Descriptions:
  -- 1. green_frogs refuse to sit next to red_frogs
  -- 2. green_frogs and red_frogs are fine sitting next to blue_frogs
  -- 3. blue_frogs can sit next to each other
  frogs = 7 → 
  ∃ arrangements : ℕ, arrangements = 72 :=
by 
  sorry

end frog_arrangement_count_l188_188373


namespace find_abcd_abs_eq_one_l188_188578

noncomputable def non_zero_real (r : ℝ) := r ≠ 0

theorem find_abcd_abs_eq_one
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : a^2 + (1/b) = b^2 + (1/c) ∧ b^2 + (1/c) = c^2 + (1/d) ∧ c^2 + (1/d) = d^2 + (1/a)) :
  |a * b * c * d| = 1 :=
sorry

end find_abcd_abs_eq_one_l188_188578


namespace relationship_a_e_l188_188671

theorem relationship_a_e (a : ℝ) (h : 0 < a ∧ a < 1) : a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end relationship_a_e_l188_188671


namespace added_water_proof_l188_188803

variable (total_volume : ℕ) (milk_ratio water_ratio : ℕ) (added_water : ℕ)

theorem added_water_proof 
  (h1 : total_volume = 45) 
  (h2 : milk_ratio = 4) 
  (h3 : water_ratio = 1) 
  (h4 : added_water = 3) 
  (milk_volume : ℕ)
  (water_volume : ℕ)
  (h5 : milk_volume = (milk_ratio * total_volume) / (milk_ratio + water_ratio))
  (h6 : water_volume = (water_ratio * total_volume) / (milk_ratio + water_ratio))
  (new_ratio : ℕ)
  (h7 : new_ratio = milk_volume / (water_volume + added_water)) : added_water = 3 :=
by
  sorry

end added_water_proof_l188_188803


namespace ratio_value_l188_188765

theorem ratio_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
(h1 : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2)) 
(h2 : (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) :
  (x + 1) / (y + 1) = 2 :=
by
  sorry

end ratio_value_l188_188765


namespace minimum_area_of_Archimedean_triangle_l188_188306

-- Define the problem statement with necessary conditions
theorem minimum_area_of_Archimedean_triangle (p : ℝ) (hp : p > 0) :
  ∃ (ABQ_area : ℝ), ABQ_area = p^2 ∧ 
    (∀ (A B Q : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * p * A.1) ∧
      (B.2 ^ 2 = 2 * p * B.1) ∧
      (0, 0) = (p / 2, p / 2) ∧
      (Q.2 = 0) → 
      ABQ_area = p^2) :=
sorry

end minimum_area_of_Archimedean_triangle_l188_188306


namespace only_solution_l188_188950

theorem only_solution (a : ℤ) : 
  (∀ x : ℤ, x > 0 → 2 * x > 4 * x - 8 → 3 * x - a > -9 → x = 2) →
  (12 ≤ a ∧ a < 15) :=
by
  sorry

end only_solution_l188_188950


namespace part1_solution_set_part2_inequality_l188_188357

-- Part (1)
theorem part1_solution_set (x : ℝ) : |x| < 2 * x - 1 ↔ 1 < x := by
  sorry

-- Part (2)
theorem part2_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + 2 * b + c = 1) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 4 := by
  sorry

end part1_solution_set_part2_inequality_l188_188357


namespace sector_max_area_l188_188122

theorem sector_max_area (P : ℝ) (R l S : ℝ) :
  (P > 0) → (2 * R + l = P) → (S = 1/2 * R * l) →
  (R = P / 4) ∧ (S = P^2 / 16) :=
by
  sorry

end sector_max_area_l188_188122


namespace final_answer_is_correct_l188_188391

-- Define the chosen number
def chosen_number : ℤ := 1376

-- Define the division by 8
def division_result : ℤ := chosen_number / 8

-- Define the final answer
def final_answer : ℤ := division_result - 160

-- Theorem statement
theorem final_answer_is_correct : final_answer = 12 := by
  sorry

end final_answer_is_correct_l188_188391


namespace theorem_227_l188_188987

theorem theorem_227 (a b c d : ℤ) (k : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧
  (a - b ≡ a - c [ZMOD d]) ∧
  (a * b ≡ a * c [ZMOD d]) :=
by
  sorry

end theorem_227_l188_188987


namespace equilateral_triangle_ratio_l188_188581

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let perimeter := 3 * s
  let area := (s * s * Real.sqrt 3) / 4
  perimeter / area = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end equilateral_triangle_ratio_l188_188581


namespace range_of_t_l188_188551

noncomputable def f (a x : ℝ) : ℝ :=
  a / x - x + a * Real.log x

noncomputable def g (a x : ℝ) : ℝ :=
  f a x + 1/2 * x^2 - (a - 1) * x - a / x

theorem range_of_t (a x₁ x₂ t : ℝ) (h1 : f a x₁ = f a x₂) (h2 : x₁ + x₂ = a)
  (h3 : x₁ * x₂ = a) (h4 : a > 4) (h5 : g a x₁ + g a x₂ > t * (x₁ + x₂)) :
  t < Real.log 4 - 3 :=
  sorry

end range_of_t_l188_188551


namespace extreme_points_sum_gt_two_l188_188253

noncomputable def f (x : ℝ) (b : ℝ) := x^2 / 2 + b * Real.exp x
noncomputable def f_prime (x : ℝ) (b : ℝ) := x + b * Real.exp x

theorem extreme_points_sum_gt_two
  (b : ℝ)
  (h_b : -1 / Real.exp 1 < b ∧ b < 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : f_prime x₁ b = 0)
  (h_x₂ : f_prime x₂ b = 0)
  (h_x₁_lt_x₂ : x₁ < x₂) :
  x₁ + x₂ > 2 := by
  sorry

end extreme_points_sum_gt_two_l188_188253


namespace measure_of_angle_C_l188_188654

theorem measure_of_angle_C
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := 
sorry

end measure_of_angle_C_l188_188654


namespace packages_of_noodles_tom_needs_l188_188027

def beef_weight : ℕ := 10
def noodles_needed_factor : ℕ := 2
def noodles_available : ℕ := 4
def noodle_package_weight : ℕ := 2

theorem packages_of_noodles_tom_needs :
  (beef_weight * noodles_needed_factor - noodles_available) / noodle_package_weight = 8 :=
by
  sorry

end packages_of_noodles_tom_needs_l188_188027


namespace celestia_badges_l188_188702

theorem celestia_badges (H L C : ℕ) (total_badges : ℕ) (h1 : H = 14) (h2 : L = 17) (h3 : total_badges = 83) (h4 : H + L + C = total_badges) : C = 52 :=
by
  sorry

end celestia_badges_l188_188702


namespace total_combinations_l188_188960

def varieties_of_wrapping_paper : Nat := 10
def colors_of_ribbon : Nat := 4
def types_of_gift_cards : Nat := 5
def kinds_of_decorative_stickers : Nat := 2

theorem total_combinations : varieties_of_wrapping_paper * colors_of_ribbon * types_of_gift_cards * kinds_of_decorative_stickers = 400 := by
  sorry

end total_combinations_l188_188960


namespace domain_of_function_l188_188992

noncomputable def domain : Set ℝ := {x | x ≥ 1/2 ∧ x ≠ 1}

theorem domain_of_function : ∀ (x : ℝ), (2 * x - 1 ≥ 0) ∧ (x ^ 2 + x - 2 ≠ 0) ↔ (x ∈ domain) :=
by 
  sorry

end domain_of_function_l188_188992


namespace smallest_prime_divisor_of_sum_l188_188511

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l188_188511


namespace certain_number_division_l188_188817

theorem certain_number_division (N G : ℤ) : 
  G = 88 ∧ (∃ k : ℤ, N = G * k + 31) ∧ (∃ m : ℤ, 4521 = G * m + 33) → 
  N = 4519 := 
by
  sorry

end certain_number_division_l188_188817


namespace teams_face_each_other_l188_188256

theorem teams_face_each_other (n : ℕ) (total_games : ℕ) (k : ℕ)
  (h1 : n = 20)
  (h2 : total_games = 760)
  (h3 : total_games = n * (n - 1) * k / 2) :
  k = 4 :=
by
  sorry

end teams_face_each_other_l188_188256


namespace number_of_eggplant_packets_l188_188669

-- Defining the problem conditions in Lean 4
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def sunflower_packets := 6
def total_plants := 116

-- Our goal is to prove the number of eggplant seed packets Shyne bought
theorem number_of_eggplant_packets : ∃ E : ℕ, E * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants ∧ E = 4 :=
sorry

end number_of_eggplant_packets_l188_188669


namespace maximum_volume_of_pyramid_l188_188245

theorem maximum_volume_of_pyramid (a b : ℝ) (hb : b > 0) (ha : a > 0):
  ∃ V_max : ℝ, V_max = (a * (4 * b ^ 2 - a ^ 2)) / 12 := 
sorry

end maximum_volume_of_pyramid_l188_188245


namespace solve_polynomial_equation_l188_188534

theorem solve_polynomial_equation :
  ∃ z, (z^5 + 40 * z^3 + 80 * z - 32 = 0) →
  ∃ x, (x = z + 4) ∧ ((x - 2)^5 + (x - 6)^5 = 32) :=
by
  sorry

end solve_polynomial_equation_l188_188534


namespace black_spools_l188_188535

-- Define the given conditions
def spools_per_beret : ℕ := 3
def red_spools : ℕ := 12
def blue_spools : ℕ := 6
def berets_made : ℕ := 11

-- Define the statement to be proved using the defined conditions
theorem black_spools (spools_per_beret red_spools blue_spools berets_made : ℕ) : (spools_per_beret * berets_made) - (red_spools + blue_spools) = 15 :=
by sorry

end black_spools_l188_188535


namespace amgm_inequality_abcd_l188_188382

-- Define the variables and their conditions
variables {a b c d : ℝ}
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)
variable (hd : 0 < d)

-- State the theorem
theorem amgm_inequality_abcd :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end amgm_inequality_abcd_l188_188382


namespace trajectory_of_P_l188_188348

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (2 * x - 3) ^ 2 + 4 * y ^ 2 = 1

theorem trajectory_of_P (m n x y : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : 2 * x = 3 + m ∧ 2 * y = n) : trajectory_equation x y :=
by 
  sorry

end trajectory_of_P_l188_188348


namespace red_or_blue_probability_is_half_l188_188661

-- Define the number of each type of marble
def num_red_marbles : ℕ := 3
def num_blue_marbles : ℕ := 2
def num_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ := num_red_marbles + num_blue_marbles + num_yellow_marbles

-- Define the number of marbles that are either red or blue
def num_red_or_blue_marbles : ℕ := num_red_marbles + num_blue_marbles

-- Define the probability of drawing a red or blue marble
def probability_red_or_blue : ℚ := num_red_or_blue_marbles / total_marbles

-- Theorem stating the probability is 0.5
theorem red_or_blue_probability_is_half : probability_red_or_blue = 0.5 := by
  sorry

end red_or_blue_probability_is_half_l188_188661


namespace tan_theta_eq_neg_sqrt_3_l188_188951

theorem tan_theta_eq_neg_sqrt_3 (theta : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (Real.cos theta, Real.sin theta))
  (h_b : b = (Real.sqrt 3, 1))
  (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.tan theta = -Real.sqrt 3 :=
sorry

end tan_theta_eq_neg_sqrt_3_l188_188951


namespace smallest_integer_in_ratio_l188_188553

theorem smallest_integer_in_ratio (a b c : ℕ) 
    (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_sum : a + b + c = 100) 
    (h_ratio : c = 5 * a / 2 ∧ b = 3 * a / 2) : 
    a = 20 := 
by
  sorry

end smallest_integer_in_ratio_l188_188553


namespace matthew_ate_8_l188_188395

variable (M P A K : ℕ)

def kimberly_ate_5 : Prop := K = 5
def alvin_eggs : Prop := A = 2 * K - 1
def patrick_eggs : Prop := P = A / 2
def matthew_eggs : Prop := M = 2 * P

theorem matthew_ate_8 (M P A K : ℕ) (h1 : kimberly_ate_5 K) (h2 : alvin_eggs A K) (h3 : patrick_eggs P A) (h4 : matthew_eggs M P) : M = 8 := by
  sorry

end matthew_ate_8_l188_188395


namespace Jim_paycheck_correct_l188_188539

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end Jim_paycheck_correct_l188_188539


namespace johns_coin_collection_value_l188_188724

theorem johns_coin_collection_value :
  ∀ (n : ℕ) (value : ℕ), n = 24 → value = 20 → 
  ((n/3) * (value/8)) = 60 :=
by
  intro n value n_eq value_eq
  sorry

end johns_coin_collection_value_l188_188724


namespace solve_polynomial_relation_l188_188568

--Given Conditions
def polynomial_relation (x y : ℤ) : Prop := y^3 = x^3 + 8 * x^2 - 6 * x + 8 

--Proof Problem
theorem solve_polynomial_relation : ∃ (x y : ℤ), (polynomial_relation x y) ∧ 
  ((y = 11 ∧ x = 9) ∨ (y = 2 ∧ x = 0)) :=
by 
  sorry

end solve_polynomial_relation_l188_188568


namespace incorrect_directions_of_opening_l188_188789

-- Define the functions
def f (x : ℝ) : ℝ := 2 * (x - 3)^2
def g (x : ℝ) : ℝ := -2 * (x - 3)^2

-- The theorem (statement) to prove
theorem incorrect_directions_of_opening :
  ¬(∀ x, (f x > 0 ∧ g x > 0) ∨ (f x < 0 ∧ g x < 0)) :=
sorry

end incorrect_directions_of_opening_l188_188789


namespace coloring_problem_l188_188931

def condition (m n : ℕ) : Prop :=
  2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 ∧ m ≠ n ∧ m % n = 0

def color (f : ℕ → ℕ) : Prop :=
  ∀ m n, condition m n → f m ≠ f n

theorem coloring_problem :
  ∃ (k : ℕ) (f : ℕ → ℕ), (∀ n, 2 ≤ n ∧ n ≤ 31 → f n ≤ k) ∧ color f ∧ k = 4 :=
by
  sorry

end coloring_problem_l188_188931


namespace positive_difference_of_two_numbers_l188_188599

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l188_188599


namespace length_of_rooms_l188_188266

-- Definitions based on conditions
def width : ℕ := 18
def num_rooms : ℕ := 20
def total_area : ℕ := 6840

-- Theorem stating the length of the rooms
theorem length_of_rooms : (total_area / num_rooms) / width = 19 := by
  sorry

end length_of_rooms_l188_188266


namespace technician_round_trip_percentage_l188_188640

theorem technician_round_trip_percentage
  (D : ℝ) 
  (H1 : D > 0) -- Assume D is positive
  (H2 : true) -- The technician completes the drive to the center
  (H3 : true) -- The technician completes 20% of the drive from the center
  : (1.20 * D / (2 * D)) * 100 = 60 := 
by
  simp [H1, H2, H3]
  sorry

end technician_round_trip_percentage_l188_188640


namespace amount_for_second_shop_l188_188696

-- Definitions based on conditions
def books_from_first_shop : Nat := 65
def amount_first_shop : Float := 1160.0
def books_from_second_shop : Nat := 50
def avg_price_per_book : Float := 18.08695652173913
def total_books : Nat := books_from_first_shop + books_from_second_shop
def total_amount_spent : Float := avg_price_per_book * (total_books.toFloat)

-- The Lean statement to prove
theorem amount_for_second_shop : total_amount_spent - amount_first_shop = 920.0 := by
  sorry

end amount_for_second_shop_l188_188696


namespace parallelepiped_volume_k_l188_188108

theorem parallelepiped_volume_k (k : ℝ) : 
    abs (3 * k^2 - 13 * k + 27) = 20 ↔ k = (13 + Real.sqrt 85) / 6 ∨ k = (13 - Real.sqrt 85) / 6 := 
by sorry

end parallelepiped_volume_k_l188_188108


namespace initial_quantity_of_A_l188_188858

theorem initial_quantity_of_A (x : ℚ) 
    (h1 : 7 * x = a)
    (h2 : 5 * x = b)
    (h3 : a + b = 12 * x)
    (h4 : a' = a - (7 / 12) * 9)
    (h5 : b' = b - (5 / 12) * 9 + 9)
    (h6 : a' / b' = 7 / 9) : 
    a = 23.625 := 
sorry

end initial_quantity_of_A_l188_188858


namespace remainder_polynomial_l188_188274

theorem remainder_polynomial (n : ℕ) (hn : n ≥ 2) : 
  ∃ Q R : Polynomial ℤ, (R.degree < 2) ∧ (X^n = Q * (X^2 - 4 * X + 3) + R) ∧ 
                       (R = (Polynomial.C ((3^n - 1) / 2) * X + Polynomial.C ((3 - 3^n) / 2))) :=
by
  sorry

end remainder_polynomial_l188_188274


namespace min_value_of_M_l188_188427

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

noncomputable def M : ℝ :=
  (Real.rpow (a / (b + c)) (1 / 4)) + (Real.rpow (b / (c + a)) (1 / 4)) + (Real.rpow (c / (b + a)) (1 / 4)) +
  Real.sqrt ((b + c) / a) + Real.sqrt ((a + c) / b) + Real.sqrt ((a + b) / c)

theorem min_value_of_M : M a b c = 3 * Real.sqrt 2 + (3 * Real.rpow 8 (1 / 4)) / 2 := sorry

end min_value_of_M_l188_188427


namespace matrix_multiplication_l188_188047

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_multiplication :
  (A - B = A * B) →
  (A * B = ![![7, -2], ![4, -3]]) →
  (B * A = ![![6, -2], ![4, -4]]) :=
by
  intros h₁ h₂
  sorry

end matrix_multiplication_l188_188047


namespace sin_gt_cos_interval_l188_188236

theorem sin_gt_cos_interval (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x > Real.cos x) : 
  Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4) :=
by
  sorry

end sin_gt_cos_interval_l188_188236


namespace bricklayer_team_size_l188_188428

/-- Problem: Prove the number of bricklayers in the team -/
theorem bricklayer_team_size
  (x : ℕ)
  (h1 : 432 = (432 * (x - 4) / x) + 9 * (x - 4)) :
  x = 16 :=
sorry

end bricklayer_team_size_l188_188428


namespace problem_arithmetic_sequence_l188_188999

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) := a + d * (n - 1)

theorem problem_arithmetic_sequence (a d : ℝ) (h₁ : d < 0) (h₂ : (arithmetic_sequence a d 1)^2 = (arithmetic_sequence a d 9)^2):
  (arithmetic_sequence a d 5) = 0 :=
by
  -- This is where the proof would go
  sorry

end problem_arithmetic_sequence_l188_188999


namespace not_mutually_exclusive_option_D_l188_188262

-- Definitions for mutually exclusive events
def mutually_exclusive (event1 event2 : Prop) : Prop := ¬ (event1 ∧ event2)

-- Conditions as given in the problem
def eventA1 : Prop := True -- Placeholder for "score is greater than 8"
def eventA2 : Prop := True -- Placeholder for "score is less than 6"

def eventB1 : Prop := True -- Placeholder for "90 seeds germinate"
def eventB2 : Prop := True -- Placeholder for "80 seeds germinate"

def eventC1 : Prop := True -- Placeholder for "pass rate is higher than 70%"
def eventC2 : Prop := True -- Placeholder for "pass rate is 70%"

def eventD1 : Prop := True -- Placeholder for "average score is not lower than 90"
def eventD2 : Prop := True -- Placeholder for "average score is not higher than 120"

-- Lean proof statement
theorem not_mutually_exclusive_option_D :
  mutually_exclusive eventA1 eventA2 ∧
  mutually_exclusive eventB1 eventB2 ∧
  mutually_exclusive eventC1 eventC2 ∧
  ¬ mutually_exclusive eventD1 eventD2 :=
sorry

end not_mutually_exclusive_option_D_l188_188262


namespace possible_perimeters_l188_188804

-- Define the condition that the side lengths satisfy the equation
def sides_satisfy_eqn (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Theorem to prove the possible perimeters
theorem possible_perimeters (x y z : ℝ) (h1 : sides_satisfy_eqn x) (h2 : sides_satisfy_eqn y) (h3 : sides_satisfy_eqn z) :
  (x + y + z = 10) ∨ (x + y + z = 6) ∨ (x + y + z = 12) := by
  sorry

end possible_perimeters_l188_188804


namespace section_b_students_can_be_any_nonnegative_integer_l188_188741

def section_a_students := 36
def avg_weight_section_a := 30
def avg_weight_section_b := 30
def avg_weight_whole_class := 30

theorem section_b_students_can_be_any_nonnegative_integer (x : ℕ) :
  let total_weight_section_a := section_a_students * avg_weight_section_a
  let total_weight_section_b := x * avg_weight_section_b
  let total_weight_whole_class := (section_a_students + x) * avg_weight_whole_class
  (total_weight_section_a + total_weight_section_b = total_weight_whole_class) :=
by 
  sorry

end section_b_students_can_be_any_nonnegative_integer_l188_188741


namespace solve_for_n_l188_188013

-- Define the problem statement
theorem solve_for_n : ∃ n : ℕ, (3 * n^2 + n = 219) ∧ (n = 9) := 
sorry

end solve_for_n_l188_188013


namespace sector_angle_radian_measure_l188_188582

theorem sector_angle_radian_measure (r l : ℝ) (h1 : r = 1) (h2 : l = 2) : l / r = 2 := by
  sorry

end sector_angle_radian_measure_l188_188582


namespace haman_dropped_trays_l188_188756

def initial_trays_to_collect : ℕ := 10
def additional_trays : ℕ := 7
def eggs_sold : ℕ := 540
def eggs_per_tray : ℕ := 30

theorem haman_dropped_trays :
  ∃ dropped_trays : ℕ,
  (initial_trays_to_collect + additional_trays - dropped_trays)*eggs_per_tray = eggs_sold → dropped_trays = 8 :=
sorry

end haman_dropped_trays_l188_188756


namespace find_fourth_vertex_l188_188926

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  is_midpoint ({x := 0, y := -9}) A C ∧ is_midpoint ({x := 2, y := 6}) B D ∧
  is_midpoint ({x := 4, y := 5}) C D ∧ is_midpoint ({x := 0, y := -9}) A D

theorem find_fourth_vertex :
  ∃ D : Point,
    (is_parallelogram ({x := 0, y := -9}) ({x := 2, y := 6}) ({x := 4, y := 5}) D)
    ∧ ((D = {x := 2, y := -10}) ∨ (D = {x := -2, y := -8}) ∨ (D = {x := 6, y := 20})) :=
sorry

end find_fourth_vertex_l188_188926


namespace functions_eq_l188_188338

open Function

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem functions_eq (h_surj : Surjective f) (h_inj : Injective g) (h_ge : ∀ n : ℕ, f n ≥ g n) : ∀ n : ℕ, f n = g n :=
sorry

end functions_eq_l188_188338


namespace angle_coloring_min_colors_l188_188973

  theorem angle_coloring_min_colors (n : ℕ) : 
    (∃ c : ℕ, (c = 2 ↔ n % 2 = 0) ∧ (c = 3 ↔ n % 2 = 1)) :=
  by
    sorry
  
end angle_coloring_min_colors_l188_188973


namespace range_of_m_l188_188283

theorem range_of_m (m : ℝ) : 
  (¬(-2 ≤ 1 - (x - 1) / 3 ∧ (1 - (x - 1) / 3 ≤ 2)) → (∀ x, m > 0 → x^2 - 2*x + 1 - m^2 > 0)) → 
  (40 ≤ m ∧ m < 50) :=
by
  sorry

end range_of_m_l188_188283


namespace sum_of_coefficients_l188_188569

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (2 * x + 1)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 →
  a₀ = 1 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 :=
by
  intros h_expand h_a₀
  sorry

end sum_of_coefficients_l188_188569


namespace least_six_digit_divisible_by_198_l188_188920

/-- The least 6-digit natural number that is divisible by 198 is 100188. -/
theorem least_six_digit_divisible_by_198 : 
  ∃ n : ℕ, n ≥ 100000 ∧ n % 198 = 0 ∧ n = 100188 :=
by
  use 100188
  sorry

end least_six_digit_divisible_by_198_l188_188920


namespace double_counted_page_number_l188_188208

theorem double_counted_page_number (n x : ℕ) 
  (h1: 1 ≤ x ∧ x ≤ n)
  (h2: (n * (n + 1) / 2) + x = 1997) : 
  x = 44 := 
by
  sorry

end double_counted_page_number_l188_188208


namespace robis_savings_in_january_l188_188455

theorem robis_savings_in_january (x : ℕ) (h: (x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) = 126)) : x = 11 := 
by {
  -- By simplification, the lean equivalent proof would include combining like
  -- terms and solving the resulting equation. For now, we'll use sorry.
  sorry
}

end robis_savings_in_january_l188_188455


namespace spring_length_at_9kg_l188_188529

theorem spring_length_at_9kg :
  (∃ (k b : ℝ), (∀ x : ℝ, y = k * x + b) ∧ 
                 (y = 10 ∧ x = 0) ∧ 
                 (y = 10.5 ∧ x = 1)) → 
  (∀ x : ℝ, x = 9 → y = 14.5) :=
sorry

end spring_length_at_9kg_l188_188529


namespace solution1_solution2_l188_188668

-- Define the first problem
def equation1 (x : ℝ) : Prop :=
  (x + 1) / 3 - 1 = (x - 1) / 2

-- Prove that x = -1 is the solution to the first problem
theorem solution1 : equation1 (-1) := 
by 
  sorry

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  x - y = 1 ∧ 3 * x + y = 7

-- Prove that x = 2 and y = 1 are the solutions to the system of equations
theorem solution2 : system_of_equations 2 1 :=
by 
  sorry

end solution1_solution2_l188_188668


namespace sub_numbers_correct_l188_188649

theorem sub_numbers_correct : 
  (500.50 - 123.45 - 55 : ℝ) = 322.05 := by 
-- The proof can be filled in here
sorry

end sub_numbers_correct_l188_188649


namespace B_starts_cycling_after_A_l188_188537

theorem B_starts_cycling_after_A (t : ℝ) : 10 * t + 20 * (2 - t) = 60 → t = 2 :=
by
  intro h
  sorry

end B_starts_cycling_after_A_l188_188537


namespace small_disks_radius_l188_188882

theorem small_disks_radius (r : ℝ) (h : r > 0) :
  (2 * r ≥ 1 + r) → (r ≥ 1 / 2) := by
  intro hr
  linarith

end small_disks_radius_l188_188882


namespace school_starts_at_8_l188_188556

def minutes_to_time (minutes : ℕ) : ℕ × ℕ :=
  let hour := minutes / 60
  let minute := minutes % 60
  (hour, minute)

def add_minutes_to_time (h : ℕ) (m : ℕ) (added_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) + added_minutes)

def subtract_minutes_from_time (h : ℕ) (m : ℕ) (subtracted_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) - subtracted_minutes)

theorem school_starts_at_8 : True := by
  let normal_commute := 30
  let red_light_stops := 3 * 4
  let construction_delay := 10
  let total_additional_time := red_light_stops + construction_delay
  let total_commute_time := normal_commute + total_additional_time
  let depart_time := (7, 15)
  let arrival_time := add_minutes_to_time depart_time.1 depart_time.2 total_commute_time
  let start_time := subtract_minutes_from_time arrival_time.1 arrival_time.2 7

  have : start_time = (8, 0) := by
    sorry

  exact trivial

end school_starts_at_8_l188_188556


namespace expand_and_simplify_l188_188782

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := 
by
  sorry

end expand_and_simplify_l188_188782


namespace line_inclination_angle_l188_188532

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + (Real.sqrt 3) * y - 1 = 0

-- Define the condition of inclination angle in radians
def inclination_angle (θ : ℝ) : Prop := θ = Real.arctan (-1 / Real.sqrt 3) + Real.pi

-- The theorem to prove the inclination angle of the line
theorem line_inclination_angle (x y θ : ℝ) (h : line_eq x y) : inclination_angle θ :=
by
  sorry

end line_inclination_angle_l188_188532


namespace smallest_natural_number_l188_188846

theorem smallest_natural_number (a : ℕ) : 
  (∃ a, a % 3 = 0 ∧ (a - 1) % 4 = 0 ∧ (a - 2) % 5 = 0) → a = 57 :=
by
  sorry

end smallest_natural_number_l188_188846


namespace range_of_a_l188_188605

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x < a → x ^ 2 > 1 ∧ ¬(x ^ 2 > 1 → x < a)) : a ≤ -1 :=
sorry

end range_of_a_l188_188605


namespace comb_eq_comb_imp_n_eq_18_l188_188718

theorem comb_eq_comb_imp_n_eq_18 {n : ℕ} (h : Nat.choose n 14 = Nat.choose n 4) : n = 18 :=
sorry

end comb_eq_comb_imp_n_eq_18_l188_188718


namespace line_quadrant_conditions_l188_188580

theorem line_quadrant_conditions (k b : ℝ) 
  (H1 : ∃ x : ℝ, x > 0 ∧ k * x + b > 0)
  (H3 : ∃ x : ℝ, x < 0 ∧ k * x + b < 0)
  (H4 : ∃ x : ℝ, x > 0 ∧ k * x + b < 0) : k > 0 ∧ b < 0 :=
sorry

end line_quadrant_conditions_l188_188580


namespace union_set_when_m_neg3_range_of_m_for_intersection_l188_188074

def setA (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def setB (x m : ℝ) : Prop := 2*m - 1 ≤ x ∧ x ≤ m + 1

theorem union_set_when_m_neg3 : 
  (∀ x, setA x ∨ setB x (-3) ↔ -7 ≤ x ∧ x ≤ 4) := 
by sorry

theorem range_of_m_for_intersection :
  (∀ m x, (setA x ∧ setB x m ↔ setB x m) → m ≥ -1) := 
by sorry

end union_set_when_m_neg3_range_of_m_for_intersection_l188_188074


namespace movie_ticket_percentage_decrease_l188_188645

theorem movie_ticket_percentage_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100) 
  (h2 : new_price = 80) :
  ((old_price - new_price) / old_price) * 100 = 20 := 
by
  sorry

end movie_ticket_percentage_decrease_l188_188645


namespace cauchy_functional_eq_l188_188199

theorem cauchy_functional_eq
  (f : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end cauchy_functional_eq_l188_188199


namespace smallest_x_condition_l188_188787

theorem smallest_x_condition (x : ℕ) : (∃ x > 0, (3 * x + 28)^2 % 53 = 0) -> x = 26 := 
by
  sorry

end smallest_x_condition_l188_188787


namespace travel_west_l188_188936

-- Define the condition
def travel_east (d: ℝ) : ℝ := d

-- Define the distance for east
def east_distance := (travel_east 3 = 3)

-- The theorem to prove that traveling west for 2km should be -2km
theorem travel_west (d: ℝ) (h: east_distance) : travel_east (-d) = -d := 
by
  sorry

-- Applying this theorem to the specific case of 2km travel
example (h: east_distance): travel_east (-2) = -2 :=
by 
  apply travel_west 2 h

end travel_west_l188_188936


namespace fraction_of_marbles_taken_away_l188_188044

theorem fraction_of_marbles_taken_away (Chris_marbles Ryan_marbles remaining_marbles total_marbles taken_away_marbles : ℕ) 
    (hChris : Chris_marbles = 12) 
    (hRyan : Ryan_marbles = 28) 
    (hremaining : remaining_marbles = 20) 
    (htotal : total_marbles = Chris_marbles + Ryan_marbles) 
    (htaken_away : taken_away_marbles = total_marbles - remaining_marbles) : 
    (taken_away_marbles : ℚ) / total_marbles = 1 / 2 := 
by 
  sorry

end fraction_of_marbles_taken_away_l188_188044


namespace max_piece_length_total_pieces_l188_188154

-- Definitions based on the problem's conditions
def length1 : ℕ := 42
def length2 : ℕ := 63
def gcd_length : ℕ := Nat.gcd length1 length2

-- Theorem statements based on the realized correct answers
theorem max_piece_length (h1 : length1 = 42) (h2 : length2 = 63) :
  gcd_length = 21 := by
  sorry

theorem total_pieces (h1 : length1 = 42) (h2 : length2 = 63) :
  (length1 / gcd_length) + (length2 / gcd_length) = 5 := by
  sorry

end max_piece_length_total_pieces_l188_188154


namespace inequality_of_abc_l188_188425

variable {a b c : ℝ}

theorem inequality_of_abc 
    (h : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : abc * (a + b + c) = ab + bc + ca) :
    5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_of_abc_l188_188425


namespace C_share_of_profit_l188_188503

variable (A B C P Rs_36000 k : ℝ)

-- Definitions as per the conditions given in the problem statement.
def investment_A := 24000
def investment_B := 32000
def investment_C := 36000
def total_profit := 92000
def C_Share := 36000

-- The Lean statement without the proof as requested.
theorem C_share_of_profit 
  (h_A : investment_A = 24000)
  (h_B : investment_B = 32000)
  (h_C : investment_C = 36000)
  (h_P : total_profit = 92000)
  (h_C_share : C_Share = 36000)
  : C_Share = (investment_C / k) / ((investment_A / k) + (investment_B / k) + (investment_C / k)) * total_profit := 
sorry

end C_share_of_profit_l188_188503


namespace degree_of_g_l188_188330

noncomputable def poly_degree (p : Polynomial ℝ) : ℕ :=
  Polynomial.natDegree p

theorem degree_of_g
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ := f.comp g - g)
  (hf : poly_degree f = 3)
  (hh : poly_degree h = 8) :
  poly_degree g = 3 :=
sorry

end degree_of_g_l188_188330


namespace not_right_triangle_l188_188444

theorem not_right_triangle (a b c : ℝ) (h : a / b = 1 / 2 ∧ b / c = 2 / 3) :
  ¬(a^2 = b^2 + c^2) :=
by sorry

end not_right_triangle_l188_188444


namespace find_ratio_of_geometric_sequence_l188_188678

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  2 * a2 = a1 + a3

theorem find_ratio_of_geometric_sequence 
  {a : ℕ → ℝ} {q : ℝ}
  (h_pos : ∀ n, 0 < a n)
  (h_geo : geometric_sequence a q)
  (h_arith : arithmetic_sequence (a 1) ((1/2) * a 3) (2 * a 2)) :
  (a 10) / (a 8) = 3 + 2 * sqrt 2 :=
sorry

end find_ratio_of_geometric_sequence_l188_188678


namespace cost_price_correct_l188_188876

variables (sp : ℕ) (profitPerMeter : ℕ) (metersSold : ℕ)

def total_profit (profitPerMeter metersSold : ℕ) : ℕ := profitPerMeter * metersSold
def total_cost_price (sp total_profit : ℕ) : ℕ := sp - total_profit
def cost_price_per_meter (total_cost_price metersSold : ℕ) : ℕ := total_cost_price / metersSold

theorem cost_price_correct (h1 : sp = 8925) (h2 : profitPerMeter = 10) (h3 : metersSold = 85) :
  cost_price_per_meter (total_cost_price sp (total_profit profitPerMeter metersSold)) metersSold = 95 :=
by
  rw [h1, h2, h3];
  sorry

end cost_price_correct_l188_188876


namespace Vasya_numbers_l188_188731

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l188_188731


namespace percentage_of_50_l188_188281

theorem percentage_of_50 (P : ℝ) :
  (0.10 * 30) + (P / 100 * 50) = 10.5 → P = 15 := by
  sorry

end percentage_of_50_l188_188281


namespace value_of_quotient_l188_188903

variable (a b c d : ℕ)

theorem value_of_quotient 
  (h1 : a = 3 * b)
  (h2 : b = 2 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 15 :=
by
  sorry

end value_of_quotient_l188_188903


namespace zeros_in_square_of_nines_l188_188467

def num_zeros (n : ℕ) (m : ℕ) : ℕ :=
  -- Count the number of zeros in the decimal representation of m
sorry

theorem zeros_in_square_of_nines :
  num_zeros 6 ((10^6 - 1)^2) = 5 :=
sorry

end zeros_in_square_of_nines_l188_188467


namespace probability_of_all_female_l188_188956

noncomputable def probability_all_females_final (females males total chosen : ℕ) : ℚ :=
  (females.choose chosen) / (total.choose chosen)

theorem probability_of_all_female:
  probability_all_females_final 5 3 8 3 = 5 / 28 :=
by
  sorry

end probability_of_all_female_l188_188956


namespace total_votes_cast_is_8200_l188_188350

variable (V : ℝ) (h1 : 0.35 * V < V) (h2 : 0.35 * V + 2460 = 0.65 * V)

theorem total_votes_cast_is_8200 (V : ℝ)
  (h1 : 0.35 * V < V)
  (h2 : 0.35 * V + 2460 = 0.65 * V) :
  V = 8200 := by
sorry

end total_votes_cast_is_8200_l188_188350


namespace point_relationship_l188_188826

variable {m : ℝ}

theorem point_relationship
    (hA : ∃ y1 : ℝ, y1 = (-4 : ℝ)^2 - 2 * (-4 : ℝ) + m)
    (hB : ∃ y2 : ℝ, y2 = (0 : ℝ)^2 - 2 * (0 : ℝ) + m)
    (hC : ∃ y3 : ℝ, y3 = (3 : ℝ)^2 - 2 * (3 : ℝ) + m) :
    (∃ y2 y3 y1 : ℝ, y2 < y3 ∧ y3 < y1) := by
  sorry

end point_relationship_l188_188826


namespace smallest_X_l188_188566

theorem smallest_X (T : ℕ) (hT_digits : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) (hX_int : ∃ (X : ℕ), T = 20 * X) : ∃ T, ∀ X, X = T / 20 → X = 55 :=
by
  sorry

end smallest_X_l188_188566


namespace length_of_bridge_l188_188091

theorem length_of_bridge 
  (lenA : ℝ) (speedA : ℝ) (lenB : ℝ) (speedB : ℝ) (timeA : ℝ) (timeB : ℝ) (startAtSameTime : Prop)
  (h1 : lenA = 120) (h2 : speedA = 12.5) (h3 : lenB = 150) (h4 : speedB = 15.28) 
  (h5 : timeA = 30) (h6 : timeB = 25) : 
  (∃ X : ℝ, X = 757) :=
by
  sorry

end length_of_bridge_l188_188091


namespace total_length_of_sticks_l188_188538

-- Definitions based on conditions
def num_sticks := 30
def length_per_stick := 25
def overlap := 6
def effective_length_per_stick := length_per_stick - overlap

-- Theorem statement
theorem total_length_of_sticks : num_sticks * effective_length_per_stick - effective_length_per_stick + length_per_stick = 576 := sorry

end total_length_of_sticks_l188_188538


namespace curve_points_satisfy_equation_l188_188691

theorem curve_points_satisfy_equation (C : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C → f p = 0) → (∀ q : ℝ × ℝ, f q ≠ 0 → q ∉ C) :=
by
  intro h₁
  intro q
  intro h₂
  sorry

end curve_points_satisfy_equation_l188_188691


namespace polynomial_divisor_l188_188657

theorem polynomial_divisor (f : Polynomial ℂ) (n : ℕ) (h : (X - 1) ∣ (f.comp (X ^ n))) : (X ^ n - 1) ∣ (f.comp (X ^ n)) :=
sorry

end polynomial_divisor_l188_188657


namespace part_a_least_moves_part_b_least_moves_l188_188711

def initial_position : Nat := 0
def total_combinations : Nat := 10^6
def excluded_combinations : List Nat := [0, 10^5, 2 * 10^5, 3 * 10^5, 4 * 10^5, 5 * 10^5, 6 * 10^5, 7 * 10^5, 8 * 10^5, 9 * 10^5]

theorem part_a_least_moves : total_combinations - 1 = 10^6 - 1 := by
  simp [total_combinations, Nat.pow]

theorem part_b_least_moves : total_combinations - excluded_combinations.length = 10^6 - 10 := by
  simp [total_combinations, excluded_combinations, Nat.pow, List.length]

end part_a_least_moves_part_b_least_moves_l188_188711


namespace system_equations_solution_l188_188786

theorem system_equations_solution (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 3) ∧ 
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) ∧ 
  (1 / (x * y * z) = 1) → 
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end system_equations_solution_l188_188786


namespace range_of_f_when_a_eq_2_max_value_implies_a_l188_188548

-- first part
theorem range_of_f_when_a_eq_2 (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 3) :
  (∀ y, (y = x^2 + 3*x - 3) → (y ≥ -21/4 ∧ y ≤ 15)) :=
by sorry

-- second part
theorem max_value_implies_a (a : ℝ) (hx : ∀ x, -1 ≤ x ∧ x ≤ 3 → x^2 + (2*a - 1)*x - 3 ≤ 1) :
  a = -1 ∨ a = -1 / 3 :=
by sorry

end range_of_f_when_a_eq_2_max_value_implies_a_l188_188548


namespace basketball_problem_l188_188586

theorem basketball_problem :
  ∃ x y : ℕ, (3 + x + y = 14) ∧ (3 * 3 + 2 * x + y = 28) ∧ (x = 8) ∧ (y = 3) :=
by
  sorry

end basketball_problem_l188_188586


namespace option_d_correct_l188_188422

theorem option_d_correct (a b : ℝ) (h : a > b) : -b > -a :=
sorry

end option_d_correct_l188_188422


namespace find_num_round_balloons_l188_188082

variable (R : ℕ) -- Number of bags of round balloons that Janeth bought
variable (RoundBalloonsPerBag : ℕ := 20)
variable (LongBalloonsPerBag : ℕ := 30)
variable (BagsLongBalloons : ℕ := 4)
variable (BurstRoundBalloons : ℕ := 5)
variable (BalloonsLeft : ℕ := 215)

def total_long_balloons : ℕ := BagsLongBalloons * LongBalloonsPerBag
def total_balloons : ℕ := R * RoundBalloonsPerBag + total_long_balloons - BurstRoundBalloons

theorem find_num_round_balloons :
  BalloonsLeft = total_balloons → R = 5 := by
  sorry

end find_num_round_balloons_l188_188082


namespace sum_pos_implies_one_pos_l188_188003

theorem sum_pos_implies_one_pos (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 := 
sorry

end sum_pos_implies_one_pos_l188_188003


namespace subtract_correctly_l188_188116

theorem subtract_correctly (x : ℕ) (h : x + 35 = 77) : x - 35 = 7 :=
sorry

end subtract_correctly_l188_188116


namespace gain_percent_l188_188943

theorem gain_percent (CP SP : ℝ) (hCP : CP = 20) (hSP : SP = 35) : 
  (SP - CP) / CP * 100 = 75 :=
by
  rw [hCP, hSP]
  sorry

end gain_percent_l188_188943


namespace reciprocal_roots_k_value_l188_188141

theorem reciprocal_roots_k_value :
  ∀ k : ℝ, (∀ r : ℝ, 5.2 * r^2 + 14.3 * r + k = 0 ∧ 5.2 * (1 / r)^2 + 14.3 * (1 / r) + k = 0) →
          k = 5.2 :=
by
  sorry

end reciprocal_roots_k_value_l188_188141


namespace family_ages_l188_188200

theorem family_ages:
  (∀ (Peter Harriet Jane Emily father: ℕ),
  ((Peter + 12 = 2 * (Harriet + 12)) ∧
   (Jane = Emily + 10) ∧
   (Peter = 60 / 3) ∧
   (Peter = Jane + 5) ∧
   (Aunt_Lucy = 52) ∧
   (Aunt_Lucy = 4 + Peter_Jane_mother) ∧
   (father - 20 = Aunt_Lucy)) →
  (Harriet = 4) ∧ (Peter = 20) ∧ (Jane = 15) ∧ (Emily = 5) ∧ (father = 72)) :=
sorry

end family_ages_l188_188200


namespace translation_is_elevator_l188_188706

-- Definitions representing the conditions
def P_A : Prop := true  -- The movement of elevators constitutes translation.
def P_B : Prop := false -- Swinging on a swing does not constitute translation.
def P_C : Prop := false -- Closing an open textbook does not constitute translation.
def P_D : Prop := false -- The swinging of a pendulum does not constitute translation.

-- The goal is to prove that Option A is the phenomenon that belongs to translation
theorem translation_is_elevator : P_A ∧ ¬P_B ∧ ¬P_C ∧ ¬P_D :=
by
  sorry -- proof not required

end translation_is_elevator_l188_188706


namespace find_annual_interest_rate_l188_188352

-- Define the given conditions
def principal : ℝ := 10000
def time : ℝ := 1  -- since 12 months is 1 year for annual rate
def simple_interest : ℝ := 800

-- Define the annual interest rate to be proved
def annual_interest_rate : ℝ := 0.08

-- The theorem stating the problem
theorem find_annual_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) : 
  P = principal → 
  T = time → 
  SI = simple_interest → 
  SI = P * annual_interest_rate * T := 
by
  intros hP hT hSI
  rw [hP, hT, hSI]
  unfold annual_interest_rate
  -- here's where we skip the proof
  sorry

end find_annual_interest_rate_l188_188352


namespace Cherie_boxes_l188_188184

theorem Cherie_boxes (x : ℕ) :
  (2 * 8 + x * (8 + 9) = 33) → x = 1 :=
by
  intros h
  have h_eq : 16 + 17 * x = 33 := by simp [mul_add, mul_comm, h]
  linarith

end Cherie_boxes_l188_188184


namespace lowest_value_meter_can_record_l188_188070

theorem lowest_value_meter_can_record (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 6) (h2 : A = 2) : A = 2 :=
by sorry

end lowest_value_meter_can_record_l188_188070


namespace sarah_saves_5_dollars_l188_188406

noncomputable def price_per_pair : ℕ := 40

noncomputable def promotion_A_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n / 2 else price_per_pair

noncomputable def promotion_B_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n - (15 * (n / 2)) else price_per_pair

noncomputable def total_price_promotion_A : ℕ :=
price_per_pair + (price_per_pair / 2)

noncomputable def total_price_promotion_B : ℕ :=
price_per_pair + (price_per_pair - 15)

theorem sarah_saves_5_dollars : total_price_promotion_B - total_price_promotion_A = 5 :=
by
  rw [total_price_promotion_B, total_price_promotion_A]
  norm_num
  sorry

end sarah_saves_5_dollars_l188_188406


namespace largest_B_at_45_l188_188848

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def B (k : ℕ) : ℝ :=
  if k ≤ 500 then (binomial_coeff 500 k) * (0.1)^k else 0

theorem largest_B_at_45 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 500 → B k ≤ B 45 :=
by
  intros k hk
  sorry

end largest_B_at_45_l188_188848


namespace vacant_seats_calculation_l188_188093

noncomputable def seats_vacant (total_seats : ℕ) (percentage_filled : ℚ) : ℚ := 
  total_seats * (1 - percentage_filled)

theorem vacant_seats_calculation: 
  seats_vacant 600 0.45 = 330 := 
by 
    -- sorry to skip the proof.
    sorry

end vacant_seats_calculation_l188_188093


namespace distance_between_chords_l188_188446

theorem distance_between_chords (R : ℝ) (AB CD : ℝ) (d : ℝ) : 
  R = 25 → AB = 14 → CD = 40 → (d = 39 ∨ d = 9) :=
by intros; sorry

end distance_between_chords_l188_188446


namespace area_triangle_ABC_correct_l188_188977

noncomputable def rectangle_area : ℝ := 42

noncomputable def area_triangle_outside_I : ℝ := 9
noncomputable def area_triangle_outside_II : ℝ := 3.5
noncomputable def area_triangle_outside_III : ℝ := 12

noncomputable def area_triangle_ABC : ℝ :=
  rectangle_area - (area_triangle_outside_I + area_triangle_outside_II + area_triangle_outside_III)

theorem area_triangle_ABC_correct : area_triangle_ABC = 17.5 := by 
  sorry

end area_triangle_ABC_correct_l188_188977


namespace new_average_doubled_l188_188769

theorem new_average_doubled (n : ℕ) (avg : ℝ) (h1 : n = 12) (h2 : avg = 50) :
  2 * avg = 100 := by
sorry

end new_average_doubled_l188_188769


namespace geometric_sequence_S6_l188_188018

variable (a : ℕ → ℝ) -- represents the geometric sequence

noncomputable def S (n : ℕ) : ℝ :=
if n = 0 then 0 else ((a 0) * (1 - (a 1 / a 0) ^ n)) / (1 - a 1 / a 0)

theorem geometric_sequence_S6 (h : ∀ n, a n = (a 0) * (a 1 / a 0) ^ n) :
  S a 2 = 6 ∧ S a 4 = 18 → S a 6 = 42 := 
by 
  intros h1
  sorry

end geometric_sequence_S6_l188_188018


namespace measure_of_B_l188_188065

-- Define the conditions (angles and their relationships)
variable (angle_P angle_R angle_O angle_B angle_L angle_S : ℝ)
variable (sum_of_angles : angle_P + angle_R + angle_O + angle_B + angle_L + angle_S = 720)
variable (supplementary_O_S : angle_O + angle_S = 180)
variable (right_angle_L : angle_L = 90)
variable (congruent_angles : angle_P = angle_R ∧ angle_R = angle_B)

-- Prove the measure of angle B
theorem measure_of_B : angle_B = 150 := by
  sorry

end measure_of_B_l188_188065


namespace geometric_sequence_k_value_l188_188131

theorem geometric_sequence_k_value
  (k : ℤ)
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h1 : ∀ n, S n = 3 * 2^n + k)
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1))
  (h3 : ∃ r, ∀ n, a (n + 1) = r * a n) : k = -3 :=
sorry

end geometric_sequence_k_value_l188_188131


namespace olivia_earning_l188_188287

theorem olivia_earning
  (cost_per_bar : ℝ)
  (total_bars : ℕ)
  (unsold_bars : ℕ)
  (sold_bars : ℕ := total_bars - unsold_bars)
  (earnings : ℝ := sold_bars * cost_per_bar) :
  cost_per_bar = 3 → total_bars = 7 → unsold_bars = 4 → earnings = 9 :=
by
  sorry

end olivia_earning_l188_188287


namespace factorize_expression_l188_188838

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l188_188838


namespace find_A_minus_B_l188_188525

variables (A B : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + B = 814.8
def condition2 : Prop := B = A / 10

-- Statement to prove
theorem find_A_minus_B (h1 : condition1 A B) (h2 : condition2 A B) : A - B = 611.1 :=
sorry

end find_A_minus_B_l188_188525


namespace find_f_one_l188_188408

-- Define the function f(x-3) = 2x^2 - 3x + 1
noncomputable def f (x : ℤ) := 2 * (x+3)^2 - 3 * (x+3) + 1

-- Declare the theorem we intend to prove
theorem find_f_one : f 1 = 21 :=
by
  -- The proof goes here (saying "sorry" because the detailed proof is skipped)
  sorry

end find_f_one_l188_188408


namespace function_solution_l188_188791

theorem function_solution (f : ℝ → ℝ) (α : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + α * x) :=
by
  sorry

end function_solution_l188_188791


namespace A_profit_share_l188_188682

theorem A_profit_share (A_shares : ℚ) (B_shares : ℚ) (C_shares : ℚ) (D_shares : ℚ) (total_profit : ℚ) (A_profit : ℚ) :
  A_shares = 1/3 → B_shares = 1/4 → C_shares = 1/5 → 
  D_shares = 1 - (A_shares + B_shares + C_shares) → total_profit = 2445 → A_profit = 815 →
  A_shares * total_profit = A_profit :=
by sorry

end A_profit_share_l188_188682


namespace sum_reciprocals_geom_seq_l188_188567

theorem sum_reciprocals_geom_seq (a₁ q : ℝ) (h_pos_a₁ : 0 < a₁) (h_pos_q : 0 < q)
    (h_sum : a₁ + a₁ * q + a₁ * q^2 + a₁ * q^3 = 9)
    (h_prod : a₁^4 * q^6 = 81 / 4) :
    (1 / a₁) + (1 / (a₁ * q)) + (1 / (a₁ * q^2)) + (1 / (a₁ * q^3)) = 2 :=
by
  sorry

end sum_reciprocals_geom_seq_l188_188567


namespace number_of_panes_l188_188389

theorem number_of_panes (length width total_area : ℕ) (h_length : length = 12) (h_width : width = 8) (h_total_area : total_area = 768) :
  total_area / (length * width) = 8 :=
by
  sorry

end number_of_panes_l188_188389


namespace probability_same_color_ball_draw_l188_188054

theorem probability_same_color_ball_draw (red white : ℕ) 
    (h_red : red = 2) (h_white : white = 2) : 
    let total_outcomes := (red + white) * (red + white)
    let same_color_outcomes := 2 * (red * red + white * white)
    same_color_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_same_color_ball_draw_l188_188054


namespace flower_count_l188_188106

theorem flower_count (R L T : ℕ) (h1 : R = L + 22) (h2 : R = T - 20) (h3 : L + R + T = 100) : R = 34 :=
by
  sorry

end flower_count_l188_188106


namespace n_in_S_implies_n2_in_S_l188_188475

def S (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a ≥ b ∧ c ≥ d ∧ e ≥ f ∧
  n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2

theorem n_in_S_implies_n2_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end n_in_S_implies_n2_in_S_l188_188475


namespace find_f_2_l188_188983

theorem find_f_2 (f : ℝ → ℝ) (h : ∀ x, f (1 / x + 1) = 2 * x + 3) : f 2 = 5 :=
by
  sorry

end find_f_2_l188_188983


namespace boys_from_Pine_l188_188349

/-
We need to prove that the number of boys from Pine Middle School is 70
given the following conditions:
1. There were 150 students in total.
2. 90 were boys and 60 were girls.
3. 50 students were from Maple Middle School.
4. 100 students were from Pine Middle School.
5. 30 of the girls were from Maple Middle School.
-/
theorem boys_from_Pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h_total : total_students = 150) (h_boys : total_boys = 90)
  (h_girls : total_girls = 60) (h_maple : maple_students = 50)
  (h_pine : pine_students = 100) (h_maple_girls : maple_girls = 30) :
  total_boys - maple_students + maple_girls = 70 :=
by
  sorry

end boys_from_Pine_l188_188349


namespace Intersection_A_B_l188_188351

open Set

theorem Intersection_A_B :
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  show A ∩ B = {x : ℝ | -3 < x ∧ x < 1}
  sorry

end Intersection_A_B_l188_188351


namespace profit_percentage_B_l188_188147

-- Definitions based on conditions:
def CP_A : ℝ := 150  -- Cost price for A
def profit_percentage_A : ℝ := 0.20  -- Profit percentage for A
def SP_C : ℝ := 225  -- Selling price for C

-- Lean statement for the problem:
theorem profit_percentage_B : (SP_C - (CP_A * (1 + profit_percentage_A))) / (CP_A * (1 + profit_percentage_A)) * 100 = 25 := 
by 
  sorry

end profit_percentage_B_l188_188147


namespace range_of_a_l188_188799

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) / (x + 2)

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, -2 < x → -2 < y → x < y → f a x < f a y) → (a > 1/2) :=
by
  sorry

end range_of_a_l188_188799


namespace tetrahedron_dihedral_face_areas_l188_188683

variables {S₁ S₂ a b : ℝ} {α φ : ℝ}

theorem tetrahedron_dihedral_face_areas :
  S₁^2 + S₂^2 - 2 * S₁ * S₂ * Real.cos α = (a * b * Real.sin φ / 4)^2 :=
sorry

end tetrahedron_dihedral_face_areas_l188_188683


namespace surface_area_with_holes_l188_188829

-- Define the cube and holes properties
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def number_faces_cube : ℕ := 6

-- Define areas
def area_face_cube := edge_length_cube ^ 2
def area_face_hole := side_length_hole ^ 2
def original_surface_area := number_faces_cube * area_face_cube
def total_hole_area := number_faces_cube * area_face_hole
def new_exposed_area := number_faces_cube * 4 * area_face_hole

-- Calculate the total surface area including holes
def total_surface_area := original_surface_area - total_hole_area + new_exposed_area

-- Lean statement for the proof
theorem surface_area_with_holes :
  total_surface_area = 168 := by
  sorry

end surface_area_with_holes_l188_188829


namespace circle_equation_unique_l188_188284

theorem circle_equation_unique {F D E : ℝ} : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 4 ∧ y = 2) → x^2 + y^2 + D * x + E * y + F = 0) → 
  (x^2 + y^2 - 8 * x + 6 * y = 0) :=
by 
  sorry

end circle_equation_unique_l188_188284


namespace cost_price_eq_560_l188_188469

variables (C SP1 SP2 : ℝ)
variables (h1 : SP1 = 0.79 * C) (h2 : SP2 = SP1 + 140) (h3 : SP2 = 1.04 * C)

theorem cost_price_eq_560 : C = 560 :=
by 
  sorry

end cost_price_eq_560_l188_188469


namespace f_divisible_by_k2_k1_l188_188823

noncomputable def f (n : ℕ) (x : ℤ) : ℤ :=
  x^(n + 2) + (x + 1)^(2 * n + 1)

theorem f_divisible_by_k2_k1 (n : ℕ) (k : ℤ) (hn : n > 0) : 
  ((k^2 + k + 1) ∣ f n k) :=
sorry

end f_divisible_by_k2_k1_l188_188823


namespace pow_mod_79_l188_188917

theorem pow_mod_79 (a : ℕ) (h : a = 7) : a^79 % 11 = 6 := by
  sorry

end pow_mod_79_l188_188917


namespace numbers_composite_l188_188923

theorem numbers_composite (a b c d : ℕ) (h : a * b = c * d) : ∃ x y : ℕ, (x > 1 ∧ y > 1) ∧ a^2000 + b^2000 + c^2000 + d^2000 = x * y := 
sorry

end numbers_composite_l188_188923


namespace triangle_area_correct_l188_188112

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_correct :
  let A : point := (3, -1)
  let B : point := (3, 6)
  let C : point := (8, 6)
  triangle_area A B C = 17.5 :=
by
  sorry

end triangle_area_correct_l188_188112


namespace temperature_difference_l188_188764

theorem temperature_difference (t_low t_high : ℝ) (h_low : t_low = -2) (h_high : t_high = 5) :
  t_high - t_low = 7 :=
by
  rw [h_low, h_high]
  norm_num

end temperature_difference_l188_188764


namespace sum_of_all_potential_real_values_of_x_l188_188309

/-- Determine the sum of all potential real values of x such that when the mean, median, 
and mode of the list [12, 3, 6, 3, 8, 3, x, 15] are arranged in increasing order, they 
form a non-constant arithmetic progression. -/
def sum_potential_x_values : ℚ :=
    let values := [12, 3, 6, 3, 8, 3, 15]
    let mean (x : ℚ) : ℚ := (50 + x) / 8
    let mode : ℚ := 3
    let median (x : ℚ) : ℚ := 
      if x ≤ 3 then 3.5 else if x < 6 then (x + 6) / 2 else 6
    let is_arithmetic_seq (a b c : ℚ) : Prop := 2 * b = a + c
    let valid_x_values : List ℚ := 
      (if is_arithmetic_seq mode 3.5 (mean (3.5)) then [] else []) ++
      (if is_arithmetic_seq mode 6 (mean 6) then [22] else []) ++
      (if is_arithmetic_seq mode (median (50 / 7)) (mean (50 / 7)) then [50 / 7] else [])
    (valid_x_values.sum)
theorem sum_of_all_potential_real_values_of_x :
  sum_potential_x_values = 204 / 7 :=
  sorry

end sum_of_all_potential_real_values_of_x_l188_188309


namespace university_cost_per_box_l188_188100

def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def num_boxes (total_volume box_volume : ℕ) : ℕ :=
  total_volume / box_volume

def cost_per_box (total_cost num_boxes : ℚ) : ℚ :=
  total_cost / num_boxes

theorem university_cost_per_box :
  let length := 20
  let width := 20
  let height := 15
  let total_volume := 3060000
  let total_cost := 459
  let box_vol := box_volume length width height
  let boxes := num_boxes total_volume box_vol
  cost_per_box total_cost boxes = 0.90 :=
by
  sorry

end university_cost_per_box_l188_188100


namespace steve_matching_pairs_l188_188372

/-- Steve's total number of socks -/
def total_socks : ℕ := 25

/-- Number of Steve's mismatching socks -/
def mismatching_socks : ℕ := 17

/-- Number of Steve's matching socks -/
def matching_socks : ℕ := total_socks - mismatching_socks

/-- Number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := matching_socks / 2

/-- Proof that Steve has 4 pairs of matching socks -/
theorem steve_matching_pairs : matching_pairs = 4 := by
  sorry

end steve_matching_pairs_l188_188372


namespace find_f_of_500_l188_188339

theorem find_f_of_500
  (f : ℕ → ℕ)
  (h_pos : ∀ x y : ℕ, f x > 0 ∧ f y > 0) 
  (h_mul : ∀ x y : ℕ, f (x * y) = f x + f y) 
  (h_f10 : f 10 = 15)
  (h_f40 : f 40 = 23) :
  f 500 = 41 :=
sorry

end find_f_of_500_l188_188339


namespace domain_of_function_l188_188922

def quadratic_inequality (x : ℝ) : Prop := -8 * x^2 - 14 * x + 9 ≥ 0

theorem domain_of_function :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 9 / 8} :=
by
  -- The detailed proof would go here, but we're focusing on the statement structure.
  sorry

end domain_of_function_l188_188922


namespace problem_solution_1_problem_solution_2_l188_188742

def Sn (n : ℕ) := n * (n + 2)

def a_n (n : ℕ) := 2 * n + 1

def b_n (n : ℕ) := 2 ^ (n - 1)

def c_n (n : ℕ) := if n % 2 = 1 then 2 / Sn n else b_n n

def T_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i => c_n (i + 1))

theorem problem_solution_1 : 
  ∀ (n : ℕ), a_n n = 2 * n + 1 ∧ b_n n = 2 ^ (n - 1) := 
  by sorry

theorem problem_solution_2 (n : ℕ) : 
  T_n (2 * n) = (2 * n) / (2 * n + 1) + (2 / 3) * (4 ^ n - 1) := 
  by sorry

end problem_solution_1_problem_solution_2_l188_188742


namespace quadratic_coefficients_sum_l188_188814

-- Definition of the quadratic function and the conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Conditions
def vertexCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 2 = 3
  
def pointCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 3 = 2

-- The theorem to prove
theorem quadratic_coefficients_sum (a b c : ℝ)
  (hv : vertexCondition a b c)
  (hp : pointCondition a b c):
  a + b + 2 * c = 2 :=
sorry

end quadratic_coefficients_sum_l188_188814


namespace pyramid_height_l188_188297

noncomputable def height_of_pyramid : ℝ :=
  let perimeter := 32
  let pb := 12
  let side := perimeter / 4
  let fb := (side * Real.sqrt 2) / 2
  Real.sqrt (pb^2 - fb^2)

theorem pyramid_height :
  height_of_pyramid = 4 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_l188_188297


namespace isosceles_triangle_relationship_l188_188974

theorem isosceles_triangle_relationship (x y : ℝ) (h1 : 2 * x + y = 30) (h2 : 7.5 < x) (h3 : x < 15) : 
  y = 30 - 2 * x :=
  by sorry

end isosceles_triangle_relationship_l188_188974


namespace joan_kittens_count_correct_l188_188914

def joan_initial_kittens : Nat := 8
def kittens_from_friends : Nat := 2
def joan_total_kittens (initial: Nat) (added: Nat) : Nat := initial + added

theorem joan_kittens_count_correct : joan_total_kittens joan_initial_kittens kittens_from_friends = 10 := 
by
  sorry

end joan_kittens_count_correct_l188_188914


namespace product_divisible_by_third_l188_188991

theorem product_divisible_by_third (a b c : Int)
    (h1 : (a + b + c)^2 = -(a * b + a * c + b * c))
    (h2 : a + b ≠ 0) (h3 : b + c ≠ 0) (h4 : a + c ≠ 0) :
    ((a + b) * (a + c) % (b + c) = 0) ∧ ((a + b) * (b + c) % (a + c) = 0) ∧ ((a + c) * (b + c) % (a + b) = 0) :=
  sorry

end product_divisible_by_third_l188_188991


namespace no_solution_for_inequality_system_l188_188342

theorem no_solution_for_inequality_system (x : ℝ) : 
  ¬ ((2 * x + 3 ≥ x + 11) ∧ (((2 * x + 5) / 3 - 1) < (2 - x))) :=
by
  sorry

end no_solution_for_inequality_system_l188_188342


namespace co_presidents_included_probability_l188_188132

-- Let the number of students in each club
def club_sizes : List ℕ := [6, 8, 9, 10]

-- Function to calculate binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Function to calculate probability for a given club size
noncomputable def co_president_probability (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4)

-- List of probabilities for each club
noncomputable def probabilities : List ℚ :=
  List.map co_president_probability club_sizes

-- Aggregate total probability by averaging the individual probabilities
noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * probabilities.sum

-- The proof problem: proving the total probability equals 119/700
theorem co_presidents_included_probability :
  total_probability = 119 / 700 := by
  sorry

end co_presidents_included_probability_l188_188132


namespace triangle_side_length_l188_188634

theorem triangle_side_length (a b c : ℝ) (B : ℝ) (ha : a = 2) (hB : B = 60) (hc : c = 3) :
  b = Real.sqrt 7 :=
by
  sorry

end triangle_side_length_l188_188634


namespace unique_combination_of_segments_l188_188195

theorem unique_combination_of_segments :
  ∃! (x y : ℤ), 7 * x + 12 * y = 100 := sorry

end unique_combination_of_segments_l188_188195


namespace algebraic_expression_value_l188_188042

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023 * a - 1 = 0) : 
  a * (a + 1) * (a - 1) + 2023 * a^2 + 1 = 1 :=
by
  sorry

end algebraic_expression_value_l188_188042


namespace minimum_value_S_l188_188971

noncomputable def S (x a : ℝ) : ℝ := (x - a)^2 + (Real.log x - a)^2

theorem minimum_value_S : ∃ x a : ℝ, x > 0 ∧ (S x a = 1 / 2) := by
  sorry

end minimum_value_S_l188_188971


namespace solve_equation_nat_numbers_l188_188384

theorem solve_equation_nat_numbers (a b c d e f g : ℕ) 
  (h : a * b * c * d * e * f * g = a + b + c + d + e + f + g) : 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 2 ∧ g = 7) ∨ (f = 7 ∧ g = 2))) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 3 ∧ g = 4) ∨ (f = 4 ∧ g = 3))) :=
sorry

end solve_equation_nat_numbers_l188_188384


namespace find_fraction_l188_188255

theorem find_fraction {a b : ℕ} 
  (h1 : 32016 + (a / b) = 2016 * 3 + (a / b)) 
  (ha : a = 2016) 
  (hb : b = 2016^3 - 1) : 
  (b + 1) / a^2 = 2016 := 
by 
  sorry

end find_fraction_l188_188255


namespace find_n_l188_188482

theorem find_n (n : ℕ) (h : n * Nat.factorial n + Nat.factorial n = 5040) : n = 6 :=
sorry

end find_n_l188_188482


namespace quadratic_equation_proof_l188_188393

def is_quadratic_equation (eqn : String) : Prop :=
  eqn = "x^2 + 2x - 1 = 0"

theorem quadratic_equation_proof :
  is_quadratic_equation "x^2 + 2x - 1 = 0" :=
sorry

end quadratic_equation_proof_l188_188393


namespace product_seqFrac_l188_188872

def seqFrac (n : ℕ) : ℚ := (n : ℚ) / (n + 5 : ℚ)

theorem product_seqFrac :
  ((List.range 53).map seqFrac).prod = 1 / 27720 := by
  sorry

end product_seqFrac_l188_188872


namespace probability_of_at_least_one_three_l188_188157

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l188_188157


namespace boys_in_class_l188_188396

theorem boys_in_class (students : ℕ) (ratio_girls_boys : ℕ → Prop)
  (h1 : students = 56)
  (h2 : ratio_girls_boys 4 ∧ ratio_girls_boys 3) :
  ∃ k : ℕ, 4 * k + 3 * k = students ∧ 3 * k = 24 :=
by
  sorry

end boys_in_class_l188_188396


namespace find_n_l188_188051

theorem find_n (m n : ℕ) (h1: m = 34)
               (h2: (1^(m+1) / 5^(m+1)) * (1^n / 4^n) = 1 / (2 * 10^35)) : 
               n = 18 :=
by
  sorry

end find_n_l188_188051


namespace largest_4_digit_number_divisible_by_24_l188_188819

theorem largest_4_digit_number_divisible_by_24 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 24 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 24 = 0 → m ≤ n :=
sorry

end largest_4_digit_number_divisible_by_24_l188_188819


namespace sum_of_cubes_l188_188653

theorem sum_of_cubes (p q r : ℝ) (h1 : p + q + r = 7) (h2 : p * q + p * r + q * r = 10) (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 :=
by
  sorry

end sum_of_cubes_l188_188653


namespace people_in_room_l188_188279

variable (total_chairs occupied_chairs people_present : ℕ)
variable (h1 : total_chairs = 28)
variable (h2 : occupied_chairs = 14)
variable (h3 : (2 / 3 : ℚ) * people_present = 14)
variable (h4 : total_chairs = 2 * occupied_chairs)

theorem people_in_room : people_present = 21 := 
by 
  --proof will be here
  sorry

end people_in_room_l188_188279


namespace marion_score_is_correct_l188_188508

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end marion_score_is_correct_l188_188508


namespace apple_distribution_l188_188146

theorem apple_distribution (total_apples : ℝ)
  (time_anya time_varya time_sveta total_time : ℝ)
  (work_anya work_varya work_sveta : ℝ) :
  total_apples = 10 →
  time_anya = 20 →
  time_varya = 35 →
  time_sveta = 45 →
  total_time = (time_anya + time_varya + time_sveta) →
  work_anya = (total_apples * time_anya / total_time) →
  work_varya = (total_apples * time_varya / total_time) →
  work_sveta = (total_apples * time_sveta / total_time) →
  work_anya = 2 ∧ work_varya = 3.5 ∧ work_sveta = 4.5 := by
  sorry

end apple_distribution_l188_188146


namespace rook_reaches_right_total_rook_reaches_right_seven_moves_l188_188737

-- Definition of the conditions for the problem
def rook_ways_total (n : Nat) :=
  2 ^ (n - 2)

def rook_ways_in_moves (n k : Nat) :=
  Nat.choose (n - 2) (k - 1)

-- Proof problem statements
theorem rook_reaches_right_total : rook_ways_total 30 = 2 ^ 28 := 
by sorry

theorem rook_reaches_right_seven_moves : rook_ways_in_moves 30 7 = Nat.choose 28 6 := 
by sorry

end rook_reaches_right_total_rook_reaches_right_seven_moves_l188_188737


namespace dance_lessons_l188_188881

theorem dance_lessons (cost_per_lesson : ℕ) (free_lessons : ℕ) (amount_paid : ℕ) 
  (H1 : cost_per_lesson = 10) 
  (H2 : free_lessons = 2) 
  (H3 : amount_paid = 80) : 
  (amount_paid / cost_per_lesson + free_lessons = 10) :=
by
  sorry

end dance_lessons_l188_188881


namespace number_of_sets_B_l188_188733

theorem number_of_sets_B (A : Set ℕ) (hA : A = {1, 2}) :
    ∃ (n : ℕ), n = 4 ∧ (∀ B : Set ℕ, A ∪ B = {1, 2} → B ⊆ A) := sorry

end number_of_sets_B_l188_188733


namespace eugene_used_six_boxes_of_toothpicks_l188_188008

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end eugene_used_six_boxes_of_toothpicks_l188_188008


namespace find_k_l188_188898

def geom_seq (c : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = c * (a n)

def sum_first_n_terms (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k {c : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ} {k : ℝ} (hGeom : geom_seq c a) (hSum : sum_first_n_terms S k) :
  k = -1 :=
by
  sorry

end find_k_l188_188898


namespace max_integers_sum_power_of_two_l188_188697

open Set

/-- Given a finite set of positive integers such that the sum of any two distinct elements is a power of two,
    the cardinality of the set is at most 2. -/
theorem max_integers_sum_power_of_two (S : Finset ℕ) (h_pos : ∀ x ∈ S, 0 < x)
  (h_sum : ∀ {a b : ℕ}, a ∈ S → b ∈ S → a ≠ b → ∃ n : ℕ, a + b = 2^n) : S.card ≤ 2 :=
sorry

end max_integers_sum_power_of_two_l188_188697


namespace g_symmetric_l188_188912

noncomputable def g (x : ℝ) : ℝ := |⌊2 * x⌋| - |⌊2 - 2 * x⌋|

theorem g_symmetric : ∀ x : ℝ, g x = g (1 - x) := by
  sorry

end g_symmetric_l188_188912


namespace percentage_difference_max_min_l188_188800

-- Definitions for the sector angles of each department
def angle_manufacturing := 162.0
def angle_sales := 108.0
def angle_research_and_development := 54.0
def angle_administration := 36.0

-- Full circle in degrees
def full_circle := 360.0

-- Compute the percentage representations of each department
def percentage_manufacturing := (angle_manufacturing / full_circle) * 100
def percentage_sales := (angle_sales / full_circle) * 100
def percentage_research_and_development := (angle_research_and_development / full_circle) * 100
def percentage_administration := (angle_administration / full_circle) * 100

-- Prove that the percentage difference between the department with the maximum and minimum number of employees is 35%
theorem percentage_difference_max_min : 
  percentage_manufacturing - percentage_administration = 35.0 :=
by
  -- placeholder for the actual proof
  sorry

end percentage_difference_max_min_l188_188800


namespace students_who_chose_water_l188_188560

-- Defining the conditions
def percent_juice : ℚ := 75 / 100
def percent_water : ℚ := 25 / 100
def students_who_chose_juice : ℚ := 90
def ratio_water_to_juice : ℚ := percent_water / percent_juice  -- This should equal 1/3

-- The theorem we need to prove
theorem students_who_chose_water : students_who_chose_juice * ratio_water_to_juice = 30 := 
by
  sorry

end students_who_chose_water_l188_188560


namespace repeating_decimal_as_fraction_l188_188513

-- Define the repeating decimal 4.25252525... as x
def repeating_decimal : ℚ := 4 + 25 / 99

-- Theorem statement to prove the equivalence
theorem repeating_decimal_as_fraction :
  repeating_decimal = 421 / 99 :=
by
  sorry

end repeating_decimal_as_fraction_l188_188513


namespace value_of_expression_l188_188722

theorem value_of_expression (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (3 * x - 4 * y) / z = 1 / 4 := 
by 
  sorry

end value_of_expression_l188_188722


namespace rahim_books_second_shop_l188_188778

variable (x : ℕ)

-- Definitions of the problem's conditions
def total_cost : ℕ := 520 + 248
def total_books (x : ℕ) : ℕ := 42 + x
def average_price : ℕ := 12

-- The problem statement in Lean 4
theorem rahim_books_second_shop : x = 22 → total_cost / total_books x = average_price :=
  sorry

end rahim_books_second_shop_l188_188778


namespace find_number_l188_188464

theorem find_number (x : ℝ) (h : 0.65 * x = 0.8 * x - 21) : x = 140 := by
  sorry

end find_number_l188_188464


namespace pete_travel_time_l188_188242

-- Definitions for the given conditions
def map_distance := 5.0          -- in inches
def scale := 0.05555555555555555 -- in inches per mile
def speed := 60.0                -- in miles per hour
def real_distance := map_distance / scale

-- The theorem to state the proof problem
theorem pete_travel_time : 
  real_distance = 90 → -- Based on condition deduced from earlier
  real_distance / speed = 1.5 := 
by 
  intro h1
  rw[h1]
  norm_num
  sorry

end pete_travel_time_l188_188242


namespace total_owed_proof_l188_188439

-- Define initial conditions
def initial_owed : ℕ := 20
def borrowed : ℕ := 8

-- Define the total amount owed
def total_owed : ℕ := initial_owed + borrowed

-- Prove the statement
theorem total_owed_proof : total_owed = 28 := 
by 
  -- Proof is omitted with sorry
  sorry

end total_owed_proof_l188_188439


namespace multiply_polynomials_l188_188413

variable {x y : ℝ}

theorem multiply_polynomials (x y : ℝ) :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 :=
by
  sorry

end multiply_polynomials_l188_188413


namespace barefoot_kids_l188_188263

theorem barefoot_kids (total_kids kids_socks kids_shoes kids_both : ℕ) 
  (h1 : total_kids = 22) 
  (h2 : kids_socks = 12) 
  (h3 : kids_shoes = 8) 
  (h4 : kids_both = 6) : 
  (total_kids - (kids_socks - kids_both + kids_shoes - kids_both + kids_both) = 8) :=
by
  -- following sorry to skip proof.
  sorry

end barefoot_kids_l188_188263


namespace lex_read_pages_l188_188327

theorem lex_read_pages (total_pages days : ℕ) (h1 : total_pages = 240) (h2 : days = 12) :
  total_pages / days = 20 :=
by sorry

end lex_read_pages_l188_188327


namespace average_score_of_remaining_students_correct_l188_188049

noncomputable def average_score_remaining_students (n : ℕ) (h_n : n > 15) (avg_all : ℚ) (avg_subgroup : ℚ) : ℚ :=
if h_avg_all : avg_all = 10 ∧ avg_subgroup = 16 then
  (10 * n - 240) / (n - 15)
else
  0

theorem average_score_of_remaining_students_correct (n : ℕ) (h_n : n > 15) :
  (average_score_remaining_students n h_n 10 16) = (10 * n - 240) / (n - 15) :=
by
  dsimp [average_score_remaining_students]
  split_ifs with h_avg
  · sorry
  · sorry

end average_score_of_remaining_students_correct_l188_188049


namespace total_weight_of_5_moles_of_cai2_l188_188883

-- Definitions based on the conditions
def weight_of_calcium : Real := 40.08
def weight_of_iodine : Real := 126.90
def iodine_atoms_in_cai2 : Nat := 2
def moles_of_calcium_iodide : Nat := 5

-- Lean 4 statement for the proof problem
theorem total_weight_of_5_moles_of_cai2 :
  (weight_of_calcium + (iodine_atoms_in_cai2 * weight_of_iodine)) * moles_of_calcium_iodide = 1469.4 := by
  sorry

end total_weight_of_5_moles_of_cai2_l188_188883


namespace coefficient_of_term_x7_in_expansion_l188_188732

theorem coefficient_of_term_x7_in_expansion:
  let general_term (r : ℕ) := (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r * (x : ℤ)^(12 - (5 * r) / 2)
  ∃ r : ℕ, 12 - (5 * r) / 2 = 7 ∧ (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r = 240 := 
sorry

end coefficient_of_term_x7_in_expansion_l188_188732


namespace borrowed_movie_price_correct_l188_188417

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def total_paid : ℝ := 20.00
def change_received : ℝ := 1.37
def tickets_cost : ℝ := number_of_tickets * ticket_price
def total_spent : ℝ := total_paid - change_received
def borrowed_movie_cost : ℝ := total_spent - tickets_cost

theorem borrowed_movie_price_correct : borrowed_movie_cost = 6.79 := by
  sorry

end borrowed_movie_price_correct_l188_188417


namespace greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l188_188933

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem greatest_divisor_of_546_smaller_than_30_and_factor_of_126 :
  ∃ (d : ℕ), d < 30 ∧ is_factor d 546 ∧ is_factor d 126 ∧ ∀ e : ℕ, e < 30 ∧ is_factor e 546 ∧ is_factor e 126 → e ≤ d := 
sorry

end greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l188_188933


namespace length_of_cube_side_l188_188740

theorem length_of_cube_side (SA : ℝ) (h₀ : SA = 600) (h₁ : SA = 6 * a^2) : a = 10 := by
  sorry

end length_of_cube_side_l188_188740


namespace regular_polygon_sides_l188_188533

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i < n → (180 * (n - 2) / n) = 174) : n = 60 := by
  sorry

end regular_polygon_sides_l188_188533


namespace sum_of_cubes_correct_l188_188317

noncomputable def expression_for_sum_of_cubes (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) : Prop :=
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + b^3 * d^3) / (a * b * c * d)

theorem sum_of_cubes_correct (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) :
  expression_for_sum_of_cubes x y z w a b c d hx hy hz hw ha hb hc hd hxy hxz hyz hxw :=
sorry

end sum_of_cubes_correct_l188_188317


namespace annual_decrease_rate_l188_188942

theorem annual_decrease_rate (P₀ P₂ : ℝ) (r : ℝ) (h₀ : P₀ = 8000) (h₂ : P₂ = 5120) :
  P₂ = P₀ * (1 - r / 100) ^ 2 → r = 20 :=
by
  intros h
  have h₀' : P₀ = 8000 := h₀
  have h₂' : P₂ = 5120 := h₂
  sorry

end annual_decrease_rate_l188_188942


namespace solve_linear_system_l188_188762

theorem solve_linear_system :
  ∃ (x1 x2 x3 : ℚ), 
  (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧ 
  (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧ 
  (5 * x1 + 5 * x2 - 7 * x3 = 27) ∧
  (x1 = 19 / 3 + x3) ∧ 
  (x2 = -14 / 15 + 2 / 5 * x3) := 
by 
  sorry

end solve_linear_system_l188_188762


namespace parabola_directrix_eq_l188_188238

noncomputable def equation_of_directrix (p : ℝ) : Prop :=
  (p > 0) ∧ (∀ (x y : ℝ), (x ≠ -5 / 4) → ¬ (y ^ 2 = 2 * p * x))

theorem parabola_directrix_eq (A_x A_y : ℝ) (hA : A_x = 2 ∧ A_y = 1)
  (h_perpendicular_bisector_fo : ∃ (f_x f_y : ℝ), f_x = 5 / 4 ∧ f_y = 0) :
  equation_of_directrix (5 / 2) :=
by {
  sorry
}

end parabola_directrix_eq_l188_188238


namespace find_y_l188_188345

def G (a b c d : ℕ) : ℕ := a ^ b + c * d

theorem find_y (y : ℕ) : G 3 y 5 10 = 350 ↔ y = 5 := by
  sorry

end find_y_l188_188345


namespace quadratic_roots_square_cube_sum_l188_188821

theorem quadratic_roots_square_cube_sum
  (a b c : ℝ) (h : a ≠ 0) (x1 x2 : ℝ)
  (hx : ∀ (x : ℝ), a * x^2 + b * x + c = 0 ↔ x = x1 ∨ x = x2) :
  (x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2) ∧
  (x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3) :=
by
  sorry

end quadratic_roots_square_cube_sum_l188_188821


namespace min_value_of_expression_l188_188230

theorem min_value_of_expression (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 5) : 
  (9 / x + 16 / y + 25 / z) ≥ 28.8 :=
by sorry

end min_value_of_expression_l188_188230


namespace no_linear_factor_l188_188597

theorem no_linear_factor : ∀ x y z : ℤ,
  ¬ ∃ a b c : ℤ, a*x + b*y + c*z + (x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z) = 0 :=
by sorry

end no_linear_factor_l188_188597


namespace constant_seq_arith_geo_l188_188249

def is_arithmetic_sequence (s : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n + d

def is_geometric_sequence (s : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n * r

theorem constant_seq_arith_geo (s : ℕ → ℝ) (d r : ℝ) :
  is_arithmetic_sequence s d →
  is_geometric_sequence s r →
  (∃ c : ℝ, ∀ n : ℕ, s n = c) ∧ r = 1 :=
by
  sorry

end constant_seq_arith_geo_l188_188249


namespace work_required_to_lift_satellite_l188_188776

noncomputable def satellite_lifting_work (m H R3 g : ℝ) : ℝ :=
  m * g * R3^2 * ((1 / R3) - (1 / (R3 + H)))

theorem work_required_to_lift_satellite :
  satellite_lifting_work (7.0 * 10^3) (200 * 10^3) (6380 * 10^3) 10 = 13574468085 :=
by sorry

end work_required_to_lift_satellite_l188_188776


namespace richmond_more_than_victoria_l188_188863

-- Defining the population of Beacon
def beacon_people : ℕ := 500

-- Defining the population of Victoria based on Beacon's population
def victoria_people : ℕ := 4 * beacon_people

-- Defining the population of Richmond
def richmond_people : ℕ := 3000

-- The proof problem: calculating the difference
theorem richmond_more_than_victoria : richmond_people - victoria_people = 1000 := by
  -- The statement of the theorem
  sorry

end richmond_more_than_victoria_l188_188863


namespace longest_tape_l188_188621

theorem longest_tape (r b y : ℚ) (h₀ : r = 11 / 6) (h₁ : b = 7 / 4) (h₂ : y = 13 / 8) : r > b ∧ r > y :=
by 
  sorry

end longest_tape_l188_188621


namespace sequence_infinite_coprime_l188_188316

theorem sequence_infinite_coprime (a : ℤ) (h : a > 1) :
  ∃ (S : ℕ → ℕ), (∀ n m : ℕ, n ≠ m → Int.gcd (a^(S n + 1) + a^S n - 1) (a^(S m + 1) + a^S m - 1) = 1) :=
sorry

end sequence_infinite_coprime_l188_188316


namespace tan_alpha_third_quadrant_l188_188407

theorem tan_alpha_third_quadrant (α : ℝ) 
  (h_eq: Real.sin α = Real.cos α) 
  (h_third: π < α ∧ α < 3 * π / 2) : Real.tan α = 1 := 
by 
  sorry

end tan_alpha_third_quadrant_l188_188407


namespace triangle_segments_equivalence_l188_188480

variable {a b c p : ℝ}

theorem triangle_segments_equivalence (h_acute : a^2 + b^2 > c^2) 
  (h_perpendicular : ∃ h: ℝ, h^2 = c^2 - (a - p)^2 ∧ h^2 = b^2 - p^2) :
  a / (c + b) = (c - b) / (a - 2 * p) := by
sorry

end triangle_segments_equivalence_l188_188480


namespace quadrilateral_sum_of_squares_l188_188808

theorem quadrilateral_sum_of_squares
  (a b c d m n t : ℝ) : 
  a^2 + b^2 + c^2 + d^2 = m^2 + n^2 + 4 * t^2 :=
sorry

end quadrilateral_sum_of_squares_l188_188808


namespace cyclist_speed_l188_188517

theorem cyclist_speed 
  (v : ℝ) 
  (hiker1_speed : ℝ := 4)
  (hiker2_speed : ℝ := 5)
  (cyclist_overtakes_hiker2_after_hiker1 : ∃ t1 t2 : ℝ, 
      t1 = 8 / (v - hiker1_speed) ∧ 
      t2 = 5 / (v - hiker2_speed) ∧ 
      t2 - t1 = 1/6)
: (v = 20 ∨ v = 7 ∨ abs (v - 6.5) < 0.1) :=
sorry

end cyclist_speed_l188_188517


namespace part_a_l188_188611

theorem part_a (p : ℕ → ℕ → ℝ) (m : ℕ) (hm : m ≥ 1) : p m 0 = (3 / 4) * p (m-1) 0 + (1 / 2) * p (m-1) 2 + (1 / 8) * p (m-1) 4 :=
by
  sorry

end part_a_l188_188611


namespace cycle_reappear_l188_188332

/-- Given two sequences with cycle lengths 6 and 4, prove the sequences will align on line number 12 -/
theorem cycle_reappear (l1 l2 : ℕ) (h1 : l1 = 6) (h2 : l2 = 4) :
  Nat.lcm l1 l2 = 12 := by
  sorry

end cycle_reappear_l188_188332


namespace expr_simplify_l188_188932

variable {a b c d m : ℚ}
variable {b_nonzero : b ≠ 0}
variable {m_nat : ℕ}
variable {m_bound : 0 ≤ m_nat ∧ m_nat < 2}

def expr_value (a b c d m : ℚ) : ℚ :=
  m - (c * d) + (a + b) / 2023 + a / b

theorem expr_simplify (h1 : a = -b) (h2 : c * d = 1) (h3 : m = (m_nat : ℚ)) :
  expr_value a b c d m = -1 ∨ expr_value a b c d m = -2 := by
  sorry

end expr_simplify_l188_188932


namespace count_divisibles_in_range_l188_188063

theorem count_divisibles_in_range :
  let lower_bound := (2:ℤ)^10
  let upper_bound := (2:ℤ)^18
  let divisor := (2:ℤ)^9 
  (upper_bound - lower_bound) / divisor + 1 = 511 :=
by 
  sorry

end count_divisibles_in_range_l188_188063


namespace bird_families_flew_away_l188_188777

def initial_families : ℕ := 41
def left_families : ℕ := 14

theorem bird_families_flew_away :
  initial_families - left_families = 27 :=
by
  -- This is a placeholder for the proof
  sorry

end bird_families_flew_away_l188_188777


namespace negation_of_proposition_l188_188209

theorem negation_of_proposition : 
  (¬ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0)) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) :=
by sorry

end negation_of_proposition_l188_188209


namespace store_profit_l188_188362

variable (m n : ℝ)
variable (h_mn : m > n)

theorem store_profit : 10 * (m - n) > 0 :=
by
  sorry

end store_profit_l188_188362


namespace handshake_max_participants_l188_188239

theorem handshake_max_participants (N : ℕ) (hN : 5 < N) (hNotAllShaken: ∃ p1 p2 : ℕ, p1 ≠ p2 ∧ p1 < N ∧ p2 < N ∧ (∀ i : ℕ, i < N → i ≠ p1 → i ≠ p2 → ∃ j : ℕ, j < N ∧ j ≠ i ∧ j ≠ p1 ∧ j ≠ p2)) :
∃ k, k = N - 2 :=
by
  sorry

end handshake_max_participants_l188_188239


namespace grown_ups_in_milburg_l188_188889

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l188_188889


namespace parabola_equation_l188_188061

-- Define the constants and the conditions
def parabola_focus : ℝ × ℝ := (3, 3)
def directrix : ℝ × ℝ × ℝ := (3, 7, -21)

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
  a > 0 ∧
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd a b) c) d) e) f = 1 ∧
  (a : ℝ) * x^2 + (b : ℝ) * x * y + (c : ℝ) * y^2 + (d : ℝ) * x + (e : ℝ) * y + (f : ℝ) = 
  49 * x^2 - 42 * x * y + 9 * y^2 - 222 * x - 54 * y + 603 := sorry

end parabola_equation_l188_188061


namespace luigi_pizza_cost_l188_188593

theorem luigi_pizza_cost (num_pizzas pieces_per_pizza cost_per_piece : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : pieces_per_pizza = 5) 
  (h3 : cost_per_piece = 4) :
  num_pizzas * pieces_per_pizza * cost_per_piece / pieces_per_pizza = 80 := by
  sorry

end luigi_pizza_cost_l188_188593


namespace multiplication_72519_9999_l188_188167

theorem multiplication_72519_9999 :
  72519 * 9999 = 725117481 :=
by
  sorry

end multiplication_72519_9999_l188_188167


namespace quadratic_root_square_of_another_l188_188637

theorem quadratic_root_square_of_another (a : ℚ) :
  (∃ x y : ℚ, x^2 - (15/4) * x + a^3 = 0 ∧ (x = y^2 ∨ y = x^2) ∧ (x*y = a^3)) →
  (a = 3/2 ∨ a = -5/2) :=
sorry

end quadratic_root_square_of_another_l188_188637


namespace range_of_a_l188_188272

variable (f : ℝ → ℝ)

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f y < f x

theorem range_of_a 
  (decreasing_f : is_decreasing f)
  (hfdef : ∀ x, -1 ≤ x ∧ x ≤ 1 → f (2 * x - 3) < f (x - 2)) :
  ∃ a : ℝ, 1 < a ∧ a ≤ 2  :=
by 
  sorry

end range_of_a_l188_188272


namespace surface_area_of_cylinder_with_square_cross_section_l188_188369

theorem surface_area_of_cylinder_with_square_cross_section
  (side_length : ℝ) (h1 : side_length = 2) : 
  (2 * Real.pi * 2 + 2 * Real.pi * 1^2) = 6 * Real.pi :=
by
  rw [←h1]
  sorry

end surface_area_of_cylinder_with_square_cross_section_l188_188369


namespace common_ratio_of_series_l188_188839

-- Define the terms and conditions for the infinite geometric series problem.
def first_term : ℝ := 500
def series_sum : ℝ := 4000

-- State the theorem that needs to be proven: the common ratio of the series is 7/8.
theorem common_ratio_of_series (a S r : ℝ) (h_a : a = 500) (h_S : S = 4000) (h_eq : S = a / (1 - r)) :
  r = 7 / 8 :=
by
  sorry

end common_ratio_of_series_l188_188839


namespace find_a_l188_188494

noncomputable def f (x : ℝ) : ℝ := x^2 + 9
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 9) : a = Real.sqrt 5 :=
by
  sorry

end find_a_l188_188494


namespace Miss_Adamson_paper_usage_l188_188222

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l188_188222


namespace midpoint_polar_coords_l188_188861

/-- 
Given two points in polar coordinates: (6, π/6) and (2, -π/6),  
the midpoint of the line segment connecting these points in polar coordinates is (√13, π/6).
-/
theorem midpoint_polar_coords :
  let A := (6, Real.pi / 6)
  let B := (2, -Real.pi / 6)
  let A_cart := (6 * Real.cos (Real.pi / 6), 6 * Real.sin (Real.pi / 6))
  let B_cart := (2 * Real.cos (-Real.pi / 6), 2 * Real.sin (-Real.pi / 6))
  let Mx := ((A_cart.fst + B_cart.fst) / 2)
  let My := ((A_cart.snd + B_cart.snd) / 2)
  let r := Real.sqrt (Mx^2 + My^2)
  let theta := Real.arctan (My / Mx)
  0 <= theta ∧ theta < 2 * Real.pi ∧ r > 0 ∧ (r = Real.sqrt 13 ∧ theta = Real.pi / 6) :=
by 
  sorry

end midpoint_polar_coords_l188_188861


namespace inequality_solution_l188_188636

theorem inequality_solution (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 :=
by 
  sorry

end inequality_solution_l188_188636


namespace circle_eq_of_hyperbola_focus_eccentricity_l188_188641

theorem circle_eq_of_hyperbola_focus_eccentricity :
  ∀ (x y : ℝ), ((y^2 - (x^2 / 3) = 1) → (x^2 + (y-2)^2 = 4)) := by
  intro x y
  intro hyp_eq
  sorry

end circle_eq_of_hyperbola_focus_eccentricity_l188_188641


namespace evaluate_expression_l188_188110

theorem evaluate_expression :
  -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := 
by
  sorry

end evaluate_expression_l188_188110


namespace hardest_vs_least_worked_hours_difference_l188_188186

-- Let x be the scaling factor for the ratio
-- The times worked are 2x, 3x, and 4x

def project_time_difference (x : ℕ) : Prop :=
  let time1 := 2 * x
  let time2 := 3 * x
  let time3 := 4 * x
  (time1 + time2 + time3 = 90) ∧ ((4 * x - 2 * x) = 20)

theorem hardest_vs_least_worked_hours_difference :
  ∃ x : ℕ, project_time_difference x :=
by
  sorry

end hardest_vs_least_worked_hours_difference_l188_188186


namespace total_cost_of_pens_and_notebooks_l188_188528

theorem total_cost_of_pens_and_notebooks (a b : ℝ) : 5 * a + 8 * b = 5 * a + 8 * b := 
by 
  sorry

end total_cost_of_pens_and_notebooks_l188_188528


namespace fermats_little_theorem_l188_188526

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) : 
  (a^p - a) % p = 0 :=
sorry

end fermats_little_theorem_l188_188526


namespace work_ratio_l188_188048

theorem work_ratio (m b : ℝ) (h1 : 12 * m + 16 * b = 1 / 5) (h2 : 13 * m + 24 * b = 1 / 4) : m = 2 * b :=
by sorry

end work_ratio_l188_188048


namespace cubic_polynomial_roots_product_l188_188862

theorem cubic_polynomial_roots_product :
  (∃ a b c : ℝ, (3*a^3 - 9*a^2 + 5*a - 15 = 0) ∧
               (3*b^3 - 9*b^2 + 5*b - 15 = 0) ∧
               (3*c^3 - 9*c^2 + 5*c - 15 = 0) ∧
               a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  ∃ a b c : ℝ, (3*a*b*c = 5) := 
sorry

end cubic_polynomial_roots_product_l188_188862


namespace fractions_simplify_to_prime_denominator_2023_l188_188158

def num_fractions_simplifying_to_prime_denominator (n: ℕ) (p q: ℕ) : ℕ :=
  let multiples (m: ℕ) : ℕ := (n - 1) / m
  multiples p + multiples (p * q)

theorem fractions_simplify_to_prime_denominator_2023 :
  num_fractions_simplifying_to_prime_denominator 2023 17 7 = 22 :=
by
  sorry

end fractions_simplify_to_prime_denominator_2023_l188_188158


namespace determine_quadrant_l188_188118

def pointInWhichQuadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On axis or origin"

theorem determine_quadrant : pointInWhichQuadrant (-7) 3 = "Second quadrant" :=
by
  sorry

end determine_quadrant_l188_188118
