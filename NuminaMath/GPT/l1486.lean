import Mathlib

namespace john_safe_weight_l1486_148603

-- Assuming the conditions provided that form the basis of our problem.
def max_capacity : ℝ := 1000
def safety_margin : ℝ := 0.20
def john_weight : ℝ := 250
def safe_weight (max_capacity safety_margin john_weight : ℝ) : ℝ := 
  (max_capacity * (1 - safety_margin)) - john_weight

-- The main theorem to prove based on the provided problem statement.
theorem john_safe_weight : safe_weight max_capacity safety_margin john_weight = 550 := by
  -- skipping the proof details as instructed
  sorry

end john_safe_weight_l1486_148603


namespace min_value_f_l1486_148657

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem min_value_f (a : ℝ) (h : -2 < a) :
  ∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ≥ m) ∧ 
  ((a ≤ 1 → m = a^2 - 2 * a) ∧ (1 < a → m = -1)) :=
by
  sorry

end min_value_f_l1486_148657


namespace books_from_second_shop_l1486_148662

-- Define the conditions
def num_books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1280
def cost_second_shop : ℕ := 880
def total_cost : ℤ := cost_first_shop + cost_second_shop
def average_price_per_book : ℤ := 18

-- Define the statement to be proved
theorem books_from_second_shop (x : ℕ) :
  (num_books_first_shop + x) * average_price_per_book = total_cost →
  x = 55 :=
by
  sorry

end books_from_second_shop_l1486_148662


namespace price_per_hotdog_l1486_148629

-- The conditions
def hot_dogs_per_hour := 10
def hours := 10
def total_sales := 200

-- Conclusion we need to prove
theorem price_per_hotdog : total_sales / (hot_dogs_per_hour * hours) = 2 := by
  sorry

end price_per_hotdog_l1486_148629


namespace necessary_not_sufficient_condition_l1486_148613

theorem necessary_not_sufficient_condition {a : ℝ} :
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) →
  (¬ (∀ x : ℝ, x ≥ a → |x - 1| < 1)) →
  a ≤ 0 :=
by
  intro h1 h2
  sorry

end necessary_not_sufficient_condition_l1486_148613


namespace hcf_of_abc_l1486_148610

-- Given conditions
variables (a b c : ℕ)
def lcm_abc := Nat.lcm (Nat.lcm a b) c
def product_abc := a * b * c

-- Statement to prove
theorem hcf_of_abc (H1 : lcm_abc a b c = 1200) (H2 : product_abc a b c = 108000) : 
  Nat.gcd (Nat.gcd a b) c = 90 :=
by
  sorry

end hcf_of_abc_l1486_148610


namespace custom_op_value_l1486_148637

variable {a b : ℤ}
def custom_op (a b : ℤ) := 1/a + 1/b

axiom h1 : a + b = 15
axiom h2 : a * b = 56

theorem custom_op_value : custom_op a b = 15/56 :=
by
  sorry

end custom_op_value_l1486_148637


namespace olaf_total_toy_cars_l1486_148659

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end olaf_total_toy_cars_l1486_148659


namespace pen_and_notebook_cost_l1486_148608

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 17 * p + 5 * n = 200 ∧ p > n ∧ p + n = 16 := 
by
  sorry

end pen_and_notebook_cost_l1486_148608


namespace probability_of_blue_face_l1486_148649

theorem probability_of_blue_face (total_faces blue_faces : ℕ) (h_total : total_faces = 8) (h_blue : blue_faces = 5) : 
  blue_faces / total_faces = 5 / 8 :=
by
  sorry

end probability_of_blue_face_l1486_148649


namespace units_digit_2_1501_5_1602_11_1703_l1486_148660

theorem units_digit_2_1501_5_1602_11_1703 : 
  (2 ^ 1501 * 5 ^ 1602 * 11 ^ 1703) % 10 = 0 :=
  sorry

end units_digit_2_1501_5_1602_11_1703_l1486_148660


namespace train_stop_time_l1486_148641

theorem train_stop_time
  (D : ℝ)
  (h1 : D > 0)
  (T_no_stop : ℝ := D / 300)
  (T_with_stop : ℝ := D / 200)
  (T_stop : ℝ := T_with_stop - T_no_stop):
  T_stop = 6 / 60 := by
    sorry

end train_stop_time_l1486_148641


namespace find_f_2011_l1486_148647

-- Definitions of given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_of_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Main theorem to be proven
theorem find_f_2011 (f : ℝ → ℝ) 
  (hf_even: is_even_function f) 
  (hf_periodic: is_periodic_of_period f 4) 
  (hf_at_1: f 1 = 1) : 
  f 2011 = 1 := 
by 
  sorry

end find_f_2011_l1486_148647


namespace determinant_of_A_l1486_148685

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![8, 5, -4], ![3, 3, 7]]  -- Defining matrix A

def A' : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![5, 4, -2], ![0, 2, 9]]  -- Defining matrix A' after row operations

theorem determinant_of_A' : Matrix.det A' = 55 := by -- Proving that the determinant of A' is 55
  sorry

end determinant_of_A_l1486_148685


namespace intersection_complement_l1486_148677

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 > 4}

-- Define set N
def N : Set ℝ := {x : ℝ | (x - 3) / (x + 1) < 0}

-- Complement of N in U
def complement_N : Set ℝ := {x : ℝ | x <= -1} ∪ {x : ℝ | x >= 3}

-- Final proof to show intersection
theorem intersection_complement :
  M ∩ complement_N = {x : ℝ | x < -2} ∪ {x : ℝ | x >= 3} :=
by
  sorry

end intersection_complement_l1486_148677


namespace skittles_distribution_l1486_148652

-- Given problem conditions
variable (Brandon_initial : ℕ := 96) (Bonnie_initial : ℕ := 4) 
variable (Brandon_loss : ℕ := 9)
variable (combined_skittles : ℕ := (Brandon_initial - Brandon_loss) + Bonnie_initial)
variable (individual_share : ℕ := combined_skittles / 4)
variable (remainder : ℕ := combined_skittles % 4)
variable (Chloe_share : ℕ := individual_share)
variable (Dylan_share_initial : ℕ := individual_share)
variable (Chloe_to_Dylan : ℕ := Chloe_share / 2)
variable (Dylan_new_share : ℕ := Dylan_share_initial + Chloe_to_Dylan)
variable (Dylan_to_Bonnie : ℕ := Dylan_new_share / 3)
variable (final_Bonnie : ℕ := individual_share + Dylan_to_Bonnie)
variable (final_Chloe : ℕ := Chloe_share - Chloe_to_Dylan)
variable (final_Dylan : ℕ := Dylan_new_share - Dylan_to_Bonnie)

-- The theorem to be proved
theorem skittles_distribution : 
  individual_share = 22 ∧ final_Bonnie = 33 ∧ final_Chloe = 11 ∧ final_Dylan = 22 :=
by
  -- The proof would go here, but it’s not required for this task.
  sorry

end skittles_distribution_l1486_148652


namespace minimum_sum_of_dimensions_l1486_148650

   theorem minimum_sum_of_dimensions (a b c : ℕ) (habc : a * b * c = 3003) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
     a + b + c = 45 :=
   sorry
   
end minimum_sum_of_dimensions_l1486_148650


namespace f_f_of_2020_l1486_148690

def f (x : ℕ) : ℕ :=
  if x ≤ 1 then 1
  else if 1 < x ∧ x ≤ 1837 then 2
  else if 1837 < x ∧ x < 2019 then 3
  else 2018

theorem f_f_of_2020 : f (f 2020) = 3 := by
  sorry

end f_f_of_2020_l1486_148690


namespace total_painting_area_correct_l1486_148679

def barn_width : ℝ := 12
def barn_length : ℝ := 15
def barn_height : ℝ := 6

def area_to_be_painted (width length height : ℝ) : ℝ := 
  2 * (width * height + length * height) + width * length

theorem total_painting_area_correct : area_to_be_painted barn_width barn_length barn_height = 828 := 
  by sorry

end total_painting_area_correct_l1486_148679


namespace smallest_n_candy_price_l1486_148653

theorem smallest_n_candy_price :
  ∃ n : ℕ, 25 * n = Nat.lcm (Nat.lcm 20 18) 24 ∧ ∀ k : ℕ, k > 0 ∧ 25 * k = Nat.lcm (Nat.lcm 20 18) 24 → n ≤ k :=
sorry

end smallest_n_candy_price_l1486_148653


namespace no_integer_solutions_l1486_148609

theorem no_integer_solutions (x y : ℤ) : 15 * x^2 - 7 * y^2 ≠ 9 :=
by
  sorry

end no_integer_solutions_l1486_148609


namespace total_boys_eq_350_l1486_148670

variable (Total : ℕ)
variable (SchoolA : ℕ)
variable (NotScience : ℕ)

axiom h1 : SchoolA = 20 * Total / 100
axiom h2 : NotScience = 70 * SchoolA / 100
axiom h3 : NotScience = 49

theorem total_boys_eq_350 : Total = 350 :=
by
  sorry

end total_boys_eq_350_l1486_148670


namespace moles_of_naoh_needed_l1486_148688

-- Define the chemical reaction
def balanced_eqn (nh4no3 naoh nano3 nh4oh : ℕ) : Prop :=
  nh4no3 = naoh ∧ nh4no3 = nano3

-- Theorem stating the moles of NaOH required to form 2 moles of NaNO3 from 2 moles of NH4NO3
theorem moles_of_naoh_needed (nh4no3 naoh nano3 nh4oh : ℕ) (h_balanced_eqn : balanced_eqn nh4no3 naoh nano3 nh4oh) 
  (h_nano3: nano3 = 2) (h_nh4no3: nh4no3 = 2) : naoh = 2 :=
by
  unfold balanced_eqn at h_balanced_eqn
  sorry

end moles_of_naoh_needed_l1486_148688


namespace bisecting_line_of_circle_l1486_148665

theorem bisecting_line_of_circle : 
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0 → x - y + 1 = 0) := 
sorry

end bisecting_line_of_circle_l1486_148665


namespace sum_first_3n_terms_is_36_l1486_148681

-- Definitions and conditions
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2
def sum_first_2n_terms (a d : ℤ) (n : ℕ) : ℤ := 2 * n * (2 * a + (2 * n - 1) * d) / 2
def sum_first_3n_terms (a d : ℤ) (n : ℕ) : ℤ := 3 * n * (2 * a + (3 * n - 1) * d) / 2

axiom h1 : ∀ (a d : ℤ) (n : ℕ), sum_first_n_terms a d n = 48
axiom h2 : ∀ (a d : ℤ) (n : ℕ), sum_first_2n_terms a d n = 60

theorem sum_first_3n_terms_is_36 (a d : ℤ) (n : ℕ) : sum_first_3n_terms a d n = 36 := by
  sorry

end sum_first_3n_terms_is_36_l1486_148681


namespace karen_cookies_grandparents_l1486_148655

theorem karen_cookies_grandparents :
  ∀ (total_cookies cookies_kept class_size cookies_per_person : ℕ)
  (cookies_given_class cookies_left cookies_to_grandparents : ℕ),
  total_cookies = 50 →
  cookies_kept = 10 →
  class_size = 16 →
  cookies_per_person = 2 →
  cookies_given_class = class_size * cookies_per_person →
  cookies_left = total_cookies - cookies_kept - cookies_given_class →
  cookies_to_grandparents = cookies_left →
  cookies_to_grandparents = 8 :=
by
  intros
  sorry

end karen_cookies_grandparents_l1486_148655


namespace train_journey_duration_l1486_148636

variable (z x : ℝ)
variable (h1 : 1.7 = 1 + 42 / 60)
variable (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7)

theorem train_journey_duration (z x : ℝ)
    (h1 : 1.7 = 1 + 42 / 60)
    (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7):
    z / x = 10 := 
by
  sorry

end train_journey_duration_l1486_148636


namespace equation_of_line_AC_l1486_148699

-- Define the given points A and B
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-3, -5)

-- Define the line m as a predicate
def line_m (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 + 6 = 0

-- Define the condition that line m is the angle bisector of ∠ACB
def is_angle_bisector (A B C : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : Prop := sorry

-- The symmetric point of B with respect to line m
def symmetric_point (B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : (ℝ × ℝ) := sorry

-- Proof statement
theorem equation_of_line_AC :
  ∀ (A B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop),
  A = (1, 1) →
  B = (-3, -5) →
  m = line_m →
  is_angle_bisector A B (symmetric_point B m) m →
  AC = {p : ℝ × ℝ | p.1 = 1} := sorry

end equation_of_line_AC_l1486_148699


namespace polar_equation_is_circle_l1486_148656

-- Define the polar coordinates equation condition
def polar_equation (r θ : ℝ) : Prop := r = 5

-- Define what it means for a set of points to form a circle centered at the origin with a radius of 5
def is_circle_radius_5 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- State the theorem we want to prove
theorem polar_equation_is_circle (r θ : ℝ) (x y : ℝ) (h1 : polar_equation r θ)
  (h2 : x = r * Real.cos θ) (h3 : y = r * Real.sin θ) : is_circle_radius_5 x y := 
sorry

end polar_equation_is_circle_l1486_148656


namespace pyramid_base_side_length_l1486_148696

theorem pyramid_base_side_length
  (lateral_face_area : Real)
  (slant_height : Real)
  (s : Real)
  (h_lateral_face_area : lateral_face_area = 200)
  (h_slant_height : slant_height = 40)
  (h_area_formula : lateral_face_area = 0.5 * s * slant_height) :
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l1486_148696


namespace polynomial_identity_l1486_148658

theorem polynomial_identity 
  (P : Polynomial ℤ)
  (a b : ℤ) 
  (h_distinct : a ≠ b)
  (h_eq : P.eval a * P.eval b = -(a - b) ^ 2) : 
  P.eval a + P.eval b = 0 := 
by
  sorry

end polynomial_identity_l1486_148658


namespace question_l1486_148671
-- Importing necessary libraries

-- Stating the problem
theorem question (x : ℤ) (h : (x + 12) / 8 = 9) : 35 - (x / 2) = 5 :=
by {
  sorry
}

end question_l1486_148671


namespace find_u_l1486_148651

variable (α β γ : ℝ)
variables (q s u : ℝ)

-- The first polynomial has roots α, β, γ
axiom roots_first_poly : ∀ x : ℝ, x^3 + 4 * x^2 + 6 * x - 8 = (x - α) * (x - β) * (x - γ)

-- Sum of the roots α + β + γ = -4
axiom sum_roots_first_poly : α + β + γ = -4

-- Product of the roots αβγ = 8
axiom product_roots_first_poly : α * β * γ = 8

-- The second polynomial has roots α + β, β + γ, γ + α
axiom roots_second_poly : ∀ x : ℝ, x^3 + q * x^2 + s * x + u = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))

theorem find_u : u = 32 :=
sorry

end find_u_l1486_148651


namespace solve_discriminant_l1486_148627

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem solve_discriminant : 
  discriminant 2 (2 + 1/2) (1/2) = 2.25 :=
by
  -- The proof can be filled in here
  -- Assuming a = 2, b = 2.5, c = 1/2
  -- discriminant 2 2.5 0.5 will be computed
  sorry

end solve_discriminant_l1486_148627


namespace perp_line_through_point_l1486_148614

variable (x y c : ℝ)

def line_perpendicular (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

def perpendicular_line (x y c : ℝ) : Prop :=
  2*x + y + c = 0

theorem perp_line_through_point :
  (line_perpendicular x y) ∧ (perpendicular_line (-2) 3 1) :=
by
  -- The first part asserts that the given line equation holds
  have h1 : line_perpendicular x y := sorry
  -- The second part asserts that our calculated line passes through the point (-2, 3) and is perpendicular
  have h2 : perpendicular_line (-2) 3 1 := sorry
  exact ⟨h1, h2⟩

end perp_line_through_point_l1486_148614


namespace FO_gt_DI_l1486_148630

-- Definitions and conditions
variables (F I D O : Type) [MetricSpace F] [MetricSpace I] [MetricSpace D] [MetricSpace O]
variables (FI DO DI FO : ℝ) (angle_FIO angle_DIO : ℝ)
variable (convex_FIDO : ConvexQuadrilateral F I D O)

-- Conditions
axiom FI_DO_equal : FI = DO
axiom FI_DO_gt_DI : FI > DI
axiom angles_equal : angle_FIO = angle_DIO

-- Goal
theorem FO_gt_DI : FO > DI :=
sorry

end FO_gt_DI_l1486_148630


namespace triangle_incircle_ratio_l1486_148619

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end triangle_incircle_ratio_l1486_148619


namespace find_number_l1486_148640

theorem find_number (x : ℝ) : 
  (3 * x / 5 - 220) * 4 + 40 = 360 → x = 500 :=
by
  intro h
  sorry

end find_number_l1486_148640


namespace no_solution_in_A_l1486_148612

def A : Set ℕ := 
  {n | ∃ k : ℤ, abs (n * Real.sqrt 2022 - 1 / 3 - k) ≤ 1 / 2022}

theorem no_solution_in_A (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) : 
  20 * x + 21 * y ≠ 22 * z := 
sorry

end no_solution_in_A_l1486_148612


namespace find_b_l1486_148695

theorem find_b (a b c : ℤ) (h1 : a + b + c = 120) (h2 : a + 4 = b - 12) (h3 : a + 4 = 3 * c) : b = 60 :=
sorry

end find_b_l1486_148695


namespace value_of_a7_minus_a8_l1486_148620

variable {a : ℕ → ℤ} (d a₁ : ℤ)

-- Definition that this is an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

-- Given condition
def condition (a : ℕ → ℤ) : Prop :=
  a 2 + a 6 + a 8 + a 10 = 80

-- The goal to prove
theorem value_of_a7_minus_a8 (a : ℕ → ℤ) (h_arith : is_arithmetic_seq a a₁ d)
  (h_cond : condition a) : a 7 - a 8 = 8 :=
sorry

end value_of_a7_minus_a8_l1486_148620


namespace number_is_divisible_by_divisor_l1486_148691

-- Defining the number after replacing y with 3
def number : ℕ := 7386038

-- Defining the divisor which we need to prove 
def divisor : ℕ := 7

-- Stating the property that 7386038 is divisible by 7
theorem number_is_divisible_by_divisor : number % divisor = 0 := by
  sorry

end number_is_divisible_by_divisor_l1486_148691


namespace price_of_fifth_basket_l1486_148604

-- Define the initial conditions
def avg_cost_of_4_baskets (total_cost_4 : ℝ) : Prop :=
  total_cost_4 / 4 = 4

def avg_cost_of_5_baskets (total_cost_5 : ℝ) : Prop :=
  total_cost_5 / 5 = 4.8

-- Theorem statement to be proved
theorem price_of_fifth_basket
  (total_cost_4 : ℝ)
  (h1 : avg_cost_of_4_baskets total_cost_4)
  (total_cost_5 : ℝ)
  (h2 : avg_cost_of_5_baskets total_cost_5) :
  total_cost_5 - total_cost_4 = 8 :=
by
  sorry

end price_of_fifth_basket_l1486_148604


namespace tenth_term_is_513_l1486_148643

def nth_term (n : ℕ) : ℕ :=
  2^(n-1) + 1

theorem tenth_term_is_513 : nth_term 10 = 513 := 
by 
  sorry

end tenth_term_is_513_l1486_148643


namespace triangle_right_angle_l1486_148631

theorem triangle_right_angle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A + B = 90)
  (h2 : (a + b) * (a - b) = c ^ 2)
  (h3 : A / B = 1 / 2) :
  C = 90 :=
sorry

end triangle_right_angle_l1486_148631


namespace tower_height_proof_l1486_148601

-- Definitions corresponding to given conditions
def elev_angle_A : ℝ := 45
def distance_AD : ℝ := 129
def elev_angle_D : ℝ := 60
def tower_height : ℝ := 305

-- Proving the height of Liaoning Broadcasting and Television Tower
theorem tower_height_proof (h : ℝ) (AC CD : ℝ) (h_eq_AC : h = AC) (h_eq_CD_sqrt3 : h = CD * (Real.sqrt 3)) (AC_CD_sum : AC + CD = 129) :
  h = 305 :=
by
  sorry

end tower_height_proof_l1486_148601


namespace fraction_computation_l1486_148674

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l1486_148674


namespace price_of_remote_controlled_airplane_l1486_148622

theorem price_of_remote_controlled_airplane (x : ℝ) (h : 300 = 0.8 * x) : x = 375 :=
by
  sorry

end price_of_remote_controlled_airplane_l1486_148622


namespace smallest_positive_period_of_f_l1486_148602

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * Real.sin (x + Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x :=
by
  intros
  -- Proof is omitted
  sorry

end smallest_positive_period_of_f_l1486_148602


namespace forester_total_trees_planted_l1486_148606

theorem forester_total_trees_planted (initial_trees monday_trees tuesday_trees wednesday_trees total_trees : ℕ)
    (h1 : initial_trees = 30)
    (h2 : total_trees = 300)
    (h3 : monday_trees = 2 * initial_trees)
    (h4 : tuesday_trees = monday_trees / 3)
    (h5 : wednesday_trees = 2 * tuesday_trees) : 
    (monday_trees + tuesday_trees + wednesday_trees = 120) := by
  sorry

end forester_total_trees_planted_l1486_148606


namespace jerusha_earnings_l1486_148623

variable (L : ℝ) 

theorem jerusha_earnings (h1 : L + 4 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l1486_148623


namespace value_of_g_of_h_at_2_l1486_148621

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -5 * x^3 + 4

theorem value_of_g_of_h_at_2 : g (h 2) = 3890 := by
  sorry

end value_of_g_of_h_at_2_l1486_148621


namespace find_number_l1486_148648

theorem find_number (some_number : ℤ) : 45 - (28 - (some_number - (15 - 19))) = 58 ↔ some_number = 37 := 
by 
  sorry

end find_number_l1486_148648


namespace gcd_problem_l1486_148654

-- Define the two numbers
def a : ℕ := 1000000000
def b : ℕ := 1000000005

-- Define the problem to prove the GCD
theorem gcd_problem : Nat.gcd a b = 5 :=
by 
  sorry

end gcd_problem_l1486_148654


namespace simplify_expression_l1486_148687

theorem simplify_expression :
  (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 :=
by
  sorry

end simplify_expression_l1486_148687


namespace inequalities_hold_l1486_148624

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b) ≥ 2 := by
  sorry

end inequalities_hold_l1486_148624


namespace question_1_question_2_l1486_148611

open Real

noncomputable def f (x a : ℝ) := abs (x - a) + 3 * x

theorem question_1 :
  {x : ℝ | f x 1 > 3 * x + 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by 
  sorry
  
theorem question_2 (h : {x : ℝ | f x a ≤ 0} = {x : ℝ | x ≤ -1}) :
  a = 2 :=
by 
  sorry

end question_1_question_2_l1486_148611


namespace meal_cost_l1486_148684

theorem meal_cost (total_people total_bill : ℕ) (h1 : total_people = 2 + 5) (h2 : total_bill = 21) :
  total_bill / total_people = 3 := by
  sorry

end meal_cost_l1486_148684


namespace students_more_than_rabbits_by_64_l1486_148663

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end students_more_than_rabbits_by_64_l1486_148663


namespace rectangle_area_l1486_148638

theorem rectangle_area (x : ℝ) (l : ℝ) (h1 : 3 * l = x^2 / 10) : 
  3 * l^2 = 3 * x^2 / 10 :=
by sorry

end rectangle_area_l1486_148638


namespace remaining_area_l1486_148668

-- Definitions based on conditions
def large_rectangle_length (x : ℝ) : ℝ := 2 * x + 8
def large_rectangle_width (x : ℝ) : ℝ := x + 6
def hole_length (x : ℝ) : ℝ := 3 * x - 4
def hole_width (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem remaining_area (x : ℝ) : (large_rectangle_length x) * (large_rectangle_width x) - (hole_length x) * (hole_width x) = -x^2 + 21 * x + 52 :=
by
  -- Proof is skipped
  sorry

end remaining_area_l1486_148668


namespace find_S2012_l1486_148626

section Problem

variable {a : ℕ → ℝ} -- Defining the sequence

-- Conditions
def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (Finset.range n).sum a

axiom a1 : a 1 = 2011
axiom recurrence_relation (n : ℕ) : a n + 2*a (n + 1) + a (n + 2) = 0

-- Proof statement
theorem find_S2012 (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ):
  geometric_sequence a →
  (∀ n, S n = sum_S a n) →
  S 2012 = 0 :=
by
  sorry

end Problem

end find_S2012_l1486_148626


namespace construct_right_triangle_l1486_148628

noncomputable def quadrilateral (A B C D : Type) : Prop :=
∃ (AB BC CA : ℝ), 
AB = BC ∧ BC = CA ∧ 
∃ (angle_D : ℝ), 
angle_D = 30

theorem construct_right_triangle (A B C D : Type) (angle_D: ℝ) (AB BC CA : ℝ) 
    (h1 : AB = BC) (h2 : BC = CA) (h3 : angle_D = 30) : 
    exists DA DB DC : ℝ, (DA * DA) + (DC * DC) = (AD * AD) :=
by sorry

end construct_right_triangle_l1486_148628


namespace fill_in_the_blanks_correctly_l1486_148633

def remote_areas_need : String := "what the remote areas need"
def children : String := "children"
def education : String := "education"
def good_textbooks : String := "good textbooks"

-- Defining the grammatical agreement condition
def subject_verb_agreement (s : String) (v : String) : Prop :=
  (s = remote_areas_need ∧ v = "is") ∨ (s = children ∧ v = "are")

-- The main theorem statement
theorem fill_in_the_blanks_correctly : 
  subject_verb_agreement remote_areas_need "is" ∧ subject_verb_agreement children "are" :=
sorry

end fill_in_the_blanks_correctly_l1486_148633


namespace min_value_l1486_148642

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_l1486_148642


namespace middle_integer_is_five_l1486_148635

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def are_consecutive_odd_integers (a b c : ℤ) : Prop :=
  a < b ∧ b < c ∧ (∃ n : ℤ, a = b - 2 ∧ c = b + 2 ∧ is_odd a ∧ is_odd b ∧ is_odd c)

def sum_is_one_eighth_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_five :
  ∃ (a b c : ℤ), are_consecutive_odd_integers a b c ∧ sum_is_one_eighth_product a b c ∧ b = 5 :=
by
  sorry

end middle_integer_is_five_l1486_148635


namespace sum_divisible_by_12_l1486_148646

theorem sum_divisible_by_12 :
  ((2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12) = 3 := by
  sorry

end sum_divisible_by_12_l1486_148646


namespace circle_represents_valid_a_l1486_148615

theorem circle_represents_valid_a (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2 * a * x - 4 * y + 5 * a = 0) → (a > 4 ∨ a < 1) :=
by
  sorry

end circle_represents_valid_a_l1486_148615


namespace sarees_with_6_shirts_l1486_148645

-- Define the prices of sarees, shirts and the equation parameters
variables (S T : ℕ) (X : ℕ)

-- Define the conditions as hypotheses
def condition1 := 2 * S + 4 * T = 1600
def condition2 := 12 * T = 2400
def condition3 := X * S + 6 * T = 1600

-- Define the theorem to prove X = 1 under these conditions
theorem sarees_with_6_shirts
  (h1 : condition1 S T)
  (h2 : condition2 T)
  (h3 : condition3 S T X) : 
  X = 1 :=
sorry

end sarees_with_6_shirts_l1486_148645


namespace packs_with_extra_red_pencils_eq_3_l1486_148673

def total_packs : Nat := 15
def regular_red_per_pack : Nat := 1
def total_red_pencils : Nat := 21
def extra_red_per_pack : Nat := 2

theorem packs_with_extra_red_pencils_eq_3 :
  ∃ (packs_with_extra : Nat), packs_with_extra * extra_red_per_pack + (total_packs - packs_with_extra) * regular_red_per_pack = total_red_pencils ∧ packs_with_extra = 3 :=
by
  sorry

end packs_with_extra_red_pencils_eq_3_l1486_148673


namespace cube_root_inequality_l1486_148607

theorem cube_root_inequality (a b : ℝ) (h : a > b) : (a ^ (1/3)) > (b ^ (1/3)) :=
sorry

end cube_root_inequality_l1486_148607


namespace rational_number_addition_l1486_148666

theorem rational_number_addition :
  (-206 : ℚ) + (401 + 3 / 4) + (-(204 + 2 / 3)) + (-(1 + 1 / 2)) = -10 - 5 / 12 :=
by
  sorry

end rational_number_addition_l1486_148666


namespace expression_square_minus_three_times_l1486_148616

-- Defining the statement
theorem expression_square_minus_three_times (a b : ℝ) : a^2 - 3 * b = a^2 - 3 * b := 
by
  sorry

end expression_square_minus_three_times_l1486_148616


namespace compute_fraction_l1486_148683

theorem compute_fraction (a b c : ℝ) (h1 : a + b = 20) (h2 : b + c = 22) (h3 : c + a = 2022) :
  (a - b) / (c - a) = 1000 :=
by
  sorry

end compute_fraction_l1486_148683


namespace sum_g_squared_l1486_148694

noncomputable def g (n : ℕ) : ℝ :=
  ∑' m, if m ≥ 3 then 1 / (m : ℝ)^n else 0

theorem sum_g_squared :
  (∑' n, if n ≥ 3 then (g n)^2 else 0) = 1 / 288 :=
by
  sorry

end sum_g_squared_l1486_148694


namespace total_cakes_served_l1486_148692

theorem total_cakes_served (l : ℝ) (p : ℝ) (s : ℝ) (total_cakes_served_today : ℝ) :
  l = 48.5 → p = 0.6225 → s = 95 → total_cakes_served_today = 108 :=
by
  intros hl hp hs
  sorry

end total_cakes_served_l1486_148692


namespace problem_conditions_l1486_148676

open Real

variable {m n : ℝ}

theorem problem_conditions (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 2 * m * n) :
  (min (m + n) = 2) ∧ (min (sqrt (m * n)) = 1) ∧
  (min ((n^2) / m + (m^2) / n) = 2) ∧ 
  (max ((sqrt m + sqrt n) / sqrt (m * n)) = 2) :=
by sorry

end problem_conditions_l1486_148676


namespace tile_difference_correct_l1486_148618

def initial_blue_tiles := 23
def initial_green_tiles := 16
def first_border_green_tiles := 6 * 1
def second_border_green_tiles := 6 * 2
def total_green_tiles := initial_green_tiles + first_border_green_tiles + second_border_green_tiles
def difference_tiling := total_green_tiles - initial_blue_tiles

theorem tile_difference_correct : difference_tiling = 11 := by
  sorry

end tile_difference_correct_l1486_148618


namespace common_root_equation_l1486_148686

theorem common_root_equation (a b r : ℝ) (h₁ : a ≠ b)
  (h₂ : r^2 + 2019 * a * r + b = 0)
  (h₃ : r^2 + 2019 * b * r + a = 0) :
  r = 1 / 2019 :=
by
  sorry

end common_root_equation_l1486_148686


namespace hexagon_perimeter_l1486_148680

-- Define the length of a side of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def perimeter (num_sides side_length : ℕ) : ℕ :=
  num_sides * side_length

-- Theorem stating the perimeter of the hexagon with given side length is 42 inches
theorem hexagon_perimeter : perimeter num_sides side_length = 42 := by
  sorry

end hexagon_perimeter_l1486_148680


namespace probability_of_non_touching_square_is_correct_l1486_148672

def square_not_touching_perimeter_or_center_probability : ℚ :=
  let total_squares := 100
  let perimeter_squares := 24
  let center_line_squares := 16
  let touching_squares := perimeter_squares + center_line_squares
  let non_touching_squares := total_squares - touching_squares
  non_touching_squares / total_squares

theorem probability_of_non_touching_square_is_correct :
  square_not_touching_perimeter_or_center_probability = 3 / 5 :=
by
  sorry

end probability_of_non_touching_square_is_correct_l1486_148672


namespace second_divisor_13_l1486_148689

theorem second_divisor_13 (N D : ℤ) (k m : ℤ) 
  (h1 : N = 39 * k + 17) 
  (h2 : N = D * m + 4) : 
  D = 13 := 
sorry

end second_divisor_13_l1486_148689


namespace smallest_sector_angle_24_l1486_148625

theorem smallest_sector_angle_24
  (a : ℕ) (d : ℕ)
  (h1 : ∀ i, i < 8 → ((a + i * d) : ℤ) > 0)
  (h2 : (2 * a + 7 * d = 90)) : a = 24 :=
by
  sorry

end smallest_sector_angle_24_l1486_148625


namespace last_score_is_87_l1486_148675

-- Definitions based on conditions:
def scores : List ℕ := [73, 78, 82, 84, 87, 95]
def total_sum := 499
def final_median := 83

-- Prove that the last score is 87 under given conditions.
theorem last_score_is_87 (h1 : total_sum = 499)
                        (h2 : ∀ n ∈ scores, (499 - n) % 6 ≠ 0)
                        (h3 : final_median = 83) :
  87 ∈ scores := sorry

end last_score_is_87_l1486_148675


namespace range_y_over_x_l1486_148697

theorem range_y_over_x {x y : ℝ} (h : (x-4)^2 + (y-2)^2 ≤ 4) : 
  ∃ k : ℝ, k = y / x ∧ 0 ≤ k ∧ k ≤ 4/3 :=
sorry

end range_y_over_x_l1486_148697


namespace find_a6_l1486_148639

variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (h1 : ∀ n ≥ 2, S n = 2 * a n)
variable (h2 : S 5 = 8)

theorem find_a6 : a 6 = 8 :=
by
  sorry

end find_a6_l1486_148639


namespace no_p_safe_numbers_l1486_148605

/-- A number n is p-safe if it differs in absolute value by more than 2 from all multiples of p. -/
def p_safe (n p : ℕ) : Prop := ∀ k : ℤ, abs (n - k * p) > 2 

/-- The main theorem stating that there are no numbers that are simultaneously 5-safe, 
    7-safe, and 9-safe from 1 to 15000. -/
theorem no_p_safe_numbers (n : ℕ) (hp : 1 ≤ n ∧ n ≤ 15000) : 
  ¬ (p_safe n 5 ∧ p_safe n 7 ∧ p_safe n 9) :=
sorry

end no_p_safe_numbers_l1486_148605


namespace part1_part2_l1486_148698

def setA : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def setB (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

theorem part1 : (setB (-1)).union setA = {x : ℝ | -1 < x ∧ x < 3 } := by
  sorry

theorem part2 (k : ℝ) : (setA ∩ setB k = setB k ↔ 0 ≤ k) := by
  sorry

end part1_part2_l1486_148698


namespace scientific_notation_of_18M_l1486_148632

theorem scientific_notation_of_18M : 18000000 = 1.8 * 10^7 :=
by
  sorry

end scientific_notation_of_18M_l1486_148632


namespace parallel_vectors_tan_l1486_148693

/-- Given vector a and vector b, and given the condition that a is parallel to b,
prove that the value of tan α is 1/4. -/
theorem parallel_vectors_tan (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.sin α, Real.cos α - 2 * Real.sin α))
  (hb : b = (1, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) : 
  Real.tan α = 1 / 4 := 
by 
  sorry

end parallel_vectors_tan_l1486_148693


namespace expected_value_of_fair_dodecahedral_die_l1486_148678

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l1486_148678


namespace stock_price_end_second_year_l1486_148669

theorem stock_price_end_second_year
  (P₀ : ℝ) (r₁ r₂ : ℝ) 
  (h₀ : P₀ = 150)
  (h₁ : r₁ = 0.80)
  (h₂ : r₂ = 0.30) :
  let P₁ := P₀ + r₁ * P₀
  let P₂ := P₁ - r₂ * P₁
  P₂ = 189 :=
by
  sorry

end stock_price_end_second_year_l1486_148669


namespace mryak_bryak_problem_l1486_148661

variable (m b : ℚ)

theorem mryak_bryak_problem
  (h1 : 3 * m = 5 * b + 10)
  (h2 : 6 * m = 8 * b + 31) :
  7 * m - 9 * b = 38 := sorry

end mryak_bryak_problem_l1486_148661


namespace find_smaller_number_l1486_148600

theorem find_smaller_number (x y : ℕ) (h₁ : y - x = 1365) (h₂ : y = 6 * x + 15) : x = 270 :=
sorry

end find_smaller_number_l1486_148600


namespace triangle_BC_length_l1486_148634

theorem triangle_BC_length
  (y_eq_2x2 : ∀ (x : ℝ), ∃ (y : ℝ), y = 2 * x ^ 2)
  (area_ABC : ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧ (∃ (a : ℝ), B = (a, 2 * a ^ 2) ∧ C = (-a, 2 * a ^ 2) ∧ 2 * a ^ 3 = 128))
  : ∃ (a : ℝ), 2 * a = 8 := 
sorry

end triangle_BC_length_l1486_148634


namespace magnitude_of_complex_l1486_148682

open Complex

theorem magnitude_of_complex :
  abs (Complex.mk (2/3) (-4/5)) = Real.sqrt 244 / 15 :=
by
  -- Placeholder for the actual proof
  sorry

end magnitude_of_complex_l1486_148682


namespace second_number_is_46_l1486_148667

theorem second_number_is_46 (sum_is_330 : ∃ (a b c d : ℕ), a + b + c + d = 330)
    (first_is_twice_second : ∀ (b : ℕ), ∃ (a : ℕ), a = 2 * b)
    (third_is_one_third_of_first : ∀ (a : ℕ), ∃ (c : ℕ), c = a / 3)
    (fourth_is_half_difference : ∀ (a b : ℕ), ∃ (d : ℕ), d = (a - b) / 2) :
  ∃ (b : ℕ), b = 46 :=
by
  -- Proof goes here, inserted for illustrating purposes only
  sorry

end second_number_is_46_l1486_148667


namespace smallest_a_l1486_148644

def f (x : ℕ) : ℕ :=
  if x % 21 = 0 then x / 21
  else if x % 7 = 0 then 3 * x
  else if x % 3 = 0 then 7 * x
  else x + 3

def f_iterate (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate f a x

theorem smallest_a (a : ℕ) : a > 1 ∧ f_iterate a 2 = f 2 ↔ a = 7 := 
sorry

end smallest_a_l1486_148644


namespace solve_system_l1486_148617

theorem solve_system (x y : ℚ) 
  (h₁ : 7 * x - 14 * y = 3) 
  (h₂ : 3 * y - x = 5) : 
  x = 79 / 7 ∧ y = 38 / 7 := 
by 
  sorry

end solve_system_l1486_148617


namespace area_of_set_K_l1486_148664

open Metric

def set_K :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Define the area function for a general set s

theorem area_of_set_K : area set_K = 24 :=
  sorry

end area_of_set_K_l1486_148664
