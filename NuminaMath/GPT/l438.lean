import Mathlib

namespace gcd_possible_values_count_l438_43868

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l438_43868


namespace first_discount_percentage_l438_43889

theorem first_discount_percentage (x : ℝ) 
  (h₁ : ∀ (p : ℝ), p = 70) 
  (h₂ : ∀ (d₁ d₂ : ℝ), d₁ = x / 100 ∧ d₂ = 0.01999999999999997 )
  (h₃ : ∀ (final_price : ℝ), final_price = 61.74):
  x = 10 := 
by
  sorry

end first_discount_percentage_l438_43889


namespace friends_behind_Yuna_l438_43893

def total_friends : ℕ := 6
def friends_in_front_of_Yuna : ℕ := 2

theorem friends_behind_Yuna : total_friends - friends_in_front_of_Yuna = 4 :=
by
  -- Proof goes here
  sorry

end friends_behind_Yuna_l438_43893


namespace hot_dogs_served_today_l438_43870

-- Define the number of hot dogs served during lunch
def h_dogs_lunch : ℕ := 9

-- Define the number of hot dogs served during dinner
def h_dogs_dinner : ℕ := 2

-- Define the total number of hot dogs served today
def total_h_dogs : ℕ := h_dogs_lunch + h_dogs_dinner

-- Theorem stating that the total number of hot dogs served today is 11
theorem hot_dogs_served_today : total_h_dogs = 11 := by
  sorry

end hot_dogs_served_today_l438_43870


namespace max_knights_among_10_l438_43806

def is_knight (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (p m ↔ (m ≥ n))

def is_liar (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (¬ p m ↔ (m ≥ n))

def greater_than (k : ℕ) (n : ℕ) := n > k

def less_than (k : ℕ) (n : ℕ) := n < k

def person_statement_1 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => greater_than 1 n
  | 2 => greater_than 2 n
  | 3 => greater_than 3 n
  | 4 => greater_than 4 n
  | 5 => greater_than 5 n
  | 6 => greater_than 6 n
  | 7 => greater_than 7 n
  | 8 => greater_than 8 n
  | 9 => greater_than 9 n
  | 10 => greater_than 10 n
  | _ => false

def person_statement_2 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => less_than 1 n
  | 2 => less_than 2 n
  | 3 => less_than 3 n
  | 4 => less_than 4 n
  | 5 => less_than 5 n
  | 6 => less_than 6 n
  | 7 => less_than 7 n
  | 8 => less_than 8 n
  | 9 => less_than 9 n
  | 10 => less_than 10 n
  | _ => false

theorem max_knights_among_10 (knights : ℕ) : 
  (∀ i < 10, (is_knight (person_statement_1 (i + 1)) (i + 1) ∨ is_liar (person_statement_1 (i + 1)) (i + 1))) ∧
  (∀ i < 10, (is_knight (person_statement_2 (i + 1)) (i + 1) ∨ is_liar (person_statement_2 (i + 1)) (i + 1))) →
  knights ≤ 8 := sorry

end max_knights_among_10_l438_43806


namespace find_distance_from_home_to_airport_l438_43829

variable (d t : ℝ)

-- Conditions
def condition1 := d = 40 * (t + 0.75)
def condition2 := d - 40 = 60 * (t - 1.25)

-- Proof statement
theorem find_distance_from_home_to_airport (hd : condition1 d t) (ht : condition2 d t) : d = 160 :=
by
  sorry

end find_distance_from_home_to_airport_l438_43829


namespace gnollish_valid_sentences_count_is_50_l438_43809

def gnollish_words : List String := ["splargh", "glumph", "amr", "blort"]

def is_valid_sentence (sentence : List String) : Prop :=
  match sentence with
  | [_, "splargh", "glumph"] => False
  | ["splargh", "glumph", _] => False
  | [_, "blort", "amr"] => False
  | ["blort", "amr", _] => False
  | _ => True

def count_valid_sentences (n : Nat) : Nat :=
  (List.replicate n gnollish_words).mapM id |>.length

theorem gnollish_valid_sentences_count_is_50 : count_valid_sentences 3 = 50 :=
by 
  sorry

end gnollish_valid_sentences_count_is_50_l438_43809


namespace range_of_a_l438_43886

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by
  sorry

end range_of_a_l438_43886


namespace base_seven_sum_l438_43812

def base_seven_sum_of_product (n m : ℕ) : ℕ :=
  let product := n * m
  let digits := product.digits 7
  digits.sum

theorem base_seven_sum (k l : ℕ) (hk : k = 5 * 7 + 3) (hl : l = 343) :
  base_seven_sum_of_product k l = 11 := by
  sorry

end base_seven_sum_l438_43812


namespace geometric_sequence_a4_l438_43880

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ (n : ℕ), a (n + 1) = a n * r

def a_3a_5_is_64 (a : ℕ → ℝ) : Prop :=
  a 3 * a 5 = 64

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h1 : is_geometric_sequence a) (h2 : a_3a_5_is_64 a) : a 4 = 8 ∨ a 4 = -8 :=
by
  sorry

end geometric_sequence_a4_l438_43880


namespace polynomial_factorization_l438_43839

noncomputable def poly_1 : Polynomial ℤ := (Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1)
noncomputable def poly_2 : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X ^ 12 - Polynomial.C 1 * Polynomial.X ^ 11 +
  Polynomial.C 1 * Polynomial.X ^ 9 - Polynomial.C 1 * Polynomial.X ^ 8 +
  Polynomial.C 1 * Polynomial.X ^ 6 - Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1
noncomputable def polynomial_expression : Polynomial ℤ := Polynomial.X ^ 15 + Polynomial.X ^ 10 + Polynomial.C 1

theorem polynomial_factorization : polynomial_expression = poly_1 * poly_2 :=
  by { sorry }

end polynomial_factorization_l438_43839


namespace problem_f_neg2_l438_43856

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2007 + b * x + 1

theorem problem_f_neg2 (a b : ℝ) (h : f a b 2 = 2) : f a b (-2) = 0 :=
by
  sorry

end problem_f_neg2_l438_43856


namespace margin_in_terms_of_selling_price_l438_43878

variable (C S M : ℝ) (n : ℕ) (h : M = (1 / 2) * (S - (1 / n) * C))

theorem margin_in_terms_of_selling_price :
  M = ((n - 1) / (2 * n - 1)) * S :=
sorry

end margin_in_terms_of_selling_price_l438_43878


namespace two_p_in_S_l438_43804

def is_in_S (a b : ℤ) : Prop :=
  ∃ k : ℤ, k = a^2 + 5 * b^2 ∧ Int.gcd a b = 1

def S : Set ℤ := { x | ∃ a b : ℤ, is_in_S a b ∧ a^2 + 5 * b^2 = x }

theorem two_p_in_S (k p n : ℤ) (hp1 : p = 4 * n + 3) (hp2 : Nat.Prime (Int.natAbs p))
  (hk : 0 < k) (hkp : k * p ∈ S) : 2 * p ∈ S := 
sorry

end two_p_in_S_l438_43804


namespace num_valid_functions_l438_43883

theorem num_valid_functions :
  ∃! (f : ℤ → ℝ), 
  (f 1 = 1) ∧ 
  (∀ (m n : ℤ), f m ^ 2 - f n ^ 2 = f (m + n) * f (m - n)) ∧ 
  (∀ n : ℤ, f n = f (n + 2013)) :=
sorry

end num_valid_functions_l438_43883


namespace circle_represents_range_l438_43842

theorem circle_represents_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 2 * y + 3 = 0 → (m > 2 * Real.sqrt 2 ∨ m < -2 * Real.sqrt 2)) :=
by
  sorry

end circle_represents_range_l438_43842


namespace evaluate_expression_l438_43847

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end evaluate_expression_l438_43847


namespace binomial_60_3_eq_34220_l438_43833

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l438_43833


namespace first_die_sides_l438_43822

theorem first_die_sides (n : ℕ) 
  (h_prob : (1 : ℝ) / n * (1 : ℝ) / 7 = 0.023809523809523808) : 
  n = 6 := by
  sorry

end first_die_sides_l438_43822


namespace verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l438_43888

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Definition for conversions from base 60 to base 10
def from_base_60 (d1 d0 : ℕ) : ℕ :=
  d1 * 60 + d0

-- Proof statements
theorem verify_21_base_60 : from_base_60 2 1 = 121 ∧ is_perfect_square 121 :=
by {
  sorry
}

theorem verify_1_base_60 : from_base_60 0 1 = 1 ∧ is_perfect_square 1 :=
by {
  sorry
}

theorem verify_2_base_60_not_square : from_base_60 0 2 = 2 ∧ ¬ is_perfect_square 2 :=
by {
  sorry
}

end verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l438_43888


namespace probability_five_digit_palindrome_div_by_11_l438_43897

noncomputable
def five_digit_palindrome_div_by_11_probability : ℚ :=
  let total_palindromes := 900
  let valid_palindromes := 80
  valid_palindromes / total_palindromes

theorem probability_five_digit_palindrome_div_by_11 :
  five_digit_palindrome_div_by_11_probability = 2 / 25 := by
  sorry

end probability_five_digit_palindrome_div_by_11_l438_43897


namespace range_of_a_l438_43808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(1 + a * x) - x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a (f a x) - x

theorem range_of_a (a : ℝ) : (F a 0 = 0 → F a e = 0) → 
  (0 < a ∧ a < (1 / (Real.exp 1 * Real.log 2))) :=
by
  sorry

end range_of_a_l438_43808


namespace clare_money_left_l438_43837

noncomputable def cost_of_bread : ℝ := 4 * 2
noncomputable def cost_of_milk : ℝ := 2 * 2
noncomputable def cost_of_cereal : ℝ := 3 * 3
noncomputable def cost_of_apples : ℝ := 1 * 4

noncomputable def total_cost_before_discount : ℝ := cost_of_bread + cost_of_milk + cost_of_cereal + cost_of_apples
noncomputable def discount_amount : ℝ := total_cost_before_discount * 0.1
noncomputable def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount
noncomputable def sales_tax : ℝ := total_cost_after_discount * 0.05
noncomputable def total_cost_after_discount_and_tax : ℝ := total_cost_after_discount + sales_tax

noncomputable def initial_amount : ℝ := 47
noncomputable def money_left : ℝ := initial_amount - total_cost_after_discount_and_tax

theorem clare_money_left : money_left = 23.37 := by
  sorry

end clare_money_left_l438_43837


namespace newton_method_approximation_bisection_method_approximation_l438_43834

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 3

theorem newton_method_approximation :
  let x0 := -1
  let x1 := x0 - f x0 / f' x0
  let x2 := x1 - f x1 / f' x1
  x2 = -7 / 5 := sorry

theorem bisection_method_approximation :
  let a := -2
  let b := -1
  let midpoint1 := (a + b) / 2
  let new_a := if f midpoint1 < 0 then midpoint1 else a
  let new_b := if f midpoint1 < 0 then b else midpoint1
  let midpoint2 := (new_a + new_b) / 2
  midpoint2 = -11 / 8 := sorry

end newton_method_approximation_bisection_method_approximation_l438_43834


namespace balls_in_third_pile_l438_43827

theorem balls_in_third_pile (a b c x : ℕ) (h1 : a + b + c = 2012) (h2 : b - x = 17) (h3 : a - x = 2 * (c - x)) : c = 665 := by
  sorry

end balls_in_third_pile_l438_43827


namespace area_of_shaded_region_l438_43846

open Real

noncomputable def line1 (x : ℝ) : ℝ := -3/10 * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -1.5 * x + 9

theorem area_of_shaded_region : 
  ∫ x in (2:ℝ)..6, (line2 x - line1 x) = 8 :=
by
  sorry

end area_of_shaded_region_l438_43846


namespace distance_between_city_and_village_l438_43807

variables (S x y : ℝ)

theorem distance_between_city_and_village (h1 : S / 2 - 2 = y * S / (2 * x))
    (h2 : 2 * S / 3 + 2 = x * S / (3 * y)) : S = 6 :=
by
  sorry

end distance_between_city_and_village_l438_43807


namespace money_has_48_l438_43810

-- Definitions derived from conditions:
def money (p : ℝ) := 
  p = (1/3 * p) + 32

-- The main theorem statement
theorem money_has_48 (p : ℝ) : money p → p = 48 := by
  intro h
  -- Skipping the proof
  sorry

end money_has_48_l438_43810


namespace sum_of_interior_angles_of_pentagon_l438_43813

theorem sum_of_interior_angles_of_pentagon :
    (5 - 2) * 180 = 540 := by 
  -- The proof goes here
  sorry

end sum_of_interior_angles_of_pentagon_l438_43813


namespace cost_price_of_ball_l438_43884

theorem cost_price_of_ball (x : ℕ) (h : 13 * x = 720 + 5 * x) : x = 90 :=
by sorry

end cost_price_of_ball_l438_43884


namespace bahs_for_1000_yahs_l438_43862

-- Definitions based on given conditions
def bahs_to_rahs_ratio (b r : ℕ) := 15 * b = 24 * r
def rahs_to_yahs_ratio (r y : ℕ) := 9 * r = 15 * y

-- Main statement to prove
theorem bahs_for_1000_yahs (b r y : ℕ) (h1 : bahs_to_rahs_ratio b r) (h2 : rahs_to_yahs_ratio r y) :
  1000 * y = 375 * b :=
by
  sorry

end bahs_for_1000_yahs_l438_43862


namespace impossible_result_l438_43887

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬ (∃ f1 f_1 : ℤ, f1 = a * Real.sin 1 + b + c ∧ f_1 = -a * Real.sin 1 - b + c ∧ (f1 = 1 ∧ f_1 = 2)) :=
by
  sorry

end impossible_result_l438_43887


namespace back_seat_people_l438_43874

-- Define the problem conditions

def leftSideSeats : ℕ := 15
def seatDifference : ℕ := 3
def peoplePerSeat : ℕ := 3
def totalBusCapacity : ℕ := 88

-- Define the formula for calculating the people at the back seat
def peopleAtBackSeat := 
  totalBusCapacity - ((leftSideSeats * peoplePerSeat) + ((leftSideSeats - seatDifference) * peoplePerSeat))

-- The statement we need to prove
theorem back_seat_people : peopleAtBackSeat = 7 :=
by
  sorry

end back_seat_people_l438_43874


namespace circle_equation_l438_43882

theorem circle_equation 
  (P : ℝ × ℝ)
  (h1 : ∀ a : ℝ, (1 - a) * 2 + (P.snd) + 2 * a - 1 = 0)
  (h2 : P = (2, -1)) :
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end circle_equation_l438_43882


namespace p_necessary_not_sufficient_for_q_l438_43852

open Real

noncomputable def p (x : ℝ) : Prop := |x| < 3
noncomputable def q (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l438_43852


namespace find_m_for_one_solution_l438_43843

theorem find_m_for_one_solution (m : ℚ) :
  (∀ x : ℝ, 3*x^2 - 7*x + m = 0 → (∃! y : ℝ, 3*y^2 - 7*y + m = 0)) → m = 49/12 := by
  sorry

end find_m_for_one_solution_l438_43843


namespace arithmetic_series_sum_l438_43863

theorem arithmetic_series_sum :
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  sum = 418 := by {
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  have h₁ : n = 11 := by sorry
  have h₂ : sum = 418 := by sorry
  exact h₂
}

end arithmetic_series_sum_l438_43863


namespace movie_ticket_cost_l438_43849

/--
Movie tickets cost a certain amount on a Monday, twice as much on a Wednesday, and five times as much as on Monday on a Saturday. If Glenn goes to the movie theater on Wednesday and Saturday, he spends $35. Prove that the cost of a movie ticket on a Monday is $5.
-/
theorem movie_ticket_cost (M : ℕ) 
  (wednesday_cost : 2 * M = 2 * M)
  (saturday_cost : 5 * M = 5 * M) 
  (total_cost : 2 * M + 5 * M = 35) : 
  M = 5 := 
sorry

end movie_ticket_cost_l438_43849


namespace parallel_vectors_l438_43890

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (P : a = (1, m) ∧ b = (m, 2) ∧ (a.1 / m = b.1 / 2)) :
  m = -Real.sqrt 2 ∨ m = Real.sqrt 2 :=
by
  sorry

end parallel_vectors_l438_43890


namespace tangent_line_at_0_2_is_correct_l438_43891

noncomputable def curve (x : ℝ) : ℝ := Real.exp (-2 * x) + 1

def tangent_line_at_0_2 (x : ℝ) : ℝ := -2 * x + 2

theorem tangent_line_at_0_2_is_correct :
  tangent_line_at_0_2 = fun x => -2 * x + 2 :=
by {
  sorry
}

end tangent_line_at_0_2_is_correct_l438_43891


namespace algebraic_expression_simplification_l438_43835

theorem algebraic_expression_simplification (x y : ℝ) (h : x + y = 1) : x^3 + y^3 + 3 * x * y = 1 := 
by
  sorry

end algebraic_expression_simplification_l438_43835


namespace intersection_point_and_distance_l438_43845

/-- Define the points A, B, C, D, and M based on the specified conditions. --/
def A := (0, 3)
def B := (6, 3)
def C := (6, 0)
def D := (0, 0)
def M := (3, 0)

/-- Define the equations of the circles. --/
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 2.25
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 25

/-- The point P that is one of the intersection points of the two circles. --/
def P := (2, 1.5)

/-- Define the line AD as the y-axis. --/
def AD := 0

/-- Calculate the distance from point P to the y-axis (AD). --/
def distance_to_ad (x : ℝ) := |x|

theorem intersection_point_and_distance :
  circle1 (2 : ℝ) (1.5 : ℝ) ∧ circle2 (2 : ℝ) (1.5 : ℝ) ∧ distance_to_ad 2 = 2 :=
by
  unfold circle1 circle2 distance_to_ad
  norm_num
  sorry

end intersection_point_and_distance_l438_43845


namespace triangle_angle_120_l438_43866

theorem triangle_angle_120 (a b c : ℝ) (B : ℝ) (hB : B = 120) :
  a^2 + a * c + c^2 - b^2 = 0 := by
sorry

end triangle_angle_120_l438_43866


namespace cos_sum_seventh_roots_of_unity_l438_43821

noncomputable def cos_sum (α : ℝ) : ℝ := 
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)

theorem cos_sum_seventh_roots_of_unity (z : ℂ) (α : ℝ)
  (hz : z^7 = 1) (hz_ne_one : z ≠ 1) (hα : z = Complex.exp (Complex.I * α)) :
  cos_sum α = -1/2 :=
by
  sorry

end cos_sum_seventh_roots_of_unity_l438_43821


namespace sqrt_144_times_3_squared_l438_43823

theorem sqrt_144_times_3_squared :
  ( (Real.sqrt 144) * 3 ) ^ 2 = 1296 := by
  sorry

end sqrt_144_times_3_squared_l438_43823


namespace problem_statement_l438_43805

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f'' (x : ℝ) : ℝ := -Real.sin x - Real.cos x

theorem problem_statement (a : ℝ) (h : f'' a = 3 * f a) : 
  (Real.sin a)^2 - 3 / (Real.cos a)^2 + 1 = -14 / 9 := 
sorry

end problem_statement_l438_43805


namespace div_by_90_l438_43876

def N : ℤ := 19^92 - 91^29

theorem div_by_90 : ∃ k : ℤ, N = 90 * k := 
sorry

end div_by_90_l438_43876


namespace number_of_triangles_l438_43828

theorem number_of_triangles (n : ℕ) : 
  ∃ k : ℕ, k = ⌊((n + 1) * (n + 3) * (2 * n + 1) : ℝ) / 24⌋ := sorry

end number_of_triangles_l438_43828


namespace order_of_f0_f1_f_2_l438_43826

noncomputable def f (m x : ℝ) := (m-1) * x^2 + 6 * m * x + 2

theorem order_of_f0_f1_f_2 (m : ℝ) (h_even : ∀ x : ℝ, f m x = f m (-x)) :
  m = 0 → f m (-2) < f m 1 ∧ f m 1 < f m 0 :=
by 
  sorry

end order_of_f0_f1_f_2_l438_43826


namespace maximum_value_l438_43858

theorem maximum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (a / (a + 1) + b / (b + 2) ≤ (5 - 2 * Real.sqrt 2) / 4) :=
sorry

end maximum_value_l438_43858


namespace correct_number_of_sequences_l438_43817

noncomputable def athlete_sequences : Nat :=
  let total_permutations := 24
  let A_first_leg := 6
  let B_fourth_leg := 6
  let A_first_and_B_fourth := 2
  total_permutations - (A_first_leg + B_fourth_leg - A_first_and_B_fourth)

theorem correct_number_of_sequences : athlete_sequences = 14 := by
  sorry

end correct_number_of_sequences_l438_43817


namespace maximum_m_value_l438_43859

theorem maximum_m_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ m, m = 4 ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (1 / a + 1 / b) ≥ m) :=
sorry

end maximum_m_value_l438_43859


namespace determine_cost_price_l438_43818

def selling_price := 16
def loss_fraction := 1 / 6

noncomputable def cost_price (CP : ℝ) : Prop :=
  selling_price = CP - (loss_fraction * CP)

theorem determine_cost_price (CP : ℝ) (h: cost_price CP) : CP = 19.2 := by
  sorry

end determine_cost_price_l438_43818


namespace find_deaf_students_l438_43840

-- Definitions based on conditions
variables (B D : ℕ)
axiom deaf_students_triple_blind_students : D = 3 * B
axiom total_students : D + B = 240

-- Proof statement
theorem find_deaf_students (h1 : D = 3 * B) (h2 : D + B = 240) : D = 180 :=
sorry

end find_deaf_students_l438_43840


namespace parking_spaces_remaining_l438_43811

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end parking_spaces_remaining_l438_43811


namespace repeating_decimal_product_l438_43851

theorem repeating_decimal_product (x : ℚ) (h : x = 4 / 9) : x * 9 = 4 := 
by
  sorry

end repeating_decimal_product_l438_43851


namespace sequence_sum_l438_43899

theorem sequence_sum (a : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, 0 < a n)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 2) = 1 + 1 / a n)
  (h₃ : a 2014 = a 2016) :
  a 13 + a 2016 = 21 / 13 + (1 + Real.sqrt 5) / 2 :=
sorry

end sequence_sum_l438_43899


namespace find_m_l438_43867

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def symmetric_about_line (x1 y1 x2 y2 m : ℝ) : Prop := (y1 - y2) / (x1 - x2) = -1
def product_y (y1 y2 : ℝ) : Prop := y1 * y2 = -1 / 2

-- Theorem to be proven
theorem find_m 
  (x1 y1 x2 y2 m : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : symmetric_about_line x1 y1 x2 y2 m)
  (h4 : product_y y1 y2) :
  m = 9 / 4 :=
sorry

end find_m_l438_43867


namespace math_problem_l438_43820

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l438_43820


namespace trapezoid_QR_length_l438_43873

variable (PQ RS Area Alt QR : ℝ)
variable (h1 : Area = 216)
variable (h2 : Alt = 9)
variable (h3 : PQ = 12)
variable (h4 : RS = 20)
variable (h5 : QR = 11)

theorem trapezoid_QR_length : 
  (∃ (PQ RS Area Alt QR : ℝ), 
    Area = 216 ∧
    Alt = 9 ∧
    PQ = 12 ∧
    RS = 20) → QR = 11 :=
by
  sorry

end trapezoid_QR_length_l438_43873


namespace addition_value_l438_43869

def certain_number : ℝ := 5.46 - 3.97

theorem addition_value : 5.46 + certain_number = 6.95 := 
  by 
    -- The proof would go here, but is replaced with sorry.
    sorry

end addition_value_l438_43869


namespace second_number_is_90_l438_43857

theorem second_number_is_90 (a b c : ℕ) 
  (h1 : a + b + c = 330) 
  (h2 : a = 2 * b) 
  (h3 : c = (1 / 3) * a) : 
  b = 90 := 
by
  sorry

end second_number_is_90_l438_43857


namespace angle_B_equal_pi_div_3_l438_43872

-- Define the conditions and the statement to be proved
theorem angle_B_equal_pi_div_3 (A B C : ℝ) 
  (h₁ : Real.sin A / Real.sin B = 5 / 7)
  (h₂ : Real.sin B / Real.sin C = 7 / 8) : 
  B = Real.pi / 3 :=
sorry

end angle_B_equal_pi_div_3_l438_43872


namespace xiaoGong_walking_speed_l438_43895

-- Defining the parameters for the problem
def distance : ℕ := 1200
def daChengExtraSpeedPerMinute : ℕ := 20
def timeUntilMeetingForDaCheng : ℕ := 12
def timeUntilMeetingForXiaoGong : ℕ := 6 + timeUntilMeetingForDaCheng

-- The main statement to prove Xiao Gong's speed
theorem xiaoGong_walking_speed : ∃ v : ℕ, 12 * (v + daChengExtraSpeedPerMinute) + 18 * v = distance ∧ v = 32 :=
by
  sorry

end xiaoGong_walking_speed_l438_43895


namespace union_of_sets_l438_43854

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2}) (hB : B = {2, 3}) : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l438_43854


namespace vector_sum_length_l438_43853

open Real

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vector_angle_cosine (v w : ℝ × ℝ) : ℝ :=
dot_product v w / (vector_length v * vector_length w)

theorem vector_sum_length (a b : ℝ × ℝ)
  (ha : vector_length a = 2)
  (hb : vector_length b = 2)
  (hab_angle : vector_angle_cosine a b = cos (π / 3)):
  vector_length (a.1 + b.1, a.2 + b.2) = 2 * sqrt 3 :=
by sorry

end vector_sum_length_l438_43853


namespace agatha_remaining_amount_l438_43824

theorem agatha_remaining_amount :
  let initial_amount := 60
  let frame_price := 15
  let frame_discount := 0.10 * frame_price
  let frame_final := frame_price - frame_discount
  let wheel_price := 25
  let wheel_discount := 0.05 * wheel_price
  let wheel_final := wheel_price - wheel_discount
  let seat_price := 8
  let seat_discount := 0.15 * seat_price
  let seat_final := seat_price - seat_discount
  let tape_price := 5
  let total_spent := frame_final + wheel_final + seat_final + tape_price
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 10.95 :=
by
  sorry

end agatha_remaining_amount_l438_43824


namespace greatest_divisor_6215_7373_l438_43855

theorem greatest_divisor_6215_7373 : 
  Nat.gcd (6215 - 23) (7373 - 29) = 144 := by
  sorry

end greatest_divisor_6215_7373_l438_43855


namespace find_unknown_rate_l438_43881

theorem find_unknown_rate :
    let n := 7 -- total number of blankets
    let avg_price := 150 -- average price of the blankets
    let total_price := n * avg_price
    let cost1 := 3 * 100
    let cost2 := 2 * 150
    let remaining := total_price - (cost1 + cost2)
    remaining / 2 = 225 :=
by sorry

end find_unknown_rate_l438_43881


namespace isosceles_triangle_time_between_9_30_and_10_l438_43841

theorem isosceles_triangle_time_between_9_30_and_10 (time : ℕ) (h_time_range : 30 ≤ time ∧ time < 60)
  (h_isosceles : ∃ x : ℝ, 0 ≤ x ∧ x + 2 * x + 2 * x = 180) :
  time = 36 :=
  sorry

end isosceles_triangle_time_between_9_30_and_10_l438_43841


namespace necessary_but_not_sufficient_l438_43877

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 > 4) → (x > 2 ∨ x < -2) ∧ ¬((x^2 > 4) ↔ (x > 2)) :=
by
  intros h
  have h1 : x > 2 ∨ x < -2 := by sorry
  have h2 : ¬((x^2 > 4) ↔ (x > 2)) := by sorry
  exact And.intro h1 h2

end necessary_but_not_sufficient_l438_43877


namespace explicit_formula_for_f_l438_43875

theorem explicit_formula_for_f (f : ℕ → ℕ) (h₀ : f 0 = 0)
  (h₁ : ∀ (n : ℕ), n % 6 = 0 ∨ n % 6 = 1 → f (n + 1) = f n + 3)
  (h₂ : ∀ (n : ℕ), n % 6 = 2 ∨ n % 6 = 5 → f (n + 1) = f n + 1)
  (h₃ : ∀ (n : ℕ), n % 6 = 3 ∨ n % 6 = 4 → f (n + 1) = f n + 2)
  (n : ℕ) : f (6 * n) = 12 * n :=
by
  sorry

end explicit_formula_for_f_l438_43875


namespace annual_increase_rate_l438_43815

theorem annual_increase_rate (r : ℝ) (h : 70400 * (1 + r)^2 = 89100) : r = 0.125 :=
sorry

end annual_increase_rate_l438_43815


namespace proof_statements_l438_43848

theorem proof_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧       -- corresponding to A
  ¬((∃ m : ℕ, 190 = 19 * m) ∧  ¬(∃ k : ℕ, 57 = 19 * k)) ∧  -- corresponding to B
  ¬((∃ p : ℕ, 90 = 30 * p) ∨ (∃ q : ℕ, 65 = 30 * q)) ∧     -- corresponding to C
  ¬((∃ r : ℕ, 33 = 11 * r) ∧ ¬(∃ s : ℕ, 55 = 11 * s)) ∧    -- corresponding to D
  (∃ t : ℕ, 162 = 9 * t) :=                                 -- corresponding to E
by {
  -- Proof steps would go here
  sorry
}

end proof_statements_l438_43848


namespace parabola_sum_l438_43838

def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - b * x + c

def f (a b c : ℝ) (x : ℝ) : ℝ := a * (x + 7) ^ 2 - b * (x + 7) + c

def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 3) ^ 2 - b * (x - 3) + c

def fg (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum (a b c x : ℝ) : fg a b c x = 2 * a * x ^ 2 + (8 * a - 2 * b) * x + (58 * a - 4 * b + 2 * c) := by
  sorry

end parabola_sum_l438_43838


namespace isosceles_right_triangle_area_l438_43800

theorem isosceles_right_triangle_area (h : ℝ) (area : ℝ) (hypotenuse_condition : h = 6 * Real.sqrt 2) : 
  area = 18 :=
  sorry

end isosceles_right_triangle_area_l438_43800


namespace power_function_value_l438_43801

theorem power_function_value (a : ℝ) (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 4) :
  f 9 = 81 :=
by
  sorry

end power_function_value_l438_43801


namespace problem_statement_period_property_symmetry_property_zero_property_l438_43836

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem problem_statement : ¬(∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + ε))
  → ∃ x : ℝ, f (x + Real.pi) = 0 :=
by
  intro h
  use Real.pi / 6
  sorry

theorem period_property : ∀ k : ℤ, f (x + 2 * k * Real.pi) = f x :=
by
  intro k
  sorry

theorem symmetry_property : ∀ y : ℝ, f (8 * Real.pi / 3 - y) = f (8 * Real.pi / 3 + y) :=
by
  intro y
  sorry

theorem zero_property : f (Real.pi / 6 + Real.pi) = 0 :=
by
  sorry

end problem_statement_period_property_symmetry_property_zero_property_l438_43836


namespace smallest_of_five_consecutive_numbers_l438_43819

theorem smallest_of_five_consecutive_numbers (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → 
  n = 18 :=
by sorry

end smallest_of_five_consecutive_numbers_l438_43819


namespace steel_more_by_l438_43885

variable {S T C k : ℝ}
variable (k_greater_than_zero : k > 0)
variable (copper_weight : C = 90)
variable (S_twice_T : S = 2 * T)
variable (S_minus_C : S = C + k)
variable (total_eq : 20 * S + 20 * T + 20 * C = 5100)

theorem steel_more_by (k): k = 20 := by
  sorry

end steel_more_by_l438_43885


namespace zachary_more_crunches_than_pushups_l438_43865

def zachary_pushups : ℕ := 46
def zachary_crunches : ℕ := 58
def zachary_crunches_more_than_pushups : ℕ := zachary_crunches - zachary_pushups

theorem zachary_more_crunches_than_pushups : zachary_crunches_more_than_pushups = 12 := by
  sorry

end zachary_more_crunches_than_pushups_l438_43865


namespace quadratic_has_real_roots_find_specific_k_l438_43825

-- Part 1: Prove the range of values for k
theorem quadratic_has_real_roots (k : ℝ) : (k ≥ 2) ↔ ∃ x1 x2 : ℝ, x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 := 
sorry

-- Part 2: Prove the specific value of k given the additional condition
theorem find_specific_k (k : ℝ) (x1 x2 : ℝ) : (x1 ^ 3 * x2 + x1 * x2 ^ 3 = 24) ∧ x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 → k = 3 :=
sorry

end quadratic_has_real_roots_find_specific_k_l438_43825


namespace problem_part_1_problem_part_2_l438_43894

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x - 1

theorem problem_part_1 (m n : ℝ) :
  (∀ x, f x m < 0 ↔ -2 < x ∧ x < n) → m = 5 / 2 ∧ n = 1 / 2 :=
sorry

theorem problem_part_2 (m : ℝ) :
  (∀ x, m ≤ x ∧ x ≤ m + 1 → f x m < 0) → m ∈ Set.Ioo (-Real.sqrt (2) / 2) 0 :=
sorry

end problem_part_1_problem_part_2_l438_43894


namespace solve_for_x_l438_43816

-- Definition of the operation
def otimes (a b : ℝ) : ℝ := a^2 + b^2 - a * b

-- The mathematical statement to be proved
theorem solve_for_x (x : ℝ) (h : otimes x (x - 1) = 3) : x = 2 ∨ x = -1 := 
by 
  sorry

end solve_for_x_l438_43816


namespace sum_YNRB_l438_43896

theorem sum_YNRB :
  ∃ (R Y B N : ℕ),
    (RY = 10 * R + Y) ∧
    (BY = 10 * B + Y) ∧
    (111 * N = (10 * R + Y) * (10 * B + Y)) →
    (Y + N + R + B = 21) :=
sorry

end sum_YNRB_l438_43896


namespace first_player_win_condition_l438_43803

def player_one_wins (p q : ℕ) : Prop :=
  p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 4 ∨
  q % 5 = 0 ∨ q % 5 = 1 ∨ q % 5 = 4

theorem first_player_win_condition (p q : ℕ) :
  player_one_wins p q ↔
  (∃ (a b : ℕ), (a, b) = (p, q) ∧ (a % 5 = 0 ∨ a % 5 = 1 ∨ a % 5 = 4 ∨ 
                                     b % 5 = 0 ∨ b % 5 = 1 ∨ b % 5 = 4)) :=
sorry

end first_player_win_condition_l438_43803


namespace solve_for_b_l438_43814

theorem solve_for_b (b : ℝ) : (∃ y x : ℝ, 4 * y - 2 * x - 6 = 0 ∧ 5 * y + b * x + 1 = 0) → b = 10 :=
by sorry

end solve_for_b_l438_43814


namespace number_of_days_l438_43830

variables (S Wx Wy : ℝ)

-- Given conditions
def condition1 : Prop := S = 36 * Wx
def condition2 : Prop := S = 45 * Wy

-- The lean statement to prove the number of days D = 20
theorem number_of_days (h1 : condition1 S Wx) (h2 : condition2 S Wy) : 
  S / (Wx + Wy) = 20 :=
by
  sorry

end number_of_days_l438_43830


namespace sum_red_equals_sum_blue_l438_43802

variable (r1 r2 r3 r4 b1 b2 b3 b4 w1 w2 w3 w4 : ℝ)

theorem sum_red_equals_sum_blue (h : (r1 + w1 / 2) + (r2 + w2 / 2) + (r3 + w3 / 2) + (r4 + w4 / 2) 
                                 = (b1 + w1 / 2) + (b2 + w2 / 2) + (b3 + w3 / 2) + (b4 + w4 / 2)) : 
  r1 + r2 + r3 + r4 = b1 + b2 + b3 + b4 :=
by sorry

end sum_red_equals_sum_blue_l438_43802


namespace fraction_of_total_cost_for_raisins_l438_43832

-- Define variables and constants
variable (R : ℝ) -- cost of a pound of raisins

-- Define the conditions as assumptions
variable (cost_of_nuts : ℝ := 4 * R)
variable (cost_of_dried_berries : ℝ := 2 * R)

variable (total_cost : ℝ := 3 * R + 4 * cost_of_nuts + 2 * cost_of_dried_berries)
variable (cost_of_raisins : ℝ := 3 * R)

-- Main statement that we want to prove
theorem fraction_of_total_cost_for_raisins :
  cost_of_raisins / total_cost = 3 / 23 := by
  sorry

end fraction_of_total_cost_for_raisins_l438_43832


namespace total_cost_computers_l438_43879

theorem total_cost_computers (B T : ℝ) 
  (cA : ℝ := 1.4 * B) 
  (cB : ℝ := B) 
  (tA : ℝ := T) 
  (tB : ℝ := T + 20) 
  (total_cost_A : ℝ := cA * tA)
  (total_cost_B : ℝ := cB * tB):
  total_cost_A = total_cost_B → 70 * B = total_cost_A := 
by
  sorry

end total_cost_computers_l438_43879


namespace set_theory_problem_l438_43898

def U : Set ℤ := {x ∈ Set.univ | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_theory_problem : 
  (A ∩ B = {4}) ∧ 
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧ 
  (U \ (A ∪ C) = {6, 8, 10}) ∧ 
  ((U \ A) ∩ (U \ B) = {3}) := 
by 
  sorry

end set_theory_problem_l438_43898


namespace car_speed_first_hour_l438_43871

theorem car_speed_first_hour (x : ℕ) (hx : x = 65) : 
  let speed_second_hour := 45 
  let average_speed := 55
  (x + 45) / 2 = 55 
  :=
  by
  sorry

end car_speed_first_hour_l438_43871


namespace find_radius_of_stationary_tank_l438_43861

theorem find_radius_of_stationary_tank
  (h_stationary : Real) (r_truck : Real) (h_truck : Real) (drop : Real) (V_truck : Real)
  (ht1 : h_stationary = 25)
  (ht2 : r_truck = 4)
  (ht3 : h_truck = 10)
  (ht4 : drop = 0.016)
  (ht5 : V_truck = π * r_truck ^ 2 * h_truck) :
  ∃ R : Real, π * R ^ 2 * drop = V_truck ∧ R = 100 :=
by
  sorry

end find_radius_of_stationary_tank_l438_43861


namespace max_k_l438_43864

def A : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}

def valid_collection (B : ℕ → Finset ℕ) (k : ℕ) : Prop :=
  ∀ i j : ℕ, i < k → j < k → i ≠ j → (B i ∩ B j).card ≤ 2

theorem max_k (B : ℕ → Finset ℕ) : ∃ k, valid_collection B k → k ≤ 175 := sorry

end max_k_l438_43864


namespace five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l438_43831

theorem five_digit_numbers_greater_than_20314_and_formable_with_0_to_5 :
  (∃ (f : Fin 6 → Fin 5) (n : ℕ), 
    (n = 120 * 3 + 24 * 4 + 6 * 3 - 1) ∧
    (n = 473) ∧ 
    (∀ (x : Fin 6), f x = 0 ∨ f x = 1 ∨ f x = 2 ∨ f x = 3 ∨ f x = 4 ∨ f x = 5) ∧
    (∀ (i j : Fin 5), i ≠ j → f i ≠ f j)) :=
sorry

end five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l438_43831


namespace Sam_dimes_remaining_l438_43892

-- Define the initial and borrowed dimes
def initial_dimes_count : Nat := 8
def borrowed_dimes_count : Nat := 4

-- State the theorem
theorem Sam_dimes_remaining : (initial_dimes_count - borrowed_dimes_count) = 4 := by
  sorry

end Sam_dimes_remaining_l438_43892


namespace total_toads_l438_43860

def pond_toads : ℕ := 12
def outside_toads : ℕ := 6

theorem total_toads : pond_toads + outside_toads = 18 :=
by
  -- Proof goes here
  sorry

end total_toads_l438_43860


namespace num_ordered_pairs_l438_43844

theorem num_ordered_pairs (M N : ℕ) (hM : M > 0) (hN : N > 0) :
  (M * N = 32) → ∃ (k : ℕ), k = 6 :=
by
  sorry

end num_ordered_pairs_l438_43844


namespace bugs_max_contacts_l438_43850

theorem bugs_max_contacts :
  ∃ a b : ℕ, (a + b = 2016) ∧ (a * b = 1008^2) :=
by
  sorry

end bugs_max_contacts_l438_43850
