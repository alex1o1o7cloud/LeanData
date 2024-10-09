import Mathlib

namespace find_two_numbers_l20_2071

theorem find_two_numbers :
  ∃ (x y : ℝ), 
  (2 * (x + y) = x^2 - y^2 ∧ 2 * (x + y) = (x * y) / 4 - 56) ∧ 
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := 
sorry

end find_two_numbers_l20_2071


namespace power_of_square_l20_2040

variable {R : Type*} [CommRing R] (a : R)

theorem power_of_square (a : R) : (3 * a^2)^2 = 9 * a^4 :=
by sorry

end power_of_square_l20_2040


namespace max_area_house_l20_2075

def price_colored := 450
def price_composite := 200
def cost_limit := 32000

def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

theorem max_area_house : 
  ∃ (x y S : ℝ), 
    (S = x * y) ∧ 
    (material_cost x y ≤ cost_limit) ∧ 
    (0 < S ∧ S ≤ 100) ∧ 
    (S = 100 → x = 20 / 3) := 
by
  sorry

end max_area_house_l20_2075


namespace total_turtles_in_lake_l20_2005

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l20_2005


namespace AlbertTookAwayCandies_l20_2082

-- Define the parameters and conditions given in the problem
def PatriciaStartCandies : ℕ := 76
def PatriciaEndCandies : ℕ := 71

-- Define the statement that proves the number of candies Albert took away
theorem AlbertTookAwayCandies :
  PatriciaStartCandies - PatriciaEndCandies = 5 := by
  sorry

end AlbertTookAwayCandies_l20_2082


namespace tracy_dog_food_l20_2090

theorem tracy_dog_food
(f : ℕ) (c : ℝ) (m : ℕ) (d : ℕ)
(hf : f = 4) (hc : c = 2.25) (hm : m = 3) (hd : d = 2) :
  (f * c / m) / d = 1.5 :=
by
  sorry

end tracy_dog_food_l20_2090


namespace cake_pieces_l20_2077

theorem cake_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) 
  (pan_dim : pan_length = 24 ∧ pan_width = 15) 
  (piece_dim : piece_length = 3 ∧ piece_width = 2) : 
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
sorry

end cake_pieces_l20_2077


namespace range_of_a_l20_2007

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (if x1 ≤ 1 then (-x1^2 + a*x1)
     else (a*x1 - 1)) = 
    (if x2 ≤ 1 then (-x2^2 + a*x2)
     else (a*x2 - 1))) → a < 2 :=
sorry

end range_of_a_l20_2007


namespace remaining_kids_l20_2048

def initial_kids : Float := 22.0
def kids_who_went_home : Float := 14.0

theorem remaining_kids : initial_kids - kids_who_went_home = 8.0 :=
by 
  sorry

end remaining_kids_l20_2048


namespace circumference_of_flower_bed_l20_2008

noncomputable def square_garden_circumference (a p s r C : ℝ) : Prop :=
  a = s^2 ∧
  p = 4 * s ∧
  a = 2 * p + 14.25 ∧
  r = s / 4 ∧
  C = 2 * Real.pi * r

theorem circumference_of_flower_bed (a p s r : ℝ) (h : square_garden_circumference a p s r (4.75 * Real.pi)) : 
  ∃ C, square_garden_circumference a p s r C ∧ C = 4.75 * Real.pi :=
sorry

end circumference_of_flower_bed_l20_2008


namespace complex_division_l20_2019

-- Define the imaginary unit 'i'
def i := Complex.I

-- Define the complex numbers as described in the problem
def num := Complex.mk 3 (-1)
def denom := Complex.mk 1 (-1)
def expected := Complex.mk 2 1

-- State the theorem to prove that the complex division is as expected
theorem complex_division : (num / denom) = expected :=
by
  sorry

end complex_division_l20_2019


namespace dropouts_correct_l20_2055

/-- Definition for initial racers, racers joining after 20 minutes, and racers at finish line. -/
def initial_racers : ℕ := 50
def joining_racers : ℕ := 30
def finishers : ℕ := 130

/-- Total racers after initial join and doubling. -/
def total_racers : ℕ := (initial_racers + joining_racers) * 2

/-- The number of people who dropped out before finishing the race. -/
def dropped_out : ℕ := total_racers - finishers

/-- Proof statement to show the number of people who dropped out before finishing is 30. -/
theorem dropouts_correct : dropped_out = 30 := by
  sorry

end dropouts_correct_l20_2055


namespace xy_y_sq_eq_y_sq_3y_12_l20_2042

variable (x y : ℝ)

theorem xy_y_sq_eq_y_sq_3y_12 (h : x * (x + y) = x^2 + 3 * y + 12) : 
  x * y + y^2 = y^2 + 3 * y + 12 := 
sorry

end xy_y_sq_eq_y_sq_3y_12_l20_2042


namespace dwarfs_truthful_count_l20_2087

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end dwarfs_truthful_count_l20_2087


namespace percent_students_with_pets_l20_2098

theorem percent_students_with_pets 
  (total_students : ℕ) (students_with_cats : ℕ) (students_with_dogs : ℕ) (students_with_both : ℕ) (h_total : total_students = 500)
  (h_cats : students_with_cats = 150) (h_dogs : students_with_dogs = 100) (h_both : students_with_both = 40) :
  (students_with_cats + students_with_dogs - students_with_both) * 100 / total_students = 42 := 
by
  sorry

end percent_students_with_pets_l20_2098


namespace geq_solution_l20_2014

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a (n+1) / a n) = (a 1 / a 0)

theorem geq_solution
  (a : ℕ → ℝ)
  (h_seq : geom_seq a)
  (h_cond : a 0 * a 2 + 2 * a 1 * a 3 + a 1 * a 5 = 9) :
  a 1 + a 3 = 3 :=
sorry

end geq_solution_l20_2014


namespace tallest_vs_shortest_height_difference_l20_2018

-- Define the heights of the trees
def pine_tree_height := 12 + 4/5
def birch_tree_height := 18 + 1/2
def maple_tree_height := 14 + 3/5

-- Calculate improper fractions
def pine_tree_improper := 64 / 5
def birch_tree_improper := 41 / 2  -- This is 82/4 but not simplified here
def maple_tree_improper := 73 / 5

-- Calculate height difference
def height_difference := (82 / 4) - (64 / 5)

-- The statement that needs to be proven
theorem tallest_vs_shortest_height_difference : height_difference = 7 + 7 / 10 :=
by 
  sorry

end tallest_vs_shortest_height_difference_l20_2018


namespace find_line_equation_l20_2059

-- Define the point (2, -1) which the line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the line perpendicular to 2x - 3y = 1
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- The equation of the line we are supposed to find
def equation_of_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

-- Proof problem: prove the equation satisfies given the conditions
theorem find_line_equation :
  (equation_of_line point.1 point.2) ∧ 
  (∃ (a b c : ℝ), ∀ (x y : ℝ), perpendicular_line x y → equation_of_line x y) := sorry

end find_line_equation_l20_2059


namespace Area_S_inequality_l20_2091

def S (t : ℝ) (x y : ℝ) : Prop :=
  let T := Real.sin (Real.pi * t)
  |x - T| + |y - T| ≤ T

theorem Area_S_inequality (t : ℝ) :
  let T := Real.sin (Real.pi * t)
  0 ≤ 2 * T^2 := by
  sorry

end Area_S_inequality_l20_2091


namespace chloe_points_first_round_l20_2058

theorem chloe_points_first_round 
  (P : ℕ)
  (second_round_points : ℕ := 50)
  (lost_points : ℕ := 4)
  (total_points : ℕ := 86)
  (h : P + second_round_points - lost_points = total_points) : 
  P = 40 := 
by 
  sorry

end chloe_points_first_round_l20_2058


namespace age_problem_l20_2004

theorem age_problem 
  (P R J M : ℕ)
  (h1 : P = 1 / 2 * R)
  (h2 : R = J + 7)
  (h3 : J + 12 = 3 * P)
  (h4 : M = J + 17)
  (h5 : M = 2 * R + 4) : 
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 :=
by sorry

end age_problem_l20_2004


namespace second_person_avg_pages_per_day_l20_2066

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end second_person_avg_pages_per_day_l20_2066


namespace dice_probability_l20_2023

-- The context that there are three six-sided dice
def total_outcomes : ℕ := 6 * 6 * 6

-- Function to count the number of favorable outcomes where two dice sum to the third
def favorable_outcomes : ℕ :=
  let sum_cases := [1, 2, 3, 4, 5]
  sum_cases.sum
  -- sum_cases is [1, 2, 3, 4, 5] each mapping to the number of ways to form that sum with a third die

theorem dice_probability : 
  (favorable_outcomes * 3) / total_outcomes = 5 / 24 := 
by 
  -- to prove: the probability that the values on two dice sum to the value on the remaining die is 5/24
  sorry

end dice_probability_l20_2023


namespace brian_commission_rate_l20_2092

noncomputable def commission_rate (sale1 sale2 sale3 commission : ℝ) : ℝ :=
  (commission / (sale1 + sale2 + sale3)) * 100

theorem brian_commission_rate :
  commission_rate 157000 499000 125000 15620 = 2 :=
by
  unfold commission_rate
  sorry

end brian_commission_rate_l20_2092


namespace inverse_proportion_relationship_l20_2053

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_relationship (h1 : x1 < 0) (h2 : 0 < x2) 
  (hy1 : y1 = 3 / x1) (hy2 : y2 = 3 / x2) : y1 < 0 ∧ 0 < y2 :=
by
  sorry

end inverse_proportion_relationship_l20_2053


namespace passengers_remaining_l20_2080

theorem passengers_remaining :
  let initial_passengers := 64
  let reduction_factor := (2 / 3)
  ∀ (n : ℕ), n = 4 → initial_passengers * reduction_factor^n = 1024 / 81 := by
sorry

end passengers_remaining_l20_2080


namespace original_amount_in_cookie_jar_l20_2049

theorem original_amount_in_cookie_jar (doris_spent martha_spent money_left_in_jar original_amount : ℕ)
  (h1 : doris_spent = 6)
  (h2 : martha_spent = doris_spent / 2)
  (h3 : money_left_in_jar = 15)
  (h4 : original_amount = money_left_in_jar + doris_spent + martha_spent) :
  original_amount = 24 := 
sorry

end original_amount_in_cookie_jar_l20_2049


namespace polynomial_negativity_l20_2050

theorem polynomial_negativity (a x : ℝ) (h₀ : 0 < x) (h₁ : x < a) (h₂ : 0 < a) : 
  (a - x)^6 - 3 * a * (a - x)^5 + (5 / 2) * a^2 * (a - x)^4 - (1 / 2) * a^4 * (a - x)^2 < 0 := 
by
  sorry

end polynomial_negativity_l20_2050


namespace largest_sum_faces_l20_2028

theorem largest_sum_faces (a b c d e f : ℕ)
  (h_ab : a + b ≤ 7) (h_ac : a + c ≤ 7) (h_ad : a + d ≤ 7) (h_ae : a + e ≤ 7) (h_af : a + f ≤ 7)
  (h_bc : b + c ≤ 7) (h_bd : b + d ≤ 7) (h_be : b + e ≤ 7) (h_bf : b + f ≤ 7)
  (h_cd : c + d ≤ 7) (h_ce : c + e ≤ 7) (h_cf : c + f ≤ 7)
  (h_de : d + e ≤ 7) (h_df : d + f ≤ 7)
  (h_ef : e + f ≤ 7) :
  ∃ x y z, 
  ((x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧ 
   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) ∧ 
   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e ∨ z = f)) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (x + y ≤ 7) ∧ (y + z ≤ 7) ∧ (x + z ≤ 7) ∧
  (x + y + z = 9) :=
sorry

end largest_sum_faces_l20_2028


namespace total_marks_more_than_physics_l20_2079

-- Definitions of variables for marks in different subjects
variables (P C M : ℕ)

-- Conditions provided in the problem
def total_marks_condition (P : ℕ) (C : ℕ) (M : ℕ) : Prop := P + C + M > P
def average_chemistry_math_marks (C : ℕ) (M : ℕ) : Prop := (C + M) / 2 = 55

-- The main proof statement: Proving the difference in total marks and physics marks
theorem total_marks_more_than_physics 
    (h1 : total_marks_condition P C M)
    (h2 : average_chemistry_math_marks C M) :
  (P + C + M) - P = 110 := 
sorry

end total_marks_more_than_physics_l20_2079


namespace simplify_expr_l20_2094

theorem simplify_expr (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x :=
by
  sorry

end simplify_expr_l20_2094


namespace Peter_can_always_ensure_three_distinct_real_roots_l20_2033

noncomputable def cubic_has_three_distinct_real_roots (b d : ℝ) : Prop :=
∃ (a : ℝ), ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
  (r1 * r2 * r3 = -a) ∧ (r1 + r2 + r3 = -b) ∧ (r1 * r2 + r2 * r3 + r3 * r1 = -d)

theorem Peter_can_always_ensure_three_distinct_real_roots (b d : ℝ) :
  cubic_has_three_distinct_real_roots b d :=
sorry

end Peter_can_always_ensure_three_distinct_real_roots_l20_2033


namespace least_value_expression_l20_2044

theorem least_value_expression (x y : ℝ) : 
  (x^2 * y + x * y^2 - 1)^2 + (x + y)^2 ≥ 1 :=
sorry

end least_value_expression_l20_2044


namespace no_such_prime_pair_l20_2022

open Prime

theorem no_such_prime_pair :
  ∀ (p q : ℕ), Prime p → Prime q → (p > 5) → (q > 5) →
  (p * q) ∣ ((5^p - 2^p) * (5^q - 2^q)) → false :=
by
  intros p q hp hq hp_gt5 hq_gt5 hdiv
  sorry

end no_such_prime_pair_l20_2022


namespace maria_payment_l20_2000

noncomputable def calculate_payment : ℝ :=
  let regular_price := 15
  let first_discount := 0.40 * regular_price
  let after_first_discount := regular_price - first_discount
  let holiday_discount := 0.10 * after_first_discount
  let after_holiday_discount := after_first_discount - holiday_discount
  after_holiday_discount + 2

theorem maria_payment : calculate_payment = 10.10 :=
by
  sorry

end maria_payment_l20_2000


namespace find_integers_a_b_c_l20_2021

theorem find_integers_a_b_c :
  ∃ (a b c : ℤ), (∀ (x : ℤ), (x - a) * (x - 8) + 4 = (x + b) * (x + c)) ∧ 
  (a = 20 ∨ a = 29) :=
 by {
      sorry 
}

end find_integers_a_b_c_l20_2021


namespace negation_of_proposition_l20_2015

theorem negation_of_proposition (x y : ℝ): (x + y > 0 → x > 0 ∧ y > 0) ↔ ¬ ((x + y ≤ 0) → (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end negation_of_proposition_l20_2015


namespace eggs_total_l20_2054

-- Definitions based on the conditions
def breakfast_eggs : Nat := 2
def lunch_eggs : Nat := 3
def dinner_eggs : Nat := 1

-- Theorem statement
theorem eggs_total : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  -- This part will be filled in with proof steps, but it's omitted here
  sorry

end eggs_total_l20_2054


namespace palindromic_condition_l20_2016

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem palindromic_condition (m n : ℕ) :
  is_palindrome (2^n + 2^m + 1) ↔ (m ≤ 9 ∨ n ≤ 9) :=
sorry

end palindromic_condition_l20_2016


namespace solve_quadratic_l20_2070

   theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 5 * x^2 + 8 * x - 24 = 0) : x = 6 / 5 :=
   sorry
   
end solve_quadratic_l20_2070


namespace ratio_pr_l20_2085

variable (p q r s : ℚ)

def ratio_pq (p q : ℚ) : Prop := p / q = 5 / 4
def ratio_rs (r s : ℚ) : Prop := r / s = 4 / 3
def ratio_sq (s q : ℚ) : Prop := s / q = 1 / 5

theorem ratio_pr (hpq : ratio_pq p q) (hrs : ratio_rs r s) (hsq : ratio_sq s q) : p / r = 75 / 16 := by
  sorry

end ratio_pr_l20_2085


namespace find_base_l20_2093

theorem find_base (b : ℕ) (h : (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 2 * b + 5) : b = 7 :=
sorry

end find_base_l20_2093


namespace probability_snow_first_week_l20_2057

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end probability_snow_first_week_l20_2057


namespace max_k_mono_incr_binom_l20_2074

theorem max_k_mono_incr_binom :
  ∀ (k : ℕ), (k ≤ 11) → 
  (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ k → (Nat.choose 10 (i - 1) < Nat.choose 10 (j - 1))) →
  k = 6 :=
by sorry

end max_k_mono_incr_binom_l20_2074


namespace angle_measure_l20_2012

variable (x : ℝ)

noncomputable def is_supplement (x : ℝ) : Prop := 180 - x = 3 * (90 - x) - 60

theorem angle_measure : is_supplement x → x = 15 :=
by
  sorry

end angle_measure_l20_2012


namespace factorize_expression_l20_2046

variable {x y : ℝ}

theorem factorize_expression :
  3 * x^2 - 27 * y^2 = 3 * (x + 3 * y) * (x - 3 * y) :=
by
  sorry

end factorize_expression_l20_2046


namespace find_acute_angle_l20_2047

theorem find_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < 90) (h2 : ∃ k : ℤ, 10 * α = α + k * 360) :
  α = 40 ∨ α = 80 :=
by
  sorry

end find_acute_angle_l20_2047


namespace percentage_of_respondents_l20_2095

variables {X Y : ℝ}
variable (h₁ : 23 <= 100 - X)

theorem percentage_of_respondents 
  (h₁ : 0 ≤ X) 
  (h₂ : X ≤ 100) 
  (h₃ : 0 ≤ 23) 
  (h₄ : 23 ≤ 23) : 
  Y = 100 - X := 
by
  sorry

end percentage_of_respondents_l20_2095


namespace repeating_decimal_to_fraction_l20_2037

noncomputable def repeating_decimal_solution : ℚ := 7311 / 999

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 7 + 318 / 999) : x = repeating_decimal_solution := 
by
  sorry

end repeating_decimal_to_fraction_l20_2037


namespace fraction_of_students_who_walk_home_l20_2063

theorem fraction_of_students_who_walk_home (bus auto bikes scooters : ℚ) 
  (hbus : bus = 2/5) (hauto : auto = 1/5) 
  (hbikes : bikes = 1/10) (hscooters : scooters = 1/10) : 
  1 - (bus + auto + bikes + scooters) = 1/5 :=
by 
  rw [hbus, hauto, hbikes, hscooters]
  sorry

end fraction_of_students_who_walk_home_l20_2063


namespace find_x_y_l20_2041

theorem find_x_y (a n x y : ℕ) (hx4 : 1000 ≤ x ∧ x < 10000) (hy4 : 1000 ≤ y ∧ y < 10000) 
  (h_yx : y > x) (h_y : y = a * 10 ^ n) 
  (h_sum : (x / 1000) + ((x % 1000) / 100) = 5 * a) 
  (ha : a = 2) (hn : n = 3) :
  x = 1990 ∧ y = 2000 := 
by 
  sorry

end find_x_y_l20_2041


namespace lattice_points_on_hyperbola_l20_2083

theorem lattice_points_on_hyperbola : 
  ∃ n, (∀ x y : ℤ, x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | 
  ∃ a b : ℤ, x = 2 * a + b ∧ y = 2 * a - b}) ∧ n = 250 := 
by {
  sorry
}

end lattice_points_on_hyperbola_l20_2083


namespace wrapping_paper_needed_l20_2039

-- Define the conditions as variables in Lean
def wrapping_paper_first := 3.5
def wrapping_paper_second := (2 / 3) * wrapping_paper_first
def wrapping_paper_third := wrapping_paper_second + 0.5 * wrapping_paper_second
def wrapping_paper_fourth := wrapping_paper_first + wrapping_paper_second
def wrapping_paper_fifth := wrapping_paper_third - 0.25 * wrapping_paper_third

-- Define the total wrapping paper needed
def total_wrapping_paper := wrapping_paper_first + wrapping_paper_second + wrapping_paper_third + wrapping_paper_fourth + wrapping_paper_fifth

-- Statement to prove the final equivalence
theorem wrapping_paper_needed : 
  total_wrapping_paper = 17.79 := 
sorry  -- Proof is omitted

end wrapping_paper_needed_l20_2039


namespace projectile_max_height_l20_2078

theorem projectile_max_height :
  ∀ (t : ℝ), -12 * t^2 + 72 * t + 45 ≤ 153 :=
by
  sorry

end projectile_max_height_l20_2078


namespace fred_current_money_l20_2003

-- Conditions
def initial_amount_fred : ℕ := 19
def earned_amount_fred : ℕ := 21

-- Question and Proof
theorem fred_current_money : initial_amount_fred + earned_amount_fred = 40 :=
by sorry

end fred_current_money_l20_2003


namespace condition_necessary_but_not_sufficient_l20_2034

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem condition_necessary_but_not_sufficient (a_1 d : ℝ) :
  (∀ n : ℕ, S_n a_1 d (n + 1) > S_n a_1 d n) ↔ (a_1 + d > 0) :=
sorry

end condition_necessary_but_not_sufficient_l20_2034


namespace invitation_methods_l20_2067

-- Definitions
def num_ways_invite_6_out_of_10 : ℕ := Nat.choose 10 6
def num_ways_both_A_and_B : ℕ := Nat.choose 8 4

-- Theorem statement
theorem invitation_methods : num_ways_invite_6_out_of_10 - num_ways_both_A_and_B = 140 :=
by
  -- Proof should be provided here
  sorry

end invitation_methods_l20_2067


namespace product_of_areas_square_of_volume_l20_2020

-- Declare the original dimensions and volume
variables (a b c : ℝ)
def V := a * b * c

-- Declare the areas of the new box
def area_bottom := (a + 2) * (b + 2)
def area_side := (b + 2) * (c + 2)
def area_front := (c + 2) * (a + 2)

-- Final theorem to prove
theorem product_of_areas_square_of_volume :
  (area_bottom a b) * (area_side b c) * (area_front c a) = V a b c ^ 2 :=
sorry

end product_of_areas_square_of_volume_l20_2020


namespace product_of_roots_l20_2036

theorem product_of_roots :
  let a := 18
  let b := 45
  let c := -500
  let prod_roots := c / a
  prod_roots = -250 / 9 := 
by
  -- Define coefficients
  let a := 18
  let c := -500

  -- Calculate product of roots
  let prod_roots := c / a

  -- Statement to prove
  have : prod_roots = -250 / 9 := sorry
  exact this

-- Adding sorry since the proof is not required according to the problem statement.

end product_of_roots_l20_2036


namespace magic_coin_l20_2086

theorem magic_coin (m n : ℕ) (h_m_prime: Nat.gcd m n = 1)
  (h_prob : (m : ℚ) / n = 1 / 158760): m + n = 158761 := by
  sorry

end magic_coin_l20_2086


namespace ball_hits_ground_at_time_l20_2072

theorem ball_hits_ground_at_time :
  ∀ (t : ℝ), (-18 * t^2 + 30 * t + 60 = 0) ↔ (t = (5 + Real.sqrt 145) / 6) :=
sorry

end ball_hits_ground_at_time_l20_2072


namespace zero_point_exists_in_interval_l20_2035

noncomputable def f (x : ℝ) : ℝ := x + 2^x

theorem zero_point_exists_in_interval :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f x = 0 :=
by
  existsi -0.5 -- This is not a formal proof; the existi -0.5 is just for example purposes
  sorry

end zero_point_exists_in_interval_l20_2035


namespace compute_expr_l20_2076

theorem compute_expr {x : ℝ} (h : x = 5) : (x^6 - 2 * x^3 + 1) / (x^3 - 1) = 124 :=
by
  sorry

end compute_expr_l20_2076


namespace distance_between_houses_l20_2027

theorem distance_between_houses (d d_JS d_QS : ℝ) (h1 : d_JS = 3) (h2 : d_QS = 1) :
  (2 ≤ d ∧ d ≤ 4) → d = 3 :=
sorry

end distance_between_houses_l20_2027


namespace find_A_from_complement_l20_2096

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define the complement of set A in U
variable (A : Set ℕ)
def complement_U_A : Set ℕ := {n | n ∈ U ∧ n ∉ A}

-- Define the condition given in the problem
axiom h : complement_U_A A = {2}

-- State the theorem to be proven
theorem find_A_from_complement : A = {0, 1} :=
sorry

end find_A_from_complement_l20_2096


namespace productivity_increase_l20_2051

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : (7/8) * b * (1 + x / 100) = 1.05 * b)

theorem productivity_increase (x : ℝ) : x = 20 := sorry

end productivity_increase_l20_2051


namespace maximum_n_for_positive_S_l20_2052

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

theorem maximum_n_for_positive_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (S : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (d_neg : d < 0)
  (S4_eq_S8 : S 4 = S 8)
  (h1 : is_arithmetic_sequence a d)
  (h2 : ∀ n, S n = sum_of_first_n_terms a n) :
  ∃ n, ∀ m, m ≤ n → S m > 0 ∧ ∀ k, k > n → S k ≤ 0 ∧ n = 11 :=
sorry

end maximum_n_for_positive_S_l20_2052


namespace solution_set_of_inequality_l20_2099

theorem solution_set_of_inequality :
  {x : ℝ | 1 / x < 1 / 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l20_2099


namespace number_of_n_values_l20_2031

-- Definition of sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

-- The main statement to prove
theorem number_of_n_values : 
  ∃ M, M = 8 ∧ ∀ n : ℕ, (n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2010) → M = 8 :=
by
  sorry

end number_of_n_values_l20_2031


namespace third_player_games_l20_2088

theorem third_player_games (p1 p2 p3 : ℕ) (h1 : p1 = 21) (h2 : p2 = 10)
  (total_games : p1 = p2 + p3) : p3 = 11 :=
by
  sorry

end third_player_games_l20_2088


namespace expression_undefined_l20_2001

theorem expression_undefined (a : ℝ) : (a = 2 ∨ a = -2) ↔ (a^2 - 4 = 0) :=
by sorry

end expression_undefined_l20_2001


namespace bunnies_out_of_burrow_l20_2061

theorem bunnies_out_of_burrow:
  (3 * 60 * 10 * 20) = 36000 :=
by 
  sorry

end bunnies_out_of_burrow_l20_2061


namespace Yan_distance_ratio_l20_2081

theorem Yan_distance_ratio (d x : ℝ) (v : ℝ) (h1 : d > 0) (h2 : x > 0) (h3 : x < d)
  (h4 : 7 * (d - x) = x + d) : 
  x / (d - x) = 3 / 4 :=
by
  sorry

end Yan_distance_ratio_l20_2081


namespace Greatest_Percentage_Difference_l20_2025

def max_percentage_difference (B W P : ℕ) : ℕ :=
  ((max B (max W P) - min B (min W P)) * 100) / (min B (min W P))

def January_B : ℕ := 6
def January_W : ℕ := 4
def January_P : ℕ := 5

def February_B : ℕ := 7
def February_W : ℕ := 5
def February_P : ℕ := 6

def March_B : ℕ := 7
def March_W : ℕ := 7
def March_P : ℕ := 7

def April_B : ℕ := 5
def April_W : ℕ := 6
def April_P : ℕ := 7

def May_B : ℕ := 3
def May_W : ℕ := 4
def May_P : ℕ := 2

theorem Greatest_Percentage_Difference :
  max_percentage_difference May_B May_W May_P >
  max (max_percentage_difference January_B January_W January_P)
      (max (max_percentage_difference February_B February_W February_P)
           (max (max_percentage_difference March_B March_W March_P)
                (max_percentage_difference April_B April_W April_P))) :=
by
  sorry

end Greatest_Percentage_Difference_l20_2025


namespace problem_y_values_l20_2038

theorem problem_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 54) :
  ∃ y : ℝ, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 7.5 ∨ y = 4.5) := by
sorry

end problem_y_values_l20_2038


namespace remainder_geometric_series_sum_l20_2009

/-- Define the sum of the geometric series. --/
def geometric_series_sum (n : ℕ) : ℕ :=
  (13^(n+1) - 1) / 12

/-- The given geometric series. --/
def series_sum := geometric_series_sum 1004

/-- Define the modulo operation. --/
def mod_op (a b : ℕ) := a % b

/-- The main statement to prove. --/
theorem remainder_geometric_series_sum :
  mod_op series_sum 1000 = 1 :=
sorry

end remainder_geometric_series_sum_l20_2009


namespace cos_lt_sin3_div_x3_l20_2062

open Real

theorem cos_lt_sin3_div_x3 (x : ℝ) (h1 : 0 < x) (h2 : x < pi / 2) : 
  cos x < (sin x / x) ^ 3 := 
  sorry

end cos_lt_sin3_div_x3_l20_2062


namespace balls_of_yarn_per_sweater_l20_2024

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

end balls_of_yarn_per_sweater_l20_2024


namespace find_x_ceil_mul_l20_2056

theorem find_x_ceil_mul (x : ℝ) (h : ⌈x⌉ * x = 75) : x = 8.333 := by
  sorry

end find_x_ceil_mul_l20_2056


namespace quadruple_perimeter_l20_2060

variable (s : ℝ) -- side length of the original square
variable (x : ℝ) -- perimeter of the original square
variable (P_new : ℝ) -- new perimeter after side length is quadrupled

theorem quadruple_perimeter (h1 : x = 4 * s) (h2 : P_new = 4 * (4 * s)) : P_new = 4 * x := 
by sorry

end quadruple_perimeter_l20_2060


namespace find_f_l20_2006

theorem find_f 
  (h_vertex : ∃ (d e f : ℝ), ∀ x, y = d * (x - 3)^2 - 5 ∧ y = d * x^2 + e * x + f)
  (h_point : y = d * (4 - 3)^2 - 5) 
  (h_value : y = -3) :
  ∃ f, f = 13 :=
sorry

end find_f_l20_2006


namespace max_val_z_lt_2_l20_2013

-- Definitions for the variables and constraints
variable {x y m : ℝ}
variable (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1)

-- Theorem statement
theorem max_val_z_lt_2 (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1) : 
  (∀ x y, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2) ↔ 1 < m ∧ m < 1 + Real.sqrt 2 :=
sorry

end max_val_z_lt_2_l20_2013


namespace desired_alcohol_percentage_is_18_l20_2069

noncomputable def final_alcohol_percentage (volume_x volume_y : ℕ) (percentage_x percentage_y : ℚ) : ℚ :=
  let total_volume := (volume_x + volume_y)
  let total_alcohol := (percentage_x * volume_x + percentage_y * volume_y)
  total_alcohol / total_volume * 100

theorem desired_alcohol_percentage_is_18 : 
  final_alcohol_percentage 300 200 0.10 0.30 = 18 := 
  sorry

end desired_alcohol_percentage_is_18_l20_2069


namespace correct_option_l20_2010

theorem correct_option :
  (∀ (a b : ℝ),  3 * a^2 * b - 4 * b * a^2 = -a^2 * b) ∧
  ¬(1 / 7 * (-7) + (-1 / 7) * 7 = 1) ∧
  ¬((-3 / 5)^2 = 9 / 5) ∧
  ¬(∀ (a b : ℝ), 3 * a + 5 * b = 8 * a * b) :=
by
  sorry

end correct_option_l20_2010


namespace bus_speed_including_stoppages_l20_2097

theorem bus_speed_including_stoppages 
  (speed_without_stoppages : ℕ) 
  (stoppage_time_per_hour : ℕ) 
  (correct_speed_including_stoppages : ℕ) :
  speed_without_stoppages = 54 →
  stoppage_time_per_hour = 10 →
  correct_speed_including_stoppages = 45 :=
by
sorry

end bus_speed_including_stoppages_l20_2097


namespace measure_angle_WYZ_l20_2011

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end measure_angle_WYZ_l20_2011


namespace fraction_not_collapsing_l20_2089

variable (total_homes : ℕ)
variable (termite_ridden_fraction collapsing_fraction : ℚ)
variable (h : termite_ridden_fraction = 1 / 3)
variable (c : collapsing_fraction = 7 / 10)

theorem fraction_not_collapsing : 
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction)) = 1 / 10 := 
by 
  rw [h, c]
  sorry

end fraction_not_collapsing_l20_2089


namespace pencils_cost_l20_2030

theorem pencils_cost (A B : ℕ) (C D : ℕ) (r : ℚ) : 
    A * 20 = 3200 → B * 20 = 960 → (A / B = 3200 / 960) → (A = 160) → (B = 48) → (C = 3200) → (D = 960) → 160 * 960 / 48 = 3200 :=
by
sorry

end pencils_cost_l20_2030


namespace trig_evaluation_trig_identity_value_l20_2043

-- Problem 1: Prove the trigonometric evaluation
theorem trig_evaluation :
  (Real.cos (9 * Real.pi / 4)) + (Real.tan (-Real.pi / 4)) + (Real.sin (21 * Real.pi)) = (Real.sqrt 2 / 2) - 1 :=
by
  sorry

-- Problem 2: Prove the value given the trigonometric identity
theorem trig_identity_value (θ : ℝ) (h : Real.sin θ = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by
  sorry

end trig_evaluation_trig_identity_value_l20_2043


namespace tangent_line_to_parabola_l20_2002

theorem tangent_line_to_parabola (l : ℝ → ℝ) (y : ℝ) (x : ℝ)
  (passes_through_P : l (-2) = 0)
  (intersects_once : ∃! x, (l x)^2 = 8*x) :
  (l = fun x => 0) ∨ (l = fun x => x + 2) ∨ (l = fun x => -x - 2) :=
sorry

end tangent_line_to_parabola_l20_2002


namespace f_max_a_zero_f_zero_range_l20_2068

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end f_max_a_zero_f_zero_range_l20_2068


namespace sum_of_roots_eq_zero_l20_2017

theorem sum_of_roots_eq_zero :
  ∀ (x : ℝ), x^2 - 7 * |x| + 6 = 0 → (∃ a b c d : ℝ, a + b + c + d = 0) :=
by
  sorry

end sum_of_roots_eq_zero_l20_2017


namespace equilateral_triangle_sum_l20_2064

theorem equilateral_triangle_sum (a u v w : ℝ)
  (h1: u^2 + v^2 = w^2):
  w^2 + Real.sqrt 3 * u * v = a^2 := 
sorry

end equilateral_triangle_sum_l20_2064


namespace combinations_eight_choose_three_l20_2065

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end combinations_eight_choose_three_l20_2065


namespace general_formula_for_a_n_l20_2084

-- Given conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
variable (h1 : ∀ n : ℕ, a n > 0)
variable (h2 : ∀ n : ℕ, 4 * S n = (a n - 1) * (a n + 3))

theorem general_formula_for_a_n :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end general_formula_for_a_n_l20_2084


namespace average_speed_x_to_z_l20_2029

theorem average_speed_x_to_z 
  (d : ℝ)
  (h1 : d > 0)
  (distance_xy : ℝ := 2 * d)
  (distance_yz : ℝ := d)
  (speed_xy : ℝ := 100)
  (speed_yz : ℝ := 75)
  (total_distance : ℝ := distance_xy + distance_yz)
  (time_xy : ℝ := distance_xy / speed_xy)
  (time_yz : ℝ := distance_yz / speed_yz)
  (total_time : ℝ := time_xy + time_yz) :
  total_distance / total_time = 90 :=
by
  sorry

end average_speed_x_to_z_l20_2029


namespace aaron_guesses_correctly_l20_2045

noncomputable def P_H : ℝ := 2 / 3
noncomputable def P_T : ℝ := 1 / 3
noncomputable def P_G_H : ℝ := 2 / 3
noncomputable def P_G_T : ℝ := 1 / 3

noncomputable def p : ℝ := P_H * P_G_H + P_T * P_G_T

theorem aaron_guesses_correctly :
  9000 * p = 5000 :=
by
  sorry

end aaron_guesses_correctly_l20_2045


namespace math_problem_l20_2032

theorem math_problem (a b : ℝ) :
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 :=
by
  sorry

end math_problem_l20_2032


namespace problem_1_problem_2_l20_2073

open Real
open Set

noncomputable def y (x : ℝ) : ℝ := (2 * sin x - cos x ^ 2) / (1 + sin x)

theorem problem_1 :
  { x : ℝ | y x = 1 ∧ sin x ≠ -1 } = { x | ∃ (k : ℤ), x = 2 * k * π + (π / 2) } :=
by
  sorry

theorem problem_2 : 
  ∃ x, y x = 1 ∧ ∀ x', y x' ≤ 1 :=
by
  sorry

end problem_1_problem_2_l20_2073


namespace smallest_integer_to_make_1008_perfect_square_l20_2026

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_integer_to_make_1008_perfect_square : ∃ k : ℕ, k > 0 ∧ 
  (∀ m : ℕ, m > 0 → (is_perfect_square (1008 * m) → m ≥ k)) ∧ is_perfect_square (1008 * k) :=
by
  sorry

end smallest_integer_to_make_1008_perfect_square_l20_2026
