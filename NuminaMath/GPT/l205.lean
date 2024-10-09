import Mathlib

namespace jenny_kenny_reunion_time_l205_20530

/-- Define initial conditions given in the problem --/
def jenny_initial_pos : ℝ × ℝ := (-60, 100)
def kenny_initial_pos : ℝ × ℝ := (-60, -100)
def building_radius : ℝ := 60
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def distance_apa : ℝ := 200
def initial_distance : ℝ := 200

theorem jenny_kenny_reunion_time : ∃ t : ℚ, 
  (t = (10 * (Real.sqrt 35)) / 7) ∧ 
  (17 = (10 + 7)) :=
by
  -- conditions to be used
  let jenny_pos (t : ℝ) := (-60 + 2 * t, 100)
  let kenny_pos (t : ℝ) := (-60 + 4 * t, -100)
  let circle_eq (x y : ℝ) := (x^2 + y^2 = building_radius^2)
  
  sorry

end jenny_kenny_reunion_time_l205_20530


namespace angle_relation_l205_20549

theorem angle_relation
  (x y z w : ℝ)
  (h_sum : x + y + z + (360 - w) = 360) :
  x = w - y - z :=
by
  sorry

end angle_relation_l205_20549


namespace prime_between_30_and_40_with_remainder_7_l205_20594

theorem prime_between_30_and_40_with_remainder_7 (n : ℕ) 
  (h1 : Nat.Prime n) 
  (h2 : 30 < n) 
  (h3 : n < 40) 
  (h4 : n % 12 = 7) : 
  n = 31 := 
sorry

end prime_between_30_and_40_with_remainder_7_l205_20594


namespace jar_and_beans_weight_is_60_percent_l205_20553

theorem jar_and_beans_weight_is_60_percent
  (J B : ℝ)
  (h1 : J = 0.10 * (J + B))
  (h2 : ∃ x : ℝ, x = 0.5555555555555556 ∧ (J + x * B = 0.60 * (J + B))) :
  J + 0.5555555555555556 * B = 0.60 * (J + B) :=
by
  sorry

end jar_and_beans_weight_is_60_percent_l205_20553


namespace xyz_cubic_expression_l205_20524

theorem xyz_cubic_expression (x y z a b c : ℝ) (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0) (h7 : a ≠ 0) (h8 : b ≠ 0) (h9 : c ≠ 0) :
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) :=
by
  sorry

end xyz_cubic_expression_l205_20524


namespace rhombus_diagonal_l205_20507

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h : d1 * d2 = 2 * area) (hd2 : d2 = 21) (h_area : area = 157.5) : d1 = 15 :=
by
  sorry

end rhombus_diagonal_l205_20507


namespace Andy_has_4_more_candies_than_Caleb_l205_20579

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end Andy_has_4_more_candies_than_Caleb_l205_20579


namespace cyclist_wait_20_minutes_l205_20596

noncomputable def cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (time_passed_minutes : ℝ) : ℝ :=
  let time_passed_hours := time_passed_minutes / 60
  let distance := cyclist_speed * time_passed_hours
  let hiker_catch_up_time := distance / hiker_speed
  hiker_catch_up_time * 60

theorem cyclist_wait_20_minutes :
  cyclist_wait_time 5 20 5 = 20 :=
by
  -- Definitions according to given conditions
  let hiker_speed := 5 -- miles per hour
  let cyclist_speed := 20 -- miles per hour
  let time_passed_minutes := 5
  -- Required result
  let result_needed := 20
  -- Using the cyclist_wait_time function
  show cyclist_wait_time hiker_speed cyclist_speed time_passed_minutes = result_needed
  sorry

end cyclist_wait_20_minutes_l205_20596


namespace part_one_part_two_l205_20557

theorem part_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 := sorry

theorem part_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := sorry

end part_one_part_two_l205_20557


namespace factorize_expression1_factorize_expression2_l205_20521

section
variable (x y : ℝ)

theorem factorize_expression1 : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

theorem factorize_expression2 : 3 * x^3 - 12 * x^2 * y + 12 * x * y^2 = 3 * x * (x - 2 * y)^2 :=
sorry
end

end factorize_expression1_factorize_expression2_l205_20521


namespace sin_x1_sub_x2_l205_20581

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sin_x1_sub_x2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < Real.pi)
  (h₄ : f x₁ = 1 / 3) (h₅ : f x₂ = 1 / 3) : 
  Real.sin (x₁ - x₂) = - (2 * Real.sqrt 2) / 3 := 
sorry

end sin_x1_sub_x2_l205_20581


namespace simplify_expression_l205_20592

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) :=
by
  sorry

end simplify_expression_l205_20592


namespace period_is_seven_l205_20538

-- Define the conditions
def apples_per_sandwich (a : ℕ) := a = 4
def sandwiches_per_day (s : ℕ) := s = 10
def total_apples (t : ℕ) := t = 280

-- Define the question to prove the period
theorem period_is_seven (a s t d : ℕ) 
  (h1 : apples_per_sandwich a)
  (h2 : sandwiches_per_day s)
  (h3 : total_apples t)
  (h4 : d = t / (a * s)) 
  : d = 7 := 
sorry

end period_is_seven_l205_20538


namespace Matias_longest_bike_ride_l205_20566

-- Define conditions in Lean
def blocks : ℕ := 4
def block_side_length : ℕ := 100
def streets : ℕ := 12

def Matias_route : Prop :=
  ∀ (intersections_used : ℕ), 
    intersections_used ≤ 4 → (streets - intersections_used/2 * 2) = 10

def correct_maximum_path_length : ℕ := 1000

-- Objective: Prove that given the conditions the longest route is 1000 meters
theorem Matias_longest_bike_ride :
  (100 * (streets - 2)) = correct_maximum_path_length :=
by
  sorry

end Matias_longest_bike_ride_l205_20566


namespace octavio_can_reach_3_pow_2023_l205_20532

theorem octavio_can_reach_3_pow_2023 (n : ℤ) (hn : n ≥ 1) :
  ∃ (steps : ℕ → ℤ), steps 0 = n ∧ (∀ k, steps (k + 1) = 3 * (steps k)) ∧
  steps 2023 = 3 ^ 2023 :=
by
  sorry

end octavio_can_reach_3_pow_2023_l205_20532


namespace correct_calculation_result_l205_20546

theorem correct_calculation_result :
  ∃ x : ℕ, 6 * x = 42 ∧ 3 * x = 21 :=
by
  sorry

end correct_calculation_result_l205_20546


namespace smallest_square_factor_2016_l205_20584

theorem smallest_square_factor_2016 : ∃ n : ℕ, (168 = n) ∧ (∃ k : ℕ, k^2 = n) ∧ (2016 ∣ k^2) :=
by
  sorry

end smallest_square_factor_2016_l205_20584


namespace probability_nearest_odd_l205_20599

def is_odd_nearest (a b : ℝ) : Prop := ∃ k : ℤ, 2 * k + 1 = Int.floor ((a - b) / (a + b))

def is_valid (a b : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1

noncomputable def probability_odd_nearest : ℝ :=
  let interval_area := 1 -- the area of the unit square [0, 1] x [0, 1]
  let odd_area := 1 / 3 -- as derived from the geometric interpretation in the problem's solution
  odd_area / interval_area

theorem probability_nearest_odd (a b : ℝ) (h : is_valid a b) :
  probability_odd_nearest = 1 / 3 := by
  sorry

end probability_nearest_odd_l205_20599


namespace problem1_problem2_l205_20597

theorem problem1 : 
  -(3^3) * ((-1 : ℚ)/ 3)^2 - 24 * (3/4 - 1/6 + 3/8) = -26 := 
by 
  sorry

theorem problem2 : 
  -(1^100 : ℚ) - (3/4) / (((-2)^2) * ((-1 / 4) ^ 2) - 1 / 2) = 2 := 
by 
  sorry

end problem1_problem2_l205_20597


namespace sum_of_areas_of_circles_l205_20517

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l205_20517


namespace roja_alone_time_l205_20545

theorem roja_alone_time (W : ℝ) (R : ℝ) :
  (1 / 60 + 1 / R = 1 / 35) → (R = 210) :=
by
  intros
  -- Proof goes here
  sorry

end roja_alone_time_l205_20545


namespace multiply_by_15_is_225_l205_20531

-- Define the condition
def number : ℕ := 15

-- State the theorem with the conditions and the expected result
theorem multiply_by_15_is_225 : 15 * number = 225 := by
  -- Insert the proof here
  sorry

end multiply_by_15_is_225_l205_20531


namespace plane_equation_l205_20575

theorem plane_equation (A B C D x y z : ℤ) (h1 : A = 15) (h2 : B = -3) (h3 : C = 2) (h4 : D = -238) 
  (h5 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1) (h6 : A > 0) :
  A * x + B * y + C * z + D = 0 ↔ 15 * x - 3 * y + 2 * z - 238 = 0 :=
by
  sorry

end plane_equation_l205_20575


namespace sum_first_15_nat_eq_120_l205_20513

-- Define a function to sum the first n natural numbers
def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define the theorem to show that the sum of the first 15 natural numbers equals 120
theorem sum_first_15_nat_eq_120 : sum_natural_numbers 15 = 120 := 
  by
    sorry

end sum_first_15_nat_eq_120_l205_20513


namespace price_equivalence_l205_20516

theorem price_equivalence : 
  (∀ a o p : ℕ, 10 * a = 5 * o ∧ 4 * o = 6 * p) → 
  (∀ a o p : ℕ, 20 * a = 15 * p) :=
by
  intro h
  sorry

end price_equivalence_l205_20516


namespace count_odd_numbers_300_600_l205_20509

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end count_odd_numbers_300_600_l205_20509


namespace g_three_fifths_l205_20558

-- Given conditions
variable (g : ℝ → ℝ)
variable (h₀ : g 0 = 0)
variable (h₁ : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
variable (h₂ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
variable (h₃ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3)

-- Proof statement
theorem g_three_fifths : g (3 / 5) = 2 / 3 := by
  sorry

end g_three_fifths_l205_20558


namespace find_length_of_wood_l205_20580

-- Definitions based on given conditions
def Area := 24  -- square feet
def Width := 6  -- feet

-- The mathematical proof problem turned into Lean 4 statement
theorem find_length_of_wood (h : Area = 24) (hw : Width = 6) : (Length : ℕ) ∈ {l | l = Area / Width ∧ l = 4} :=
by {
  sorry
}

end find_length_of_wood_l205_20580


namespace find_x2_y2_l205_20589

theorem find_x2_y2 (x y : ℝ) (h₁ : (x + y)^2 = 9) (h₂ : x * y = -6) : x^2 + y^2 = 21 := 
by
  sorry

end find_x2_y2_l205_20589


namespace not_solution_of_equation_l205_20536

theorem not_solution_of_equation (a : ℝ) (h : a ≠ 0) : ¬ (a^2 * 1^2 + (a + 1) * 1 + 1 = 0) :=
by {
  sorry
}

end not_solution_of_equation_l205_20536


namespace sum_set_15_l205_20518

noncomputable def sum_nth_set (n : ℕ) : ℕ :=
  let first_element := 1 + (n - 1) * n / 2
  let last_element := first_element + n - 1
  n * (first_element + last_element) / 2

theorem sum_set_15 : sum_nth_set 15 = 1695 :=
  by sorry

end sum_set_15_l205_20518


namespace vector_subtraction_result_l205_20504

-- Defining the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- The main theorem stating that a - 2b results in the expected coordinates
theorem vector_subtraction_result :
  a - 2 • b = (7, -2) := by
  sorry

end vector_subtraction_result_l205_20504


namespace option_a_solution_l205_20511

theorem option_a_solution (x y : ℕ) (h₁: x = 2) (h₂: y = 2) : 2 * x + y = 6 := by
sorry

end option_a_solution_l205_20511


namespace cos_240_eq_neg_half_l205_20548

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l205_20548


namespace b_1001_value_l205_20535

theorem b_1001_value (b : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) 
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 888 = 17 + Real.sqrt 11) : 
  b 1001 = 7 * Real.sqrt 11 - 20 := sorry

end b_1001_value_l205_20535


namespace quadratic_function_inequality_l205_20528

variable (a x x₁ x₂ : ℝ)

def f (x : ℝ) := a * x^2 + 2 * a * x + 4

theorem quadratic_function_inequality
  (h₀ : 0 < a) (h₁ : a < 3)
  (h₂ : x₁ + x₂ = 0)
  (h₃ : x₁ < x₂) :
  f a x₁ < f a x₂ := 
sorry

end quadratic_function_inequality_l205_20528


namespace number_of_white_balls_l205_20569

theorem number_of_white_balls (total_balls yellow_frequency : ℕ) (h1 : total_balls = 10) (h2 : yellow_frequency = 60) :
  (total_balls - (total_balls * yellow_frequency / 100) = 4) :=
by
  sorry

end number_of_white_balls_l205_20569


namespace solution_l205_20552

variable (a : ℕ → ℝ)

noncomputable def pos_sequence (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → a k > 0

noncomputable def recursive_relation (n : ℕ) : Prop :=
  ∀ n : ℕ, (n > 0) → (n+1) * a (n+1)^2 - n * a n^2 + a (n+1) * a n = 0

noncomputable def sequence_condition (n : ℕ) : Prop :=
  a 1 = 1 ∧ pos_sequence a n ∧ recursive_relation a n

theorem solution : ∀ n : ℕ, n > 0 → sequence_condition a n → a n = 1 / n :=
by
  intros n hn h
  sorry

end solution_l205_20552


namespace neg_q_sufficient_not_necc_neg_p_l205_20540

variable (p q : Prop)

theorem neg_q_sufficient_not_necc_neg_p (hp: p → q) (hnpq: ¬(q → p)) : (¬q → ¬p) ∧ (¬(¬p → ¬q)) :=
by
  sorry

end neg_q_sufficient_not_necc_neg_p_l205_20540


namespace sum_first_seven_terms_of_arith_seq_l205_20503

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Conditions: a_2 = 10 and a_5 = 1
def a_2 := 10
def a_5 := 1

-- The sum of the first 7 terms of the sequence
theorem sum_first_seven_terms_of_arith_seq (a d : ℤ) :
  arithmetic_seq a d 1 = a_2 →
  arithmetic_seq a d 4 = a_5 →
  (7 * a + (7 * 6 / 2) * d = 28) :=
by
  sorry

end sum_first_seven_terms_of_arith_seq_l205_20503


namespace range_of_k_for_circle_l205_20526

theorem range_of_k_for_circle (x y : ℝ) (k : ℝ) : 
  (x^2 + y^2 - 4*x + 2*y + 5*k = 0) → k < 1 :=
by 
  sorry

end range_of_k_for_circle_l205_20526


namespace equation_solutions_l205_20501

theorem equation_solutions :
  ∀ x y : ℤ, x^2 + x * y + y^2 + x + y - 5 = 0 → (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -3) ∨ (x = -3 ∧ y = 1) :=
by
  intro x y h
  sorry

end equation_solutions_l205_20501


namespace remainder_of_sum_of_first_150_numbers_l205_20577

def sum_of_first_n_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem remainder_of_sum_of_first_150_numbers :
  (sum_of_first_n_natural_numbers 150) % 5000 = 1275 :=
by
  sorry

end remainder_of_sum_of_first_150_numbers_l205_20577


namespace num_spacy_subsets_15_l205_20541

def spacy_subsets (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | 3     => 4
  | n + 1 => spacy_subsets n + if n ≥ 2 then spacy_subsets (n - 2) else 1

theorem num_spacy_subsets_15 : spacy_subsets 15 = 406 := by
  sorry

end num_spacy_subsets_15_l205_20541


namespace product_of_all_possible_values_l205_20514

theorem product_of_all_possible_values (x : ℝ) (h : 2 * |x + 3| - 4 = 2) :
  ∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = 0 :=
by
  sorry

end product_of_all_possible_values_l205_20514


namespace no_solution_exists_l205_20598

theorem no_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x^y + 3 = y^x ∧ 3 * x^y = y^x + 8) :=
by
  intro h
  obtain ⟨eq1, eq2⟩ := h
  sorry

end no_solution_exists_l205_20598


namespace area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l205_20572

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
  (1 / 2) * b * h

noncomputable def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

theorem area_triangle_ACD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 →
  area_of_triangle C 20 = 100 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

theorem area_trapezoid_ABCD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 → 
  area_trapezoid 24 10 24 = 260 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

end area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l205_20572


namespace vector_subtraction_proof_l205_20593

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end vector_subtraction_proof_l205_20593


namespace percentage_of_one_pair_repeated_digits_l205_20525

theorem percentage_of_one_pair_repeated_digits (n : ℕ) (h1 : 10000 ≤ n) (h2 : n ≤ 99999) :
  ∃ (percentage : ℝ), percentage = 56.0 :=
by
  sorry

end percentage_of_one_pair_repeated_digits_l205_20525


namespace green_peaches_count_l205_20510

def red_peaches : ℕ := 17
def green_peaches (x : ℕ) : Prop := red_peaches = x + 1

theorem green_peaches_count (x : ℕ) (h : green_peaches x) : x = 16 :=
by
  sorry

end green_peaches_count_l205_20510


namespace correct_formula_l205_20502

-- Given conditions
def table : List (ℕ × ℕ) := [(2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Candidate formulas
def formulaA (x : ℕ) : ℕ := 2 * x - 4
def formulaB (x : ℕ) : ℕ := x^2 - 3 * x + 2
def formulaC (x : ℕ) : ℕ := x^3 - 3 * x^2 + 2 * x
def formulaD (x : ℕ) : ℕ := x^2 - 4 * x
def formulaE (x : ℕ) : ℕ := x^2 - 4

-- The statement to be proven
theorem correct_formula : ∀ (x y : ℕ), (x, y) ∈ table → y = formulaB x :=
by
  sorry

end correct_formula_l205_20502


namespace ball_total_distance_l205_20519

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (bounces : ℕ) : ℝ :=
  let rec loop (height : ℝ) (total : ℝ) (remaining : ℕ) : ℝ :=
    if remaining = 0 then total
    else loop (height * bounce_factor) (total + height + height * bounce_factor) (remaining - 1)
  loop initial_height 0 bounces

theorem ball_total_distance : 
  total_distance 20 0.8 4 = 106.272 :=
by
  sorry

end ball_total_distance_l205_20519


namespace average_mark_of_excluded_students_l205_20534

theorem average_mark_of_excluded_students
  (N : ℕ) (A A_remaining : ℕ)
  (num_excluded : ℕ)
  (hN : N = 9)
  (hA : A = 60)
  (hA_remaining : A_remaining = 80)
  (h_excluded : num_excluded = 5) :
  (N * A - (N - num_excluded) * A_remaining) / num_excluded = 44 :=
by
  sorry

end average_mark_of_excluded_students_l205_20534


namespace product_of_roots_l205_20555

theorem product_of_roots :
  ∀ (x : ℝ), (|x|^2 - 3 * |x| - 10 = 0) →
  (∃ a b : ℝ, a ≠ b ∧ (|a| = 5 ∧ |b| = 5) ∧ a * b = -25) :=
by {
  sorry
}

end product_of_roots_l205_20555


namespace common_number_exists_l205_20520

def sum_of_list (l : List ℚ) : ℚ := l.sum

theorem common_number_exists (l1 l2 : List ℚ) (commonNumber : ℚ) 
    (h1 : l1.length = 5) 
    (h2 : l2.length = 5) 
    (h3 : sum_of_list l1 / 5 = 7) 
    (h4 : sum_of_list l2 / 5 = 10) 
    (h5 : (sum_of_list l1 + sum_of_list l2 - commonNumber) / 9 = 74 / 9) 
    : commonNumber = 11 :=
sorry

end common_number_exists_l205_20520


namespace binomial_expression_value_l205_20588

theorem binomial_expression_value :
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end binomial_expression_value_l205_20588


namespace regular_pentagon_cannot_cover_floor_completely_l205_20578

theorem regular_pentagon_cannot_cover_floor_completely
  (hexagon_interior_angle : ℝ)
  (pentagon_interior_angle : ℝ)
  (square_interior_angle : ℝ)
  (triangle_interior_angle : ℝ)
  (hexagon_condition : 360 / hexagon_interior_angle = 3)
  (square_condition : 360 / square_interior_angle = 4)
  (triangle_condition : 360 / triangle_interior_angle = 6)
  (pentagon_condition : 360 / pentagon_interior_angle ≠ 3)
  (pentagon_condition2 : 360 / pentagon_interior_angle ≠ 4)
  (pentagon_condition3 : 360 / pentagon_interior_angle ≠ 6) :
  pentagon_interior_angle = 108 := 
  sorry

end regular_pentagon_cannot_cover_floor_completely_l205_20578


namespace ratio_of_volumes_l205_20537

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem ratio_of_volumes :
  let r_C := 10
  let h_C := 20
  let r_D := 18
  let h_D := 12
  volume_cone r_C h_C / volume_cone r_D h_D = 125 / 243 :=
by
  sorry

end ratio_of_volumes_l205_20537


namespace ball_arrangement_divisibility_l205_20523

theorem ball_arrangement_divisibility :
  ∀ (n : ℕ), (∀ (i : ℕ), i < n → (∃ j k l m : ℕ, j < k ∧ k < l ∧ l < m ∧ m < n ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ j
    ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m)) →
  ¬((n = 2021) ∨ (n = 2022) ∨ (n = 2023) ∨ (n = 2024)) :=
sorry

end ball_arrangement_divisibility_l205_20523


namespace total_rainfall_2019_to_2021_l205_20567

theorem total_rainfall_2019_to_2021 :
  let R2019 := 50
  let R2020 := R2019 + 5
  let R2021 := R2020 - 3
  12 * R2019 + 12 * R2020 + 12 * R2021 = 1884 :=
by
  sorry

end total_rainfall_2019_to_2021_l205_20567


namespace farey_sequence_mediant_l205_20527

theorem farey_sequence_mediant (a b x y c d : ℕ) (h₁ : a * y < b * x) (h₂ : b * x < y * c) (farey_consecutiveness: bx - ay = 1 ∧ cy - dx = 1) : (x / y) = (a+c) / (b+d) := 
by
  sorry

end farey_sequence_mediant_l205_20527


namespace percentage_return_is_25_l205_20544

noncomputable def percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ) : ℝ :=
  (dividend_rate / 100 * face_value / purchase_price) * 100

theorem percentage_return_is_25 :
  percentage_return_on_investment 18.5 50 37 = 25 := 
by
  sorry

end percentage_return_is_25_l205_20544


namespace problem1_problem2_l205_20562

-- First problem
theorem problem1 (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := 
by sorry

-- Second problem
theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : ∃ k, a^x = k ∧ b^y = k ∧ c^z = k) (h_sum : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := 
by sorry

end problem1_problem2_l205_20562


namespace find_a_plus_b_l205_20570

theorem find_a_plus_b (a b : ℤ) (h1 : 2 * a = 0) (h2 : a^2 - b = 25) : a + b = -25 :=
by 
  sorry

end find_a_plus_b_l205_20570


namespace rate_per_sq_meter_l205_20583

theorem rate_per_sq_meter (length width : ℝ) (total_cost : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : total_cost = 16500) : 
  total_cost / (length * width) = 800 :=
by
  sorry

end rate_per_sq_meter_l205_20583


namespace num_of_elements_l205_20508

-- Lean statement to define and prove the problem condition
theorem num_of_elements (n S : ℕ) (h1 : (S + 26) / n = 5) (h2 : (S + 36) / n = 6) : n = 10 := by
  sorry

end num_of_elements_l205_20508


namespace city_a_location_l205_20533

theorem city_a_location (ϕ A_latitude : ℝ) (m : ℝ) (h_eq_height : true)
  (h_shadows_3x : true) 
  (h_angle: true) (h_southern : A_latitude < 0) 
  (h_rad_lat : ϕ = abs A_latitude):

  ϕ = 45 ∨ ϕ = 7.14 :=
by 
  sorry

end city_a_location_l205_20533


namespace directrix_eqn_of_parabola_l205_20542

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l205_20542


namespace boxes_used_l205_20559

-- Define the given conditions
def oranges_per_box : ℕ := 10
def total_oranges : ℕ := 2650

-- Define the proof statement
theorem boxes_used : total_oranges / oranges_per_box = 265 :=
by
  -- Proof goes here
  sorry

end boxes_used_l205_20559


namespace tiling_condition_l205_20539

theorem tiling_condition (a b n : ℕ) : 
  (∃ f : ℕ → ℕ × ℕ, ∀ i < (a * b) / n, (f i).fst < a ∧ (f i).snd < b) ↔ (n ∣ a ∨ n ∣ b) :=
sorry

end tiling_condition_l205_20539


namespace find_equation_of_line_l205_20591

open Real

noncomputable def equation_of_line : Prop :=
  ∃ c : ℝ, (∀ (x y : ℝ), (3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 → 2 * x + 3 * y + c = 0)) ∧
  ∃ x y : ℝ, 3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 ∧
              (2 * x + 3 * y + c = 0 → 6 * x + 9 * y - 7 = 0)

theorem find_equation_of_line : equation_of_line :=
sorry

end find_equation_of_line_l205_20591


namespace system_has_infinite_solutions_l205_20506

theorem system_has_infinite_solutions :
  ∀ (x y : ℝ), (3 * x - 4 * y = 5) ↔ (6 * x - 8 * y = 10) ∧ (9 * x - 12 * y = 15) :=
by
  sorry

end system_has_infinite_solutions_l205_20506


namespace evaluate_expression_l205_20543

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 :=
by sorry

end evaluate_expression_l205_20543


namespace find_length_of_PC_l205_20554

theorem find_length_of_PC (P A B C D : ℝ × ℝ) (h1 : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 25)
                            (h2 : (P.1 - D.1)^2 + (P.2 - D.2)^2 = 36)
                            (h3 : (P.1 - B.1)^2 + (P.2 - B.2)^2 = 49)
                            (square_ABCD : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) :
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = 38 :=
by
  sorry

end find_length_of_PC_l205_20554


namespace floors_above_l205_20587

theorem floors_above (dennis_floor charlie_floor frank_floor : ℕ)
  (h1 : dennis_floor = 6)
  (h2 : frank_floor = 16)
  (h3 : charlie_floor = frank_floor / 4) :
  dennis_floor - charlie_floor = 2 :=
by
  sorry

end floors_above_l205_20587


namespace term_number_l205_20556

theorem term_number (n : ℕ) : 
  (n ≥ 1) ∧ (5 * Real.sqrt 3 = Real.sqrt (3 + 4 * (n - 1))) → n = 19 :=
by
  intro h
  let h1 := h.1
  let h2 := h.2
  have h3 : (5 * Real.sqrt 3)^2 = (Real.sqrt (3 + 4 * (n - 1)))^2 := by sorry
  sorry

end term_number_l205_20556


namespace twenty_five_percent_of_2004_l205_20586

theorem twenty_five_percent_of_2004 : (1 / 4 : ℝ) * 2004 = 501 := by
  sorry

end twenty_five_percent_of_2004_l205_20586


namespace equation_has_three_solutions_l205_20561

theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ x^2 * (x - 1) * (x - 2) = 0 := 
by
  sorry

end equation_has_three_solutions_l205_20561


namespace volume_of_prism_l205_20595

-- Given conditions
def length : ℕ := 12
def width : ℕ := 8
def depth : ℕ := 8

-- Proving the volume of the rectangular prism
theorem volume_of_prism : length * width * depth = 768 := by
  sorry

end volume_of_prism_l205_20595


namespace circle_center_radius_l205_20564

/-
Given:
- The endpoints of a diameter are (2, -3) and (-8, 7).

Prove:
- The center of the circle is (-3, 2).
- The radius of the circle is 5√2.
-/

noncomputable def center_and_radius (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let Cx := ((A.1 + B.1) / 2)
  let Cy := ((A.2 + B.2) / 2)
  let radius := Real.sqrt ((A.1 - Cx) * (A.1 - Cx) + (A.2 - Cy) * (A.2 - Cy))
  (Cx, Cy, radius)

theorem circle_center_radius :
  center_and_radius (2, -3) (-8, 7) = (-3, 2, 5 * Real.sqrt 2) :=
by
  sorry

end circle_center_radius_l205_20564


namespace f_value_at_4_l205_20500

def f : ℝ → ℝ := sorry  -- Define f as a function from ℝ to ℝ

-- Specify the condition that f satisfies for all real numbers x
axiom f_condition (x : ℝ) : f (2^x) + x * f (2^(-x)) = 3

-- Statement to be proven: f(4) = -3
theorem f_value_at_4 : f 4 = -3 :=
by {
  -- Proof goes here
  sorry
}

end f_value_at_4_l205_20500


namespace combined_experience_l205_20582

noncomputable def james_experience : ℕ := 20
noncomputable def john_experience_8_years_ago : ℕ := 2 * (james_experience - 8)
noncomputable def john_current_experience : ℕ := john_experience_8_years_ago + 8
noncomputable def mike_experience : ℕ := john_current_experience - 16

theorem combined_experience :
  james_experience + john_current_experience + mike_experience = 68 :=
by
  sorry

end combined_experience_l205_20582


namespace height_of_pole_l205_20590

theorem height_of_pole (pole_shadow tree_shadow tree_height : ℝ) 
                       (ratio_equal : pole_shadow = 84 ∧ tree_shadow = 32 ∧ tree_height = 28) : 
                       round (tree_height * (pole_shadow / tree_shadow)) = 74 :=
by
  sorry

end height_of_pole_l205_20590


namespace reciprocal_neg_half_l205_20515

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l205_20515


namespace find_students_l205_20529

theorem find_students (n : ℕ) (h1 : n % 8 = 5) (h2 : n % 6 = 1) (h3 : n < 50) : n = 13 :=
sorry

end find_students_l205_20529


namespace platform_length_l205_20573

theorem platform_length
  (train_length : ℝ := 360) -- The train is 360 meters long
  (train_speed_kmh : ℝ := 45) -- The train runs at a speed of 45 km/hr
  (time_to_pass_platform : ℝ := 60) -- It takes 60 seconds to pass the platform
  (platform_length : ℝ) : platform_length = 390 :=
by
  sorry

end platform_length_l205_20573


namespace sin_double_angle_l205_20585

-- Lean code to define the conditions and represent the problem
variable (α : ℝ)
variable (x y : ℝ) 
variable (r : ℝ := Real.sqrt (x^2 + y^2))

-- Given conditions
def point_on_terminal_side (x y : ℝ) (h : x = 1 ∧ y = -2) : Prop :=
  ∃ α, (⟨1, -2⟩ : ℝ × ℝ) = ⟨Real.cos α * (Real.sqrt (1^2 + (-2)^2)), Real.sin α * (Real.sqrt (1^2 + (-2)^2))⟩

-- The theorem to prove
theorem sin_double_angle (h : point_on_terminal_side 1 (-2) ⟨rfl, rfl⟩) : 
  Real.sin (2 * α) = -4 / 5 := 
sorry

end sin_double_angle_l205_20585


namespace crayons_difference_l205_20574

theorem crayons_difference (total_crayons : ℕ) (given_crayons : ℕ) (lost_crayons : ℕ) (h1 : total_crayons = 589) (h2 : given_crayons = 571) (h3 : lost_crayons = 161) : (given_crayons - lost_crayons) = 410 := by
  sorry

end crayons_difference_l205_20574


namespace angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l205_20505

-- Definitions of the sides and conditions in triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: a + b = 6
axiom sum_of_sides : a + b = 6

-- Condition: Area of triangle ABC is 2 * sqrt(3)
axiom area_of_triangle : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3

-- Condition: a cos B + b cos A = 2c cos C
axiom cos_condition : (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C

-- Proof problem 1: Prove that C = π/3
theorem angle_C_is_pi_div_3 (h_cos : Real.cos C = 1/2) : C = Real.pi / 3 :=
sorry

-- Proof problem 2: Prove that c = 2 sqrt(3)
theorem side_c_is_2_sqrt_3 (h_sin : Real.sin C = Real.sqrt 3 / 2) : c = 2 * Real.sqrt 3 :=
sorry

end angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l205_20505


namespace sum_first_six_terms_l205_20568

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Define the existence of a geometric sequence with given properties
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given Condition: a_3 = 2a_4 = 2
def cond1 (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ a 4 = 1

-- Define the sum of the first n terms of the sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

-- We need to prove that under these conditions, S_6 = 63/4
theorem sum_first_six_terms 
  (hq : q = 1 / 2) 
  (ha : is_geometric_sequence a q) 
  (hcond1 : cond1 a) 
  (hS : geometric_sum a q S) : 
  S 6 = 63 / 4 := 
sorry

end sum_first_six_terms_l205_20568


namespace interest_rate_A_to_B_l205_20560

theorem interest_rate_A_to_B :
  ∀ (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ) (interest_C : ℝ) (interest_A : ℝ),
    principal = 3500 →
    rate_C = 0.13 →
    time = 3 →
    gain_B = 315 →
    interest_C = principal * rate_C * time →
    gain_B = interest_C - interest_A →
    interest_A = principal * (R / 100) * time →
    R = 10 := by
  sorry

end interest_rate_A_to_B_l205_20560


namespace girls_count_l205_20571

-- Definition of the conditions
variables (B G : ℕ)

def college_conditions (B G : ℕ) : Prop :=
  (B + G = 416) ∧ (B = (8 * G) / 5)

-- Statement to prove
theorem girls_count (B G : ℕ) (h : college_conditions B G) : G = 160 :=
by
  sorry

end girls_count_l205_20571


namespace compute_fraction_l205_20512

theorem compute_fraction (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) (sum_eq : x + y + z = 12) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end compute_fraction_l205_20512


namespace exchange_ways_10_dollar_l205_20565

theorem exchange_ways_10_dollar (p q : ℕ) (H : 2 * p + 5 * q = 200) : 
  ∃ (n : ℕ), n = 20 :=
by {
  sorry
}

end exchange_ways_10_dollar_l205_20565


namespace brenda_friends_l205_20576

def total_slices (pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ := pizzas * slices_per_pizza
def total_people (total_slices : ℕ) (slices_per_person : ℕ) : ℕ := total_slices / slices_per_person
def friends (total_people : ℕ) : ℕ := total_people - 1

theorem brenda_friends (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (slices_per_person : ℕ) (pizzas_ordered : pizzas = 5) 
  (slices_per_pizza_value : slices_per_pizza = 4) 
  (slices_per_person_value : slices_per_person = 2) :
  friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) = 9 :=
by
  rw [pizzas_ordered, slices_per_pizza_value, slices_per_person_value]
  sorry

end brenda_friends_l205_20576


namespace avogadro_constant_problem_l205_20550

theorem avogadro_constant_problem 
  (N_A : ℝ) -- Avogadro's constant
  (mass1 : ℝ := 18) (molar_mass1 : ℝ := 20) (moles1 : ℝ := mass1 / molar_mass1) 
  (atoms_D2O_molecules : ℝ := 2) (atoms_D2O : ℝ := moles1 * atoms_D2O_molecules * N_A)
  (mass2 : ℝ := 14) (molar_mass_N2CO : ℝ := 28) (moles2 : ℝ := mass2 / molar_mass_N2CO)
  (electrons_per_molecule : ℝ := 14) (total_electrons_mixture : ℝ := moles2 * electrons_per_molecule * N_A)
  (volume3 : ℝ := 2.24) (temp_unk : Prop := true) -- unknown temperature
  (pressure_unk : Prop := true) -- unknown pressure
  (carbonate_molarity : ℝ := 0.1) (volume_solution : ℝ := 1) (moles_carbonate : ℝ := carbonate_molarity * volume_solution) 
  (anions_carbonate_solution : ℝ := moles_carbonate * N_A) :
  (atoms_D2O ≠ 2 * N_A) ∧ (anions_carbonate_solution > 0.1 * N_A) ∧ (total_electrons_mixture = 7 * N_A) -> 
  True := sorry

end avogadro_constant_problem_l205_20550


namespace max_sheets_one_participant_l205_20522

theorem max_sheets_one_participant
  (n : ℕ) (avg_sheets : ℕ) (h1 : n = 40) (h2 : avg_sheets = 7) 
  (h3 : ∀ i : ℕ, i < n → 1 ≤ 1) : 
  ∃ max_sheets : ℕ, max_sheets = 241 :=
by
  sorry

end max_sheets_one_participant_l205_20522


namespace paint_per_color_equal_l205_20547

theorem paint_per_color_equal (total_paint : ℕ) (num_colors : ℕ) (paint_per_color : ℕ) : 
  total_paint = 15 ∧ num_colors = 3 → paint_per_color = 5 := by
  sorry

end paint_per_color_equal_l205_20547


namespace deepaks_age_l205_20563

theorem deepaks_age (R D : ℕ) (h1 : R / D = 5 / 2) (h2 : R + 6 = 26) : D = 8 := 
sorry

end deepaks_age_l205_20563


namespace cupboard_selling_percentage_l205_20551

theorem cupboard_selling_percentage (CP SP : ℝ) (h1 : CP = 6250) (h2 : SP + 1500 = 6250 * 1.12) :
  ((CP - SP) / CP) * 100 = 12 := by
sorry

end cupboard_selling_percentage_l205_20551
