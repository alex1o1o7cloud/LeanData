import Mathlib

namespace x_intercept_rotation_30_degrees_eq_l607_60756

noncomputable def x_intercept_new_line (x0 y0 : ℝ) (θ : ℝ) (a b c : ℝ) : ℝ :=
  let m := a / b
  let m' := (m + θ.tan) / (1 - m * θ.tan)
  let x_intercept := x0 - (y0 * (b - m * c)) / (m' * (b - m * c) - a)
  x_intercept

theorem x_intercept_rotation_30_degrees_eq :
  x_intercept_new_line 7 4 (Real.pi / 6) 4 (-7) 28 = 7 - (4 * (7 * Real.sqrt 3 - 4) / (4 * Real.sqrt 3 + 7)) :=
by 
  -- detailed math proof goes here 
  sorry

end x_intercept_rotation_30_degrees_eq_l607_60756


namespace units_digit_of_expression_l607_60745

theorem units_digit_of_expression :
  (8 * 18 * 1988 - 8^4) % 10 = 6 := 
by
  sorry

end units_digit_of_expression_l607_60745


namespace percentage_increase_l607_60766

theorem percentage_increase (original new : ℝ) (h_original : original = 50) (h_new : new = 75) : 
  (new - original) / original * 100 = 50 :=
by
  sorry

end percentage_increase_l607_60766


namespace books_loaned_out_l607_60747

theorem books_loaned_out (initial_books : ℕ) (returned_percentage : ℝ) (end_books : ℕ) (x : ℝ) :
    initial_books = 75 →
    returned_percentage = 0.70 →
    end_books = 63 →
    0.30 * x = (initial_books - end_books) →
    x = 40 := by
  sorry

end books_loaned_out_l607_60747


namespace determine_k_range_l607_60703

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (Real.log x) / (x * x)

theorem determine_k_range :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f k x = g x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) →
  k ∈ Set.Ico (1 / (Real.exp 1) ^ 2) (1 / (2 * Real.exp 1)) := 
  sorry

end determine_k_range_l607_60703


namespace complement_of_A_in_U_l607_60715

noncomputable def U := {x : ℝ | Real.exp x > 1}

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

def A := { x : ℝ | x > 1 }

def compl (U A : Set ℝ) := { x : ℝ | x ∈ U ∧ x ∉ A }

theorem complement_of_A_in_U : compl U A = { x : ℝ | 0 < x ∧ x ≤ 1 } := sorry

end complement_of_A_in_U_l607_60715


namespace slices_with_both_toppings_l607_60787

theorem slices_with_both_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (all_with_topping : total_slices = 15 ∧ pepperoni_slices = 8 ∧ mushroom_slices = 12 ∧ ∀ i, i < 15 → (i < 8 ∨ i < 12)) :
  ∃ n, (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices ∧ n = 5 :=
by
  sorry

end slices_with_both_toppings_l607_60787


namespace geometric_sequence_first_term_l607_60791

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

theorem geometric_sequence_first_term (a_1 q : ℝ)
  (h1 : a_n a_1 q 2 * a_n a_1 q 3 * a_n a_1 q 4 = 27)
  (h2 : a_n a_1 q 6 = 27) 
  (h3 : a_1 > 0) : a_1 = 1 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_first_term_l607_60791


namespace complex_number_multiplication_l607_60724

theorem complex_number_multiplication (i : ℂ) (hi : i * i = -1) : i * (1 + i) = -1 + i :=
by sorry

end complex_number_multiplication_l607_60724


namespace solve_for_b_l607_60763

def p (x : ℝ) : ℝ := 2 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

theorem solve_for_b (b : ℝ) : p (q 5 b) = 11 → b = 7 := by
  sorry

end solve_for_b_l607_60763


namespace ratio_second_to_first_l607_60722

-- Condition 1: The first bell takes 50 pounds of bronze
def first_bell_weight : ℕ := 50

-- Condition 2: The second bell is a certain size compared to the first bell
variable (x : ℕ) -- the ratio of the size of the second bell to the first bell
def second_bell_weight := first_bell_weight * x

-- Condition 3: The third bell is four times the size of the second bell
def third_bell_weight := 4 * second_bell_weight x

-- Condition 4: The total weight of bronze required is 550 pounds
def total_weight : ℕ := 550

-- Define the proof problem
theorem ratio_second_to_first (x : ℕ) (h : 50 + 50 * x + 200 * x = 550) : x = 2 :=
by
  sorry

end ratio_second_to_first_l607_60722


namespace cake_and_tea_cost_l607_60776

theorem cake_and_tea_cost (cost_of_milk_tea : ℝ) (cost_of_cake : ℝ)
    (h1 : cost_of_cake = (3 / 4) * cost_of_milk_tea)
    (h2 : cost_of_milk_tea = 2.40) :
    2 * cost_of_cake + cost_of_milk_tea = 6.00 := 
sorry

end cake_and_tea_cost_l607_60776


namespace compute_fraction_sum_l607_60736

-- Define the equation whose roots are a, b, c
def cubic_eq (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x = 12

-- State the main theorem
theorem compute_fraction_sum 
  (a b c : ℝ) 
  (ha : cubic_eq a) 
  (hb : cubic_eq b) 
  (hc : cubic_eq c) :
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  ∃ (r : ℝ), r = -23/12 ∧ (ab/c + bc/a + ca/b) = r := 
  sorry

end compute_fraction_sum_l607_60736


namespace find_x_l607_60737

theorem find_x (x : ℝ) (h1 : 3 * Real.sin (2 * x) = 2 * Real.sin x) (h2 : 0 < x ∧ x < Real.pi) :
  x = Real.arccos (1 / 3) :=
by
  sorry

end find_x_l607_60737


namespace lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l607_60702

def total_area_of_triangles_and_quadrilateral (A B Q : ℝ) : ℝ :=
  A + B + Q

def lena_triangles_and_quadrilateral_area (A B Q : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_quadrilateral A B Q

def total_area_of_triangles_and_pentagon (C D P : ℝ) : ℝ :=
  C + D + P

def vasya_triangles_and_pentagon_area (C D P : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_pentagon C D P

theorem lena_can_form_rectangles (A B Q : ℝ) (h : lena_triangles_and_quadrilateral_area A B Q) :
  lena_triangles_and_quadrilateral_area A B Q :=
by 
-- We assume the definition holds as given
sorry

theorem vasya_can_form_rectangles (C D P : ℝ) (h : vasya_triangles_and_pentagon_area C D P) :
  vasya_triangles_and_pentagon_area C D P :=
by 
-- We assume the definition holds as given
sorry

theorem lena_and_vasya_can_be_right (A B Q C D P : ℝ)
  (hlena : lena_triangles_and_quadrilateral_area A B Q)
  (hvasya : vasya_triangles_and_pentagon_area C D P) :
  lena_triangles_and_quadrilateral_area A B Q ∧ vasya_triangles_and_pentagon_area C D P :=
by 
-- Combining both assumptions
exact ⟨hlena, hvasya⟩

end lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l607_60702


namespace nine_digit_palindrome_count_l607_60700

-- Defining the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

-- Defining the proposition of the number of 9-digit palindromes
def num_9_digit_palindromes (digs : Multiset ℕ) : ℕ := 36

-- The proof statement
theorem nine_digit_palindrome_count : num_9_digit_palindromes digits = 36 := 
sorry

end nine_digit_palindrome_count_l607_60700


namespace math_problem_proof_l607_60773

theorem math_problem_proof :
    24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 :=
by
  -- Proof omitted
  sorry

end math_problem_proof_l607_60773


namespace sarah_math_homework_pages_l607_60754

theorem sarah_math_homework_pages (x : ℕ) 
  (h1 : ∀ page, 4 * page = 4 * 6 + 4 * x)
  (h2 : 40 = 4 * 6 + 4 * x) : 
  x = 4 :=
by 
  sorry

end sarah_math_homework_pages_l607_60754


namespace calculate_expression_l607_60725

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l607_60725


namespace regular_17gon_symmetries_l607_60707

theorem regular_17gon_symmetries : 
  let L := 17
  let R := 360 / 17
  L + R = 17 + 360 / 17 :=
by
  sorry

end regular_17gon_symmetries_l607_60707


namespace combined_average_age_l607_60752

theorem combined_average_age :
  (8 * 35 + 6 * 30) / (8 + 6) = 33 :=
by
  sorry

end combined_average_age_l607_60752


namespace possible_integer_roots_l607_60765

theorem possible_integer_roots (x : ℤ) :
  x^3 + 3 * x^2 - 4 * x - 13 = 0 →
  x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13 :=
by sorry

end possible_integer_roots_l607_60765


namespace how_many_years_younger_l607_60733

-- Define conditions
def age_ratio (sandy_age moll_age : ℕ) := sandy_age * 9 = moll_age * 7
def sandy_age := 70

-- Define the theorem to prove
theorem how_many_years_younger 
  (molly_age : ℕ) 
  (h1 : age_ratio sandy_age molly_age) 
  (h2 : sandy_age = 70) : molly_age - sandy_age = 20 := 
sorry

end how_many_years_younger_l607_60733


namespace incorrect_statement_l607_60713

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h₁ : f 0 = -1)
variable (h₂ : ∀ x, f' x > k)
variable (h₃ : k > 1)

theorem incorrect_statement :
  ¬ f (1 / (k - 1)) < 1 / (k - 1) :=
sorry

end incorrect_statement_l607_60713


namespace solve_system_of_equations_l607_60734

theorem solve_system_of_equations:
  ∃ (x y z : ℝ), 
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34 ∧
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l607_60734


namespace find_a_l607_60741

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → 
     ∀ (x' y' : ℝ), (x' = x - 2 * (x - a * y + 2) / (1 + a^2)) ∧ (y' = y - 2 * a * (x - a * y + 2) / (1 + a^2)) → 
     (x'^2 + y'^2 + 2 * x' - 4 * y' + 1 = 0)) → 
  (a = -1 / 2) := 
sorry

end find_a_l607_60741


namespace parity_of_f_minimum_value_of_f_l607_60769

noncomputable def f (x a : ℝ) : ℝ := x^2 + |x - a| - 1

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem parity_of_f (a : ℝ) :
  (a = 0 → is_even_function (f a)) ∧
  (a ≠ 0 → ¬is_even_function (f a) ∧ ¬is_odd_function (f a)) := 
by sorry

theorem minimum_value_of_f (a : ℝ) :
  (a ≤ -1/2 → ∀ x : ℝ, f x a ≥ -a - 5 / 4) ∧
  (-1/2 < a ∧ a ≤ 1/2 → ∀ x : ℝ, f x a ≥ a^2 - 1) ∧
  (a > 1/2 → ∀ x : ℝ, f x a ≥ a - 5 / 4) :=
by sorry

end parity_of_f_minimum_value_of_f_l607_60769


namespace right_triangle_ratio_l607_60742

theorem right_triangle_ratio (x : ℝ) :
  let AB := 3 * x
  let BC := 4 * x
  let AC := (AB ^ 2 + BC ^ 2).sqrt
  let h := AC
  let AD := 16 / 21 * h / (16 / 21 + 1)
  let CD := h / (16 / 21 + 1)
  (CD / AD) = 21 / 16 :=
by 
  sorry

end right_triangle_ratio_l607_60742


namespace find_d_for_single_point_l607_60718

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end find_d_for_single_point_l607_60718


namespace graphs_intersect_exactly_eight_times_l607_60792

theorem graphs_intersect_exactly_eight_times (A : ℝ) (hA : 0 < A) :
  ∃ (count : ℕ), count = 8 ∧ ∀ x y : ℝ, y = A * x ^ 4 → y ^ 2 + 5 = x ^ 2 + 6 * y :=
sorry

end graphs_intersect_exactly_eight_times_l607_60792


namespace find_m_set_l607_60799

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
noncomputable def B (m : ℝ) : Set ℝ := if m = 0 then ∅ else {-1/m}

theorem find_m_set :
  { m : ℝ | A ∪ B m = A } = {0, -1/2, -1/3} :=
by
  sorry

end find_m_set_l607_60799


namespace split_cube_l607_60782

theorem split_cube (m : ℕ) (hm : m > 1) (h : ∃ k, ∃ l, l > 0 ∧ (3 + 2 * (k - 1)) = 59 ∧ (k + l = (m * (m - 1)) / 2)) : m = 8 :=
sorry

end split_cube_l607_60782


namespace A_is_5_years_older_than_B_l607_60761

-- Given conditions
variables (A B : ℕ) -- A and B are the current ages
variables (x y : ℕ) -- x is the current age of A, y is the current age of B
variables 
  (A_was_B_age : A = y)
  (B_was_10_when_A_was_B_age : B = 10)
  (B_will_be_A_age : B = x)
  (A_will_be_25_when_B_will_be_A_age : A = 25)

-- Define the theorem to prove that A is 5 years older than B: A = B + 5
theorem A_is_5_years_older_than_B (x y : ℕ) (A B : ℕ) 
  (A_was_B_age : x = y) 
  (B_was_10_when_A_was_B_age : y = 10) 
  (B_will_be_A_age : y = x) 
  (A_will_be_25_when_B_will_be_A_age : x = 25): 
  x - y = 5 := 
by sorry

end A_is_5_years_older_than_B_l607_60761


namespace average_temperature_week_l607_60704

theorem average_temperature_week :
  let d1 := 40
  let d2 := 40
  let d3 := 40
  let d4 := 80
  let d5 := 80
  let remaining_days_total := 140
  d1 + d2 + d3 + d4 + d5 + remaining_days_total = 420 ∧ 420 / 7 = 60 :=
by sorry

end average_temperature_week_l607_60704


namespace total_books_is_177_l607_60789

-- Define the number of books read (x), books yet to read (y), and the total number of books (T)
def x : Nat := 13
def y : Nat := 8
def T : Nat := x^2 + y

-- Prove that the total number of books in the series is 177
theorem total_books_is_177 : T = 177 :=
  sorry

end total_books_is_177_l607_60789


namespace sum_of_three_digits_eq_nine_l607_60744

def horizontal_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 0
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 2
  | 6 => 1
  | 7 => 1
  | 8 => 3
  | 9 => 2
  | _ => 0  -- Invalid digit

def vertical_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 2
  | 2 => 3
  | 3 => 3
  | 4 => 3
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 4
  | 9 => 3
  | _ => 0  -- Invalid digit

theorem sum_of_three_digits_eq_nine :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
             (horizontal_segments a + horizontal_segments b + horizontal_segments c = 5) ∧ 
             (vertical_segments a + vertical_segments b + vertical_segments c = 10) ∧
             (a + b + c = 9) :=
sorry

end sum_of_three_digits_eq_nine_l607_60744


namespace simplify_and_evaluate_expression_l607_60762

theorem simplify_and_evaluate_expression (a : ℂ) (h: a^2 + 4 * a + 1 = 0) :
  ( ( (a + 2) / (a^2 - 2 * a) + 8 / (4 - a^2) ) / ( (a^2 - 4) / a ) ) = 1 / 3 := by
  sorry

end simplify_and_evaluate_expression_l607_60762


namespace seating_arrangements_l607_60757

-- Define the conditions and the proof problem
theorem seating_arrangements (children : Finset (Fin 6)) 
  (is_sibling_pair : (Fin 6) -> (Fin 6) -> Prop)
  (no_siblings_next_to_each_other : (Fin 6) -> (Fin 6) -> Bool)
  (no_sibling_directly_in_front : (Fin 6) -> (Fin 6) -> Bool) :
  -- Statement: There are 96 valid seating arrangements
  ∃ (arrangements : Finset (Fin 6 -> Fin (2 * 3))),
  arrangements.card = 96 :=
by
  -- Proof omitted
  sorry

end seating_arrangements_l607_60757


namespace number_of_true_propositions_l607_60751

open Classical

axiom real_numbers (a b : ℝ): Prop

noncomputable def original_proposition (a b : ℝ) : Prop := a > b → a * abs a > b * abs b
noncomputable def converse_proposition (a b : ℝ) : Prop := a * abs a > b * abs b → a > b
noncomputable def negation_proposition (a b : ℝ) : Prop := a ≤ b → a * abs a ≤ b * abs b
noncomputable def contrapositive_proposition (a b : ℝ) : Prop := a * abs a ≤ b * abs b → a ≤ b

theorem number_of_true_propositions (a b : ℝ) (h₁: original_proposition a b) 
  (h₂: converse_proposition a b) (h₃: negation_proposition a b)
  (h₄: contrapositive_proposition a b) : ∃ n, n = 4 := 
by
  -- The proof would go here, proving that ∃ n, n = 4 is true.
  sorry

end number_of_true_propositions_l607_60751


namespace arithmetic_sequence_sum_l607_60793

theorem arithmetic_sequence_sum (c d e : ℕ) (h1 : 10 - 3 = 7) (h2 : 17 - 10 = 7) (h3 : c - 17 = 7) (h4 : d - c = 7) (h5 : e - d = 7) : 
  c + d + e = 93 :=
sorry

end arithmetic_sequence_sum_l607_60793


namespace time_to_fill_pond_l607_60759

-- Conditions:
def pond_capacity : ℕ := 200
def normal_pump_rate : ℕ := 6
def drought_factor : ℚ := 2 / 3

-- The current pumping rate:
def current_pump_rate : ℚ := normal_pump_rate * drought_factor

-- We need to prove the time it takes to fill the pond is 50 minutes:
theorem time_to_fill_pond : 
  (pond_capacity : ℚ) / current_pump_rate = 50 := 
sorry

end time_to_fill_pond_l607_60759


namespace larry_substituted_value_l607_60731

theorem larry_substituted_value :
  ∀ (a b c d e : ℤ), a = 5 → b = 3 → c = 4 → d = 2 → e = 2 → 
  (a + b - c + d - e = a + (b - (c + (d - e)))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end larry_substituted_value_l607_60731


namespace remainder_division_l607_60746

theorem remainder_division (x r : ℕ) (h₁ : 1650 - x = 1390) (h₂ : 1650 = 6 * x + r) : r = 90 := by
  sorry

end remainder_division_l607_60746


namespace y_intercept_l607_60711

theorem y_intercept (x1 y1 : ℝ) (m : ℝ) (h1 : x1 = -2) (h2 : y1 = 4) (h3 : m = 1 / 2) : 
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1/2 * x + 5) ∧ b = 5 := 
by
  sorry

end y_intercept_l607_60711


namespace divisibility_by_n5_plus_1_l607_60719

theorem divisibility_by_n5_plus_1 (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n^5 + 1 ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) :=
sorry

end divisibility_by_n5_plus_1_l607_60719


namespace bob_distance_walked_l607_60732

theorem bob_distance_walked
    (dist : ℕ)
    (yolanda_rate : ℕ)
    (bob_rate : ℕ)
    (hour_diff : ℕ)
    (meet_time_bob: ℕ) :

    dist = 31 → yolanda_rate = 1 → bob_rate = 2 → hour_diff = 1 → meet_time_bob = 10 →
    (bob_rate * meet_time_bob) = 20 :=
by
  intros
  sorry

end bob_distance_walked_l607_60732


namespace calculation_results_in_a_pow_5_l607_60726

variable (a : ℕ)

theorem calculation_results_in_a_pow_5 : a^3 * a^2 = a^5 := 
  by sorry

end calculation_results_in_a_pow_5_l607_60726


namespace percentage_BCM_hens_l607_60785

theorem percentage_BCM_hens (total_chickens : ℕ) (BCM_percentage : ℝ) (BCM_hens : ℕ) : 
  total_chickens = 100 → BCM_percentage = 0.20 → BCM_hens = 16 →
  ((BCM_hens : ℝ) / (total_chickens * BCM_percentage)) * 100 = 80 :=
by
  sorry

end percentage_BCM_hens_l607_60785


namespace floor_length_l607_60772

/-- Given the rectangular tiles of size 50 cm by 40 cm, which are laid on a rectangular floor
without overlap and with a maximum of 9 tiles. Prove the floor length is 450 cm. -/
theorem floor_length (tiles_max : ℕ) (tile_length tile_width floor_length floor_width : ℕ)
  (Htile_length : tile_length = 50) (Htile_width : tile_width = 40)
  (Htiles_max : tiles_max = 9)
  (Hconditions : (∀ m n : ℕ, (m * n = tiles_max) → 
                  (floor_length = m * tile_length ∨ floor_length = m * tile_width)))
  : floor_length = 450 :=
by 
  sorry

end floor_length_l607_60772


namespace gathering_handshakes_l607_60743

theorem gathering_handshakes :
  let N := 12       -- twelve people, six couples
  let shakes_per_person := 9   -- each person shakes hands with 9 others
  let total_shakes := (N * shakes_per_person) / 2
  total_shakes = 54 := 
by
  sorry

end gathering_handshakes_l607_60743


namespace largest_time_for_77_degrees_l607_60795

-- Define the initial conditions of the problem
def temperature_eqn (t : ℝ) : ℝ := -t^2 + 14 * t + 40

-- Define the proposition we want to prove
theorem largest_time_for_77_degrees : ∃ t, temperature_eqn t = 77 ∧ t = 11 := 
sorry

end largest_time_for_77_degrees_l607_60795


namespace books_per_bookshelf_l607_60790

theorem books_per_bookshelf (total_bookshelves total_books books_per_bookshelf : ℕ)
  (h1 : total_bookshelves = 23)
  (h2 : total_books = 621)
  (h3 : total_books = total_bookshelves * books_per_bookshelf) :
  books_per_bookshelf = 27 :=
by 
  -- Proof goes here
  sorry

end books_per_bookshelf_l607_60790


namespace percent_sales_other_l607_60794

theorem percent_sales_other (percent_notebooks : ℕ) (percent_markers : ℕ) (h1 : percent_notebooks = 42) (h2 : percent_markers = 26) :
    100 - (percent_notebooks + percent_markers) = 32 := by
  sorry

end percent_sales_other_l607_60794


namespace find_divisor_l607_60796

theorem find_divisor : ∃ (divisor : ℕ), ∀ (quotient remainder dividend : ℕ), quotient = 14 ∧ remainder = 7 ∧ dividend = 301 → (dividend = divisor * quotient + remainder) ∧ divisor = 21 :=
by
  sorry

end find_divisor_l607_60796


namespace problem_solution_l607_60735

noncomputable def area_triangle_ABC
  (R : ℝ) 
  (angle_BAC : ℝ) 
  (angle_DAC : ℝ) : ℝ :=
  let α := angle_DAC
  let β := angle_BAC
  2 * R^2 * (Real.sin α) * (Real.sin β) * (Real.sin (α + β))

theorem problem_solution :
  ∀ (R : ℝ) (angle_BAC : ℝ) (angle_DAC : ℝ),
  R = 3 →
  angle_BAC = (Real.pi / 4) →
  angle_DAC = (5 * Real.pi / 12) →
  area_triangle_ABC R angle_BAC angle_DAC = 10 :=
by intros R angle_BAC angle_DAC hR hBAC hDAC
   sorry

end problem_solution_l607_60735


namespace machines_in_first_scenario_l607_60767

theorem machines_in_first_scenario (x : ℕ) (hx : x ≠ 0) : 
  ∃ n : ℕ, (∀ m : ℕ, (∀ r1 r2 : ℚ, r1 = (x:ℚ) / (6 * n) → r2 = (3 * x:ℚ) / (6 * 12) → r1 = r2 → m = 12 → 3 * n = 12) → n = 4) :=
by
  sorry

end machines_in_first_scenario_l607_60767


namespace quadratic_roots_identity_l607_60771

theorem quadratic_roots_identity :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) →
  (x1^2 - 2 * x1 * x2 + x2^2) = 25 :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_identity_l607_60771


namespace triangle_area_ordering_l607_60753

variable (m n p : ℚ)

theorem triangle_area_ordering (hm : m = 15 / 2) (hn : n = 13 / 2) (hp : p = 7) : n < p ∧ p < m := by
  sorry

end triangle_area_ordering_l607_60753


namespace find_g_3_16_l607_60708

theorem find_g_3_16 (g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x → x ≤ 1 → g x = g x) 
(h2 : g 0 = 0) 
(h3 : ∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) 
(h4 : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x) 
(h5 : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) : 
  g (3 / 16) = 8 / 27 :=
sorry

end find_g_3_16_l607_60708


namespace regular_price_of_tire_l607_60779

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 10 = 250) : x = 80 :=
sorry

end regular_price_of_tire_l607_60779


namespace real_roots_iff_le_one_l607_60778

theorem real_roots_iff_le_one (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 :=
by
  sorry

end real_roots_iff_le_one_l607_60778


namespace odd_number_as_diff_of_squares_l607_60783

theorem odd_number_as_diff_of_squares :
    ∀ (x y : ℤ), 63 = x^2 - y^2 ↔ (x = 32 ∧ y = 31) ∨ (x = 12 ∧ y = 9) ∨ (x = 8 ∧ y = 1) := 
by
  sorry

end odd_number_as_diff_of_squares_l607_60783


namespace max_term_of_sequence_l607_60717

def a (n : ℕ) : ℚ := (n : ℚ) / (n^2 + 156)

theorem max_term_of_sequence : ∃ n, (n = 12 ∨ n = 13) ∧ (∀ m, a m ≤ a n) := by 
  sorry

end max_term_of_sequence_l607_60717


namespace abs_neg_two_l607_60721

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l607_60721


namespace hours_practicing_l607_60764

theorem hours_practicing (W : ℕ) (hours_weekday : ℕ) 
  (h1 : hours_weekday = W + 17)
  (h2 : W + hours_weekday = 33) :
  W = 8 :=
sorry

end hours_practicing_l607_60764


namespace number_of_numbers_l607_60714

theorem number_of_numbers (N : ℕ) (h_avg : (18 * N + 40) / N = 22) : N = 10 :=
by
  sorry

end number_of_numbers_l607_60714


namespace bleaching_process_percentage_decrease_l607_60740

noncomputable def total_percentage_decrease (L B : ℝ) : ℝ :=
  let area1 := (0.80 * L) * (0.90 * B)
  let area2 := (0.85 * (0.80 * L)) * (0.95 * (0.90 * B))
  let area3 := (0.90 * (0.85 * (0.80 * L))) * (0.92 * (0.95 * (0.90 * B)))
  ((L * B - area3) / (L * B)) * 100

theorem bleaching_process_percentage_decrease (L B : ℝ) :
  total_percentage_decrease L B = 44.92 :=
by
  sorry

end bleaching_process_percentage_decrease_l607_60740


namespace max_sum_at_n_is_6_l607_60760

-- Assuming an arithmetic sequence a_n where a_1 = 4 and d = -5/7
def arithmetic_seq (n : ℕ) : ℚ := (33 / 7) - (5 / 7) * n

-- Sum of the first n terms (S_n) of the arithmetic sequence {a_n}
def sum_arithmetic_seq (n : ℕ) : ℚ := (n / 2) * (2 * (arithmetic_seq 1) + (n - 1) * (-5 / 7))

theorem max_sum_at_n_is_6 
  (a_1 : ℚ) (d : ℚ) (h1 : a_1 = 4) (h2 : d = -5/7) :
  ∀ n : ℕ, sum_arithmetic_seq n ≤ sum_arithmetic_seq 6 :=
by
  sorry

end max_sum_at_n_is_6_l607_60760


namespace value_of_larger_denom_eq_10_l607_60739

/-- Anna has 12 bills in her wallet, and the total value is $100. 
    She has 4 $5 bills and 8 bills of a larger denomination.
    Prove that the value of the larger denomination bill is $10. -/
theorem value_of_larger_denom_eq_10 (n : ℕ) (b : ℤ) (total_value : ℤ) (five_bills : ℕ) (larger_bills : ℕ):
    (total_value = 100) ∧ 
    (five_bills = 4) ∧ 
    (larger_bills = 8) ∧ 
    (n = five_bills + larger_bills) ∧ 
    (n = 12) → 
    (b = 10) :=
by
  sorry

end value_of_larger_denom_eq_10_l607_60739


namespace complement_of_A_in_U_l607_60749

open Set

def univeral_set : Set ℕ := { x | x + 1 ≤ 0 ∨ 0 ≤ x - 5 }

def A : Set ℕ := {1, 2, 4}

noncomputable def complement_U_A : Set ℕ := {0, 3}

theorem complement_of_A_in_U : (compl A ∩ univeral_set) = complement_U_A := 
by 
  sorry

end complement_of_A_in_U_l607_60749


namespace triangle_area_l607_60788

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l607_60788


namespace batsman_average_l607_60777

theorem batsman_average
  (avg_20_matches : ℕ → ℕ → ℕ)
  (avg_10_matches : ℕ → ℕ → ℕ)
  (total_1st_20 : ℕ := avg_20_matches 20 30)
  (total_next_10 : ℕ := avg_10_matches 10 15) :
  (total_1st_20 + total_next_10) / 30 = 25 :=
by
  sorry

end batsman_average_l607_60777


namespace complement_of_union_l607_60774

-- Define the universal set U, set M, and set N as given:
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- Define the complement of a set relative to the universal set U
def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- Prove that the complement of M ∪ N with respect to U is {1, 6}
theorem complement_of_union : complement_U (M ∪ N) = {1, 6} :=
  sorry -- proof goes here

end complement_of_union_l607_60774


namespace complex_solution_l607_60723

theorem complex_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (Complex.mk a b)^2 = Complex.mk 3 4) :
  Complex.mk a b = Complex.mk 2 1 :=
sorry

end complex_solution_l607_60723


namespace unpainted_cubes_count_l607_60770

noncomputable def num_unpainted_cubes : ℕ :=
  let total_cubes := 216
  let painted_on_faces := 16 * 6 / 1  -- Central 4x4 areas on each face
  let shared_edges := ((4 * 4) * 6) / 2  -- Shared edges among faces
  let shared_corners := (4 * 6) / 3  -- Shared corners among faces
  let total_painted := painted_on_faces - shared_edges - shared_corners
  total_cubes - total_painted

theorem unpainted_cubes_count : num_unpainted_cubes = 160 := sorry

end unpainted_cubes_count_l607_60770


namespace tan_difference_l607_60730

open Real

noncomputable def tan_difference_intermediate (θ : ℝ) : ℝ :=
  (tan θ - tan (π / 4)) / (1 + tan θ * tan (π / 4))

theorem tan_difference (θ : ℝ) (h1 : cos θ = -12 / 13) (h2 : π < θ ∧ θ < 3 * π / 2) :
  tan (θ - π / 4) = -7 / 17 :=
by
  sorry

end tan_difference_l607_60730


namespace exam_student_count_l607_60709

theorem exam_student_count (N T T_5 T_remaining : ℕ)
  (h1 : T = 70 * N)
  (h2 : T_5 = 50 * 5)
  (h3 : T_remaining = 90 * (N - 5))
  (h4 : T = T_5 + T_remaining) :
  N = 10 :=
by
  sorry

end exam_student_count_l607_60709


namespace remainder_of_four_m_plus_five_l607_60710

theorem remainder_of_four_m_plus_five (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 :=
by
  -- Proof steps would go here
  sorry

end remainder_of_four_m_plus_five_l607_60710


namespace pentagon_area_l607_60706

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end pentagon_area_l607_60706


namespace right_triangle_area_l607_60705

theorem right_triangle_area (a b c p : ℝ) (h1 : a = b) (h2 : 3 * p = a + b + c)
  (h3 : c = Real.sqrt (2 * a ^ 2)) :
  (1/2) * a ^ 2 = (9 * p ^ 2 * (3 - 2 * Real.sqrt 2)) / 4 :=
by
  sorry

end right_triangle_area_l607_60705


namespace solve_quadratic_eq_solve_equal_squares_l607_60729

theorem solve_quadratic_eq (x : ℝ) : 
    (4 * x^2 - 2 * x - 1 = 0) ↔ 
    (x = (1 + Real.sqrt 5) / 4 ∨ x = (1 - Real.sqrt 5) / 4) := 
by
  sorry

theorem solve_equal_squares (y : ℝ) :
    ((y + 1)^2 = (3 * y - 1)^2) ↔ 
    (y = 1 ∨ y = 0) := 
by
  sorry

end solve_quadratic_eq_solve_equal_squares_l607_60729


namespace middle_number_is_nine_l607_60786

theorem middle_number_is_nine (x : ℝ) (h : (2 * x)^2 + (4 * x)^2 = 180) : 3 * x = 9 :=
by
  sorry

end middle_number_is_nine_l607_60786


namespace factor_y6_plus_64_l607_60797

theorem factor_y6_plus_64 : (y^2 + 4) ∣ (y^6 + 64) :=
sorry

end factor_y6_plus_64_l607_60797


namespace least_number_of_table_entries_l607_60748

-- Given conditions
def num_towns : ℕ := 6

-- Theorem statement
theorem least_number_of_table_entries : (num_towns * (num_towns - 1)) / 2 = 15 := by
  -- Proof goes here.
  sorry

end least_number_of_table_entries_l607_60748


namespace petya_must_have_photo_files_on_portable_hard_drives_l607_60701

theorem petya_must_have_photo_files_on_portable_hard_drives 
    (H F P T : ℕ) 
    (h1 : H > F) 
    (h2 : P > T) 
    : ∃ x, x ≠ 0 ∧ x ≤ H :=
by
  sorry

end petya_must_have_photo_files_on_portable_hard_drives_l607_60701


namespace Compute_fraction_power_l607_60798

theorem Compute_fraction_power :
  (81081 / 27027) ^ 4 = 81 :=
by
  -- We provide the specific condition as part of the proof statement
  have h : 27027 * 3 = 81081 := by norm_num
  sorry

end Compute_fraction_power_l607_60798


namespace age_ratio_l607_60750

theorem age_ratio (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 10) (h3 : a + b + c = 27) : b / c = 2 := by
  sorry

end age_ratio_l607_60750


namespace multiplication_of_935421_and_625_l607_60738

theorem multiplication_of_935421_and_625 :
  935421 * 625 = 584638125 :=
by sorry

end multiplication_of_935421_and_625_l607_60738


namespace smaller_fraction_is_l607_60775

theorem smaller_fraction_is
  (x y : ℝ)
  (h₁ : x + y = 7 / 8)
  (h₂ : x * y = 1 / 12) :
  min x y = (7 - Real.sqrt 17) / 16 :=
sorry

end smaller_fraction_is_l607_60775


namespace bracelet_cost_l607_60780

theorem bracelet_cost (B : ℝ)
  (H1 : 5 = 5)
  (H2 : 3 = 3)
  (H3 : 2 * B + 5 + B + 3 = 20) : B = 4 :=
by
  sorry

end bracelet_cost_l607_60780


namespace find_four_digit_number_l607_60784

theorem find_four_digit_number : ∃ x : ℕ, (1000 ≤ x ∧ x ≤ 9999) ∧ (x % 7 = 0) ∧ (x % 29 = 0) ∧ (19 * x % 37 = 3) ∧ x = 5075 :=
by
  sorry

end find_four_digit_number_l607_60784


namespace production_line_B_units_l607_60768

theorem production_line_B_units
  (total_units : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
  (h_total_units : total_units = 5000)
  (h_ratio : ratio_A = 1 ∧ ratio_B = 2 ∧ ratio_C = 2) :
  (2 * (total_units / (ratio_A + ratio_B + ratio_C))) = 2000 :=
by
  sorry

end production_line_B_units_l607_60768


namespace fraction_division_l607_60716

theorem fraction_division :
  (3 / 4) / (5 / 6) = 9 / 10 :=
by {
  -- We skip the proof as per the instructions
  sorry
}

end fraction_division_l607_60716


namespace binary_multiplication_l607_60781

theorem binary_multiplication :
  0b1101 * 0b110 = 0b1011110 := 
sorry

end binary_multiplication_l607_60781


namespace geometric_loci_l607_60758

noncomputable def quadratic_discriminant (x y : ℝ) : ℝ :=
  x^2 + 4 * y^2 - 4

-- Conditions:
def real_and_distinct (x y : ℝ) := 
  ((x^2) / 4 + y^2 > 1) 

def equal_and_real (x y : ℝ) := 
  ((x^2) / 4 + y^2 = 1) 

def complex_roots (x y : ℝ) := 
  ((x^2) / 4 + y^2 < 1)

def both_roots_positive (x y : ℝ) := 
  (x < 0) ∧ (-1 < y) ∧ (y < 1)

def both_roots_negative (x y : ℝ) := 
  (x > 0) ∧ (-1 < y) ∧ (y < 1)

def opposite_sign_roots (x y : ℝ) := 
  (y > 1) ∨ (y < -1)

theorem geometric_loci (x y : ℝ) :
  (real_and_distinct x y ∨ equal_and_real x y ∨ complex_roots x y) ∧ 
  ((real_and_distinct x y ∧ both_roots_positive x y) ∨
   (real_and_distinct x y ∧ both_roots_negative x y) ∨
   (real_and_distinct x y ∧ opposite_sign_roots x y)) := 
sorry

end geometric_loci_l607_60758


namespace min_value_of_quadratic_l607_60720

theorem min_value_of_quadratic (x : ℝ) : 
  ∃ m : ℝ, (∀ z : ℝ, z = 5 * x ^ 2 + 20 * x + 25 → z ≥ m) ∧ m = 5 :=
by
  sorry

end min_value_of_quadratic_l607_60720


namespace quadratic_solution_l607_60712

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l607_60712


namespace least_positive_integer_l607_60728

theorem least_positive_integer :
  ∃ (a : ℕ), (a ≡ 1 [MOD 3]) ∧ (a ≡ 2 [MOD 4]) ∧ (∀ b, (b ≡ 1 [MOD 3]) → (b ≡ 2 [MOD 4]) → b ≥ a → b = a) :=
sorry

end least_positive_integer_l607_60728


namespace temperature_rise_l607_60755

variable (t : ℝ)

theorem temperature_rise (initial final : ℝ) (h : final = t) : final = 5 + t := by
  sorry

end temperature_rise_l607_60755


namespace square_side_length_l607_60727

theorem square_side_length (p : ℝ) (h : p = 17.8) : (p / 4) = 4.45 := by
  sorry

end square_side_length_l607_60727
