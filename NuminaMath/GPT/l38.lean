import Mathlib

namespace base7_to_base10_l38_38592

open Nat

theorem base7_to_base10 : (3 * 7^2 + 5 * 7^1 + 1 * 7^0 = 183) :=
by
  sorry

end base7_to_base10_l38_38592


namespace domain_of_composed_function_l38_38797

theorem domain_of_composed_function
  (f : ℝ → ℝ)
  (dom_f : ∀ x, 0 ≤ x ∧ x ≤ 4 → f x ≠ 0) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → f (x^2) ≠ 0 :=
by
  sorry

end domain_of_composed_function_l38_38797


namespace real_number_a_pure_imaginary_l38_38780

-- Definition of an imaginary number
def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Given conditions and the proof problem statement
theorem real_number_a_pure_imaginary (a : ℝ) :
  pure_imaginary (⟨(a + 1) / 2, (1 - a) / 2⟩) → a = -1 :=
by
  sorry

end real_number_a_pure_imaginary_l38_38780


namespace rhombus_side_length_l38_38497

-- Definitions
def is_rhombus_perimeter (P s : ℝ) : Prop := P = 4 * s

-- Theorem to prove
theorem rhombus_side_length (P : ℝ) (hP : P = 4) : ∃ s : ℝ, is_rhombus_perimeter P s ∧ s = 1 :=
by
  sorry

end rhombus_side_length_l38_38497


namespace final_price_for_tiffany_l38_38853

noncomputable def calculate_final_price (n : ℕ) (c : ℝ) (d : ℝ) (s : ℝ) : ℝ :=
  let total_cost := n * c
  let discount := d * total_cost
  let discounted_price := total_cost - discount
  let sales_tax := s * discounted_price
  let final_price := discounted_price + sales_tax
  final_price

theorem final_price_for_tiffany :
  calculate_final_price 9 4.50 0.20 0.07 = 34.67 :=
by
  sorry

end final_price_for_tiffany_l38_38853


namespace probability_five_dice_same_l38_38270

-- Define a function that represents the probability problem
noncomputable def probability_all_dice_same : ℚ :=
  (1 / 6) * (1 / 6) * (1 / 6) * (1 / 6)

-- The main theorem to state the proof problem
theorem probability_five_dice_same : probability_all_dice_same = 1 / 1296 :=
by
  sorry

end probability_five_dice_same_l38_38270


namespace possible_side_values_l38_38602

noncomputable def valid_isosceles_triangle (x : ℝ) := 
  ∃ (a b c : ℝ), 
    a = real.sin x ∧ 
    b = real.sin x ∧ 
    c = real.sin (7 * x) ∧ 
    (a = b ∨ b = c ∨ c = a) ∧
    ∃ (theta : ℝ), theta = 3 * x ∧ theta < 90 ∧ theta > 0

theorem possible_side_values :
  ∀ (x : ℝ), valid_isosceles_triangle x ↔ x ∈ {10, 30, 50} :=
by
  sorry

end possible_side_values_l38_38602


namespace domain_of_f_l38_38856

def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ∈ set.Ioo (-∞) 6 ∪ set.Ioo 6 ∞) :=
by
  sorry

end domain_of_f_l38_38856


namespace probability_x_gt_3y_l38_38677

theorem probability_x_gt_3y :
  let width := 3000
  let height := 3001
  let triangle_area := (1 / 2 : ℚ) * width * (width / 3)
  let rectangle_area := (width : ℚ) * height
  triangle_area / rectangle_area = 1500 / 9003 :=
by 
  sorry

end probability_x_gt_3y_l38_38677


namespace john_total_cost_l38_38347

-- The total cost John incurs to rent a car, buy gas, and drive 320 miles
def total_cost (rental_cost gas_cost_per_gallon cost_per_mile miles driven_gallons : ℝ): ℝ :=
  rental_cost + (gas_cost_per_gallon * driven_gallons) + (cost_per_mile * miles)

theorem john_total_cost :
  let rental_cost := 150
  let gallons := 8
  let gas_cost_per_gallon := 3.50
  let cost_per_mile := 0.50
  let miles := 320
  total_cost rental_cost gas_cost_per_gallon cost_per_mile miles gallons = 338 := 
by
  -- The detailed proof is skipped here
  sorry

end john_total_cost_l38_38347


namespace pieces_to_same_point_l38_38264

theorem pieces_to_same_point :
  ∀ (x y z : ℤ), (∃ (final_pos : ℤ), (x = final_pos ∧ y = final_pos ∧ z = final_pos)) ↔ 
  (x, y, z) = (1, 2009, 2010) ∨ 
  (x, y, z) = (0, 2009, 2010) ∨ 
  (x, y, z) = (2, 2009, 2010) ∨ 
  (x, y, z) = (3, 2009, 2010) := 
by {
  sorry
}

end pieces_to_same_point_l38_38264


namespace sum_of_roots_quadratic_eq_l38_38837

theorem sum_of_roots_quadratic_eq : ∀ P Q : ℝ, (3 * P^2 - 9 * P + 6 = 0) ∧ (3 * Q^2 - 9 * Q + 6 = 0) → P + Q = 3 :=
by
  sorry

end sum_of_roots_quadratic_eq_l38_38837


namespace difference_of_squares_l38_38552

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end difference_of_squares_l38_38552


namespace is_not_age_of_child_l38_38359

-- Initial conditions
def mrs_smith_child_ages : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Given number
def n : Nat := 1124

-- Mrs. Smith's age 
noncomputable def mrs_smith_age : Nat := 46

-- Divisibility check
def is_divisible (n k : Nat) : Bool := n % k = 0

-- Prove the statement
theorem is_not_age_of_child (child_age : Nat) : 
  child_age ∈ mrs_smith_child_ages ∧ ¬ is_divisible n child_age → child_age = 3 :=
by
  intros h
  sorry

end is_not_age_of_child_l38_38359


namespace find_solutions_l38_38466

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l38_38466


namespace find_number_l38_38105

theorem find_number (x : ℝ) (h : 0.4 * x + 60 = x) : x = 100 :=
by
  sorry

end find_number_l38_38105


namespace base5_division_l38_38459

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end base5_division_l38_38459


namespace donald_laptop_cost_l38_38752

theorem donald_laptop_cost (original_price : ℕ) (reduction_percent : ℕ) (reduced_price : ℕ) (h1 : original_price = 800) (h2 : reduction_percent = 15) : reduced_price = 680 :=
by
  -- Definitions of the conditions
  have h3 : reduction_percent / 100 * original_price = 120 := sorry  -- Calculation of the discount (15/100)*800
  have h4 : original_price - 120 = 680 := sorry  -- Subtracting discount from original price
  -- Conclusion
  exact h4

end donald_laptop_cost_l38_38752


namespace lunch_combinations_l38_38409

theorem lunch_combinations :
  let num_meat := 4 in
  let num_veg := 7 in
  let choose := @nat.choose in
  (choose num_meat 2 * choose num_veg 2) + 
  (choose num_meat 1 * choose num_veg 2) = 210 :=
by
  sorry

end lunch_combinations_l38_38409


namespace cube_painted_surface_l38_38281

theorem cube_painted_surface (n : ℕ) (hn : n > 2) 
: 6 * (n - 2) ^ 2 = (n - 2) ^ 3 → n = 8 :=
by
  sorry

end cube_painted_surface_l38_38281


namespace problem1_problem2_problem3_problem4_l38_38739

-- Problem 1: Prove (1 * -6) + -13 = -19
theorem problem1 : (1 * -6) + -13 = -19 := by 
  sorry

-- Problem 2: Prove (3/5) + (-3/4) = -3/20
theorem problem2 : (3/5 : ℚ) + (-3/4) = -3/20 := by 
  sorry

-- Problem 3: Prove 4.7 + (-0.8) + 5.3 + (-8.2) = 1
theorem problem3 : (4.7 + (-0.8) + 5.3 + (-8.2) : ℝ) = 1 := by 
  sorry

-- Problem 4: Prove (-1/6) + (1/3) + (-1/12) = 1/12
theorem problem4 : (-1/6 : ℚ) + (1/3) + (-1/12) = 1/12 := by 
  sorry

end problem1_problem2_problem3_problem4_l38_38739


namespace secretary_longest_time_l38_38544

theorem secretary_longest_time (h_ratio : ∃ x : ℕ, ∃ y : ℕ, ∃ z : ℕ, y = 2 * x ∧ z = 3 * x ∧ (5 * x = 40)) :
  5 * x = 40 := sorry

end secretary_longest_time_l38_38544


namespace recurring_decimal_to_fraction_l38_38070

theorem recurring_decimal_to_fraction :
  let x := 0.4 + 67 / (99 : ℝ)
  (∀ y : ℝ, y = x ↔ y = 463 / 990) := 
by
  sorry

end recurring_decimal_to_fraction_l38_38070


namespace initial_number_is_correct_l38_38716

theorem initial_number_is_correct (x : ℝ) (h : 8 * x - 4 = 2.625) : x = 0.828125 :=
by
  sorry

end initial_number_is_correct_l38_38716


namespace number_of_sides_of_polygon_l38_38254

theorem number_of_sides_of_polygon (n : ℕ) (h : 3 * (n * (n - 3) / 2) - n = 21) : n = 6 :=
by sorry

end number_of_sides_of_polygon_l38_38254


namespace max_value_of_x2_y3_z_l38_38513

theorem max_value_of_x2_y3_z (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 1) : x^2 * y^3 * z ≤ 1 / 3888 :=
begin
  -- Proof omitted
  sorry
end

end max_value_of_x2_y3_z_l38_38513


namespace unique_sequence_l38_38461

/-- Define an infinite sequence of positive real numbers -/
def infinite_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n, 0 < X n

/-- Define the recurrence relation for the sequence -/
def recurrence_relation (X : ℕ → ℝ) : Prop :=
  ∀ n, X (n + 2) = (1 / 2) * (1 / X (n + 1) + X n)

/-- Prove that the only infinite sequence satisfying the recurrence relation is the constant sequence 1 -/
theorem unique_sequence (X : ℕ → ℝ) (h_seq : infinite_sequence X) (h_recur : recurrence_relation X) :
  ∀ n, X n = 1 :=
by
  sorry

end unique_sequence_l38_38461


namespace rate_of_stream_l38_38405

theorem rate_of_stream (x : ℝ) (h1 : ∀ (distance : ℝ), (24 : ℝ) > 0) (h2 : ∀ (distance : ℝ), (distance / (24 - x)) = 3 * (distance / (24 + x))) : x = 12 :=
by
  sorry

end rate_of_stream_l38_38405


namespace who_wears_which_dress_l38_38138

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l38_38138


namespace cafeteria_extra_fruits_l38_38850

def extra_fruits (ordered wanted : Nat) : Nat :=
  ordered - wanted

theorem cafeteria_extra_fruits :
  let red_apples_ordered := 6
  let red_apples_wanted := 5
  let green_apples_ordered := 15
  let green_apples_wanted := 8
  let oranges_ordered := 10
  let oranges_wanted := 6
  let bananas_ordered := 8
  let bananas_wanted := 7
  extra_fruits red_apples_ordered red_apples_wanted = 1 ∧
  extra_fruits green_apples_ordered green_apples_wanted = 7 ∧
  extra_fruits oranges_ordered oranges_wanted = 4 ∧
  extra_fruits bananas_ordered bananas_wanted = 1 := 
by
  sorry

end cafeteria_extra_fruits_l38_38850


namespace problem_g_eq_l38_38839

noncomputable def g : ℝ → ℝ := sorry

theorem problem_g_eq :
  (∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x + x) →
  g 3 = ( -31 - 3 * 3^(1/3)) / 8 :=
by
  intro h
  -- proof goes here
  sorry

end problem_g_eq_l38_38839


namespace miles_tankful_highway_l38_38108

variable (miles_tankful_city : ℕ)
variable (mpg_city : ℕ)
variable (mpg_highway : ℕ)

-- Relationship between miles per gallon in city and highway
axiom h_mpg_relation : mpg_highway = mpg_city + 18

-- Given the car travels 336 miles per tankful of gasoline in the city
axiom h_miles_tankful_city : miles_tankful_city = 336

-- Given the car travels 48 miles per gallon in the city
axiom h_mpg_city : mpg_city = 48

-- Prove the car travels 462 miles per tankful of gasoline on the highway
theorem miles_tankful_highway : ∃ (miles_tankful_highway : ℕ), miles_tankful_highway = (mpg_highway * (miles_tankful_city / mpg_city)) := 
by 
  exists (66 * (336 / 48)) -- Since 48 + 18 = 66 and 336 / 48 = 7, 66 * 7 = 462
  sorry

end miles_tankful_highway_l38_38108


namespace hannah_total_savings_l38_38916

theorem hannah_total_savings :
  let a1 := 4
  let a2 := 2 * a1
  let a3 := 2 * a2
  let a4 := 2 * a3
  let a5 := 20
  a1 + a2 + a3 + a4 + a5 = 80 :=
by
  sorry

end hannah_total_savings_l38_38916


namespace real_part_of_complex_pow_l38_38313

open Complex

theorem real_part_of_complex_pow (a b : ℝ) : a = 1 → b = -2 → (realPart ((a : ℂ) + (b : ℂ) * Complex.I)^5) = 41 :=
by
  sorry

end real_part_of_complex_pow_l38_38313


namespace fraction_spent_at_toy_store_l38_38014

theorem fraction_spent_at_toy_store 
  (total_allowance : ℝ)
  (arcade_fraction : ℝ)
  (candy_store_amount : ℝ) 
  (remaining_allowance : ℝ)
  (toy_store_amount : ℝ)
  (H1 : total_allowance = 2.40)
  (H2 : arcade_fraction = 3 / 5)
  (H3 : candy_store_amount = 0.64)
  (H4 : remaining_allowance = total_allowance - (arcade_fraction * total_allowance))
  (H5 : toy_store_amount = remaining_allowance - candy_store_amount) :
  toy_store_amount / remaining_allowance = 1 / 3 := 
sorry

end fraction_spent_at_toy_store_l38_38014


namespace inequality_proof_l38_38044

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d):
    1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) :=
by
  sorry

end inequality_proof_l38_38044


namespace gross_pay_calculation_l38_38185

theorem gross_pay_calculation
    (NetPay : ℕ) (Taxes : ℕ) (GrossPay : ℕ) 
    (h1 : NetPay = 315) 
    (h2 : Taxes = 135) 
    (h3 : GrossPay = NetPay + Taxes) : 
    GrossPay = 450 :=
by
    -- We need to prove this part
    sorry

end gross_pay_calculation_l38_38185


namespace pencils_bought_l38_38672

theorem pencils_bought (total_spent notebook_cost ruler_cost pencil_cost : ℕ)
  (h_total : total_spent = 74)
  (h_notebook : notebook_cost = 35)
  (h_ruler : ruler_cost = 18)
  (h_pencil : pencil_cost = 7) :
  (total_spent - (notebook_cost + ruler_cost)) / pencil_cost = 3 :=
by
  sorry

end pencils_bought_l38_38672


namespace num_regions_of_lines_l38_38657

theorem num_regions_of_lines (R : ℕ → ℕ) :
  R 1 = 2 ∧ 
  (∀ n, R (n + 1) = R n + (n + 1)) →
  (∀ n, R n = (n * (n + 1)) / 2 + 1) :=
by
  intro h
  sorry

end num_regions_of_lines_l38_38657


namespace correct_yeast_population_change_statement_l38_38597

def yeast_produces_CO2 (aerobic : Bool) : Bool := 
  True

def yeast_unicellular_fungus : Bool := 
  True

def boiling_glucose_solution_purpose : Bool := 
  True

def yeast_facultative_anaerobe : Bool := 
  True

theorem correct_yeast_population_change_statement : 
  (∀ (aerobic : Bool), yeast_produces_CO2 aerobic) →
  yeast_unicellular_fungus →
  boiling_glucose_solution_purpose →
  yeast_facultative_anaerobe →
  "D is correct" = "D is correct" :=
by
  intros
  exact rfl

end correct_yeast_population_change_statement_l38_38597


namespace find_prime_pairs_l38_38308

def is_prime (n : ℕ) := n ≥ 2 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def has_prime_root (m n : ℕ) : Prop :=
  ∃ (p: ℕ), is_prime p ∧ (p * p - m * p - n = 0)

theorem find_prime_pairs :
  ∀ (m n : ℕ), (is_prime m ∧ is_prime n) → has_prime_root m n → (m, n) = (2, 3) :=
by sorry

end find_prime_pairs_l38_38308


namespace f_increasing_interval_l38_38546

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end f_increasing_interval_l38_38546


namespace simplify_expression_l38_38078

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 - 11 * x + 13 * x^2 - 15 + 17 * x + 19 * x^2 = 25 * x^2 + x - 3 := 
by
  sorry

end simplify_expression_l38_38078


namespace pen_count_l38_38049

-- Define the conditions
def total_pens := 140
def difference := 20

-- Define the quantities to be proven
def ballpoint_pens := (total_pens - difference) / 2
def fountain_pens := total_pens - ballpoint_pens

-- The theorem to be proved
theorem pen_count :
  ballpoint_pens = 60 ∧ fountain_pens = 80 :=
by
  -- Proof omitted
  sorry

end pen_count_l38_38049


namespace hannah_highest_score_l38_38789

-- Definitions based on conditions
def total_questions : ℕ := 40
def wrong_questions : ℕ := 3
def correct_percent_student_1 : ℝ := 0.95

-- The Lean statement representing the proof problem
theorem hannah_highest_score :
  ∃ q : ℕ, (q > (total_questions - wrong_questions) ∧ q > (total_questions * correct_percent_student_1)) ∧ q = 39 :=
by
  sorry

end hannah_highest_score_l38_38789


namespace mean_yoga_practice_days_l38_38526

noncomputable def mean_number_of_days (counts : List ℕ) (days : List ℕ) : ℚ :=
  let total_days := List.zipWith (λ c d => c * d) counts days |>.sum
  let total_students := counts.sum
  total_days / total_students

def counts : List ℕ := [2, 4, 5, 3, 2, 1, 3]
def days : List ℕ := [1, 2, 3, 4, 5, 6, 7]

theorem mean_yoga_practice_days : mean_number_of_days counts days = 37 / 10 := 
by 
  sorry

end mean_yoga_practice_days_l38_38526


namespace lloyd_excess_rate_multiple_l38_38521

theorem lloyd_excess_rate_multiple :
  let h_regular := 7.5
  let r := 4.00
  let h_total := 10.5
  let e_total := 48
  let e_regular := h_regular * r
  let excess_hours := h_total - h_regular
  let e_excess := e_total - e_regular
  let m := e_excess / (excess_hours * r)
  m = 1.5 :=
by
  sorry

end lloyd_excess_rate_multiple_l38_38521


namespace triangle_with_positive_area_l38_38042

noncomputable def num_triangles_with_A (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) : ℕ :=
  let points_excluding_A := total_points.erase A
  let total_pairs := points_excluding_A.card.choose 2
  let collinear_pairs := 20  -- Derived from the problem; in practice this would be calculated
  total_pairs - collinear_pairs

theorem triangle_with_positive_area (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) (h : total_points.card = 25):
  num_triangles_with_A total_points A = 256 :=
by
  sorry

end triangle_with_positive_area_l38_38042


namespace range_of_k_l38_38705

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → (-2 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l38_38705


namespace kim_initial_classes_l38_38805

-- Necessary definitions for the problem
def hours_per_class := 2
def total_hours_after_dropping := 6
def classes_after_dropping := total_hours_after_dropping / hours_per_class
def initial_classes := classes_after_dropping + 1

theorem kim_initial_classes : initial_classes = 4 :=
by
  -- Proof will be derived here
  sorry

end kim_initial_classes_l38_38805


namespace part1_l38_38277

theorem part1 (n : ℕ) (m : ℕ) (h_form : m = 2 ^ (n - 2) * 5 ^ n) (h : 6 * 10 ^ n + m = 25 * m) :
  ∃ k : ℕ, 6 * 10 ^ n + m = 625 * 10 ^ (n - 2) :=
by
  sorry

end part1_l38_38277


namespace no_odd_total_rows_columns_l38_38475

open Function

def array_odd_column_row_count (n : ℕ) (array : ℕ → ℕ → ℤ) : Prop :=
  n % 2 = 1 ∧
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ array i j = -1 ∨ array i j = 1) →
  (∃ (rows cols : Finset ℕ),
    rows.card + cols.card = n ∧
    ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r) k = -1 ∧
    ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c) k = -1
    )

theorem no_odd_total_rows_columns (n : ℕ) (array : ℕ → ℕ → ℤ) :
  n % 2 = 1 →
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ (array i j = -1 ∨ array i j = 1)) →
  ¬ (∃ rows cols : Finset ℕ,
       rows.card + cols.card = n ∧
       ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r k = -1) ∧
       ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c k = -1)) :=
by
  intros h_array
  sorry

end no_odd_total_rows_columns_l38_38475


namespace find_sum_of_a_b_c_l38_38543

theorem find_sum_of_a_b_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(h4 : (a + b + c) ^ 3 - a ^ 3 - b ^ 3 - c ^ 3 = 210) : a + b + c = 11 :=
sorry

end find_sum_of_a_b_c_l38_38543


namespace min_value_exists_l38_38776

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9 ∧ y ≥ 2

theorem min_value_exists : ∃ x y : ℝ, point_on_circle x y ∧ x + Real.sqrt 3 * y = 2 * Real.sqrt 3 - 2 := 
sorry

end min_value_exists_l38_38776


namespace number_of_strikers_l38_38725

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l38_38725


namespace linear_function_point_l38_38321

theorem linear_function_point (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 :=
by
  sorry

end linear_function_point_l38_38321


namespace dress_assignment_l38_38144

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l38_38144


namespace number_of_hens_l38_38578

-- Conditions as Lean definitions
def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 136

-- Mathematically equivalent proof problem
theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 28 :=
by
  sorry

end number_of_hens_l38_38578


namespace sara_schavenger_hunt_l38_38734

theorem sara_schavenger_hunt :
  let monday := 1 -- Sara rearranges the books herself
  let tuesday := 2 -- Sara can choose from Liam or Mia
  let wednesday := 4 -- There are 4 classmates
  let thursday := 3 -- There are 3 new volunteers
  let friday := 1 -- Sara and Zoe do it together
  monday * tuesday * wednesday * thursday * friday = 24 :=
by
  sorry

end sara_schavenger_hunt_l38_38734


namespace find_angle_A_l38_38936

theorem find_angle_A (a b : ℝ) (B A : ℝ)
  (h1 : a = 2) 
  (h2 : b = Real.sqrt 3) 
  (h3 : B = Real.pi / 3) : 
  A = Real.pi / 2 := 
sorry

end find_angle_A_l38_38936


namespace solve_quadratic_eq_l38_38959

theorem solve_quadratic_eq : ∃ (a b : ℕ), a = 145 ∧ b = 7 ∧ a + b = 152 ∧ 
  ∀ x, x = Real.sqrt a - b → x^2 + 14 * x = 96 :=
by 
  use 145, 7
  simp
  sorry

end solve_quadratic_eq_l38_38959


namespace product_sin_eq_one_eighth_l38_38887

theorem product_sin_eq_one_eighth (h1 : Real.sin (3 * Real.pi / 8) = Real.cos (Real.pi / 8))
                                  (h2 : Real.sin (Real.pi / 8) = Real.cos (3 * Real.pi / 8)) :
  ((1 - Real.sin (Real.pi / 8)) * (1 - Real.sin (3 * Real.pi / 8)) * 
   (1 + Real.sin (Real.pi / 8)) * (1 + Real.sin (3 * Real.pi / 8)) = 1 / 8) :=
by {
  sorry
}

end product_sin_eq_one_eighth_l38_38887


namespace no_nat_numbers_satisfy_eqn_l38_38045

theorem no_nat_numbers_satisfy_eqn (a b : ℕ) : a^2 - 3 * b^2 ≠ 8 := by
  sorry

end no_nat_numbers_satisfy_eqn_l38_38045


namespace fraction_equivalent_to_0_46_periodic_l38_38758

theorem fraction_equivalent_to_0_46_periodic :
  let a := (46 : ℚ) / 100
  let r := (1 : ℚ) / 100
  let geometric_series_sum (a r : ℚ) :=
    if r.abs < 1 then a / (1 - r) else 0
  geometric_series_sum a r = 46 / 99 := by
    sorry

end fraction_equivalent_to_0_46_periodic_l38_38758


namespace solve_x4_minus_16_eq_0_l38_38463

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l38_38463


namespace correct_truth_values_l38_38706

open Real

def proposition_p : Prop := ∀ (a : ℝ), 0 < a → a^2 ≠ 0

def converse_p : Prop := ∀ (a : ℝ), a^2 ≠ 0 → 0 < a

def inverse_p : Prop := ∀ (a : ℝ), ¬(0 < a) → a^2 = 0

def contrapositive_p : Prop := ∀ (a : ℝ), a^2 = 0 → ¬(0 < a)

def negation_p : Prop := ∃ (a : ℝ), 0 < a ∧ a^2 = 0

theorem correct_truth_values : 
  (converse_p = False) ∧ 
  (inverse_p = False) ∧ 
  (contrapositive_p = True) ∧ 
  (negation_p = False) := by
  sorry

end correct_truth_values_l38_38706


namespace problem_equivalent_l38_38220

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l38_38220


namespace area_triangle_QCA_l38_38301

noncomputable def area_of_triangle_QCA (p : ℝ) : ℝ :=
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let QA := 3
  let QC := 12 - p
  (1/2) * QA * QC

theorem area_triangle_QCA (p : ℝ) : area_of_triangle_QCA p = (3/2) * (12 - p) :=
  sorry

end area_triangle_QCA_l38_38301


namespace percent_of_men_tenured_l38_38708

theorem percent_of_men_tenured (total_professors : ℕ) (women_percent tenured_percent women_tenured_or_both_percent men_percent tenured_men_percent : ℝ)
  (h1 : women_percent = 70 / 100)
  (h2 : tenured_percent = 70 / 100)
  (h3 : women_tenured_or_both_percent = 90 / 100)
  (h4 : men_percent = 30 / 100)
  (h5 : total_professors > 0)
  (h6 : tenured_men_percent = (2/3)) :
  tenured_men_percent * 100 = 66.67 :=
by sorry

end percent_of_men_tenured_l38_38708


namespace system_of_equations_solve_l38_38910

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end system_of_equations_solve_l38_38910


namespace frac_inequality_l38_38159

theorem frac_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : (a/c) > (b/d) := 
sorry

end frac_inequality_l38_38159


namespace butter_mixture_price_l38_38969

theorem butter_mixture_price :
  let cost1 := 48 * 150
  let cost2 := 36 * 125
  let cost3 := 24 * 100
  let revenue1 := cost1 + cost1 * (20 / 100)
  let revenue2 := cost2 + cost2 * (30 / 100)
  let revenue3 := cost3 + cost3 * (50 / 100)
  let total_weight := 48 + 36 + 24
  (revenue1 + revenue2 + revenue3) / total_weight = 167.5 :=
by
  sorry

end butter_mixture_price_l38_38969


namespace find_points_A_C_find_equation_line_l_l38_38027

variables (A B C : ℝ × ℝ)
variables (l : ℝ → ℝ)

-- Condition: the coordinates of point B are (2, 1)
def B_coord : Prop := B = (2, 1)

-- Condition: the equation of the line containing the altitude on side BC is x - 2y - 1 = 0
def altitude_BC (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Condition: the equation of the angle bisector of angle A is y = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0

-- Statement of the theorems to be proved
theorem find_points_A_C
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0) :
  (A = (1, 0)) ∧ (C = (4, -3)) :=
sorry

theorem find_equation_line_l
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0)
    (hA : A = (1, 0)) :
  ((∀ x : ℝ, l x = x - 1)) :=
sorry

end find_points_A_C_find_equation_line_l_l38_38027


namespace second_solution_volume_l38_38103

theorem second_solution_volume
  (V : ℝ)
  (h1 : 0.20 * 6 + 0.60 * V = 0.36 * (6 + V)) : 
  V = 4 :=
sorry

end second_solution_volume_l38_38103


namespace pebbles_collected_by_tenth_day_l38_38360

-- Define the initial conditions
def a : ℕ := 2
def r : ℕ := 2
def n : ℕ := 10

-- Total pebbles collected by the end of the 10th day
def total_pebbles (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Proof statement
theorem pebbles_collected_by_tenth_day : total_pebbles a r n = 2046 :=
  by sorry

end pebbles_collected_by_tenth_day_l38_38360


namespace contact_probability_l38_38762

variable (m : ℕ := 6) (n : ℕ := 7) (p : ℝ)

theorem contact_probability :
  let total_pairs := m * n in
  let probability_no_contact := (1 - p) ^ total_pairs in
  let probability_contact := 1 - probability_no_contact in
  probability_contact = 1 - (1 - p) ^ 42 :=
by
  -- This is where the proof would go
  sorry

end contact_probability_l38_38762


namespace solution_set_of_equation_l38_38309

theorem solution_set_of_equation :
  {p : ℝ × ℝ | p.1 * p.2 + 1 = p.1 + p.2} = {p : ℝ × ℝ | p.1 = 1 ∨ p.2 = 1} :=
by 
  sorry

end solution_set_of_equation_l38_38309


namespace find_n_l38_38978

-- Defining necessary conditions and declarations
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sumOfDigits (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def productOfDigits (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem find_n (n : ℕ) (s : ℕ) (p : ℕ) 
  (h1 : isThreeDigit n) 
  (h2 : isPerfectSquare n) 
  (h3 : sumOfDigits n = s) 
  (h4 : productOfDigits n = p) 
  (h5 : 10 ≤ s ∧ s < 100)
  (h6 : ∀ m : ℕ, isThreeDigit m → isPerfectSquare m → sumOfDigits m = s → productOfDigits m = p → (m = n → false))
  (h7 : ∃ m : ℕ, isThreeDigit m ∧ isPerfectSquare m ∧ sumOfDigits m = s ∧ productOfDigits m = p ∧ (∃ k : ℕ, k ≠ m → true)) :
  n = 841 :=
sorry

end find_n_l38_38978


namespace sara_total_spent_l38_38245

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end sara_total_spent_l38_38245


namespace percentage_of_500_l38_38712

theorem percentage_of_500 (P : ℝ) : 0.1 * (500 * P / 100) = 25 → P = 50 :=
by
  sorry

end percentage_of_500_l38_38712


namespace seq_1964_l38_38474

theorem seq_1964 (a : ℕ → ℤ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = -1)
  (h4 : ∀ n ≥ 4, a n = a (n - 1) * a (n - 3)) :
  a 1964 = -1 :=
by {
  sorry
}

end seq_1964_l38_38474


namespace point_on_line_l38_38779

theorem point_on_line (x : ℝ) : 
    (∃ k : ℝ, (-4) = k * (-4) + 8) → 
    (-4 = 2 * x + 8) → 
    x = -6 := 
sorry

end point_on_line_l38_38779


namespace part1_part2_l38_38903

def is_regressive_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

theorem part1 (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = 3 ^ n) :
  ¬ is_regressive_sequence a := by
  sorry

theorem part2 (b : ℕ → ℝ) (h_reg : is_regressive_sequence b) (h_inc : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d := by
  sorry

end part1_part2_l38_38903


namespace Jerry_has_36_stickers_l38_38030

variable (FredStickers GeorgeStickers JerryStickers CarlaStickers : ℕ)
variable (h1 : FredStickers = 18)
variable (h2 : GeorgeStickers = FredStickers - 6)
variable (h3 : JerryStickers = 3 * GeorgeStickers)
variable (h4 : CarlaStickers = JerryStickers + JerryStickers / 4)
variable (h5 : GeorgeStickers + FredStickers = CarlaStickers ^ 2)

theorem Jerry_has_36_stickers : JerryStickers = 36 := by
  sorry

end Jerry_has_36_stickers_l38_38030


namespace dress_assignments_l38_38136

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l38_38136


namespace mean_of_integers_neg3_to_6_l38_38074

theorem mean_of_integers_neg3_to_6 : 
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ) in
  let n := (6 - (-3) + 1 : ℤ) in
  s / n = 1.5 :=
by
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ)
  let n := (6 - (-3) + 1 : ℤ)
  simp
  sorry

end mean_of_integers_neg3_to_6_l38_38074


namespace problem_statement_l38_38001

theorem problem_statement (a b : ℝ) (C : ℝ) (sin_C : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_C = (Real.sqrt 15) / 4) :
  Real.cos C = 1 / 4 :=
sorry

end problem_statement_l38_38001


namespace sqrt_64_eq_8_l38_38427

theorem sqrt_64_eq_8 : Real.sqrt 64 = 8 := 
by
  sorry

end sqrt_64_eq_8_l38_38427


namespace find_value_of_a_l38_38230

def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_value_of_a (a : ℝ) :
  pure_imaginary ((a^3 - a) + (a / (1 - a)) * Complex.I) ↔ a = -1 := 
sorry

end find_value_of_a_l38_38230


namespace hexagon_perimeter_l38_38843

def side_length : ℝ := 4
def number_of_sides : ℕ := 6

theorem hexagon_perimeter :
  6 * side_length = 24 := by
    sorry

end hexagon_perimeter_l38_38843


namespace history_book_cost_is_correct_l38_38069

-- Define the conditions
def total_books : ℕ := 80
def math_book_cost : ℕ := 4
def total_price : ℕ := 390
def math_books_purchased : ℕ := 10

-- The number of history books
def history_books_purchased : ℕ := total_books - math_books_purchased

-- The total cost of math books
def total_cost_math_books : ℕ := math_books_purchased * math_book_cost

-- The total cost of history books
def total_cost_history_books : ℕ := total_price - total_cost_math_books

-- Define the cost of each history book
def history_book_cost : ℕ := total_cost_history_books / history_books_purchased

-- The theorem to be proven
theorem history_book_cost_is_correct : history_book_cost = 5 := 
by
  sorry

end history_book_cost_is_correct_l38_38069


namespace imaginary_part_z1_mul_z2_l38_38489

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end imaginary_part_z1_mul_z2_l38_38489


namespace barley_percentage_is_80_l38_38825

variables (T C : ℝ) -- Total land and cleared land
variables (B : ℝ) -- Percentage of cleared land planted with barley

-- Given conditions
def cleared_land (T : ℝ) : ℝ := 0.9 * T
def total_land_approx : ℝ := 1000
def potato_land (C : ℝ) : ℝ := 0.1 * C
def tomato_land : ℝ := 90
def barley_percentage (C : ℝ) (B : ℝ) : Prop := C - (potato_land C) - tomato_land = (B / 100) * C

-- Theorem statement to prove
theorem barley_percentage_is_80 :
  cleared_land total_land_approx = 900 → barley_percentage 900 80 :=
by
  intros hC
  rw [cleared_land, total_land_approx] at hC
  simp [barley_percentage, potato_land, tomato_land]
  sorry

end barley_percentage_is_80_l38_38825


namespace rhombus_area_in_rectangle_l38_38451

theorem rhombus_area_in_rectangle :
  ∀ (l w : ℝ), 
  (∀ (A B C D : ℝ), 
    (2 * w = l) ∧ 
    (l * w = 72) →
    let diag1 := w 
    let diag2 := l 
    (1/2 * diag1 * diag2 = 36)) :=
by
  intros
  sorry

end rhombus_area_in_rectangle_l38_38451


namespace sectionBSeats_l38_38659

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end sectionBSeats_l38_38659


namespace fraction_sum_is_ten_l38_38571

theorem fraction_sum_is_ten :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (55 / 10) = 10 :=
by
  sorry

end fraction_sum_is_ten_l38_38571


namespace divide_set_into_disjoint_subsets_l38_38919

theorem divide_set_into_disjoint_subsets {α : Type} [Fintype α] (h : Fintype.card α = 6) :
  (∃ (A B C : Finset α), A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
                         A ∪ B ∪ C = Finset.univ ∧ 
                         (A.card = 2 ∧ B.card = 2 ∧ C.card = 2)) →
  fintype.card (quotient (λ (P1 P2 : {s // s.card = 2} × {s // s.card = 2} × {s // s.card = 2}), 
    ∃ σ : Sym (Fin 3), let ⟨⟨a1, a2⟩, a3⟩ := P1, ⟨⟨b1, b2⟩, b3⟩ := P2 in 
    (a1 = b1 ∘ σ) ∧ (a2 = b2 ∘ σ) ∧ (a3 = b3 ∘ σ))) = 15 :=
by
  sorry

end divide_set_into_disjoint_subsets_l38_38919


namespace avg_speed_ratio_l38_38373

theorem avg_speed_ratio 
  (dist_tractor : ℝ) (time_tractor : ℝ) 
  (dist_car : ℝ) (time_car : ℝ) 
  (speed_factor : ℝ) :
  dist_tractor = 575 -> 
  time_tractor = 23 ->
  dist_car = 450 ->
  time_car = 5 ->
  speed_factor = 2 ->

  (dist_car / time_car) / (speed_factor * (dist_tractor / time_tractor)) = 9/5 := 
by
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  sorry

end avg_speed_ratio_l38_38373


namespace cos_315_eq_sqrt2_div_2_l38_38429

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l38_38429


namespace find_n_given_sum_l38_38902

noncomputable def geometric_sequence_general_term (n : ℕ) : ℝ :=
  if n ≥ 2 then 2^(2 * n - 3) else 0

def b_n (n : ℕ) : ℝ :=
  2 * n - 3

def sum_b_n (n : ℕ) : ℝ :=
  n^2 - 2 * n

theorem find_n_given_sum : ∃ n : ℕ, sum_b_n n = 360 :=
  by { use 20, sorry }

end find_n_given_sum_l38_38902


namespace pentagon_angles_l38_38504

theorem pentagon_angles (M T H A S : ℝ) 
  (h1 : M = T) 
  (h2 : T = H) 
  (h3 : A + S = 180) 
  (h4 : M + A + T + H + S = 540) : 
  H = 120 := 
by 
  -- The proof would be inserted here.
  sorry

end pentagon_angles_l38_38504


namespace cos_315_is_sqrt2_div_2_l38_38444

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l38_38444


namespace triangle_centers_exist_l38_38608

structure Triangle (α : Type _) [OrderedCommSemiring α] :=
(A B C : α × α)

noncomputable def circumcenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def incenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def excenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def centroid {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

theorem triangle_centers_exist {α : Type _} [OrderedCommSemiring α] (T : Triangle α) :
  ∃ K O Oc S : α × α, K = circumcenter T ∧ O = incenter T ∧ Oc = excenter T ∧ S = centroid T :=
by
  refine ⟨circumcenter T, incenter T, excenter T, centroid T, ⟨rfl, rfl, rfl, rfl⟩⟩

end triangle_centers_exist_l38_38608


namespace rectangle_length_l38_38867

theorem rectangle_length (w l : ℝ) (hP : (2 * l + 2 * w) / w = 5) (hA : l * w = 150) : l = 15 :=
by
  sorry

end rectangle_length_l38_38867


namespace domain_of_function_l38_38057

theorem domain_of_function :
  {x : ℝ | 2^x - 8 ≥ 0} = set.Ici 3 :=
begin
  ext x,
  simp [set.Ici, ge_iff_le],
  have h : 0 < 2 := by linarith,
  split,
  { intro h1,
    rw [← log_le_log_iff (pow_pos h x) (by norm_num)] at h1,
    rw [log_pow h] at h1,
    linarith, },
  { intro h2,
    rw [← log_le_log_iff (pow_pos h x) (by norm_num)],
    rw [log_pow h],
    linarith, }
end

end domain_of_function_l38_38057


namespace total_ways_to_choose_president_and_vice_president_of_same_gender_l38_38531

theorem total_ways_to_choose_president_and_vice_president_of_same_gender :
  let boys := 12
  let girls := 12
  (boys * (boys - 1) + girls * (girls - 1)) = 264 :=
by
  sorry

end total_ways_to_choose_president_and_vice_president_of_same_gender_l38_38531


namespace final_selling_price_l38_38288

def actual_price : ℝ := 9356.725146198829
def price_after_first_discount (P : ℝ) : ℝ := P * 0.80
def price_after_second_discount (P1 : ℝ) : ℝ := P1 * 0.90
def price_after_third_discount (P2 : ℝ) : ℝ := P2 * 0.95

theorem final_selling_price :
  (price_after_third_discount (price_after_second_discount (price_after_first_discount actual_price))) = 6400 :=
by 
  -- Here we would need to provide the proof, but it is skipped with sorry
  sorry

end final_selling_price_l38_38288


namespace who_wears_which_dress_l38_38150

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l38_38150


namespace exists_odd_k_l_m_l38_38568

def odd_nat (n : ℕ) : Prop := n % 2 = 1

theorem exists_odd_k_l_m : 
  ∃ (k l m : ℕ), 
  odd_nat k ∧ odd_nat l ∧ odd_nat m ∧ 
  (k ≠ 0) ∧ (l ≠ 0) ∧ (m ≠ 0) ∧ 
  (1991 * (l * m + k * m + k * l) = k * l * m) :=
by
  sorry

end exists_odd_k_l_m_l38_38568


namespace solution_system_l38_38179

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end solution_system_l38_38179


namespace halfway_between_ratios_l38_38547

theorem halfway_between_ratios :
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by
  sorry

end halfway_between_ratios_l38_38547


namespace jenny_chocolate_squares_l38_38195

theorem jenny_chocolate_squares (mike_chocolates : ℕ) (jenny_chocolates : ℕ) 
  (h_mike : mike_chocolates = 20) 
  (h_jenny : jenny_chocolates = 3 * mike_chocolates + 5) :
  jenny_chocolates = 65 :=
by
  sorry

end jenny_chocolate_squares_l38_38195


namespace find_constants_l38_38804

open BigOperators

theorem find_constants (a b c : ℕ) :
  (∀ n : ℕ, n > 0 → (∑ k in Finset.range n, k.succ * (k.succ + 1) ^ 2) = (n * (n + 1) * (a * n^2 + b * n + c)) / 12) →
  (a = 3 ∧ b = 11 ∧ c = 10) :=
by
  sorry

end find_constants_l38_38804


namespace finite_set_elements_at_least_half_m_l38_38950

theorem finite_set_elements_at_least_half_m (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ) 
  (hm : 2 ≤ m) 
  (hB : ∀ k : ℕ, 1 ≤ k → k ≤ m → (B k).sum id = (m : ℤ) ^ k) : 
  ∃ n : ℕ, (A.card ≥ n) ∧ (n ≥ m / 2) :=
by
  sorry

end finite_set_elements_at_least_half_m_l38_38950


namespace upstream_distance_calc_l38_38579

noncomputable def speed_in_still_water : ℝ := 10.5
noncomputable def downstream_distance : ℝ := 45
noncomputable def downstream_time : ℝ := 3
noncomputable def upstream_time : ℝ := 3

theorem upstream_distance_calc : 
  ∃ (d v : ℝ), (10.5 + v) * downstream_time = downstream_distance ∧ 
               v = 4.5 ∧ 
               d = (10.5 - v) * upstream_time ∧ 
               d = 18 :=
by
  sorry

end upstream_distance_calc_l38_38579


namespace frequency_calculation_l38_38583

-- Define the given conditions
def sample_capacity : ℕ := 20
def group_frequency : ℚ := 0.25

-- The main theorem statement
theorem frequency_calculation :
  sample_capacity * group_frequency = 5 :=
by sorry

end frequency_calculation_l38_38583


namespace dress_assignment_l38_38141

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l38_38141


namespace alex_integer_list_count_l38_38732

theorem alex_integer_list_count : 
  let n := 12 
  let least_multiple := 2^6 * 3^3
  let count := least_multiple / n
  count = 144 :=
by
  sorry

end alex_integer_list_count_l38_38732


namespace different_total_scores_l38_38107

noncomputable def basket_scores (x y z : ℕ) : ℕ := x + 2 * y + 3 * z

def total_baskets := 7
def score_range := {n | 7 ≤ n ∧ n ≤ 21}

theorem different_total_scores : 
  ∃ (count : ℕ), count = 15 ∧ 
  ∀ n ∈ score_range, ∃ (x y z : ℕ), x + y + z = total_baskets ∧ basket_scores x y z = n :=
sorry

end different_total_scores_l38_38107


namespace mouse_jump_frog_jump_diff_l38_38841

open Nat

theorem mouse_jump_frog_jump_diff :
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  mouse_jump - frog_jump = 20 :=
by
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  have h1 : frog_jump = 29 := by decide
  have h2 : mouse_jump = 49 := by decide
  have h3 : mouse_jump - frog_jump = 20 := by decide
  exact h3

end mouse_jump_frog_jump_diff_l38_38841


namespace hyperbola_center_l38_38310

theorem hyperbola_center :
  ∃ (h : ℝ × ℝ), h = (9 / 2, 2) ∧
  (∃ (x y : ℝ), 9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0) :=
  sorry

end hyperbola_center_l38_38310


namespace sin_of_angle_l38_38631

theorem sin_of_angle (α : ℝ) (h : Real.cos (π + α) = -(1/3)) : Real.sin ((3 * π / 2) - α) = -(1/3) := 
by
  sorry

end sin_of_angle_l38_38631


namespace quadrilateral_type_l38_38167

theorem quadrilateral_type (m n p q : ℝ) (h : m^2 + n^2 + p^2 + q^2 = 2 * m * n + 2 * p * q) : 
  (m = n ∧ p = q) ∨ (m ≠ n ∧ p ≠ q ∧ ∃ k : ℝ, k^2 * (m^2 + n^2) = p^2 + q^2) := 
sorry

end quadrilateral_type_l38_38167


namespace complement_union_eq_ge_two_l38_38221

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l38_38221


namespace solution_system_l38_38180

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end solution_system_l38_38180


namespace question_proof_l38_38215

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l38_38215


namespace complement_of_A_in_U_l38_38170

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- Define the complement of A within U
def complement_U_A : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_of_A_in_U : complement_U_A = {x | -1 < x ∧ x ≤ 0} :=
  sorry

end complement_of_A_in_U_l38_38170


namespace integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l38_38906

theorem integer_roots_k_values (k : ℤ) :
  (∀ x : ℤ, k * x ^ 2 + (2 * k - 1) * x + k - 1 = 0) →
  k = 0 ∨ k = -1 :=
sorry

theorem y1_y2_squared_sum_k_0 (m y1 y2: ℝ) :
  (m > -2) →
  (k = 0) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 + 2 * m :=
sorry

theorem y1_y2_squared_sum_k_neg1 (m y1 y2: ℝ) :
  (m > -2) →
  (k = -1) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 / 4 + m :=
sorry

end integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l38_38906


namespace angle_PQRS_l38_38053

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end angle_PQRS_l38_38053


namespace total_students_l38_38194

theorem total_students (h1 : ∀ (n : ℕ), n = 5 → Jaya_ranks_nth_from_top)
                       (h2 : ∀ (m : ℕ), m = 49 → Jaya_ranks_mth_from_bottom) :
  ∃ (total : ℕ), total = 53 :=
by
  sorry

end total_students_l38_38194


namespace rectangle_perimeter_l38_38283

theorem rectangle_perimeter :
  ∃ (a b : ℕ), (a ≠ b) ∧ (a * b = 2 * (a + b) - 4) ∧ (2 * (a + b) = 26) :=
by {
  sorry
}

end rectangle_perimeter_l38_38283


namespace taxi_service_charge_l38_38508

theorem taxi_service_charge (initial_fee : ℝ) (additional_charge : ℝ) (increment : ℝ) (total_charge : ℝ) 
  (h_initial_fee : initial_fee = 2.25) 
  (h_additional_charge : additional_charge = 0.4) 
  (h_increment : increment = 2 / 5) 
  (h_total_charge : total_charge = 5.85) : 
  ∃ distance : ℝ, distance = 3.6 :=
by
  sorry

end taxi_service_charge_l38_38508


namespace propositions_p_q_l38_38155

theorem propositions_p_q
  (p q : Prop)
  (h : ¬(p ∧ q) = False) : p ∧ q :=
by
  sorry

end propositions_p_q_l38_38155


namespace total_monthly_feed_l38_38678

def daily_feed (pounds_per_pig_per_day : ℕ) (number_of_pigs : ℕ) : ℕ :=
  pounds_per_pig_per_day * number_of_pigs

def monthly_feed (daily_feed : ℕ) (days_per_month : ℕ) : ℕ :=
  daily_feed * days_per_month

theorem total_monthly_feed :
  let pounds_per_pig_per_day := 15
  let number_of_pigs := 4
  let days_per_month := 30
  monthly_feed (daily_feed pounds_per_pig_per_day number_of_pigs) days_per_month = 1800 :=
by
  sorry

end total_monthly_feed_l38_38678


namespace apples_harvested_from_garden_l38_38819

def number_of_pies : ℕ := 10
def apples_per_pie : ℕ := 8
def apples_to_buy : ℕ := 30

def total_apples_needed : ℕ := number_of_pies * apples_per_pie

theorem apples_harvested_from_garden : total_apples_needed - apples_to_buy = 50 :=
by
  sorry

end apples_harvested_from_garden_l38_38819


namespace integer_in_range_l38_38095

theorem integer_in_range (x : ℤ) 
  (h1 : 0 < x) 
  (h2 : x < 7)
  (h3 : 0 < x)
  (h4 : x < 15)
  (h5 : -1 < x)
  (h6 : x < 5)
  (h7 : 0 < x)
  (h8 : x < 3)
  (h9 : x + 2 < 4) : x = 1 := 
sorry

end integer_in_range_l38_38095


namespace rainfall_thursday_l38_38944

theorem rainfall_thursday : 
  let monday_rain := 0.9
  let tuesday_rain := monday_rain - 0.7
  let wednesday_rain := tuesday_rain * 1.5
  let thursday_rain := wednesday_rain * 0.8
  thursday_rain = 0.24 :=
by
  sorry

end rainfall_thursday_l38_38944


namespace old_camera_model_cost_l38_38237

theorem old_camera_model_cost (C new_model_cost discounted_lens_cost : ℝ)
  (h1 : new_model_cost = 1.30 * C)
  (h2 : discounted_lens_cost = 200)
  (h3 : new_model_cost + discounted_lens_cost = 5400)
  : C = 4000 := by
sorry

end old_camera_model_cost_l38_38237


namespace inequality_proof_l38_38535

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hx1y1 : x1 * y1 - z1^2 > 0) (hx2y2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 - z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l38_38535


namespace probability_difference_l38_38713

theorem probability_difference (red_marbles black_marbles : ℤ) (h_red : red_marbles = 1500) (h_black : black_marbles = 1500) :
  |(22485 / 44985 : ℚ) - (22500 / 44985 : ℚ)| = 15 / 44985 := 
by {
  sorry
}

end probability_difference_l38_38713


namespace four_fives_to_hundred_case1_four_fives_to_hundred_case2_l38_38273

theorem four_fives_to_hundred_case1 : (5 + 5) * (5 + 5) = 100 :=
by sorry

theorem four_fives_to_hundred_case2 : (5 * 5 - 5) * 5 = 100 :=
by sorry

end four_fives_to_hundred_case1_four_fives_to_hundred_case2_l38_38273


namespace find_hourly_charge_l38_38240

variable {x : ℕ}

--Assumptions and conditions
def fixed_charge := 17
def total_paid := 80
def rental_hours := 9

-- Proof problem
theorem find_hourly_charge (h : fixed_charge + rental_hours * x = total_paid) : x = 7 :=
sorry

end find_hourly_charge_l38_38240


namespace percentage_exceeds_l38_38184

-- Defining the constants and conditions
variables {y z x : ℝ}

-- Conditions
def condition1 (y x : ℝ) : Prop := x = 0.6 * y
def condition2 (x z : ℝ) : Prop := z = 1.25 * x

-- Proposition to prove
theorem percentage_exceeds (hyx : condition1 y x) (hxz : condition2 x z) : y = 4/3 * z :=
by 
  -- We skip the proof as requested
  sorry

end percentage_exceeds_l38_38184


namespace angle_PQRS_l38_38054

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end angle_PQRS_l38_38054


namespace polynomial_root_condition_l38_38755

theorem polynomial_root_condition (a : ℝ) :
  (∃ x1 x2 x3 : ℝ,
    (x1^3 - 6 * x1^2 + a * x1 + a = 0) ∧
    (x2^3 - 6 * x2^2 + a * x2 + a = 0) ∧
    (x3^3 - 6 * x3^2 + a * x3 + a = 0) ∧
    ((x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0)) →
  a = -9 :=
by
  sorry

end polynomial_root_condition_l38_38755


namespace length_of_XY_l38_38200

-- Defining the points on the circle
variables (A B C D P Q X Y : Type*)
-- Lengths given in the problem
variables (AB_len CD_len AP_len CQ_len PQ_len : ℕ)
-- Points and lengths conditions
variables (h1 : AB_len = 11) (h2 : CD_len = 19)
variables (h3 : AP_len = 6) (h4 : CQ_len = 7)
variables (h5 : PQ_len = 27)

-- Assuming the Power of a Point theorem applied to P and Q
variables (PX_len PY_len QX_len QY_len : ℕ)
variables (h6 : PX_len = 1) (h7 : QY_len = 3)
variables (h8 : PX_len + PQ_len + QY_len = XY_len)

-- The final length of XY is to be found
def XY_len : ℕ := PX_len + PQ_len + QY_len

-- The goal is to show XY = 31
theorem length_of_XY : XY_len = 31 :=
  by
    sorry

end length_of_XY_l38_38200


namespace hydrogen_moles_l38_38616

-- Define the balanced chemical reaction as a relation between moles
def balanced_reaction (NaH H₂O NaOH H₂ : ℕ) : Prop :=
  NaH = NaOH ∧ H₂ = NaOH ∧ NaH = H₂

-- Given conditions
def given_conditions (NaH H₂O : ℕ) : Prop :=
  NaH = 2 ∧ H₂O = 2

-- Problem statement to prove
theorem hydrogen_moles (NaH H₂O NaOH H₂ : ℕ)
  (h₁ : balanced_reaction NaH H₂O NaOH H₂)
  (h₂ : given_conditions NaH H₂O) :
  H₂ = 2 :=
by sorry

end hydrogen_moles_l38_38616


namespace binom_18_10_l38_38745

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : 0 ≤ k ∧ k ≤ n then nat.choose n k else 0

theorem binom_18_10 :
  binom 16 7 = 11440 ∧
  binom 16 9 = 11440 →
  binom 18 10 = 47190 :=
by
  intro h
  sorry

end binom_18_10_l38_38745


namespace sum_of_digits_least_N_l38_38814

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 * N : ℚ) / 5⌉) / (N + 1)

theorem sum_of_digits_least_N (k : ℕ) (h_k : k = 2) (h1 : ∀ N, P N k < 8 / 10 ) :
  ∃ N : ℕ, (N % 10) + (N / 10) = 1 ∧ (P N k < 8 / 10) ∧ (∀ M : ℕ, M < N → P M k ≥ 8 / 10) := by
  sorry

end sum_of_digits_least_N_l38_38814


namespace sum_of_arithmetic_sequence_l38_38649

noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a1 : ℤ) (d : ℤ)
  (h1 : a1 = -2010)
  (h2 : (S 2011 a1 d) / 2011 - (S 2009 a1 d) / 2009 = 2) :
  S 2010 a1 d = -2010 := 
sorry

end sum_of_arithmetic_sequence_l38_38649


namespace binom_18_10_eq_43758_l38_38742

theorem binom_18_10_eq_43758
  (h1 : nat.choose 16 7 = 11440)
  (h2 : nat.choose 16 9 = 11440) :
  nat.choose 18 10 = 43758 :=
sorry

end binom_18_10_eq_43758_l38_38742


namespace ratio_value_l38_38178

theorem ratio_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
(h1 : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2)) 
(h2 : (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) :
  (x + 1) / (y + 1) = 2 :=
by
  sorry

end ratio_value_l38_38178


namespace complement_union_M_N_eq_ge_2_l38_38208

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l38_38208


namespace max_volume_pyramid_l38_38858

/-- Given:
  AB = 3,
  AC = 5,
  sin ∠BAC = 3/5,
  All lateral edges SA, SB, SC form the same angle with the base plane, not exceeding 60°.
  Prove: the maximum volume of pyramid SABC is 5sqrt(174)/4. -/
theorem max_volume_pyramid 
    (A B C S : Type) 
    (AB : ℝ) 
    (AC : ℝ) 
    (alpha : ℝ) 
    (h : ℝ)
    (V : ℝ)
    (sin_BAC : ℝ) :
    AB = 3 →
    AC = 5 →
    sin_BAC = 3 / 5 →
    alpha ≤ 60 →
    V = (1 / 3) * (1 / 2 * AB * AC * sin_BAC) * h →
    V = 5 * sqrt 174 / 4 :=
by
  intros
  sorry

end max_volume_pyramid_l38_38858


namespace interest_rate_borrowed_l38_38284

variables {P : Type} [LinearOrderedField P]

def borrowed_amount : P := 9000
def lent_interest_rate : P := 0.06
def gain_per_year : P := 180
def per_cent : P := 100

theorem interest_rate_borrowed (r : P) (h : borrowed_amount * lent_interest_rate - gain_per_year = borrowed_amount * r) : 
  r = 0.04 :=
by sorry

end interest_rate_borrowed_l38_38284


namespace dress_assignment_l38_38143

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l38_38143


namespace speed_of_second_train_is_16_l38_38066

def speed_second_train (v : ℝ) : Prop :=
  ∃ t : ℝ, 
    (20 * t = v * t + 70) ∧ -- Condition: the first train traveled 70 km more than the second train
    (20 * t + v * t = 630)  -- Condition: total distance between stations

theorem speed_of_second_train_is_16 : speed_second_train 16 :=
by
  sorry

end speed_of_second_train_is_16_l38_38066


namespace probability_of_draw_l38_38407

-- Define the probabilities as given conditions
def P_A : ℝ := 0.4
def P_A_not_losing : ℝ := 0.9

-- Define the probability of drawing
def P_draw : ℝ :=
  P_A_not_losing - P_A

-- State the theorem to be proved
theorem probability_of_draw : P_draw = 0.5 := by
  sorry

end probability_of_draw_l38_38407


namespace arithmetic_sequence_no_geometric_progression_l38_38233

theorem arithmetic_sequence_no_geometric_progression {r : ℝ} (a : ℕ → ℝ) (k l : ℕ) (h_arith : ∀ n, a (n+1) - a n = r)
(h_contains_terms : a k = 1 ∧ a l = real.sqrt 2)
: ¬∃ m n p : ℕ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ (a m, a n, a p) form_geometric_sequence := sorry

end arithmetic_sequence_no_geometric_progression_l38_38233


namespace Mikaela_savings_l38_38236

theorem Mikaela_savings
  (hourly_rate : ℕ)
  (first_month_hours : ℕ)
  (additional_hours_second_month : ℕ)
  (spending_fraction : ℚ)
  (earnings_first_month := hourly_rate * first_month_hours)
  (hours_second_month := first_month_hours + additional_hours_second_month)
  (earnings_second_month := hourly_rate * hours_second_month)
  (total_earnings := earnings_first_month + earnings_second_month)
  (amount_spent := spending_fraction * total_earnings)
  (amount_saved := total_earnings - amount_spent) :
  hourly_rate = 10 →
  first_month_hours = 35 →
  additional_hours_second_month = 5 →
  spending_fraction = 4 / 5 →
  amount_saved = 150 :=
by
  sorry

end Mikaela_savings_l38_38236


namespace count_3_digit_numbers_divisible_by_7_l38_38176

theorem count_3_digit_numbers_divisible_by_7 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.to_finset.card = 128 := 
  sorry

end count_3_digit_numbers_divisible_by_7_l38_38176


namespace ratio_both_to_onlyB_is_2_l38_38847

variables (num_A num_B both: ℕ)

-- Given conditions
axiom A_eq_2B : num_A = 2 * num_B
axiom both_eq_500 : both = 500
axiom both_multiple_of_only_B : ∃ k : ℕ, both = k * (num_B - both)
axiom only_A_eq_1000 : (num_A - both) = 1000

-- Define the Lean theorem statement
theorem ratio_both_to_onlyB_is_2 : (both : ℝ) / (num_B - both : ℝ) = 2 := 
sorry

end ratio_both_to_onlyB_is_2_l38_38847


namespace remainder_when_dividing_polynomial_l38_38763

noncomputable def P(x : ℝ) := x^5 + 3
noncomputable def Q(x : ℝ) := (x - 3)^2

theorem remainder_when_dividing_polynomial :
  ∃ (R : ℝ → ℝ), (λ x, R x) = (λ x, 405 * x - 969) ∧ (∃ (S : ℝ → ℝ), P = λ x, Q(x) * S(x) + R(x)) :=
sorry

end remainder_when_dividing_polynomial_l38_38763


namespace vector_parallel_l38_38788

variables {t : ℝ}

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, t)

theorem vector_parallel (h : (1 : ℝ) / (3 : ℝ) = (3 : ℝ) / t) : t = 9 :=
by 
  sorry

end vector_parallel_l38_38788


namespace parabola_focus_l38_38796

theorem parabola_focus (a : ℝ) (h1 : ∀ x y, x^2 = a * y ↔ y = x^2 / a)
(h2 : focus_coordinates = (0, 5)) : a = 20 := 
sorry

end parabola_focus_l38_38796


namespace longest_chord_line_eq_l38_38311

/-- Prove that the longest chord intercepted by the circle x^2 + y^2 - 2x + 4y = 0 passes through the point (2,1) and lies on the line 3x - y - 5 = 0. -/
theorem longest_chord_line_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 2*x + 4*y = 0) →
    (3*x - y - 5 = 0) →
    ∃ p : ℝ × ℝ, p = (2, 1) :=
sorry

end longest_chord_line_eq_l38_38311


namespace correct_omega_l38_38403

theorem correct_omega (Ω : ℕ) (h : Ω * Ω = 2 * 2 * 2 * 2 * 3 * 3) : Ω = 2 * 2 * 3 :=
by
  sorry

end correct_omega_l38_38403


namespace fifteenth_number_with_digit_sum_15_is_294_l38_38520

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def numbers_with_digit_sum (s : ℕ) : List ℕ :=
  List.filter (λ n => digit_sum n = s) (List.range (10 ^ 3)) -- Assume a maximum of 3-digit numbers

def fifteenth_number_with_digit_sum (s : ℕ) : ℕ :=
  (numbers_with_digit_sum s).get! 14 -- Get the 15th element (0-indexed)

theorem fifteenth_number_with_digit_sum_15_is_294 : fifteenth_number_with_digit_sum 15 = 294 :=
by
  sorry -- Proof is omitted

end fifteenth_number_with_digit_sum_15_is_294_l38_38520


namespace count_3_digit_integers_with_product_36_l38_38917

theorem count_3_digit_integers_with_product_36 : 
  ∃ n, n = 21 ∧ 
         (∀ d1 d2 d3 : ℕ, 
           1 ≤ d1 ∧ d1 ≤ 9 ∧ 
           1 ≤ d2 ∧ d2 ≤ 9 ∧ 
           1 ≤ d3 ∧ d3 ≤ 9 ∧
           d1 * d2 * d3 = 36 → 
           (d1 ≠ 0 ∨ d2 ≠ 0 ∨ d3 ≠ 0)) := sorry

end count_3_digit_integers_with_product_36_l38_38917


namespace sum_of_digits_of_77_is_14_l38_38588

-- Define the conditions given in the problem
def triangular_array_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define what it means to be the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- The actual Lean theorem statement
theorem sum_of_digits_of_77_is_14 (N : ℕ) (h : triangular_array_sum N = 3003) : sum_of_digits N = 14 :=
by
  sorry  -- Proof to be completed here

end sum_of_digits_of_77_is_14_l38_38588


namespace prob_not_adjacent_l38_38501

theorem prob_not_adjacent :
  let total_ways := Nat.choose 10 2
  let adjacent_ways := 9
  let prob_adjacent := (adjacent_ways / total_ways : ℚ)
  let prob_not_adjacent := 1 - prob_adjacent
  prob_not_adjacent = 4 / 5 := by
  unfold total_ways adjacent_ways prob_adjacent prob_not_adjacent
  norm_num
  sorry

end prob_not_adjacent_l38_38501


namespace cos_alpha_in_fourth_quadrant_l38_38000

theorem cos_alpha_in_fourth_quadrant (α : ℝ) (P : ℝ × ℝ) (h_angle_quadrant : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi)
(h_point : P = (Real.sqrt 5, 2)) (h_sin : Real.sin α = (Real.sqrt 2 / 4) * 2) :
  Real.cos α = Real.sqrt 10 / 4 :=
sorry

end cos_alpha_in_fourth_quadrant_l38_38000


namespace fish_eaten_by_new_fish_l38_38510

def initial_original_fish := 14
def added_fish := 2
def exchange_new_fish := 3
def total_fish_now := 11

theorem fish_eaten_by_new_fish : initial_original_fish - (total_fish_now - exchange_new_fish) = 6 := by
  -- This is where the proof would go
  sorry

end fish_eaten_by_new_fish_l38_38510


namespace remainder_7_pow_150_mod_12_l38_38998

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end remainder_7_pow_150_mod_12_l38_38998


namespace complement_union_eq_l38_38202

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l38_38202


namespace invest_today_for_future_value_l38_38015

-- Define the given future value, interest rate, and number of years as constants
def FV : ℝ := 600000
def r : ℝ := 0.04
def n : ℕ := 15
def target : ℝ := 333087.66

-- Define the present value calculation
noncomputable def PV : ℝ := FV / (1 + r)^n

-- State the theorem that PV is approximately equal to the target value
theorem invest_today_for_future_value : PV = target := 
by sorry

end invest_today_for_future_value_l38_38015


namespace remainder_when_x_plus_2uy_div_y_l38_38272

theorem remainder_when_x_plus_2uy_div_y (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) :
  (x + 2 * u * y) % y = v := 
sorry

end remainder_when_x_plus_2uy_div_y_l38_38272


namespace arithmetic_problem_l38_38297

theorem arithmetic_problem : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end arithmetic_problem_l38_38297


namespace max_elements_ge_distance_5_l38_38667

open Finset

-- Definitions based on the conditions:
def S : Finset (Fin 8 → Bool) := 
  univ.filter (λ A, ∀ i : Fin 8, A i = true ∨ A i = false)

def d (A B: Fin 8 → Bool) : ℕ :=
  (univ.filter (λ i, A i ≠ B i)).card

-- Lean 4 statement:
theorem max_elements_ge_distance_5 : ∃ S' ⊆ S, (∀ (A B ∈ S'), d A B ≥ 5) ∧ S'.card = 4 := 
sorry

end max_elements_ge_distance_5_l38_38667


namespace area_of_square_BDEF_l38_38656

noncomputable def right_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
∃ (AB BC AC : ℝ), AB = 15 ∧ BC = 20 ∧ AC = Real.sqrt (AB^2 + BC^2)

noncomputable def is_square (B D E F : Type*) [MetricSpace B] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
∃ (BD DE EF FB : ℝ), BD = DE ∧ DE = EF ∧ EF = FB

noncomputable def height_of_triangle (E H M : Type*) [MetricSpace E] [MetricSpace H] [MetricSpace M] : Prop :=
∃ (EH : ℝ), EH = 2

theorem area_of_square_BDEF (A B C D E F H M N : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F]
  [MetricSpace H] [MetricSpace M] [MetricSpace N]
  (H1 : right_triangle A B C)
  (H2 : is_square B D E F)
  (H3 : height_of_triangle E H M) :
  ∃ (area : ℝ), area = 100 :=
by
  sorry

end area_of_square_BDEF_l38_38656


namespace fraction_allocated_for_school_l38_38701

-- Conditions
def days_per_week : ℕ := 5
def hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 5
def allocation_for_school : ℕ := 75

-- Proof statement
theorem fraction_allocated_for_school :
  let weekly_hours := days_per_week * hours_per_day
  let weekly_earnings := weekly_hours * earnings_per_hour
  allocation_for_school / weekly_earnings = 3 / 4 := 
by
  sorry

end fraction_allocated_for_school_l38_38701


namespace division_in_base_5_l38_38456

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end division_in_base_5_l38_38456


namespace range_of_b_l38_38768

variable (a b c : ℝ)

theorem range_of_b (h1 : a * c = b^2) (h2 : a + b + c = 3) : -3 ≤ b ∧ b ≤ 1 :=
sorry

end range_of_b_l38_38768


namespace percentage_increase_in_expenses_l38_38721

-- Define the variables and conditions
def monthly_salary : ℝ := 7272.727272727273
def original_savings_percentage : ℝ := 0.10
def new_savings : ℝ := 400
def original_savings : ℝ := original_savings_percentage * monthly_salary
def savings_difference : ℝ := original_savings - new_savings
def original_expenses : ℝ := (1 - original_savings_percentage) * monthly_salary

-- Formalize the question as a theorem
theorem percentage_increase_in_expenses (P : ℝ) :
  P = (savings_difference / original_expenses) * 100 ↔ P = 5 := 
sorry

end percentage_increase_in_expenses_l38_38721


namespace sufficient_but_not_necessary_condition_for_q_l38_38320

theorem sufficient_but_not_necessary_condition_for_q (k : ℝ) :
  (∀ x : ℝ, x ≥ k → x^2 - x > 2) ∧ (∃ x : ℝ, x < k ∧ x^2 - x > 2) ↔ k > 2 :=
sorry

end sufficient_but_not_necessary_condition_for_q_l38_38320


namespace new_pressure_of_helium_l38_38424

noncomputable def helium_pressure (p V p' V' : ℝ) (k : ℝ) : Prop :=
  p * V = k ∧ p' * V' = k

theorem new_pressure_of_helium :
  ∀ (p V p' V' k : ℝ), 
  p = 8 ∧ V = 3.5 ∧ V' = 7 ∧ k = 28 →
  helium_pressure p V p' V' k →
  p' = 4 :=
by
  intros p V p' V' k h1 h2
  sorry

end new_pressure_of_helium_l38_38424


namespace jelly_bean_ratio_l38_38961

theorem jelly_bean_ratio 
  (Napoleon_jelly_beans : ℕ)
  (Sedrich_jelly_beans : ℕ)
  (Mikey_jelly_beans : ℕ)
  (h1 : Napoleon_jelly_beans = 17)
  (h2 : Sedrich_jelly_beans = Napoleon_jelly_beans + 4)
  (h3 : Mikey_jelly_beans = 19) :
  2 * (Napoleon_jelly_beans + Sedrich_jelly_beans) / Mikey_jelly_beans = 4 := 
sorry

end jelly_bean_ratio_l38_38961


namespace fraction_equals_repeating_decimal_l38_38757

noncomputable def repeating_decimal_fraction : ℚ :=
  let a : ℚ := 46 / 100
  let r : ℚ := 1 / 100
  (a / (1 - r))

theorem fraction_equals_repeating_decimal :
  repeating_decimal_fraction = 46 / 99 :=
by
  sorry

end fraction_equals_repeating_decimal_l38_38757


namespace fedya_initial_deposit_l38_38238

theorem fedya_initial_deposit (n k : ℕ) (h₁ : k < 30) (h₂ : n * (100 - k) = 84700) : 
  n = 1100 :=
by
  sorry

end fedya_initial_deposit_l38_38238


namespace four_pow_expression_l38_38703

theorem four_pow_expression : 4 ^ (3 ^ 2) / (4 ^ 3) ^ 2 = 64 := by
  sorry

end four_pow_expression_l38_38703


namespace product_of_numbers_l38_38339

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := 
by 
  sorry

end product_of_numbers_l38_38339


namespace f_l38_38494

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x

-- Define the derivative f'(x)
def f' (a b x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x - 1

-- Problem statement: Prove that f'(-1) = -5 given the conditions
theorem f'_neg_one_value (a b : ℝ) (h : f' a b 1 = 3) : f' a b (-1) = -5 :=
by
  -- Placeholder for the proof
  sorry

end f_l38_38494


namespace binom_18_10_l38_38741

open Nat -- Open the Nat namespace to use natural number properties and functions

theorem binom_18_10 :
  (binom 16 7 = 11440) →
  (binom 16 9 = 11440) →
  binom 18 10 = 32318 := ! sorry

end binom_18_10_l38_38741


namespace find_value_of_a_l38_38003

theorem find_value_of_a
  (a : ℝ)
  (h : (a + 3) * 2 * (-2 / 3) = -4) :
  a = -3 :=
sorry

end find_value_of_a_l38_38003


namespace percentage_excess_calculation_l38_38651

theorem percentage_excess_calculation (A B : ℝ) (x : ℝ) 
  (h1 : (A * (1 + x / 100)) * (B * 0.95) = A * B * 1.007) : 
  x = 6.05 :=
by
  sorry

end percentage_excess_calculation_l38_38651


namespace intersection_A_B_complement_l38_38511

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ 1}
def B_complement : Set ℝ := U \ B

theorem intersection_A_B_complement : A ∩ B_complement = {x | x > 1} := 
by 
  sorry

end intersection_A_B_complement_l38_38511


namespace range_of_m_l38_38767

noncomputable def A := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
noncomputable def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1}

theorem range_of_m (m : ℝ) (h : B m ⊆ A) : -2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l38_38767


namespace amount_subtracted_l38_38714

theorem amount_subtracted (N A : ℝ) (h1 : N = 100) (h2 : 0.80 * N - A = 60) : A = 20 :=
by 
  sorry

end amount_subtracted_l38_38714


namespace value_of_a_l38_38011

theorem value_of_a (P Q : Set ℝ) (a : ℝ) :
  (P = {x | x^2 = 1}) →
  (Q = {x | ax = 1}) →
  (Q ⊆ P) →
  (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l38_38011


namespace number_of_sad_children_l38_38824

-- Definitions of the given conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20

-- The main statement to be proved
theorem number_of_sad_children : 
  total_children - happy_children - neither_happy_nor_sad_children = 10 := 
by 
  sorry

end number_of_sad_children_l38_38824


namespace work_equivalence_l38_38928

variable (m d r : ℕ)

theorem work_equivalence (h : d > 0) : (m * d) / (m + r^2) = d := sorry

end work_equivalence_l38_38928


namespace exist_nonzero_ints_l38_38034

theorem exist_nonzero_ints (m n : ℤ) (h_m : m ≥ 2) (h_n : n ≥ 2)
  (a : ℕ → ℤ) (h_a : ∀ i, 1 ≤ i ∧ i ≤ n → ¬ m^(n-1) ∣ a i) :
  ∃ e : ℕ → ℤ, (∀ i, 1 ≤ i ∧ i ≤ n → 0 < |e i| ∧ |e i| < m) ∧ (m^n ∣ ∑ i in finset.range n, e i * a i) :=
by
  sorry

end exist_nonzero_ints_l38_38034


namespace speed_limit_inequality_l38_38876

theorem speed_limit_inequality (v : ℝ) : (v ≤ 40) :=
sorry

end speed_limit_inequality_l38_38876


namespace total_viewing_time_amaya_l38_38289

/-- The total viewing time Amaya spent, including rewinding, was 170 minutes. -/
theorem total_viewing_time_amaya 
  (u1 u2 u3 u4 u5 r1 r2 r3 r4 : ℕ)
  (h1 : u1 = 35)
  (h2 : u2 = 45)
  (h3 : u3 = 25)
  (h4 : u4 = 15)
  (h5 : u5 = 20)
  (hr1 : r1 = 5)
  (hr2 : r2 = 7)
  (hr3 : r3 = 10)
  (hr4 : r4 = 8) :
  u1 + u2 + u3 + u4 + u5 + r1 + r2 + r3 + r4 = 170 :=
by
  sorry

end total_viewing_time_amaya_l38_38289


namespace problem_equivalent_l38_38219

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l38_38219


namespace determine_polynomial_l38_38462

theorem determine_polynomial (p : ℝ → ℝ) (h : ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2) :
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b * x + c := by
  sorry

end determine_polynomial_l38_38462


namespace percentage_more_l38_38958

variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.90 * J
def Mary_income : Prop := M = 1.44 * J

-- Theorem to be proved
theorem percentage_more (h1 : Tim_income J T) (h2 : Mary_income J M) :
  ((M - T) / T) * 100 = 60 :=
sorry

end percentage_more_l38_38958


namespace solve_expression_l38_38860

theorem solve_expression : 3 ^ (1 ^ (0 ^ 2)) - ((3 ^ 1) ^ 0) ^ 2 = 2 := by
  sorry

end solve_expression_l38_38860


namespace sequence_remainder_4_l38_38576

def sequence_of_numbers (n : ℕ) : ℕ :=
  7 * n + 4

theorem sequence_remainder_4 (n : ℕ) : (sequence_of_numbers n) % 7 = 4 := by
  sorry

end sequence_remainder_4_l38_38576


namespace problem_1_problem_2_l38_38229

-- Define f as an odd function on ℝ 
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the main property given in the problem
def property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ≠ 0 → (f a + f b) / (a + b) > 0

-- Problem 1: Prove that if a > b then f(a) > f(b)
theorem problem_1 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  ∀ a b : ℝ, a > b → f a > f b := sorry

-- Problem 2: Prove that given f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x in [0, +∞), the range of k is k < 1
theorem problem_2 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  (∀ x : ℝ, 0 ≤ x → f (9 ^ x - 2 * 3 ^ x) + f (2 * 9 ^ x - k) > 0) → k < 1 := sorry

end problem_1_problem_2_l38_38229


namespace tan_2alpha_and_cos_beta_l38_38898

theorem tan_2alpha_and_cos_beta
    (α β : ℝ)
    (h1 : 0 < β ∧ β < α ∧ α < (Real.pi / 2))
    (h2 : Real.sin α = (4 * Real.sqrt 3) / 7)
    (h3 : Real.cos (β - α) = 13 / 14) :
    Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ Real.cos β = 1 / 2 := by
  sorry

end tan_2alpha_and_cos_beta_l38_38898


namespace f_periodic_function_l38_38161

noncomputable def f : ℝ → ℝ := sorry

theorem f_periodic_function (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x : ℝ, f (x + 4) = f x + f 2)
    (h3 : f 1 = 2) : 
    f 2013 = 2 := sorry

end f_periodic_function_l38_38161


namespace admission_methods_count_l38_38870

theorem admission_methods_count :
  let students := 4
  let universities := 3
  let condition := (∃ (f : Fin students → Fin universities), ∀ u : Fin universities, ∃ s : Fin students, f s = u)
  condition → (count_permutations students universities = 36) := by
  intros students universities condition
  sorry

end admission_methods_count_l38_38870


namespace pascal_sixth_element_row_20_l38_38612

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
by 
  sorry

end pascal_sixth_element_row_20_l38_38612


namespace range_of_a_l38_38479

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x a ≥ a) ↔ -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 := 
sorry

end range_of_a_l38_38479


namespace probability_of_selecting_at_least_one_female_l38_38897

theorem probability_of_selecting_at_least_one_female :
  let total_students := 5
  let total_ways := nat.choose total_students 2
  let male_students := 3
  let female_students := 2
  let ways_2_males := nat.choose male_students 2
  (1 - ((ways_2_males : ℚ) / (total_ways : ℚ))) = 7 / 10 := by
  sorry

end probability_of_selecting_at_least_one_female_l38_38897


namespace complement_union_M_N_eq_ge_2_l38_38206

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l38_38206


namespace simplify_expression_l38_38538

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l38_38538


namespace enter_exit_ways_eq_sixteen_l38_38987

theorem enter_exit_ways_eq_sixteen (n : ℕ) (h : n = 4) : n * n = 16 :=
by sorry

end enter_exit_ways_eq_sixteen_l38_38987


namespace area_S_l38_38813

open Real

variables {P : Type*} [Real : metric_space P]

def Point := P

def Set (p : P) : Prop :=
  -- Define the circle Omega with radius 8 centered at point O
  let Omega := {p : P | dist p (0 : P) ≤ 8} in
  -- Define point M on the circle Omega
  let M : P := (8, 0) in
  -- Define the set S containing points P
  let S := {P : P |
    ((8 : Point) ∈ Omega ∨
    (∃ (A B C D : Point), A.1 = 4 ∧ B.2 = 5 ∧ B.3 = 5 ∧
    dist E (0 : Point) ≤ 8 
    ∧
    (dist A O ≤ AB ∨ dist P B) 
    ∧ ((B) (X.O)))⟩ 

def area (S : Set) : ℝ :=
  164 + 64 * π 

-- The theorem which states that the area of the set S is 164 + 64π
theorem area_S : 
  let Omega := {P : Point | dist p (0 : Point) ≤ 8}
  ∃ S : Set
  (∀ p : P, Set contains p ) 
  show (area Set. p) 
  (area S =164 + 64 * π :=
sorry

end area_S_l38_38813


namespace concentration_of_concentrated_kola_is_correct_l38_38872

noncomputable def concentration_of_concentrated_kola_added 
  (initial_volume : ℝ) (initial_pct_sugar : ℝ)
  (sugar_added : ℝ) (water_added : ℝ)
  (required_pct_sugar : ℝ) (new_sugar_volume : ℝ) : ℝ :=
  let initial_sugar := initial_volume * initial_pct_sugar / 100
  let total_sugar := initial_sugar + sugar_added
  let new_total_volume := initial_volume + sugar_added + water_added
  let total_volume_with_kola := new_total_volume + (new_sugar_volume / required_pct_sugar * 100 - total_sugar) / (100 / required_pct_sugar - 1)
  total_volume_with_kola - new_total_volume

noncomputable def problem_kola : ℝ :=
  concentration_of_concentrated_kola_added 340 7 3.2 10 7.5 27

theorem concentration_of_concentrated_kola_is_correct : 
  problem_kola = 6.8 :=
by
  unfold problem_kola concentration_of_concentrated_kola_added
  sorry

end concentration_of_concentrated_kola_is_correct_l38_38872


namespace pages_read_tonight_l38_38894

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem pages_read_tonight :
  let pages_3_nights_ago := 20
  let pages_2_nights_ago := 20^2 + 5
  let pages_last_night := sum_of_digits pages_2_nights_ago * 3
  let total_pages := 500
  total_pages - (pages_3_nights_ago + pages_2_nights_ago + pages_last_night) = 48 :=
by
  sorry

end pages_read_tonight_l38_38894


namespace ahmed_total_distance_l38_38116

theorem ahmed_total_distance (d : ℝ) (h : (3 / 4) * d = 12) : d = 16 := 
by 
  sorry

end ahmed_total_distance_l38_38116


namespace remainder_of_3_pow_2023_mod_7_l38_38859

theorem remainder_of_3_pow_2023_mod_7 :
  3 ^ 2023 % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l38_38859


namespace regular_pay_limit_l38_38722

theorem regular_pay_limit (x : ℝ) : 3 * x + 6 * 13 = 198 → x = 40 :=
by
  intro h
  -- proof skipped
  sorry

end regular_pay_limit_l38_38722


namespace additional_grassy_ground_l38_38582

theorem additional_grassy_ground (r1 r2 : ℝ) (π : ℝ) :
  r1 = 12 → r2 = 18 → π = Real.pi →
  (π * r2^2 - π * r1^2) = 180 * π := by
sorry

end additional_grassy_ground_l38_38582


namespace range_of_a_l38_38493

def decreasing_range (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ 4 → y ≤ 4 → x < y → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)

theorem range_of_a (a : ℝ) : decreasing_range a ↔ a ≤ -3 := 
  sorry

end range_of_a_l38_38493


namespace sum_of_words_l38_38037

-- Definitions to represent the conditions
def ХЛЕБ : List Char := ['Х', 'Л', 'Е', 'Б']
def КАША : List Char := ['К', 'А', 'Ш', 'А']

-- Function to compute factorial
def factorial : Nat -> Nat
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- Permutations considering repetition (as in multiset permutations)
def permutations_with_repetition (n : Nat) (reps : Nat) : Nat :=
  factorial n / factorial reps

-- The theorem to prove
theorem sum_of_words : (factorial 4) + (permutations_with_repetition 4 2) = 36 := by
  sorry

end sum_of_words_l38_38037


namespace parabola_shifts_down_decrease_c_real_roots_l38_38702

-- The parabolic function and conditions
variables {a b c k : ℝ}

-- Assumption that a is positive
axiom ha : a > 0

-- Parabola shifts down when constant term c is decreased
theorem parabola_shifts_down (c : ℝ) (k : ℝ) (hk : k > 0) :
  ∀ x, (a * x^2 + b * x + (c - k)) = (a * x^2 + b * x + c) - k :=
by sorry

-- Discriminant of quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- If the discriminant is negative, decreasing c can result in real roots
theorem decrease_c_real_roots (b c : ℝ) (hb : b^2 < 4 * a * c) (k : ℝ) (hk : k > 0) :
  discriminant a b (c - k) ≥ 0 :=
by sorry

end parabola_shifts_down_decrease_c_real_roots_l38_38702


namespace complement_union_eq_ge2_l38_38212

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l38_38212


namespace solution_in_quadrant_I_l38_38231

theorem solution_in_quadrant_I (k x y : ℝ) (h1 : 2 * x - y = 5) (h2 : k * x^2 + y = 4) (h4 : x > 0) (h5 : y > 0) : k > 0 :=
sorry

end solution_in_quadrant_I_l38_38231


namespace distribute_volunteers_l38_38294

theorem distribute_volunteers (volunteers venues : ℕ) (h_vol : volunteers = 5) (h_venues : venues = 3) :
  ∃ (distributions : ℕ), (∀ v : ℕ, 1 ≤ v → v ≤ venues) ∧ 
  (∑ v in finset.range venues, v) = volunteers ∧ 
  distributions = 150 :=
by
  use 150
  sorry

end distribute_volunteers_l38_38294


namespace first_term_arithmetic_series_l38_38188

theorem first_term_arithmetic_series 
  (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 240)
  (h2 : 30 * (2 * a + 179 * d) = 3600) : 
  a = -353 / 15 :=
by
  have eq1 : 2 * a + 59 * d = 8 := by sorry
  have eq2 : 2 * a + 179 * d = 120 := by sorry
  sorry

end first_term_arithmetic_series_l38_38188


namespace contact_prob_correct_l38_38760

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l38_38760


namespace total_students_are_45_l38_38419

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end total_students_are_45_l38_38419


namespace root_of_quadratic_gives_value_l38_38774

theorem root_of_quadratic_gives_value (a : ℝ) (h : a^2 + 3 * a - 5 = 0) : a^2 + 3 * a + 2021 = 2026 :=
by {
  -- We will skip the proof here.
  sorry
}

end root_of_quadratic_gives_value_l38_38774


namespace sister_ages_l38_38050

theorem sister_ages (x y : ℕ) (h1 : x = y + 4) (h2 : x^3 - y^3 = 988) : y = 7 ∧ x = 11 :=
by
  sorry

end sister_ages_l38_38050


namespace cookie_radius_l38_38981

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 13 := 
sorry

end cookie_radius_l38_38981


namespace annual_interest_earned_l38_38365
noncomputable section

-- Define the total money
def total_money : ℝ := 3200

-- Define the first part of the investment
def P1 : ℝ := 800

-- Define the second part of the investment as total money minus the first part
def P2 : ℝ := total_money - P1

-- Define the interest rates for both parts
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Define the time period (in years)
def time_period : ℝ := 1

-- Define the interest earned from each part
def interest1 : ℝ := P1 * rate1 * time_period
def interest2 : ℝ := P2 * rate2 * time_period

-- The total interest earned from both investments
def total_interest : ℝ := interest1 + interest2

-- The proof statement
theorem annual_interest_earned : total_interest = 144 := by
  sorry

end annual_interest_earned_l38_38365


namespace estimated_total_score_l38_38654

noncomputable def regression_score (x : ℝ) : ℝ := 7.3 * x - 96.9

theorem estimated_total_score (x : ℝ) (h : x = 95) : regression_score x = 596 :=
by
  rw [h]
  -- skipping the actual calculation steps
  sorry

end estimated_total_score_l38_38654


namespace magdalena_fraction_picked_l38_38818

noncomputable def fraction_picked_first_day
  (produced_apples: ℕ)
  (remaining_apples: ℕ)
  (fraction_picked: ℚ) : Prop :=
  ∃ (f : ℚ),
  produced_apples = 200 ∧
  remaining_apples = 20 ∧
  (f = fraction_picked) ∧
  (200 * f + 2 * 200 * f + (200 * f + 20)) = 200 - remaining_apples ∧
  fraction_picked = 1 / 5

theorem magdalena_fraction_picked :
  fraction_picked_first_day 200 20 (1 / 5) :=
sorry

end magdalena_fraction_picked_l38_38818


namespace find_point_W_coordinates_l38_38615

theorem find_point_W_coordinates 
(O U S V : ℝ × ℝ)
(hO : O = (0, 0))
(hU : U = (3, 3))
(hS : S = (3, 0))
(hV : V = (0, 3))
(hSquare : (O.1 - U.1)^2 + (O.2 - U.2)^2 = 18)
(hArea_Square : 3 * 3 = 9) :
  ∃ W : ℝ × ℝ, W = (3, 9) ∧ 1 / 2 * (abs (S.1 - V.1) * abs (W.2 - S.2)) = 9 :=
by
  sorry

end find_point_W_coordinates_l38_38615


namespace number_of_bags_of_chips_l38_38395

theorem number_of_bags_of_chips (friends : ℕ) (amount_per_friend : ℕ) (cost_per_bag : ℕ) (total_amount : ℕ) (number_of_bags : ℕ) : 
  friends = 3 → amount_per_friend = 5 → cost_per_bag = 3 → total_amount = friends * amount_per_friend → number_of_bags = total_amount / cost_per_bag → number_of_bags = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_bags_of_chips_l38_38395


namespace positive_integer_solution_l38_38315

theorem positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 = y^2 + 71) :
  x = 6 ∧ y = 35 :=
by
  sorry

end positive_integer_solution_l38_38315


namespace train_pass_tree_in_time_l38_38417

-- Definitions from the given conditions
def train_length : ℚ := 270  -- length in meters
def train_speed_km_per_hr : ℚ := 108  -- speed in km/hr

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (v : ℚ) : ℚ := v * (5 / 18)

-- Speed of the train in m/s
def train_speed_m_per_s : ℚ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Question translated into a proof problem
theorem train_pass_tree_in_time :
  train_length / train_speed_m_per_s = 9 :=
by
  sorry

end train_pass_tree_in_time_l38_38417


namespace imaginary_part_z1z2_l38_38492

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end imaginary_part_z1z2_l38_38492


namespace total_price_increase_percentage_l38_38487

theorem total_price_increase_percentage 
    (P : ℝ) 
    (h1 : P > 0) 
    (P_after_first_increase : ℝ := P * 1.2) 
    (P_after_second_increase : ℝ := P_after_first_increase * 1.15) :
    ((P_after_second_increase - P) / P) * 100 = 38 :=
by
  sorry

end total_price_increase_percentage_l38_38487


namespace cost_of_fencing_l38_38565

/-- Define given conditions: -/
def sides_ratio (length width : ℕ) : Prop := length = 3 * width / 2

def park_area : ℕ := 3750

def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

/-- Prove that the cost of fencing the park is 150 rupees: -/
theorem cost_of_fencing 
  (length width : ℕ) 
  (h : sides_ratio length width) 
  (h_area : length * width = park_area) 
  (cost_per_meter_paise : ℕ := 60) : 
  (length + width) * 2 * (paise_to_rupees cost_per_meter_paise) = 150 :=
by sorry

end cost_of_fencing_l38_38565


namespace examination_is_30_hours_l38_38189

noncomputable def examination_time_in_hours : ℝ :=
  let total_questions := 200
  let type_a_problems := 10
  let total_time_on_type_a := 17.142857142857142
  let time_per_type_a := total_time_on_type_a / type_a_problems
  let time_per_type_b := time_per_type_a / 2
  let type_b_problems := total_questions - type_a_problems
  let total_time_on_type_b := time_per_type_b * type_b_problems
  let total_time_in_minutes := total_time_on_type_a * type_a_problems + total_time_on_type_b
  total_time_in_minutes / 60

theorem examination_is_30_hours :
  examination_time_in_hours = 30 := by
  sorry

end examination_is_30_hours_l38_38189


namespace lana_average_speed_l38_38810

theorem lana_average_speed (initial_reading : ℕ) (final_reading : ℕ) (time_first_day : ℕ) (time_second_day : ℕ) :
  initial_reading = 1991 → 
  final_reading = 2332 → 
  time_first_day = 5 → 
  time_second_day = 7 → 
  (final_reading - initial_reading) / (time_first_day + time_second_day : ℝ) = 28.4 :=
by
  intros h_init h_final h_first h_second
  rw [h_init, h_final, h_first, h_second]
  norm_num
  sorry

end lana_average_speed_l38_38810


namespace sequence_sum_129_l38_38024

/-- 
  In an increasing sequence of four positive integers where the first three terms form an arithmetic
  progression and the last three terms form a geometric progression, and where the first and fourth
  terms differ by 30, the sum of the four terms is 129.
-/
theorem sequence_sum_129 :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a < a + d) ∧ (a + d < a + 2 * d) ∧ 
    (a + 2 * d < a + 30) ∧ 30 = (a + 30) - a ∧ 
    (a + d) * (a + 30) = (a + 2 * d) ^ 2 ∧ 
    a + (a + d) + (a + 2 * d) + (a + 30) = 129 :=
sorry

end sequence_sum_129_l38_38024


namespace chord_length_intercepted_by_line_on_circle_l38_38655

theorem chord_length_intercepted_by_line_on_circle :
  ∀ (ρ θ : ℝ), (ρ = 4) →
  (ρ * Real.sin (θ + (Real.pi / 4)) = 2) →
  (4 * Real.sqrt (16 - (2 ^ 2)) = 4 * Real.sqrt 3) :=
by
  intros ρ θ hρ hline_eq
  sorry

end chord_length_intercepted_by_line_on_circle_l38_38655


namespace sara_initial_quarters_l38_38683

theorem sara_initial_quarters (total_quarters dad_gift initial_quarters : ℕ) (h1 : dad_gift = 49) (h2 : total_quarters = 70) (h3 : total_quarters = initial_quarters + dad_gift) : initial_quarters = 21 :=
by sorry

end sara_initial_quarters_l38_38683


namespace find_a_l38_38784

theorem find_a (a : ℝ) (t : ℝ) :
  (4 = 1 + 3 * t) ∧ (3 = a * t^2 + 2) → a = 1 :=
by
  sorry

end find_a_l38_38784


namespace problem_statement_l38_38794

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 5) : 
  -a - m * c * d - b = -5 ∨ -a - m * c * d - b = 5 := 
  sorry

end problem_statement_l38_38794


namespace binom_18_10_l38_38748

-- Given conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Objective statement
theorem binom_18_10 : nat.choose 18 10 = 43758 :=
by {
  sorry
}

end binom_18_10_l38_38748


namespace jonah_added_yellow_raisins_l38_38948

variable (y : ℝ)

theorem jonah_added_yellow_raisins (h : y + 0.4 = 0.7) : y = 0.3 := by
  sorry

end jonah_added_yellow_raisins_l38_38948


namespace farm_corn_cobs_l38_38110

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end farm_corn_cobs_l38_38110


namespace geometric_sequence_a4_a7_l38_38190

theorem geometric_sequence_a4_a7 (a : ℕ → ℝ) (h1 : ∃ a₁ a₁₀, a₁ * a₁₀ = -6 ∧ a 1 = a₁ ∧ a 10 = a₁₀) :
  a 4 * a 7 = -6 :=
sorry

end geometric_sequence_a4_a7_l38_38190


namespace range_of_a_l38_38638

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (x^2 + 1) * Real.exp x ≥ a * x^2

theorem range_of_a (a : ℝ) : satisfies_inequality a ↔ a ≤ 2 * Real.exp 1 :=
by
  sorry

end range_of_a_l38_38638


namespace inequality_always_holds_l38_38895

theorem inequality_always_holds (a : ℝ) (h : a ≥ -2) : ∀ (x : ℝ), x^2 + a * |x| + 1 ≥ 0 :=
by
  sorry

end inequality_always_holds_l38_38895


namespace condition_necessary_but_not_sufficient_l38_38868

variable (a b : ℝ)

theorem condition_necessary_but_not_sufficient (h : a ≠ 1 ∨ b ≠ 2) : (a + b ≠ 3) ∧ ¬(a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  --Proof will go here
  sorry

end condition_necessary_but_not_sufficient_l38_38868


namespace remainder_when_divided_by_7_l38_38983

theorem remainder_when_divided_by_7 :
  let a := -1234
  let b := 1984
  let c := -1460
  let d := 2008
  (a * b * c * d) % 7 = 0 :=
by
  sorry

end remainder_when_divided_by_7_l38_38983


namespace inequality_proof_l38_38775

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b+c-a)^2 / ((b+c)^2+a^2) + (c+a-b)^2 / ((c+a)^2+b^2) + (a+b-c)^2 / ((a+b)^2+c^2) ≥ 3 / 5 :=
by sorry

end inequality_proof_l38_38775


namespace greatest_large_chips_l38_38279

theorem greatest_large_chips :
  ∃ (l : ℕ), (∃ (s : ℕ), ∃ (p : ℕ), s + l = 70 ∧ s = l + p ∧ Nat.Prime p) ∧ 
  (∀ (l' : ℕ), (∃ (s' : ℕ), ∃ (p' : ℕ), s' + l' = 70 ∧ s' = l' + p' ∧ Nat.Prime p') → l' ≤ 34) :=
sorry

end greatest_large_chips_l38_38279


namespace travel_ways_l38_38392

theorem travel_ways (highways : ℕ) (railways : ℕ) (n : ℕ) :
  highways = 3 → railways = 2 → n = highways + railways → n = 5 :=
by
  intros h_eq r_eq n_eq
  rw [h_eq, r_eq] at n_eq
  exact n_eq

end travel_ways_l38_38392


namespace inequality_abc_l38_38772

theorem inequality_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) :=
sorry

end inequality_abc_l38_38772


namespace number_of_combinations_with_odd_sum_l38_38909

open Finset

/-- The set of numbers from 1 to 9 -/
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The set of odd numbers in A -/
def odd_numbers : Finset ℕ := {1, 3, 5, 7, 9}

/-- The set of even numbers in A -/
def even_numbers : Finset ℕ := {2, 4, 6, 8}

/-- The number of ways to select four numbers from A such that the sum is odd -/
theorem number_of_combinations_with_odd_sum : 
  (choose 5 1) * (choose 4 3) + (choose 5 3) * (choose 4 1) = 60 :=
by 
  simp [choose]
  sorry

end number_of_combinations_with_odd_sum_l38_38909


namespace hyperbola_equation_l38_38783

variable (a b c : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (asymptote_cond : -b / a = -1 / 2)
variable (foci_cond : c = 5)
variable (hyperbola_rel : a^2 + b^2 = c^2)

theorem hyperbola_equation : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ -b / a = -1 / 2 ∧ c = 5 ∧ a^2 + b^2 = c^2 
  ∧ ∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1)) := 
sorry

end hyperbola_equation_l38_38783


namespace vertex_of_parabola_l38_38834

/-- The given parabola y = -3(x-1)^2 - 2 has its vertex at (1, -2). -/
theorem vertex_of_parabola : ∃ h k : ℝ, (h = 1 ∧ k = -2) ∧ ∀ x : ℝ, y = -3 * (x - h) ^ 2 + k :=
begin
  use [1, -2],
  split,
  { split; refl },
  { intro x,
    refl }
end

end vertex_of_parabola_l38_38834


namespace james_ride_time_l38_38507

theorem james_ride_time (distance speed : ℝ) (h_distance : distance = 200) (h_speed : speed = 25) : distance / speed = 8 :=
by
  rw [h_distance, h_speed]
  norm_num

end james_ride_time_l38_38507


namespace total_cost_of_products_l38_38596

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end total_cost_of_products_l38_38596


namespace dress_assignment_l38_38140

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l38_38140


namespace solve_for_n_l38_38719

theorem solve_for_n (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2) ^ 2 = 12 * 12 * (n - 2)) :
  n = 26 :=
by {
  sorry
}

end solve_for_n_l38_38719


namespace find_a_in_geometric_sequence_l38_38770

theorem find_a_in_geometric_sequence (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = 3^(n+1) + a) →
  (∃ a, ∀ n, S n = 3^(n+1) + a ∧ (18 : ℝ) ^ 2 = (S 1 - (S 1 - S 2)) * (S 2 - S 3) → a = -3) := 
by
  sorry

end find_a_in_geometric_sequence_l38_38770


namespace tan_double_angle_l38_38905

theorem tan_double_angle (θ : ℝ) (h1 : θ = Real.arctan (-2)) : Real.tan (2 * θ) = 4 / 3 :=
by
  sorry

end tan_double_angle_l38_38905


namespace pie_difference_l38_38561

theorem pie_difference (s1 s3 : ℚ) (h1 : s1 = 7/8) (h3 : s3 = 3/4) :
  s1 - s3 = 1/8 :=
by
  sorry

end pie_difference_l38_38561


namespace smallest_n_satisfying_conditions_l38_38509

variable (n : ℕ)
variable (h1 : 100 ≤ n ∧ n < 1000)
variable (h2 : (n + 7) % 6 = 0)
variable (h3 : (n - 5) % 9 = 0)

theorem smallest_n_satisfying_conditions : n = 113 := by
  sorry

end smallest_n_satisfying_conditions_l38_38509


namespace who_wears_which_dress_l38_38151

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l38_38151


namespace pratyya_payel_min_difference_l38_38827

theorem pratyya_payel_min_difference (n m : ℕ) (h : n > m ∧ n - m ≥ 4) :
  ∀ t : ℕ, (2^(t+1) * n - 2^(t+1)) > 2^(t+1) * m + 2^(t+1) :=
by
  sorry

end pratyya_payel_min_difference_l38_38827


namespace tricycles_count_l38_38883

theorem tricycles_count {s t : Nat} (h1 : s + t = 10) (h2 : 2 * s + 3 * t = 26) : t = 6 :=
sorry

end tricycles_count_l38_38883


namespace find_y_l38_38047

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (h3 : x = 1) : y = 13 := by
  sorry

end find_y_l38_38047


namespace total_tiles_correct_l38_38671

-- Definitions for room dimensions
def room_length : ℕ := 24
def room_width : ℕ := 18

-- Definitions for tile dimensions
def border_tile_side : ℕ := 2
def inner_tile_side : ℕ := 1

-- Definitions for border and inner area calculations
def border_width : ℕ := 2 * border_tile_side
def inner_length : ℕ := room_length - border_width
def inner_width : ℕ := room_width - border_width

-- Calculation of the number of tiles needed
def border_area : ℕ := (room_length * room_width) - (inner_length * inner_width)
def num_border_tiles : ℕ := border_area / (border_tile_side * border_tile_side)
def inner_area : ℕ := inner_length * inner_width
def num_inner_tiles : ℕ := inner_area / (inner_tile_side * inner_tile_side)

-- Total number of tiles
def total_tiles : ℕ := num_border_tiles + num_inner_tiles

-- The proof statement
theorem total_tiles_correct : total_tiles = 318 := by
  -- Lean code to check the calculations, proof is omitted.
  sorry

end total_tiles_correct_l38_38671


namespace three_digit_multiples_of_24_l38_38918

theorem three_digit_multiples_of_24 : 
  let lower_bound := 100
  let upper_bound := 999
  let div_by := 24
  let first := lower_bound + (div_by - lower_bound % div_by) % div_by
  let last := upper_bound - (upper_bound % div_by)
  ∃ n : ℕ, (n + 1) = (last - first) / div_by + 1 := 
sorry

end three_digit_multiples_of_24_l38_38918


namespace weight_in_kilograms_l38_38278

-- Definitions based on conditions
def weight_of_one_bag : ℕ := 250
def number_of_bags : ℕ := 8

-- Converting grams to kilograms (1000 grams = 1 kilogram)
def grams_to_kilograms (grams : ℕ) : ℕ := grams / 1000

-- Total weight in grams
def total_weight_in_grams : ℕ := weight_of_one_bag * number_of_bags

-- Proof that the total weight in kilograms is 2
theorem weight_in_kilograms : grams_to_kilograms total_weight_in_grams = 2 :=
by
  sorry

end weight_in_kilograms_l38_38278


namespace D_cows_grazed_l38_38406

-- Defining the given conditions:
def A_cows := 24
def A_months := 3
def A_rent := 1440

def B_cows := 10
def B_months := 5

def C_cows := 35
def C_months := 4

def D_months := 3

def total_rent := 6500

-- Calculate the cost per cow per month (CPCM)
def CPCM := A_rent / (A_cows * A_months)

-- Proving the number of cows D grazed
theorem D_cows_grazed : ∃ x : ℕ, (x * D_months * CPCM + A_rent + (B_cows * B_months * CPCM) + (C_cows * C_months * CPCM) = total_rent) ∧ x = 21 := by
  sorry

end D_cows_grazed_l38_38406


namespace trig_identity_proof_l38_38737

theorem trig_identity_proof :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 3 :=
by sorry

end trig_identity_proof_l38_38737


namespace decimal_to_binary_45_l38_38298

theorem decimal_to_binary_45 :
  (45 : ℕ) = (0b101101 : ℕ) :=
sorry

end decimal_to_binary_45_l38_38298


namespace kylie_and_nelly_total_stamps_l38_38033

theorem kylie_and_nelly_total_stamps :
  ∀ (kylie_stamps nelly_delta: ℕ),
  kylie_stamps = 34 →
  nelly_delta = 44 →
  (kylie_stamps + (kylie_stamps + nelly_delta) = 112) :=
by
  intros kylie_stamps nelly_delta h_kylie h_delta
  rw [h_kylie, h_delta]
  rw [add_assoc]
  sorry

end kylie_and_nelly_total_stamps_l38_38033


namespace cost_of_books_purchasing_plans_l38_38962

theorem cost_of_books (x y : ℕ) (h1 : 4 * x + 2 * y = 480) (h2 : 2 * x + 3 * y = 520) : x = 50 ∧ y = 140 :=
by
  -- proof can be filled in later
  sorry

theorem purchasing_plans (a b : ℕ) (h_total_cost : 50 * a + 140 * (20 - a) ≤ 1720) (h_quantity : a ≤ 2 * (20 - b)) : (a = 12 ∧ b = 8) ∨ (a = 13 ∧ b = 7) :=
by
  -- proof can be filled in later
  sorry

end cost_of_books_purchasing_plans_l38_38962


namespace red_basket_fruit_count_l38_38262

-- Defining the basket counts
def blue_basket_bananas := 12
def blue_basket_apples := 4
def blue_basket_fruits := blue_basket_bananas + blue_basket_apples
def red_basket_fruits := blue_basket_fruits / 2

-- Statement of the proof problem
theorem red_basket_fruit_count : red_basket_fruits = 8 := by
  sorry

end red_basket_fruit_count_l38_38262


namespace triangle_inequality_l38_38949

theorem triangle_inequality (S R r : ℝ) (h : S^2 = 2 * R^2 + 8 * R * r + 3 * r^2) : 
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := 
by 
  sorry

end triangle_inequality_l38_38949


namespace circle_radii_l38_38836

noncomputable def smaller_circle_radius (r : ℝ) :=
  r = 4

noncomputable def larger_circle_radius (r : ℝ) :=
  r = 9

theorem circle_radii (r : ℝ) (h1 : ∀ (r: ℝ), (r + 5) - r = 5) (h2 : ∀ (r: ℝ), 2.4 * r = 2.4 * r):
  smaller_circle_radius r → larger_circle_radius (r + 5) :=
by
  sorry

end circle_radii_l38_38836


namespace min_value_expression_min_value_is_7_l38_38617

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  6 * x + 1 / (x^6) ≥ 7 :=
sorry

theorem min_value_is_7 : 
  6 * 1 + 1 / (1^6) = 7 :=
by norm_num

end min_value_expression_min_value_is_7_l38_38617


namespace cos_sum_identity_cosine_30_deg_l38_38606

theorem cos_sum_identity : 
  (Real.cos (Real.pi * 43 / 180) * Real.cos (Real.pi * 13 / 180) + 
   Real.sin (Real.pi * 43 / 180) * Real.sin (Real.pi * 13 / 180)) = 
   (Real.cos (Real.pi * 30 / 180)) :=
sorry

theorem cosine_30_deg : 
  Real.cos (Real.pi * 30 / 180) = (Real.sqrt 3 / 2) :=
sorry

end cos_sum_identity_cosine_30_deg_l38_38606


namespace tetrahedron_vertex_angle_sum_l38_38828

theorem tetrahedron_vertex_angle_sum (A B C D : Type) (angles_at : Type → Type → Type → ℝ) :
  (∃ A, (∀ X Y Z W, X = A ∨ Y = A ∨ Z = A ∨ W = A → angles_at X Y A + angles_at Z W A > 180)) →
  ¬ (∃ A B, A ≠ B ∧ 
    (∀ X Y, X = A ∨ Y = A → angles_at X Y A + angles_at Y X A > 180) ∧ 
    (∀ X Y, X = B ∨ Y = B → angles_at X Y B + angles_at Y X B > 180)) := 
sorry

end tetrahedron_vertex_angle_sum_l38_38828


namespace negation_of_exists_cube_pos_l38_38846

theorem negation_of_exists_cube_pos :
  (¬ (∃ x : ℝ, x^3 > 0)) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by
  sorry

end negation_of_exists_cube_pos_l38_38846


namespace who_wears_which_dress_l38_38139

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l38_38139


namespace percentage_increase_numerator_l38_38496

variable (N D : ℝ) (P : ℝ)
variable (h1 : N / D = 0.75)
variable (h2 : (N * (1 + P / 100)) / (D * 0.92) = 15 / 16)

theorem percentage_increase_numerator :
  P = 15 :=
by
  sorry

end percentage_increase_numerator_l38_38496


namespace cleaning_time_is_100_l38_38822

def time_hosing : ℕ := 10
def time_shampoo_per : ℕ := 15
def num_shampoos : ℕ := 3
def time_drying : ℕ := 20
def time_brushing : ℕ := 25

def total_time : ℕ :=
  time_hosing + (num_shampoos * time_shampoo_per) + time_drying + time_brushing

theorem cleaning_time_is_100 :
  total_time = 100 :=
by
  sorry

end cleaning_time_is_100_l38_38822


namespace complement_union_M_N_eq_ge_2_l38_38205

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l38_38205


namespace total_students_l38_38422

-- Defining the conditions
variable (H : ℕ) -- Number of students who ordered hot dogs
variable (students_ordered_burgers : ℕ) -- Number of students who ordered burgers

-- Given conditions
def burger_condition := students_ordered_burgers = 30
def hotdog_condition := students_ordered_burgers = 2 * H

-- Theorem to prove the total number of students
theorem total_students (H : ℕ) (students_ordered_burgers : ℕ) 
  (h1 : burger_condition students_ordered_burgers) 
  (h2 : hotdog_condition students_ordered_burgers H) : 
  students_ordered_burgers + H = 45 := 
by
  sorry

end total_students_l38_38422


namespace gcd_pow_sub_one_l38_38991

theorem gcd_pow_sub_one (a b : ℕ) 
  (h_a : a = 2^2004 - 1) 
  (h_b : b = 2^1995 - 1) : 
  Int.gcd a b = 511 :=
by
  sorry

end gcd_pow_sub_one_l38_38991


namespace line_bisects_circle_l38_38382

theorem line_bisects_circle
  (C : Type)
  [MetricSpace C]
  (x y : ℝ)
  (h : ∀ {x y : ℝ}, x^2 + y^2 - 2*x - 4*y + 1 = 0) : 
  x - y + 1 = 0 → True :=
by
  intro h_line
  sorry

end line_bisects_circle_l38_38382


namespace alice_bob_total_dollars_l38_38733

-- Define Alice's amount in dollars
def alice_amount : ℚ := 5 / 8

-- Define Bob's amount in dollars
def bob_amount : ℚ := 3 / 5

-- Define the total amount in dollars
def total_amount : ℚ := alice_amount + bob_amount

theorem alice_bob_total_dollars : (alice_amount + bob_amount : ℚ) = 1.225 := by
    sorry

end alice_bob_total_dollars_l38_38733


namespace subset_exists_l38_38516

theorem subset_exists (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ) (hA : A.card = p - 1) 
  (hA_div : ∀ a ∈ A, ¬ p ∣ a) :
  ∀ n ∈ Finset.range p, ∃ B ⊆ A, (B.sum id) % p = n :=
by
  -- Proof goes here
  sorry

end subset_exists_l38_38516


namespace problem_equivalent_l38_38217

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l38_38217


namespace area_triangle_MDA_l38_38802

noncomputable def area_of_triangle_MDA (r : ℝ) : ℝ := 
  let AM := r / 3
  let OM := (r ^ 2 - (AM ^ 2)).sqrt
  let AD := AM / 2
  let DM := AD / (1 / 2)
  1 / 2 * AD * DM

theorem area_triangle_MDA (r : ℝ) : area_of_triangle_MDA r = r ^ 2 / 36 := by
  sorry

end area_triangle_MDA_l38_38802


namespace tan_of_acute_angle_and_cos_pi_add_alpha_l38_38625

theorem tan_of_acute_angle_and_cos_pi_add_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2)
  (h2 : Real.cos (π + α) = -Real.sqrt (3) / 2) : 
  Real.tan α = Real.sqrt (3) / 3 :=
by
  sorry

end tan_of_acute_angle_and_cos_pi_add_alpha_l38_38625


namespace factor_of_quadratic_expression_l38_38182

def is_factor (a b : ℤ) : Prop := ∃ k, b = k * a

theorem factor_of_quadratic_expression (m : ℤ) :
  is_factor (m - 8) (m^2 - 5 * m - 24) :=
sorry

end factor_of_quadratic_expression_l38_38182


namespace cos_315_eq_l38_38435

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l38_38435


namespace weight_of_one_bowling_ball_l38_38979

def weight_of_one_canoe : ℕ := 35

def ten_bowling_balls_equal_four_canoes (W: ℕ) : Prop :=
  ∀ w, (10 * w = 4 * W)

theorem weight_of_one_bowling_ball (W: ℕ) (h : W = weight_of_one_canoe) : 
  (10 * 14 = 4 * W) → 14 = 140 / 10 :=
by
  intros H
  sorry

end weight_of_one_bowling_ball_l38_38979


namespace angle_measure_l38_38051

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end angle_measure_l38_38051


namespace shoveling_time_l38_38065

theorem shoveling_time :
  let kevin_time := 12
  let dave_time := 8
  let john_time := 6
  let allison_time := 4
  let kevin_rate := 1 / kevin_time
  let dave_rate := 1 / dave_time
  let john_rate := 1 / john_time
  let allison_rate := 1 / allison_time
  let combined_rate := kevin_rate + dave_rate + john_rate + allison_rate
  let total_minutes := 60
  let combined_rate_per_minute := combined_rate / total_minutes
  (1 / combined_rate_per_minute = 96) := 
  sorry

end shoveling_time_l38_38065


namespace vegetables_harvest_problem_l38_38559

theorem vegetables_harvest_problem
  (same_area : ∀ (a b : ℕ), a = b)
  (first_field_harvest : ℕ := 900)
  (second_field_harvest : ℕ := 1500)
  (less_harvest_per_acre : ∀ (x : ℕ), x - 300 = y) :
  x = y ->
  900 / x = 1500 / (x + 300) :=
by
  sorry

end vegetables_harvest_problem_l38_38559


namespace charge_per_mile_l38_38670

theorem charge_per_mile (rental_fee total_amount_paid : ℝ) (num_miles : ℕ) (charge_per_mile : ℝ) : 
  rental_fee = 20.99 →
  total_amount_paid = 95.74 →
  num_miles = 299 →
  (total_amount_paid - rental_fee) / num_miles = charge_per_mile →
  charge_per_mile = 0.25 :=
by 
  intros r_fee t_amount n_miles c_per_mile h1 h2 h3 h4
  sorry

end charge_per_mile_l38_38670


namespace yang_hui_rect_eq_l38_38370

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end yang_hui_rect_eq_l38_38370


namespace cos_315_proof_l38_38442

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l38_38442


namespace complement_union_eq_l38_38204

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l38_38204


namespace point_P_coordinates_l38_38325

theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 < 0 ∧ abs P.2 = 3 ∧ abs P.1 = 8 ∧ P = (8, -3) :=
sorry

end point_P_coordinates_l38_38325


namespace chenny_friends_l38_38740

theorem chenny_friends (initial_candies : ℕ) (needed_candies : ℕ) (candies_per_friend : ℕ) (h1 : initial_candies = 10) (h2 : needed_candies = 4) (h3 : candies_per_friend = 2) :
  (initial_candies + needed_candies) / candies_per_friend = 7 :=
by
  sorry

end chenny_friends_l38_38740


namespace find_z_l38_38357

variable (x y z : ℝ)

-- Define x, y as given in the problem statement
def x_def : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) := by
  sorry

def y_def : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) := by
  sorry

-- Define the equation relating z to x and y
def z_eq : 192 * z = x^4 + y^4 + (x + y)^4 := by 
  sorry

-- Theorem stating the value of z
theorem find_z (h1 : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3))
               (h2 : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3))
               (h3 : 192 * z = x^4 + y^4 + (x + y)^4) :
  z = 6 := by 
  sorry

end find_z_l38_38357


namespace max_value_pq_qr_rs_sp_l38_38848

def max_pq_qr_rs_sp (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + s * p

theorem max_value_pq_qr_rs_sp :
  ∀ (p q r s : ℕ), (p = 1 ∨ p = 5 ∨ p = 3 ∨ p = 6) → 
                    (q = 1 ∨ q = 5 ∨ q = 3 ∨ q = 6) →
                    (r = 1 ∨ r = 5 ∨ r = 3 ∨ r = 6) → 
                    (s = 1 ∨ s = 5 ∨ s = 3 ∨ s = 6) →
                    p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
                    max_pq_qr_rs_sp p q r s ≤ 56 := by
  sorry

end max_value_pq_qr_rs_sp_l38_38848


namespace Nick_raising_money_l38_38239

theorem Nick_raising_money :
  let chocolate_oranges := 20
  let oranges_price := 10
  let candy_bars := 160
  let bars_price := 5
  let total_amount := chocolate_oranges * oranges_price + candy_bars * bars_price
  total_amount = 1000 := 
by
  sorry

end Nick_raising_money_l38_38239


namespace find_second_number_l38_38104

theorem find_second_number (a : ℕ) (c : ℕ) (x : ℕ) : 
  3 * a + 3 * x + 3 * c + 11 = 170 → a = 16 → c = 20 → x = 17 := 
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  simp at h1
  sorry

end find_second_number_l38_38104


namespace weight_of_10_moles_approx_l38_38333

def atomic_mass_C : ℝ := 12.01
def atomic_mass_H : ℝ := 1.008
def atomic_mass_O : ℝ := 16.00

def molar_mass_C6H8O6 : ℝ := 
  (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)

def moles : ℝ := 10
def given_total_weight : ℝ := 1760

theorem weight_of_10_moles_approx (ε : ℝ) (hε : ε > 0) :
  abs ((moles * molar_mass_C6H8O6) - given_total_weight) < ε := by
  -- proof will go here.
  sorry

end weight_of_10_moles_approx_l38_38333


namespace value_of_x_squared_plus_inverse_squared_l38_38164

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x ≠ 0) (h : x^4 + (1 / x^4) = 2) : x^2 + (1 / x^2) = 2 :=
sorry

end value_of_x_squared_plus_inverse_squared_l38_38164


namespace tip_count_proof_l38_38589

def initial_customers : ℕ := 29
def additional_customers : ℕ := 20
def customers_who_tipped : ℕ := 15
def total_customers : ℕ := initial_customers + additional_customers
def customers_didn't_tip : ℕ := total_customers - customers_who_tipped

theorem tip_count_proof : customers_didn't_tip = 34 :=
by
  -- This is a proof outline, not the actual proof.
  sorry

end tip_count_proof_l38_38589


namespace units_digit_7_pow_3_l38_38080

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end units_digit_7_pow_3_l38_38080


namespace problem_statement_l38_38327

noncomputable theory

def xy_is_perfect_square (x y : ℕ) : Prop :=
  (x * y + 4) = (x + 2) * (x + 2)

theorem problem_statement (x y : ℕ) (h : x > 0 ∧ y > 0) : 
  (1/x + 1/y + 1/(x * y) = 1/(x + 4) + 1/(y - 4) + 1/((x + 4) * (y - 4))) → xy_is_perfect_square x y :=
by
  sorry

end problem_statement_l38_38327


namespace mass_of_CaSO4_formed_correct_l38_38610

noncomputable def mass_CaSO4_formed 
(mass_CaO : ℝ) (mass_H2SO4 : ℝ)
(molar_mass_CaO : ℝ) (molar_mass_H2SO4 : ℝ) (molar_mass_CaSO4 : ℝ) : ℝ :=
  let moles_CaO := mass_CaO / molar_mass_CaO
  let moles_H2SO4 := mass_H2SO4 / molar_mass_H2SO4
  let limiting_reactant_moles := min moles_CaO moles_H2SO4
  limiting_reactant_moles * molar_mass_CaSO4

theorem mass_of_CaSO4_formed_correct :
  mass_CaSO4_formed 25 35 56.08 98.09 136.15 = 48.57 :=
by
  rw [mass_CaSO4_formed]
  sorry

end mass_of_CaSO4_formed_correct_l38_38610


namespace P_zero_value_l38_38198

noncomputable def P (x b c : ℚ) : ℚ := x ^ 2 + b * x + c

theorem P_zero_value (b c : ℚ)
  (h1 : P (P 1 b c) b c = 0)
  (h2 : P (P (-2) b c) b c = 0)
  (h3 : P 1 b c ≠ P (-2) b c) :
  P 0 b c = -5 / 2 :=
sorry

end P_zero_value_l38_38198


namespace union_A_B_l38_38773

def setA : Set ℝ := { x | Real.log x / Real.log (1/2) > -1 }
def setB : Set ℝ := { x | 2^x > Real.sqrt 2 }

theorem union_A_B : setA ∪ setB = { x | 0 < x } := by
  sorry

end union_A_B_l38_38773


namespace maurice_riding_times_l38_38821

variable (M : ℕ) -- The number of times Maurice had been horseback riding before visiting Matt
variable (h1 : 8) -- The times Maurice rode during his visit
variable (h2 : 8) -- The times Matt rode with Maurice
variable (h3 : 16) -- The additional times Matt rode
variable (h4 : 24 = 3 * M) -- The total number of times Matt rode during the two weeks is three times the number of times Maurice had ridden before his visit

theorem maurice_riding_times : M = 8 :=
by
  sorry

end maurice_riding_times_l38_38821


namespace calculate_difference_l38_38811

def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem calculate_difference :
  f (g 5) - g (f 5) = -2 := by
  sorry

end calculate_difference_l38_38811


namespace unique_sum_of_squares_power_of_two_l38_38533

theorem unique_sum_of_squares_power_of_two (n : ℕ) :
  ∃! (a b : ℕ), 2^n = a^2 + b^2 := 
sorry

end unique_sum_of_squares_power_of_two_l38_38533


namespace negation_of_proposition_l38_38253

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_proposition_l38_38253


namespace dress_assignment_l38_38148

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l38_38148


namespace Julio_spent_on_limes_l38_38348

theorem Julio_spent_on_limes
  (days : ℕ)
  (lime_cost_per_3 : ℕ)
  (mocktails_per_day : ℕ)
  (lime_juice_per_lime_tbsp : ℕ)
  (lime_juice_per_mocktail_tbsp : ℕ)
  (limes_per_set : ℕ)
  (days_eq_30 : days = 30)
  (lime_cost_per_3_eq_1 : lime_cost_per_3 = 1)
  (mocktails_per_day_eq_1 : mocktails_per_day = 1)
  (lime_juice_per_lime_tbsp_eq_2 : lime_juice_per_lime_tbsp = 2)
  (lime_juice_per_mocktail_tbsp_eq_1 : lime_juice_per_mocktail_tbsp = 1)
  (limes_per_set_eq_3 : limes_per_set = 3) :
  days * mocktails_per_day * lime_juice_per_mocktail_tbsp / lime_juice_per_lime_tbsp / limes_per_set * lime_cost_per_3 = 5 :=
sorry

end Julio_spent_on_limes_l38_38348


namespace difference_qr_l38_38864

-- Definitions of p, q, r in terms of the common multiplier x
def p (x : ℕ) := 3 * x
def q (x : ℕ) := 7 * x
def r (x : ℕ) := 12 * x

-- Given condition that the difference between p and q's share is 4000
def condition1 (x : ℕ) := q x - p x = 4000

-- Theorem stating that the difference between q and r's share is 5000
theorem difference_qr (x : ℕ) (h : condition1 x) : r x - q x = 5000 :=
by
  -- Proof placeholder
  sorry

end difference_qr_l38_38864


namespace count_valid_m_values_l38_38316

theorem count_valid_m_values : ∃ (count : ℕ), count = 72 ∧
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5000 →
     (⌊Real.sqrt m⌋ = ⌊Real.sqrt (m+125)⌋)) ↔ count = 72 :=
by
  sorry

end count_valid_m_values_l38_38316


namespace polynomial_quotient_l38_38312

open Polynomial

noncomputable def dividend : ℤ[X] := 5 * X^4 - 9 * X^3 + 3 * X^2 + 7 * X - 6
noncomputable def divisor : ℤ[X] := X - 1

theorem polynomial_quotient :
  dividend /ₘ divisor = 5 * X^3 - 4 * X^2 + 7 * X + 7 :=
by
  sorry

end polynomial_quotient_l38_38312


namespace complement_union_eq_ge2_l38_38209

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l38_38209


namespace jasmine_milk_gallons_l38_38193

theorem jasmine_milk_gallons (G : ℝ) 
  (coffee_cost_per_pound : ℝ) (milk_cost_per_gallon : ℝ) (total_cost : ℝ)
  (coffee_pounds : ℝ) :
  coffee_cost_per_pound = 2.50 →
  milk_cost_per_gallon = 3.50 →
  total_cost = 17 →
  coffee_pounds = 4 →
  total_cost - coffee_pounds * coffee_cost_per_pound = G * milk_cost_per_gallon →
  G = 2 :=
by
  intros
  sorry

end jasmine_milk_gallons_l38_38193


namespace mrs_hilt_total_distance_l38_38039

def total_distance_walked (d n : ℕ) : ℕ := 2 * d * n

theorem mrs_hilt_total_distance :
  total_distance_walked 30 4 = 240 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_total_distance_l38_38039


namespace no_polyhedron_with_surface_area_2015_l38_38029

theorem no_polyhedron_with_surface_area_2015 : 
  ¬ ∃ (n k : ℤ), 6 * n - 2 * k = 2015 :=
by
  sorry

end no_polyhedron_with_surface_area_2015_l38_38029


namespace total_cost_l38_38593

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end total_cost_l38_38593


namespace opposite_of_2023_is_neg_2023_l38_38694

theorem opposite_of_2023_is_neg_2023 (x : ℝ) (h : x = 2023) : -x = -2023 :=
by
  /- proof begins here, but we are skipping it with sorry -/
  sorry

end opposite_of_2023_is_neg_2023_l38_38694


namespace cookie_cost_1_l38_38945

theorem cookie_cost_1 (C : ℝ) 
  (h1 : ∀ c, c > 0 → 1.2 * c = c + 0.2 * c)
  (h2 : 50 * (1.2 * C) = 60) :
  C = 1 :=
by
  sorry

end cookie_cost_1_l38_38945


namespace prime_difference_fourth_powers_is_not_prime_l38_38295

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_difference_fourth_powers_is_not_prime (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) : 
  ¬ is_prime (p^4 - q^4) :=
sorry

end prime_difference_fourth_powers_is_not_prime_l38_38295


namespace length_of_rope_l38_38271

-- Define the given conditions
variable (L : ℝ)
variable (h1 : 0.6 * L = 0.69)

-- The theorem to prove
theorem length_of_rope (L : ℝ) (h1 : 0.6 * L = 0.69) : L = 1.15 :=
by
  sorry

end length_of_rope_l38_38271


namespace tims_seashells_now_l38_38064

def initial_seashells : ℕ := 679
def seashells_given_away : ℕ := 172

theorem tims_seashells_now : (initial_seashells - seashells_given_away) = 507 :=
by
  sorry

end tims_seashells_now_l38_38064


namespace farm_corn_cobs_l38_38111

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end farm_corn_cobs_l38_38111


namespace ratio_is_7_to_10_l38_38736

-- Given conditions in the problem translated to Lean definitions
def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 10 * leopards
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := 670
def other_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + alligators
def cheetahs : ℕ := total_animals - other_animals

-- The ratio of cheetahs to snakes to be proven
def ratio_cheetahs_to_snakes (cheetahs snakes : ℕ) : ℚ := cheetahs / snakes

theorem ratio_is_7_to_10 : ratio_cheetahs_to_snakes cheetahs snakes = 7 / 10 :=
by
  sorry

end ratio_is_7_to_10_l38_38736


namespace distinct_roots_and_ratios_l38_38163

open Real

theorem distinct_roots_and_ratios (a b : ℝ) (h1 : a^2 - 3*a - 1 = 0) (h2 : b^2 - 3*b - 1 = 0) (h3 : a ≠ b) :
  b/a + a/b = -11 :=
sorry

end distinct_roots_and_ratios_l38_38163


namespace complement_union_eq_ge_two_l38_38223

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l38_38223


namespace num_pos_3_digits_div_by_7_l38_38177

theorem num_pos_3_digits_div_by_7 : 
  let lower_bound := 100
  let upper_bound := 999 
  let divisor := 7
  let smallest_3_digit := 105 -- or can be computed explicitly by: (lower_bound + divisor - 1) / divisor * divisor
  let largest_3_digit := 994  -- or can be computed explicitly by: upper_bound / divisor * divisor
  List.length (List.filter (λ n, n % divisor = 0) (List.range' smallest_3_digit (largest_3_digit + 1))) = 128 :=
by
  sorry

end num_pos_3_digits_div_by_7_l38_38177


namespace average_running_time_l38_38646

variable (s : ℕ) -- Number of seventh graders

-- let sixth graders run 20 minutes per day
-- let seventh graders run 18 minutes per day
-- let eighth graders run 15 minutes per day
-- sixth graders = 3 * seventh graders
-- eighth graders = 2 * seventh graders

def sixthGradersRunningTime : ℕ := 20 * (3 * s)
def seventhGradersRunningTime : ℕ := 18 * s
def eighthGradersRunningTime : ℕ := 15 * (2 * s)

def totalRunningTime : ℕ := sixthGradersRunningTime s + seventhGradersRunningTime s + eighthGradersRunningTime s
def totalStudents : ℕ := 3 * s + s + 2 * s

theorem average_running_time : totalRunningTime s / totalStudents s = 18 :=
by sorry

end average_running_time_l38_38646


namespace find_beta_l38_38922

variable (α β : ℝ)

theorem find_beta 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) : β = Real.pi / 3 := sorry

end find_beta_l38_38922


namespace minimum_price_to_cover_costs_l38_38718

variable (P : ℝ)

-- Conditions
def prod_cost_A := 80
def ship_cost_A := 2
def prod_cost_B := 60
def ship_cost_B := 3
def fixed_costs := 16200
def units_A := 200
def units_B := 300

-- Cost calculations
def total_cost_A := units_A * prod_cost_A + units_A * ship_cost_A
def total_cost_B := units_B * prod_cost_B + units_B * ship_cost_B
def total_costs := total_cost_A + total_cost_B + fixed_costs

-- Revenue requirement
def revenue (P_A P_B : ℝ) := units_A * P_A + units_B * P_B

theorem minimum_price_to_cover_costs :
  (units_A + units_B) * P ≥ total_costs ↔ P ≥ 103 :=
sorry

end minimum_price_to_cover_costs_l38_38718


namespace compute_infinite_series_l38_38512

noncomputable def infinite_series (c d : ℝ) (hcd : c > d) : ℝ :=
  ∑' n, 1 / (((n - 1 : ℝ) * c - (n - 2 : ℝ) * d) * (n * c - (n - 1 : ℝ) * d))

theorem compute_infinite_series (c d : ℝ) (hcd : c > d) :
  infinite_series c d hcd = 1 / ((c - d) * d) :=
by
  sorry

end compute_infinite_series_l38_38512


namespace problem_statement_l38_38225

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l38_38225


namespace mens_wages_l38_38871

theorem mens_wages
  (M : ℝ) (WW : ℝ) (B : ℝ)
  (h1 : 5 * M = WW)
  (h2 : WW = 8 * B)
  (h3 : 5 * M + WW + 8 * B = 60) :
  5 * M = 30 :=
by
  sorry

end mens_wages_l38_38871


namespace simplify_and_evaluate_expression_l38_38250

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = 3) :
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l38_38250


namespace doug_money_l38_38094

def money_problem (J D B: ℝ) : Prop :=
  J + D + B = 68 ∧
  J = 2 * B ∧
  J = (3 / 4) * D

theorem doug_money (J D B: ℝ) (h: money_problem J D B): D = 36.27 :=
by sorry

end doug_money_l38_38094


namespace rectangles_excluding_squares_in_5x5_grid_l38_38172

-- Definition for a grid and counting rectangles excluding squares in a 5x5 grid
def count_rectangles_excluding_squares : Nat :=
  let total_rectangles := (Nat.choose 5 2) * (Nat.choose 5 2)
  let total_squares := (4 * (5 - 1)^2) + (3 * (5 - 2)^2) + (2 * (5 - 3)^2) + (1 * (5 - 4)^2) 
  total_rectangles - total_squares

-- Statement of the theorem
theorem rectangles_excluding_squares_in_5x5_grid :
  count_rectangles_excluding_squares = 70 :=
begin
  -- Proof is omitted
  sorry
end

end rectangles_excluding_squares_in_5x5_grid_l38_38172


namespace bottle_caps_per_friend_l38_38296

-- The context where Catherine has 18 bottle caps
def bottle_caps : Nat := 18

-- Catherine distributes these bottle caps among 6 friends
def number_of_friends : Nat := 6

-- We need to prove that each friend gets 3 bottle caps
theorem bottle_caps_per_friend : bottle_caps / number_of_friends = 3 :=
by sorry

end bottle_caps_per_friend_l38_38296


namespace point_location_l38_38624

variables {A B C m n : ℝ}

theorem point_location (h1 : A > 0) (h2 : B < 0) (h3 : A * m + B * n + C < 0) : 
  -- Statement: the point P(m, n) is on the upper right side of the line Ax + By + C = 0
  true :=
sorry

end point_location_l38_38624


namespace dress_assignment_l38_38145

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l38_38145


namespace hyperbola_focal_length_l38_38380

theorem hyperbola_focal_length (m : ℝ) 
  (h0 : (∀ x y, x^2 / 16 - y^2 / m = 1)) 
  (h1 : (2 * Real.sqrt (16 + m) = 4 * Real.sqrt 5)) : 
  m = 4 := 
by sorry

end hyperbola_focal_length_l38_38380


namespace ratio_of_border_to_tile_l38_38645

variable {s d : ℝ}

theorem ratio_of_border_to_tile (h1 : 900 = 30 * 30)
  (h2 : 0.81 = (900 * s^2) / (30 * s + 60 * d)^2) :
  d / s = 1 / 18 := by {
  sorry }

end ratio_of_border_to_tile_l38_38645


namespace intervals_of_monotonicity_range_of_values_for_a_l38_38008

noncomputable def f (a x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioi 0, a ≤ -1 → deriv (f a) x > 0) ∧
  (∀ x ∈ Set.Ioc 0 (1 + a), -1 < a → deriv (f a) x < 0) ∧
  (∀ x ∈ Set.Ioi (1 + a), -1 < a → deriv (f a) x > 0) :=
sorry

theorem range_of_values_for_a (a : ℝ) (e : ℝ) (h : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 1 e, f a x ≤ 0) → (a ≤ -2 ∨ a ≥ (e^2 + 1) / (e - 1)) :=
sorry

end intervals_of_monotonicity_range_of_values_for_a_l38_38008


namespace evaluate_polynomial_at_4_l38_38068

-- Define the polynomial f
noncomputable def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Given x = 4, prove that f(4) = 1559
theorem evaluate_polynomial_at_4 : f 4 = 1559 :=
  by
    sorry

end evaluate_polynomial_at_4_l38_38068


namespace equilateral_triangle_of_roots_of_unity_l38_38517

open Complex

/-- Given three distinct non-zero complex numbers z1, z2, z3 such that z1 * z2 = z3 ^ 2 and z2 * z3 = z1 ^ 2.
Prove that if z2 = z1 * alpha, then alpha is a cube root of unity and the points corresponding to z1, z2, z3
form an equilateral triangle in the complex plane -/
theorem equilateral_triangle_of_roots_of_unity {z1 z2 z3 : ℂ} (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : z3 ≠ 0)
  (h_distinct : z1 ≠ z2 ∧ z2 ≠ z3 ∧ z1 ≠ z3)
  (h1_2 : z1 * z2 = z3 ^ 2) (h2_3 : z2 * z3 = z1 ^ 2) (alpha : ℂ) (hz2 : z2 = z1 * alpha) :
  alpha^3 = 1 ∧ ∃ (w1 w2 w3 : ℂ), (w1 = z1) ∧ (w2 = z2) ∧ (w3 = z3) ∧ ((w1, w2, w3) = (z1, z1 * α, z3) 
  ∨ (w1, w2, w3) = (z3, z1, z1 * α) ∨ (w1, w2, w3) = (z1 * α, z3, z1)) 
  ∧ dist w1 w2 = dist w2 w3 ∧ dist w2 w3 = dist w3 w1 := sorry

end equilateral_triangle_of_roots_of_unity_l38_38517


namespace present_population_l38_38256

theorem present_population (P : ℝ) (h : 1.04 * P = 1289.6) : P = 1240 :=
by
  sorry

end present_population_l38_38256


namespace compute_g_five_times_l38_38197

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then - x^3 else x + 10

theorem compute_g_five_times (x : ℤ) (h : x = 2) : g (g (g (g (g x)))) = -8 := by
  sorry

end compute_g_five_times_l38_38197


namespace find_f_zero_l38_38495

theorem find_f_zero (f : ℝ → ℝ) (h : ∀ x, f ((x + 1) / (x - 1)) = x^2 + 3) : f 0 = 4 :=
by
  -- The proof goes here.
  sorry

end find_f_zero_l38_38495


namespace find_x_l38_38019

variables (x y : ℚ)  -- Declare x and y as rational numbers

theorem find_x (h1 : 3 * x - 2 * y = 7) (h2 : 2 * x + 3 * y = 8) : 
  x = 37 / 13 :=
by
  sorry

end find_x_l38_38019


namespace find_C_l38_38879

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 320) : 
  C = 20 := 
by 
  sorry

end find_C_l38_38879


namespace no_nonzero_integer_solution_l38_38967

theorem no_nonzero_integer_solution 
(a b c n : ℤ) (h : 6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * n ^ 2) : 
a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
sorry

end no_nonzero_integer_solution_l38_38967


namespace intersection_l38_38488

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def setB : Set ℝ := { x | x^2 - 2 * x ≥ 0 }

theorem intersection: setA ∩ setB = { x : ℝ | x ≤ 0 } := by
  sorry

end intersection_l38_38488


namespace intersected_squares_and_circles_l38_38120

def is_intersected_by_line (p q : ℕ) : Prop :=
  p = q

def total_intersections : ℕ := 504 * 2

theorem intersected_squares_and_circles :
  total_intersections = 1008 :=
by
  sorry

end intersected_squares_and_circles_l38_38120


namespace cos_315_eq_sqrt2_div_2_l38_38430

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l38_38430


namespace rectangle_area_l38_38371

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end rectangle_area_l38_38371


namespace dress_assignment_l38_38142

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l38_38142


namespace towels_after_a_week_l38_38090

theorem towels_after_a_week 
  (initial_green : ℕ) (initial_white : ℕ) (initial_blue : ℕ) 
  (daily_green : ℕ) (daily_white : ℕ) (daily_blue : ℕ) 
  (days : ℕ) 
  (H1 : initial_green = 35)
  (H2 : initial_white = 21)
  (H3 : initial_blue = 15)
  (H4 : daily_green = 3)
  (H5 : daily_white = 1)
  (H6 : daily_blue = 1)
  (H7 : days = 7) :
  (initial_green - daily_green * days) + (initial_white - daily_white * days) + (initial_blue - daily_blue * days) = 36 :=
by 
  sorry

end towels_after_a_week_l38_38090


namespace solve_derivative_equation_l38_38058

theorem solve_derivative_equation :
  (∃ n : ℤ, ∀ x,
    x = 2 * n * Real.pi ∨
    x = 2 * n * Real.pi - 2 * Real.arctan (3 / 5)) :=
by
  sorry

end solve_derivative_equation_l38_38058


namespace intersection_complement_l38_38785

open Set

variable (U A B : Set ℕ)

-- Definitions based on conditions given in the problem
def universal_set : Set ℕ := {1, 2, 3, 4, 5}
def set_A : Set ℕ := {2, 4}
def set_B : Set ℕ := {4, 5}

-- Proof statement
theorem intersection_complement :
  A = set_A → 
  B = set_B → 
  U = universal_set → 
  (A ∩ (U \ B)) = {2} := 
by
  intros hA hB hU
  sorry

end intersection_complement_l38_38785


namespace greatest_integer_x_l38_38075

theorem greatest_integer_x (x : ℤ) : 
  (∃ n : ℤ, (x^2 + 4*x + 10) = n * (x - 4)) → x ≤ 46 := 
by
  sorry

end greatest_integer_x_l38_38075


namespace maximum_value_expression_l38_38391

theorem maximum_value_expression (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 + 3 * a * b)) + Real.sqrt (Real.sqrt (b^2 + 3 * b * c)) +
   Real.sqrt (Real.sqrt (c^2 + 3 * c * d)) + Real.sqrt (Real.sqrt (d^2 + 3 * d * a))) ≤ 4 * Real.sqrt 2 :=
by 
  sorry

end maximum_value_expression_l38_38391


namespace part_a_sequence_l38_38573

def circle_sequence (n m : ℕ) : List ℕ :=
  List.replicate m 1 -- Placeholder: Define the sequence computation properly

theorem part_a_sequence :
  circle_sequence 5 12 = [1, 6, 11, 4, 9, 2, 7, 12, 5, 10, 3, 8, 1] := 
sorry

end part_a_sequence_l38_38573


namespace isosceles_triangle_l38_38340

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b^2 - 2 * b * c + c^2) = 0) : 
  (a = b) ∨ (b = c) :=
by sorry

end isosceles_triangle_l38_38340


namespace domain_of_f_l38_38855

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = x} = {x : ℝ | x ≠ 6} := by
  sorry

end domain_of_f_l38_38855


namespace maria_money_difference_l38_38084

-- Defining constants for Maria's money when she arrived and left the fair
def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

-- Calculating the expected difference
def expected_difference : ℕ := 71

-- Statement: proving that the difference between money_at_arrival and money_at_departure is expected_difference
theorem maria_money_difference : money_at_arrival - money_at_departure = expected_difference := by
  sorry

end maria_money_difference_l38_38084


namespace num_words_sum_l38_38036

/-
  Definitions:
  - word_kasha is a multiset with the letters "К", "А", "Ш", "А".
  - word_hleb is a set with the letters "Х", "Л", "Е", "Б".
  - num_distinct_perms is the function to calculate permutations of distinct items.
  - num_perms_with_repetition is the function to calculate permutations of multiset.
-/

def word_kasha : Multiset Char := {'К', 'А', 'Ш', 'А'}
def word_hleb : Finset Char := {'Х', 'Л', 'Е', 'Б'}

def num_distinct_perms (s : Finset Char) : ℕ :=
  (Finset.card s).factorial

def num_perms_with_repetition (m : Multiset Char) : ℕ :=
  Multiset.card m ! / m.dedup.card.factorial

theorem num_words_sum : 
  num_distinct_perms word_hleb + num_perms_with_repetition word_kasha = 36 :=
by {
  sorry
}

end num_words_sum_l38_38036


namespace compare_abc_l38_38515

open Real

theorem compare_abc
  (a b c : ℝ)
  (ha : 0 < a ∧ a < π / 2)
  (hb : 0 < b ∧ b < π / 2)
  (hc : 0 < c ∧ c < π / 2)
  (h1 : cos a = a)
  (h2 : sin (cos b) = b)
  (h3 : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end compare_abc_l38_38515


namespace multiplication_24_12_l38_38574

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

end multiplication_24_12_l38_38574


namespace remaining_tickets_l38_38884

def initial_tickets : ℝ := 49.0
def lost_tickets : ℝ := 6.0
def spent_tickets : ℝ := 25.0

theorem remaining_tickets : initial_tickets - lost_tickets - spent_tickets = 18.0 := by
  sorry

end remaining_tickets_l38_38884


namespace complement_union_eq_ge_two_l38_38224

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l38_38224


namespace part_one_min_f_value_part_two_range_a_l38_38907

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x + a|

theorem part_one_min_f_value (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≥ (3/2) :=
  sorry

theorem part_two_range_a (a : ℝ) : (11/2 < a) ∧ (a < 4.5) :=
  sorry

end part_one_min_f_value_part_two_range_a_l38_38907


namespace new_student_weight_l38_38710

theorem new_student_weight (avg_weight : ℝ) (x : ℝ) :
  (avg_weight * 10 - 120) = ((avg_weight - 6) * 10 + x) → x = 60 :=
by
  intro h
  -- The proof would go here, but it's skipped.
  sorry

end new_student_weight_l38_38710


namespace exists_inequality_l38_38035

theorem exists_inequality (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (hx1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1) 
  (h_n : 2 ≤ n) : 
  ∃ i : Fin n, x i * (1 - x (i + 1)) ≥ (1 / 4) * x 0 * (1 - x n) :=
sorry

end exists_inequality_l38_38035


namespace complete_residue_system_mod_l38_38246

open Nat

theorem complete_residue_system_mod (m : ℕ) (x : Fin m → ℕ)
  (h : ∀ i j : Fin m, i ≠ j → ¬ ((x i) % m = (x j) % m)) :
  (Finset.image (λ i => x i % m) (Finset.univ : Finset (Fin m))) = Finset.range m :=
by
  -- Skipping the proof steps.
  sorry

end complete_residue_system_mod_l38_38246


namespace inequality_half_l38_38924

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end inequality_half_l38_38924


namespace derivative_at_pi_l38_38328

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_at_pi :
  deriv f π = -1 / (π^2) :=
sorry

end derivative_at_pi_l38_38328


namespace find_alpha_beta_sum_l38_38165

theorem find_alpha_beta_sum
  (a : ℝ) (α β φ : ℝ)
  (h1 : 3 * Real.sin α + 4 * Real.cos α = a)
  (h2 : 3 * Real.sin β + 4 * Real.cos β = a)
  (h3 : α ≠ β)
  (h4 : 0 < α ∧ α < 2 * Real.pi)
  (h5 : 0 < β ∧ β < 2 * Real.pi)
  (hφ : φ = Real.arcsin (4/5)) :
  α + β = Real.pi - 2 * φ ∨ α + β = 3 * Real.pi - 2 * φ :=
by
  sorry

end find_alpha_beta_sum_l38_38165


namespace only_one_correct_guess_l38_38502

-- Define the contestants
inductive Contestant : Type
| person : ℕ → Contestant

def A_win_first (c: Contestant) : Prop :=
c = Contestant.person 4 ∨ c = Contestant.person 5

def B_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 3 

def C_win_first (c: Contestant) : Prop :=
c = Contestant.person 1 ∨ c = Contestant.person 2 ∨ c = Contestant.person 6

def D_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 4 ∧ c ≠ Contestant.person 5 ∧ c ≠ Contestant.person 6

-- The main theorem: Only one correct guess among A, B, C, and D
theorem only_one_correct_guess (win: Contestant) :
  (A_win_first win ↔ false) ∧ (B_not_win_first win ↔ false) ∧ (C_win_first win ↔ false) ∧ D_not_win_first win
:=
by
  sorry

end only_one_correct_guess_l38_38502


namespace smallest_Q_value_l38_38059

noncomputable def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 + 4*x + 6

theorem smallest_Q_value :
  min (Q (-1)) (min (6) (min (1 + 2 - 1 + 4 + 6) (sorry))) = Q (-1) :=
by
  sorry

end smallest_Q_value_l38_38059


namespace pool_width_l38_38765

variable (length : ℝ) (depth : ℝ) (chlorine_per_cubic_foot : ℝ) (chlorine_cost_per_quart : ℝ) (total_spent : ℝ)
variable (w : ℝ)

-- defining the conditions
def pool_conditions := length = 10 ∧ depth = 6 ∧ chlorine_per_cubic_foot = 120 ∧ chlorine_cost_per_quart = 3 ∧ total_spent = 12

-- goal statement
theorem pool_width : pool_conditions length depth chlorine_per_cubic_foot chlorine_cost_per_quart total_spent →
  w = 8 :=
by
  sorry

end pool_width_l38_38765


namespace smallest_mu_exists_l38_38453

theorem smallest_mu_exists (a b c d : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :
  ∃ μ : ℝ, μ = (3 / 2) - (3 / (4 * Real.sqrt 2)) ∧ 
    (a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + μ * b^2 * c + c^2 * d) :=
by
  sorry

end smallest_mu_exists_l38_38453


namespace sum_of_first_15_terms_l38_38942

theorem sum_of_first_15_terms (a : ℕ → ℝ) (r : ℝ)
    (h_geom : ∀ n, a (n + 1) = a n * r)
    (h1 : a 1 + a 2 + a 3 = 1)
    (h2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 +
   a 10 + a 11 + a 12 + a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_first_15_terms_l38_38942


namespace contact_probability_l38_38759

theorem contact_probability (p : ℝ) :
  let m := 6 in
  let n := 7 in
  let number_of_pairs := m * n in
  1 - (1 - p) ^ number_of_pairs = 1 - (1 - p) ^ 42 :=
by
  let m := 6
  let n := 7
  let number_of_pairs := m * n
  have h1 : number_of_pairs = 42 := by norm_num
  rw h1
  sorry

end contact_probability_l38_38759


namespace solution_set_of_inequality_l38_38387

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end solution_set_of_inequality_l38_38387


namespace total_cost_l38_38594

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end total_cost_l38_38594


namespace grayson_unanswered_l38_38915

noncomputable def unanswered_questions : ℕ :=
  let total_questions := 200
  let first_set_questions := 50
  let first_set_time := first_set_questions * 1 -- 1 minute per question
  let second_set_questions := 50
  let second_set_time := second_set_questions * (90 / 60) -- convert 90 seconds to minutes
  let third_set_questions := 25
  let third_set_time := third_set_questions * 2 -- 2 minutes per question
  let total_answered_time := first_set_time + second_set_time + third_set_time
  let total_time_available := 4 * 60 -- 4 hours in minutes 
  let unanswered := total_questions - (first_set_questions + second_set_questions + third_set_questions)
  unanswered

theorem grayson_unanswered : unanswered_questions = 75 := 
by 
  sorry

end grayson_unanswered_l38_38915


namespace positive_difference_of_squares_l38_38553

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l38_38553


namespace sum_infinite_series_eq_half_l38_38607

theorem sum_infinite_series_eq_half :
  (∑' n : ℕ, (n^5 + 2*n^3 + 5*n^2 + 20*n + 20) / (2^(n + 1) * (n^5 + 5))) = 1 / 2 := 
sorry

end sum_infinite_series_eq_half_l38_38607


namespace susan_spaces_to_win_l38_38974

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end susan_spaces_to_win_l38_38974


namespace total_amount_shared_l38_38411

theorem total_amount_shared (a b c d : ℝ) (h1 : a = (1/3) * (b + c + d)) 
    (h2 : b = (2/7) * (a + c + d)) (h3 : c = (4/9) * (a + b + d)) 
    (h4 : d = (5/11) * (a + b + c)) (h5 : a = b + 20) (h6 : c = d - 15) 
    (h7 : (a + b + c + d) % 10 = 0) : a + b + c + d = 1330 :=
by
  sorry

end total_amount_shared_l38_38411


namespace point_distance_to_focus_of_parabola_with_focus_distance_l38_38777

def parabola_with_focus_distance (focus_distance : ℝ) (p : ℝ × ℝ) : Prop :=
  let f := (0, focus_distance)
  let directrix := -focus_distance
  let (x, y) := p
  let distance_to_focus := Real.sqrt ((x - 0)^2 + (y - focus_distance)^2)
  let distance_to_directrix := abs (y - directrix)
  distance_to_focus = distance_to_directrix

theorem point_distance_to_focus_of_parabola_with_focus_distance 
  (focus_distance : ℝ) (y_axis_distance : ℝ) (p : ℝ × ℝ)
  (h_focus_distance : focus_distance = 4)
  (h_y_axis_distance : abs (p.1) = 1) :
  parabola_with_focus_distance focus_distance p →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - focus_distance)^2) = 5 :=
by
  sorry

end point_distance_to_focus_of_parabola_with_focus_distance_l38_38777


namespace equilateral_triangle_perimeter_l38_38548

theorem equilateral_triangle_perimeter (a P : ℕ) 
  (h1 : 2 * a + 10 = 40)  -- Condition: perimeter of isosceles triangle is 40
  (h2 : P = 3 * a) :      -- Definition of perimeter of equilateral triangle
  P = 45 :=               -- Expected result
by
  sorry

end equilateral_triangle_perimeter_l38_38548


namespace cost_per_person_l38_38048

theorem cost_per_person (total_cost : ℕ) (num_people : ℕ) (h1 : total_cost = 30000) (h2 : num_people = 300) : total_cost / num_people = 100 := by
  -- No proof provided, only the theorem statement
  sorry

end cost_per_person_l38_38048


namespace sphere_radius_vol_eq_area_l38_38061

noncomputable def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r ^ 3
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

theorem sphere_radius_vol_eq_area (r : ℝ) :
  volume r = surface_area r → r = 3 :=
by
  sorry

end sphere_radius_vol_eq_area_l38_38061


namespace system_of_equations_solve_l38_38911

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end system_of_equations_solve_l38_38911


namespace price_reduction_equation_l38_38652

theorem price_reduction_equation (x : ℝ) :
  63800 * (1 - x)^2 = 3900 :=
sorry

end price_reduction_equation_l38_38652


namespace pencils_given_away_l38_38601

-- Define the basic values and conditions
def initial_pencils : ℕ := 39
def bought_pencils : ℕ := 22
def final_pencils : ℕ := 43

-- Let x be the number of pencils Brian gave away
variable (x : ℕ)

-- State the theorem we need to prove
theorem pencils_given_away : (initial_pencils - x) + bought_pencils = final_pencils → x = 18 := by
  sorry

end pencils_given_away_l38_38601


namespace complex_number_real_imaginary_opposite_l38_38337

theorem complex_number_real_imaginary_opposite (a : ℝ) (i : ℂ) (comp : z = (1 - a * i) * i):
  (z.re = -z.im) → a = 1 :=
by 
  sorry

end complex_number_real_imaginary_opposite_l38_38337


namespace vector_properties_l38_38935

-- Definitions of vectors
def vec_a : ℝ × ℝ := (3, 11)
def vec_b : ℝ × ℝ := (-1, -4)
def vec_c : ℝ × ℝ := (1, 3)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Linear combination of vector scaling and addition
def vec_sub_scal (u v : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (u.1 - k * v.1, u.2 - k * v.2)

-- Check if two vectors are parallel
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2

-- Lean statement for the proof problem
theorem vector_properties :
  dot_product vec_a vec_b = -47 ∧
  vec_sub_scal vec_a vec_b 2 = (5, 19) ∧
  dot_product (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2) vec_c ≠ 0 ∧
  parallel (vec_sub_scal vec_a vec_c 1) vec_b :=
by sorry

end vector_properties_l38_38935


namespace total_students_l38_38421

-- Defining the conditions
variable (H : ℕ) -- Number of students who ordered hot dogs
variable (students_ordered_burgers : ℕ) -- Number of students who ordered burgers

-- Given conditions
def burger_condition := students_ordered_burgers = 30
def hotdog_condition := students_ordered_burgers = 2 * H

-- Theorem to prove the total number of students
theorem total_students (H : ℕ) (students_ordered_burgers : ℕ) 
  (h1 : burger_condition students_ordered_burgers) 
  (h2 : hotdog_condition students_ordered_burgers H) : 
  students_ordered_burgers + H = 45 := 
by
  sorry

end total_students_l38_38421


namespace isosceles_triangle_base_length_l38_38383

theorem isosceles_triangle_base_length
  (perimeter_eq_triangle : ℕ)
  (perimeter_isosceles_triangle : ℕ)
  (side_eq_triangle_isosceles : ℕ)
  (side_eq : side_eq_triangle_isosceles = perimeter_eq_triangle / 3)
  (perimeter_eq : perimeter_isosceles_triangle = 2 * side_eq_triangle_isosceles + 15) :
  15 = perimeter_isosceles_triangle - 2 * side_eq_triangle_isosceles :=
sorry

end isosceles_triangle_base_length_l38_38383


namespace neg_p_is_necessary_but_not_sufficient_for_neg_q_l38_38016

variables (p q : Prop)

-- Given conditions: (p → q) and ¬(q → p)
theorem neg_p_is_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) :
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q) :=
sorry

end neg_p_is_necessary_but_not_sufficient_for_neg_q_l38_38016


namespace quadratic_roots_ratio_l38_38356

theorem quadratic_roots_ratio (r1 r2 p q n : ℝ) (h1 : p = r1 * r2) (h2 : q = -(r1 + r2)) (h3 : p ≠ 0) (h4 : q ≠ 0) (h5 : n ≠ 0) (h6 : r1 ≠ 0) (h7 : r2 ≠ 0) (h8 : x^2 + q * x + p = 0) (h9 : x^2 + p * x + n = 0) :
  n / q = -3 :=
by
  sorry

end quadratic_roots_ratio_l38_38356


namespace sphere_tangent_radius_l38_38696

variables (a b : ℝ) (h : b > a)

noncomputable def radius (a b : ℝ) : ℝ := a * (b - a) / Real.sqrt (b^2 - a^2)

theorem sphere_tangent_radius (a b : ℝ) (h : b > a) : 
  radius a b = a * (b - a) / Real.sqrt (b^2 - a^2) :=
by sorry

end sphere_tangent_radius_l38_38696


namespace solve_quadratic_eq_l38_38428

theorem solve_quadratic_eq (x : ℝ) (h : (x + 5) ^ 2 = 16) : x = -1 ∨ x = -9 :=
sorry

end solve_quadratic_eq_l38_38428


namespace lines_intersect_ellipse_at_2_or_4_points_l38_38396

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1

def line_intersects_ellipse (line : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  ellipse_eq x y ∧ line x y

def number_of_intersections (line1 line2 : ℝ → ℝ → Prop) (n : ℕ) : Prop :=
  ∃ pts : Finset (ℝ × ℝ), (∀ pt ∈ pts, (line_intersects_ellipse line1 pt.1 pt.2 ∨
                                        line_intersects_ellipse line2 pt.1 pt.2)) ∧
                           pts.card = n ∧ 
                           (∀ pt ∈ pts, line1 pt.1 pt.2 ∨ line2 pt.1 pt.2) ∧
                           (∀ (pt1 pt2 : ℝ × ℝ), pt1 ∈ pts → pt2 ∈ pts → pt1 ≠ pt2 → pt1 ≠ pt2)

theorem lines_intersect_ellipse_at_2_or_4_points 
  (line1 line2 : ℝ → ℝ → Prop)
  (h1 : ∃ x1 y1, line1 x1 y1 ∧ ellipse_eq x1 y1)
  (h2 : ∃ x2 y2, line2 x2 y2 ∧ ellipse_eq x2 y2)
  (h3: ¬ ∀ x y, line1 x y ∧ ellipse_eq x y → false)
  (h4: ¬ ∀ x y, line2 x y ∧ ellipse_eq x y → false) :
  ∃ n : ℕ, (n = 2 ∨ n = 4) ∧ number_of_intersections line1 line2 n := sorry

end lines_intersect_ellipse_at_2_or_4_points_l38_38396


namespace students_taking_neither_l38_38675

theorem students_taking_neither (total students_cs students_electronics students_both : ℕ)
  (h1 : total = 60) (h2 : students_cs = 42) (h3 : students_electronics = 35) (h4 : students_both = 25) :
  total - (students_cs - students_both + students_electronics - students_both + students_both) = 8 :=
by {
  sorry
}

end students_taking_neither_l38_38675


namespace linear_function_m_l38_38637

theorem linear_function_m (m : ℤ) (h₁ : |m| = 1) (h₂ : m + 1 ≠ 0) : m = 1 := by
  sorry

end linear_function_m_l38_38637


namespace sectionB_seats_correct_l38_38661

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end sectionB_seats_correct_l38_38661


namespace coterminal_angle_equivalence_l38_38117

theorem coterminal_angle_equivalence (k : ℤ) : ∃ n : ℤ, -463 % 360 = (k * 360 + 257) % 360 :=
by
  sorry

end coterminal_angle_equivalence_l38_38117


namespace focus_of_parabola_l38_38128

-- Define the equation of the given parabola
def given_parabola (x y : ℝ) : Prop := y = - (1 / 8) * x^2

-- Define the condition for the focus of the parabola
def is_focus (focus : ℝ × ℝ) : Prop := focus = (0, -2)

-- State the theorem
theorem focus_of_parabola : ∃ (focus : ℝ × ℝ), given_parabola x y → is_focus focus :=
by
  -- Placeholder proof
  sorry

end focus_of_parabola_l38_38128


namespace cos_315_eq_sqrt2_div_2_l38_38449

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l38_38449


namespace factorize_expression_l38_38306

theorem factorize_expression (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4 * x * y = (x * y - 1 + x + y) * (x * y - 1 - x - y) :=
by sorry

end factorize_expression_l38_38306


namespace shifted_parabola_sum_l38_38085

theorem shifted_parabola_sum (a b c : ℝ) :
  (∃ (a b c : ℝ), ∀ x : ℝ, 3 * x^2 + 2 * x - 5 = 3 * (x - 6)^2 + 2 * (x - 6) - 5 → y = a * x^2 + b * x + c) → a + b + c = 60 :=
sorry

end shifted_parabola_sum_l38_38085


namespace rhombus_area_l38_38984

theorem rhombus_area (side diagonal₁ : ℝ) (h_side : side = 20) (h_diagonal₁ : diagonal₁ = 16) : 
  ∃ (diagonal₂ : ℝ), (2 * diagonal₂ * diagonal₂ + 8 * 8 = side * side) ∧ 
  (1 / 2 * diagonal₁ * diagonal₂ = 64 * Real.sqrt 21) := by
  sorry

end rhombus_area_l38_38984


namespace captivating_quadruples_count_l38_38452

theorem captivating_quadruples_count :
  (∃ n : ℕ, n = 682) ↔ 
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d < b + c :=
sorry

end captivating_quadruples_count_l38_38452


namespace part_a_part_b_l38_38567

variable (a b : ℝ)

-- Given conditions
variable (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4)

-- Requirement (a): Prove that a > b
theorem part_a : a > b := by 
  sorry

-- Requirement (b): Prove that a^2 + b^2 ≥ 2
theorem part_b : a^2 + b^2 ≥ 2 := by 
  sorry

end part_a_part_b_l38_38567


namespace dress_assignment_l38_38146

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l38_38146


namespace inequality_div_half_l38_38927

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end inequality_div_half_l38_38927


namespace clyde_picked_bushels_l38_38408

theorem clyde_picked_bushels (weight_per_bushel : ℕ) (weight_per_cob : ℕ) (cobs_picked : ℕ) :
  weight_per_bushel = 56 →
  weight_per_cob = 1 / 2 →
  cobs_picked = 224 →
  cobs_picked * weight_per_cob / weight_per_bushel = 2 :=
by
  intros
  sorry

end clyde_picked_bushels_l38_38408


namespace find_number_l38_38929

theorem find_number (x : ℤ) :
  45 - (x - (37 - (15 - 18))) = 57 → x = 28 :=
by
  sorry

end find_number_l38_38929


namespace arithmetic_sequence_general_term_l38_38026

theorem arithmetic_sequence_general_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 3) :
  ∃ a_n, a_n = a₁ + (n - 1) * d ∧ a_n = 3 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l38_38026


namespace exists_number_added_to_sum_of_digits_gives_2014_l38_38717

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem exists_number_added_to_sum_of_digits_gives_2014 : 
  ∃ (n : ℕ), n + sum_of_digits n = 2014 :=
sorry

end exists_number_added_to_sum_of_digits_gives_2014_l38_38717


namespace mean_score_is_82_l38_38527

noncomputable def mean_score 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : ℝ := 
  (M * m + A * a) / (m + a)

theorem mean_score_is_82 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : 
  mean_score M A m a hM hA hm = 82 := 
    sorry

end mean_score_is_82_l38_38527


namespace spherical_coords_standard_form_l38_38025

theorem spherical_coords_standard_form :
  ∀ (ρ θ φ : ℝ), ρ > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi → 0 ≤ φ ∧ φ ≤ Real.pi →
  (5, (5 * Real.pi) / 7, (11 * Real.pi) / 6) = (ρ, θ, φ) →
  (ρ, (12 * Real.pi) / 7, Real.pi / 6) = (ρ, θ, φ) :=
by 
  intros ρ θ φ hρ hθ hφ h_eq
  sorry

end spherical_coords_standard_form_l38_38025


namespace correct_assignment_l38_38133

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l38_38133


namespace infinitely_many_a_not_prime_l38_38684

theorem infinitely_many_a_not_prime (a: ℤ) (n: ℤ) : ∃ (b: ℤ), b ≥ 0 ∧ (∃ (N: ℕ) (a: ℤ), a = 4*(N:ℤ)^4 ∧ ∀ (n: ℤ), ¬Prime (n^4 + a)) :=
by { sorry }

end infinitely_many_a_not_prime_l38_38684


namespace oil_leakage_calculation_l38_38882

def total_oil_leaked : ℕ := 11687
def oil_leaked_while_worked : ℕ := 5165
def oil_leaked_before_work : ℕ := 6522

theorem oil_leakage_calculation :
  oil_leaked_before_work = total_oil_leaked - oil_leaked_while_work :=
sorry

end oil_leakage_calculation_l38_38882


namespace lcm_of_fractions_l38_38738

-- Definitions based on the problem's conditions
def numerators : List ℕ := [7, 8, 3, 5, 13, 15, 22, 27]
def denominators : List ℕ := [10, 9, 8, 12, 14, 100, 45, 35]

-- LCM and GCD functions for lists of natural numbers
def list_lcm (l : List ℕ) : ℕ := l.foldr lcm 1
def list_gcd (l : List ℕ) : ℕ := l.foldr gcd 0

-- Main proposition
theorem lcm_of_fractions : list_lcm numerators / list_gcd denominators = 13860 :=
by {
  -- to be proven
  sorry
}

end lcm_of_fractions_l38_38738


namespace probability_third_smallest_is_five_l38_38124

open Finset

noncomputable def prob_third_smallest_is_five : ℚ :=
  let total_ways := choose 15 8
  let favorable_ways := (choose 4 2) * (choose 10 5)
  in favorable_ways / total_ways

theorem probability_third_smallest_is_five :
  prob_third_smallest_is_five = 72 / 307 :=
by sorry

end probability_third_smallest_is_five_l38_38124


namespace math_pages_l38_38968

def total_pages := 7
def reading_pages := 2

theorem math_pages : total_pages - reading_pages = 5 := by
  sorry

end math_pages_l38_38968


namespace cycle_price_reduction_l38_38849

theorem cycle_price_reduction (original_price : ℝ) :
  let price_after_first_reduction := original_price * 0.75
  let price_after_second_reduction := price_after_first_reduction * 0.60
  (original_price - price_after_second_reduction) / original_price = 0.55 :=
by
  sorry

end cycle_price_reduction_l38_38849


namespace yang_hui_rect_eq_l38_38369

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end yang_hui_rect_eq_l38_38369


namespace factorize_m_l38_38555

theorem factorize_m (m : ℝ) : m^2 - 4 * m - 5 = (m + 1) * (m - 5) := 
sorry

end factorize_m_l38_38555


namespace correct_statement_l38_38793

-- Define the necessary variables
variables {a b c : ℝ}

-- State the theorem including the condition and the conclusion
theorem correct_statement (h : a > b) : b - c < a - c :=
by linarith


end correct_statement_l38_38793


namespace integer_solutions_of_inequality_count_l38_38791

theorem integer_solutions_of_inequality_count :
  let a := -2 - Real.sqrt 6
  let b := -2 + Real.sqrt 6
  ∃ n, n = 5 ∧ ∀ x : ℤ, x < a ∨ b < x ↔ (4 * x^2 + 16 * x + 15 ≤ 23) → n = 5 :=
by sorry

end integer_solutions_of_inequality_count_l38_38791


namespace c_share_l38_38102

theorem c_share (A B C : ℕ) (h1 : A = B / 2) (h2 : B = C / 2) (h3 : A + B + C = 392) : C = 224 :=
by
  sorry

end c_share_l38_38102


namespace cost_effective_for_3000_cost_equal_at_2500_l38_38803

def cost_company_A (x : Nat) : Nat :=
  2 * x / 10 + 500

def cost_company_B (x : Nat) : Nat :=
  4 * x / 10

theorem cost_effective_for_3000 : cost_company_A 3000 < cost_company_B 3000 := 
by {
  sorry
}

theorem cost_equal_at_2500 : cost_company_A 2500 = cost_company_B 2500 := 
by {
  sorry
}

end cost_effective_for_3000_cost_equal_at_2500_l38_38803


namespace find_value_of_x_squared_plus_y_squared_l38_38012

theorem find_value_of_x_squared_plus_y_squared (x y : ℝ) (h : (x^2 + y^2 + 1)^2 - 4 = 0) : x^2 + y^2 = 1 :=
by
  sorry

end find_value_of_x_squared_plus_y_squared_l38_38012


namespace divisible_by_five_l38_38249

theorem divisible_by_five (a : ℤ) (h : ¬ (5 ∣ a)) :
    (5 ∣ (a^2 - 1)) ↔ ¬ (5 ∣ (a^2 + 1)) :=
by
  -- Begin the proof here (proof not required according to instructions)
  sorry

end divisible_by_five_l38_38249


namespace jack_paycheck_l38_38192

theorem jack_paycheck (P : ℝ) (h1 : 0.15 * 150 + 0.25 * (P - 150) + 30 + 70 / 100 * (P - (0.15 * 150 + 0.25 * (P - 150) + 30)) * 30 / 100 = 50) : P = 242.22 :=
sorry

end jack_paycheck_l38_38192


namespace susan_remaining_spaces_to_win_l38_38975

/-- Susan's board game has 48 spaces. She makes three moves:
 1. She moves forward 8 spaces
 2. She moves forward 2 spaces and then back 5 spaces
 3. She moves forward 6 spaces
 Prove that the remaining spaces she has to move to reach the end is 37.
-/
theorem susan_remaining_spaces_to_win :
  let total_spaces := 48
  let first_turn := 8
  let second_turn := 2 - 5
  let third_turn := 6
  let total_moved := first_turn + second_turn + third_turn
  total_spaces - total_moved = 37 :=
by
  sorry

end susan_remaining_spaces_to_win_l38_38975


namespace algebraic_expression_l38_38324

theorem algebraic_expression (m : ℝ) (hm : m^2 + m - 1 = 0) : 
  m^3 + 2 * m^2 + 2014 = 2015 := 
by
  sorry

end algebraic_expression_l38_38324


namespace maximum_cars_quotient_l38_38361

theorem maximum_cars_quotient
  (car_length : ℕ) (m_speed : ℕ) (half_hour_distance : ℕ) 
  (unit_length : ℕ) (max_units : ℕ) (N : ℕ) :
  (car_length = 5) →
  (half_hour_distance = 10000) →
  (unit_length = 5 * (m_speed + 1)) →
  (max_units = half_hour_distance / unit_length) →
  (N = max_units) →
  (N / 10 = 200) :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end maximum_cars_quotient_l38_38361


namespace incorrect_value_l38_38692

theorem incorrect_value:
  ∀ (n : ℕ) (initial_mean corrected_mean : ℚ) (correct_value incorrect_value : ℚ),
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.5 →
  correct_value = 48 →
  incorrect_value = correct_value - (corrected_mean * n - initial_mean * n) →
  incorrect_value = 23 :=
by
  intros n initial_mean corrected_mean correct_value incorrect_value
  intros h1 h2 h3 h4 h5
  sorry

end incorrect_value_l38_38692


namespace complement_of_angle_A_l38_38899

theorem complement_of_angle_A (A : ℝ) (h : A = 76) : 90 - A = 14 := by
  sorry

end complement_of_angle_A_l38_38899


namespace question_proof_l38_38216

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l38_38216


namespace compare_A_B_l38_38086

noncomputable def A (x : ℝ) := x / (x^2 - x + 1)
noncomputable def B (y : ℝ) := y / (y^2 - y + 1)

theorem compare_A_B (x y : ℝ) (hx : x > y) (hx_val : x = 2.00 * 10^1998 + 4) (hy_val : y = 2.00 * 10^1998 + 2) : 
  A x < B y := 
by 
  sorry

end compare_A_B_l38_38086


namespace find_length_of_field_l38_38875

variables (L : ℝ) -- Length of the field
variables (width_field : ℝ := 55) -- Width of the field, given as 55 meters.
variables (width_path : ℝ := 2.5) -- Width of the path around the field, given as 2.5 meters.
variables (area_path : ℝ := 1200) -- Area of the path, given as 1200 square meters.

theorem find_length_of_field
  (h : area_path = (L + 2 * width_path) * (width_field + 2 * width_path) - L * width_field)
  : L = 180 :=
by sorry

end find_length_of_field_l38_38875


namespace bridge_length_correct_l38_38267

def train_length : ℕ := 256
def train_speed_kmh : ℕ := 72
def crossing_time : ℕ := 20

noncomputable def convert_speed (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600 -- Conversion from km/h to m/s

noncomputable def bridge_length (train_length : ℕ) (speed_m : ℕ) (time_s : ℕ) : ℕ :=
  (speed_m * time_s) - train_length

theorem bridge_length_correct :
  bridge_length train_length (convert_speed train_speed_kmh) crossing_time = 144 :=
by
  sorry

end bridge_length_correct_l38_38267


namespace deepak_present_age_l38_38097

theorem deepak_present_age (x : ℕ) (Rahul_age Deepak_age : ℕ) 
  (h1 : Rahul_age = 4 * x) (h2 : Deepak_age = 3 * x) 
  (h3 : Rahul_age + 4 = 32) : Deepak_age = 21 := by
  sorry

end deepak_present_age_l38_38097


namespace greatest_k_dividing_abcdef_l38_38514

theorem greatest_k_dividing_abcdef {a b c d e f : ℤ}
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = f^2) :
  ∃ k, (∀ a b c d e f, a^2 + b^2 + c^2 + d^2 + e^2 = f^2 → k ∣ (a * b * c * d * e * f)) ∧ k = 24 :=
sorry

end greatest_k_dividing_abcdef_l38_38514


namespace cuckoo_sounds_from_10_to_16_l38_38873

-- Define a function for the cuckoo sounds per hour considering the clock
def cuckoo_sounds (h : ℕ) : ℕ :=
  if h ≤ 12 then h else h - 12

-- Define the total number of cuckoo sounds from 10:00 to 16:00
def total_cuckoo_sounds : ℕ :=
  (List.range' 10 (16 - 10 + 1)).map cuckoo_sounds |>.sum

theorem cuckoo_sounds_from_10_to_16 : total_cuckoo_sounds = 43 := by
  sorry

end cuckoo_sounds_from_10_to_16_l38_38873


namespace determine_values_l38_38478

-- Define variables and conditions
variable {x v w y z : ℕ}

-- Define the conditions
def condition1 := v * x = 8 * 9
def condition2 := y^2 = x^2 + 81
def condition3 := z^2 = 20^2 - x^2
def condition4 := w^2 = 8^2 + v^2
def condition5 := v * 20 = y * 8

-- Theorem to prove
theorem determine_values : 
  x = 12 ∧ y = 15 ∧ z = 16 ∧ v = 6 ∧ w = 10 :=
by
  -- Insert necessary logic or 
  -- produce proof steps here
  sorry

end determine_values_l38_38478


namespace rectangle_quadratic_eq_l38_38286

variable {L W : ℝ}

theorem rectangle_quadratic_eq (h1 : L + W = 15) (h2 : L * W = 2 * W^2) : 
    (∃ x : ℝ, (x - L) * (x - W) = x^2 - 15 * x + 50) :=
by
  sorry

end rectangle_quadratic_eq_l38_38286


namespace range_of_m_real_roots_l38_38933

theorem range_of_m_real_roots (m : ℝ) : 
  (∀ x : ℝ, ∃ k l : ℝ, k = 2*x ∧ l = m - x^2 ∧ k^2 - 4*l ≥ 0) ↔ m ≤ 1 := 
sorry

end range_of_m_real_roots_l38_38933


namespace only_n_equal_1_l38_38130

theorem only_n_equal_1 (n : ℕ) (h : n ≥ 1) : Nat.Prime (9^n - 2^n) ↔ n = 1 := by
  sorry

end only_n_equal_1_l38_38130


namespace find_a_perpendicular_lines_l38_38787

variable (a : ℝ)

theorem find_a_perpendicular_lines :
  (∃ a : ℝ, ∀ x y : ℝ, (a * x - y + 2 * a = 0) ∧ ((2 * a - 1) * x + a * y + a = 0) → a = 0 ∨ a = 1) := 
sorry

end find_a_perpendicular_lines_l38_38787


namespace inequality_proof_l38_38964

theorem inequality_proof
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : c^2 + a * b = a^2 + b^2) :
  c^2 + a * b ≤ a * c + b * c := sorry

end inequality_proof_l38_38964


namespace molecular_weight_of_one_mole_l38_38269

theorem molecular_weight_of_one_mole (molecular_weight_8_moles : ℝ) (h : molecular_weight_8_moles = 992) : 
  molecular_weight_8_moles / 8 = 124 :=
by
  -- proof goes here
  sorry

end molecular_weight_of_one_mole_l38_38269


namespace alpha_value_l38_38028

theorem alpha_value
  (β γ δ α : ℝ) 
  (h1 : β = 100)
  (h2 : γ = 30)
  (h3 : δ = 150)
  (h4 : α + β + γ + 0.5 * γ = 360) : 
  α = 215 :=
by
  sorry

end alpha_value_l38_38028


namespace smallest_positive_debt_l38_38989

theorem smallest_positive_debt :
  ∃ (D : ℕ) (p g : ℤ), 0 < D ∧ D = 350 * p + 240 * g ∧ D = 10 := sorry

end smallest_positive_debt_l38_38989


namespace perfect_square_count_between_20_and_150_l38_38792

theorem perfect_square_count_between_20_and_150 :
  let lower_bound := 20
  let upper_bound := 150
  let smallest_ps := 25
  let largest_ps := 144
  let count_squares (a b : Nat) := b - a
  count_squares 4 12 = 8 := sorry

end perfect_square_count_between_20_and_150_l38_38792


namespace average_members_remaining_l38_38937

theorem average_members_remaining :
  let initial_members := [7, 8, 10, 13, 6, 10, 12, 9]
  let members_leaving := [1, 2, 1, 2, 1, 2, 1, 2]
  let remaining_members := List.map (λ (x, y) => x - y) (List.zip initial_members members_leaving)
  let total_remaining := List.foldl Nat.add 0 remaining_members
  let num_families := initial_members.length
  total_remaining / num_families = 63 / 8 := by
    sorry

end average_members_remaining_l38_38937


namespace inequality_solution_l38_38385

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := |2 - 3 * x| ≥ 4

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ -2/3 ∨ x ≥ 2

-- The theorem that we need to prove
theorem inequality_solution : {x : ℝ | inequality_condition x} = {x : ℝ | solution_set x} :=
by sorry

end inequality_solution_l38_38385


namespace fraction_exceeds_l38_38580

theorem fraction_exceeds (x : ℚ) (h : 64 = 64 * x + 40) : x = 3 / 8 := 
by
  sorry

end fraction_exceeds_l38_38580


namespace no_adjacent_standing_prob_l38_38833

def coin_flip_probability : ℚ :=
  let a2 := 3
  let a3 := 4
  let a4 := a3 + a2
  let a5 := a4 + a3
  let a6 := a5 + a4
  let a7 := a6 + a5
  let a8 := a7 + a6
  let a9 := a8 + a7
  let a10 := a9 + a8
  let favorable_outcomes := a10
  favorable_outcomes / (2 ^ 10)

theorem no_adjacent_standing_prob :
  coin_flip_probability = (123 / 1024 : ℚ) :=
by sorry

end no_adjacent_standing_prob_l38_38833


namespace incorrect_ac_bc_impl_a_b_l38_38564

theorem incorrect_ac_bc_impl_a_b : ∀ (a b c : ℝ), (ac = bc → a = b) ↔ c ≠ 0 :=
by sorry

end incorrect_ac_bc_impl_a_b_l38_38564


namespace john_duck_price_l38_38031

theorem john_duck_price
  (n_ducks : ℕ)
  (cost_per_duck : ℕ)
  (weight_per_duck : ℕ)
  (total_profit : ℕ)
  (total_cost : ℕ)
  (total_weight : ℕ)
  (total_revenue : ℕ)
  (price_per_pound : ℕ)
  (h1 : n_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : total_profit = 300)
  (h5 : total_cost = n_ducks * cost_per_duck)
  (h6 : total_weight = n_ducks * weight_per_duck)
  (h7 : total_revenue = total_cost + total_profit)
  (h8 : price_per_pound = total_revenue / total_weight) :
  price_per_pound = 5 := 
sorry

end john_duck_price_l38_38031


namespace overall_ranking_l38_38878

-- Define the given conditions
def total_participants := 99
def rank_number_theory := 16
def rank_combinatorics := 30
def rank_geometry := 23
def exams := ["geometry", "number_theory", "combinatorics"]
def final_ranking_strategy := "sum_of_scores"

-- Given: best possible rank and worst possible rank should be the same in this specific problem (from solution steps).
def best_possible_rank := 67
def worst_possible_rank := 67

-- Mathematically prove that 100 * best possible rank + worst possible rank = 167
theorem overall_ranking :
  100 * best_possible_rank + worst_possible_rank = 167 :=
by {
  -- Add the "sorry" here to skip the proof, as required:
  sorry
}

end overall_ranking_l38_38878


namespace seq_sum_11_l38_38006

noncomputable def S (n : ℕ) : ℕ := sorry

noncomputable def a (n : ℕ) : ℕ := sorry

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem seq_sum_11 :
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) ∧
  (is_arithmetic_sequence a) ∧
  (3 * (a 2 + a 4) + 2 * (a 6 + a 9 + a 12) = 12) →
  S 11 = 11 :=
by
  sorry

end seq_sum_11_l38_38006


namespace proof_case_a_proof_case_b_l38_38536

noncomputable def proof_problem_a (x y z p q : ℝ) (n : ℕ) 
  (h1 : y = x^n + p*x + q) 
  (h2 : z = y^n + p*y + q) 
  (h3 : x = z^n + p*z + q) : Prop :=
  x^2 * y + y^2 * z + z^2 * x >= x^2 * z + y^2 * x + z^2 * y

theorem proof_case_a (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q) 
  (h2 : z = y^2 + p*y + q) 
  (h3 : x = z^2 + p*z + q) : 
  proof_problem_a x y z p q 2 h1 h2 h3 := 
sorry

theorem proof_case_b (x y z p q : ℝ) 
  (h1 : y = x^2010 + p*x + q) 
  (h2 : z = y^2010 + p*y + q) 
  (h3 : x = z^2010 + p*z + q) : 
  proof_problem_a x y z p q 2010 h1 h2 h3 := 
sorry

end proof_case_a_proof_case_b_l38_38536


namespace Chloe_total_points_l38_38604

-- Define the points scored in each round
def first_round_points : ℕ := 40
def second_round_points : ℕ := 50
def last_round_points : ℤ := -4

-- Define total points calculation
def total_points := first_round_points + second_round_points + last_round_points

-- The final statement to prove
theorem Chloe_total_points : total_points = 86 := by
  -- This proof is to be completed
  sorry

end Chloe_total_points_l38_38604


namespace cos_315_is_sqrt2_div_2_l38_38445

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l38_38445


namespace equal_probabilities_hearts_clubs_l38_38304

/-- Define the total number of cards in a standard deck including two Jokers -/
def total_cards := 52 + 2

/-- Define the counts of specific card types -/
def num_jokers := 2
def num_spades := 13
def num_tens := 4
def num_hearts := 13
def num_clubs := 13

/-- Define the probabilities of drawing specific card types -/
def prob_joker := num_jokers / total_cards
def prob_spade := num_spades / total_cards
def prob_ten := num_tens / total_cards
def prob_heart := num_hearts / total_cards
def prob_club := num_clubs / total_cards

theorem equal_probabilities_hearts_clubs :
  prob_heart = prob_club :=
by
  sorry

end equal_probabilities_hearts_clubs_l38_38304


namespace pythagorean_triples_l38_38088

theorem pythagorean_triples:
  (∃ a b c : ℝ, (a = 1 ∧ b = 2 ∧ c = sqrt 5 ∧ a^2 + b^2 = c^2) ∨
   (a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 ≠ c^2) ∨
   (a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∨
   (a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 ≠ c^2)) ∧
  (∃ a' b' c' : ℝ, a' = 3 ∧ b' = 4 ∧ c' = 5) :=
by
  sorry

end pythagorean_triples_l38_38088


namespace sample_second_grade_l38_38695

theorem sample_second_grade (r1 r2 r3 sample_size : ℕ) (h1 : r1 = 3) (h2 : r2 = 3) (h3 : r3 = 4) (h_sample_size : sample_size = 50) : (r2 * sample_size) / (r1 + r2 + r3) = 15 := by
  sorry

end sample_second_grade_l38_38695


namespace stratified_sampling_l38_38416

theorem stratified_sampling 
  (total_teachers : ℕ)
  (senior_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (junior_teachers : ℕ)
  (sample_size : ℕ)
  (x y z : ℕ) 
  (h1 : total_teachers = 150)
  (h2 : senior_teachers = 45)
  (h3 : intermediate_teachers = 90)
  (h4 : junior_teachers = 15)
  (h5 : sample_size = 30)
  (h6 : x + y + z = sample_size)
  (h7 : x * 10 = sample_size / 5)
  (h8 : y * 10 = (2 * sample_size) / 5)
  (h9 : z * 10 = sample_size / 15) :
  (x, y, z) = (9, 18, 3) := sorry

end stratified_sampling_l38_38416


namespace number_of_distinct_trees_7_vertices_l38_38330

theorem number_of_distinct_trees_7_vertices : ∃ (n : ℕ), n = 7 ∧ (Tree.enumeration n).card = 11 :=
by
  sorry

end number_of_distinct_trees_7_vertices_l38_38330


namespace cost_for_23_days_l38_38930

-- Define the cost structure
def costFirstWeek : ℕ → ℝ := λ days => if days <= 7 then days * 18 else 7 * 18
def costAdditionalDays : ℕ → ℝ := λ days => if days > 7 then (days - 7) * 14 else 0

-- Total cost equation
def totalCost (days : ℕ) : ℝ := costFirstWeek days + costAdditionalDays days

-- Declare the theorem to prove
theorem cost_for_23_days : totalCost 23 = 350 := by
  sorry

end cost_for_23_days_l38_38930


namespace sum_of_three_numbers_l38_38079

theorem sum_of_three_numbers :
  1.35 + 0.123 + 0.321 = 1.794 :=
sorry

end sum_of_three_numbers_l38_38079


namespace axis_of_symmetry_l38_38633

variable (f : ℝ → ℝ)

theorem axis_of_symmetry (h : ∀ x, f x = f (5 - x)) :  ∀ x y, y = f x ↔ (x = 2.5 ∧ y = f 2.5) := 
sorry

end axis_of_symmetry_l38_38633


namespace draw_4_balls_score_at_least_5_l38_38062

theorem draw_4_balls_score_at_least_5:
  let red_points := 2  -- Points for a red ball
  let white_points := 1  -- Points for a white ball
  let total_balls := 10
  let red_balls := 4
  let white_balls := 6
  let draw_count := 4
  let choices := 
    Nat.choose red_balls 4 +         -- Choosing 4 reds from 4 reds
    Nat.choose red_balls 3 *         -- Choosing 3 reds from 4 reds
    Nat.choose white_balls 1 +       -- Choosing 1 white from 6 whites
    Nat.choose red_balls 2 *         -- Choosing 2 reds from 4 reds
    Nat.choose white_balls 2 +       -- Choosing 2 whites from 6 whites
    Nat.choose red_balls 1 *         -- Choosing 1 red from 4 reds
    Nat.choose white_balls 3         -- Choosing 3 whites from 6 whites
  in choices = 195 := sorry

end draw_4_balls_score_at_least_5_l38_38062


namespace binomial_sum_equal_36_l38_38004

theorem binomial_sum_equal_36 (n : ℕ) (h : n > 0) :
  (n + n * (n - 1) / 2 = 36) → n = 8 :=
by
  sorry

end binomial_sum_equal_36_l38_38004


namespace range_of_a_for_quadratic_eq_l38_38336

theorem range_of_a_for_quadratic_eq (a : ℝ) (h : ∀ x : ℝ, ax^2 = (x+1)*(x-1)) : a ≠ 1 :=
by
  sorry

end range_of_a_for_quadratic_eq_l38_38336


namespace seulgi_stack_higher_l38_38920

-- Define the conditions
def num_red_boxes : ℕ := 15
def num_yellow_boxes : ℕ := 20
def height_red_box : ℝ := 4.2
def height_yellow_box : ℝ := 3.3

-- Define the total height for each stack
def total_height_hyunjeong : ℝ := num_red_boxes * height_red_box
def total_height_seulgi : ℝ := num_yellow_boxes * height_yellow_box

-- Lean statement to prove the comparison of their heights
theorem seulgi_stack_higher : total_height_seulgi > total_height_hyunjeong :=
by
  -- Proof will be inserted here
  sorry

end seulgi_stack_higher_l38_38920


namespace two_digit_number_ratio_l38_38345

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b
def swapped_two_digit_number (a b : ℕ) : ℕ := 10 * b + a

theorem two_digit_number_ratio (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) (h_ratio : 6 * two_digit_number a b = 5 * swapped_two_digit_number a b) : 
  two_digit_number a b = 45 :=
by
  sorry

end two_digit_number_ratio_l38_38345


namespace number_of_sides_of_polygon_l38_38635

-- Given definition about angles and polygons
def exterior_angle (sides: ℕ) : ℝ := 30

-- The sum of exterior angles of any polygon
def sum_exterior_angles : ℝ := 360

-- The proof statement
theorem number_of_sides_of_polygon (k : ℕ) 
  (h1 : exterior_angle k = 30) 
  (h2 : sum_exterior_angles = 360):
  k = 12 :=
sorry

end number_of_sides_of_polygon_l38_38635


namespace solution_set_of_inequality_l38_38386

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end solution_set_of_inequality_l38_38386


namespace inequality_sum_sq_A_le_4_sum_sq_a_l38_38769

open BigOperators

variables {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

def A (i : ℕ) : α :=
  if i = 0 then 0 else (i : α) / (i * i + i - 1) * ∑ k in finset.range i, a (k + 1)

theorem inequality_sum_sq_A_le_4_sum_sq_a
  (ha : ∀ i, i ≠ 0 → a i > 0) :
  (∑ k in finset.range n, (A a k) ^ 2) ≤ 4 * (∑ k in finset.range n, (a (k + 1)) ^ 2) :=
sorry

end inequality_sum_sq_A_le_4_sum_sq_a_l38_38769


namespace rationalize_denominator_sum_A_B_C_D_l38_38679

theorem rationalize_denominator :
  (1 / (5 : ℝ)^(1/3) - (2 : ℝ)^(1/3)) = 
  ((25 : ℝ)^(1/3) + (10 : ℝ)^(1/3) + (4 : ℝ)^(1/3)) / (3 : ℝ) := 
sorry

theorem sum_A_B_C_D : 25 + 10 + 4 + 3 = 42 := 
by norm_num

end rationalize_denominator_sum_A_B_C_D_l38_38679


namespace who_wears_which_dress_l38_38137

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l38_38137


namespace least_possible_n_l38_38874

noncomputable def d (n : ℕ) := 105 * n - 90

theorem least_possible_n :
  ∀ n : ℕ, d n > 0 → (45 - (d n + 90) / n = 150) → n ≥ 2 :=
by
  sorry

end least_possible_n_l38_38874


namespace find_k_l38_38484

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

theorem find_k (k : ℝ) (h : f 5 - g 5 k = 24) : k = -16.36 := by
  sorry

end find_k_l38_38484


namespace trader_profit_percentage_l38_38585

-- Definitions for the conditions
def original_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.80 * P
def selling_price (P : ℝ) : ℝ := 0.80 * P * 1.45

-- Theorem statement including the problem's question and the correct answer
theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) : 
  (selling_price P - original_price P) / original_price P * 100 = 16 :=
by
  sorry

end trader_profit_percentage_l38_38585


namespace solve_equation_l38_38687

theorem solve_equation :
  ∃ a b x : ℤ, 
  ((a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0) 
  ↔ (a = -6 ∧ b = -5 ∧ x = -2) :=
by {
  sorry
}

end solve_equation_l38_38687


namespace sum_of_first_three_terms_l38_38378

theorem sum_of_first_three_terms 
  (a d : ℤ) 
  (h1 : a + 4 * d = 15) 
  (h2 : d = 3) : 
  a + (a + d) + (a + 2 * d) = 18 :=
by
  sorry

end sum_of_first_three_terms_l38_38378


namespace length_dg_l38_38680

theorem length_dg (a b k l S : ℕ) (h1 : S = 47 * (a + b)) 
                   (h2 : S = a * k) (h3 : S = b * l) (h4 : b = S / l) 
                   (h5 : a = S / k) (h6 : k * l = 47 * k + 47 * l + 2209) : 
  k = 2256 :=
by sorry

end length_dg_l38_38680


namespace imaginary_part_z1_mul_z2_l38_38490

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end imaginary_part_z1_mul_z2_l38_38490


namespace correct_assignment_l38_38131

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l38_38131


namespace problem_statement_l38_38674

theorem problem_statement :
  ∀ m n : ℕ, (m = 9) → (n = m^2 + 1) → n - m = 73 :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end problem_statement_l38_38674


namespace question_proof_l38_38214

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l38_38214


namespace max_volume_pyramid_l38_38857

theorem max_volume_pyramid 
  (AB AC : ℝ)
  (sin_BAC : ℝ)
  (angle_cond : ∀ (SA SB SC : ℝ), SA = SB ∧ SB = SC ∧ SC = SA → ∀ θ, θ ≤ 60 → true)
  (h : ℝ)
  (V : ℝ)
  (AB_eq : AB = 3)
  (AC_eq : AC = 5)
  (sin_BAC_eq : sin_BAC = 3/5)
  (height_cond : h = (5 * Real.sqrt 3) / 2)
  (volume_cond : V = (1/3) * (1/2 * 3 * 5 * (3/5)) * h) :
  V = (5 * Real.sqrt 174) / 4 := sorry

end max_volume_pyramid_l38_38857


namespace mean_equal_implication_l38_38691

theorem mean_equal_implication (y : ℝ) :
  (7 + 10 + 15 + 23 = 55) →
  (55 / 4 = 13.75) →
  (18 + y + 30 = 48 + y) →
  (48 + y) / 3 = 13.75 →
  y = -6.75 :=
by 
  intros h1 h2 h3 h4
  -- The steps would be applied here to prove y = -6.75
  sorry

end mean_equal_implication_l38_38691


namespace problem1_problem2_l38_38152

variable (a b : ℝ)

theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  1/a + 1/(b+1) ≥ 4/5 := by
  sorry

theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  4/(a*b) + a/b ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end problem1_problem2_l38_38152


namespace max_blue_cells_n2_max_blue_cells_n25_l38_38251

noncomputable def max_blue_cells (table_size n : ℕ) : ℕ :=
  if h : (table_size = 50 ∧ n = 2) then 2450
  else if h : (table_size = 50 ∧ n = 25) then 1300
  else 0 -- Default case that should not happen for this problem

theorem max_blue_cells_n2 : max_blue_cells 50 2 = 2450 := 
by
  sorry

theorem max_blue_cells_n25 : max_blue_cells 50 25 = 1300 :=
by
  sorry

end max_blue_cells_n2_max_blue_cells_n25_l38_38251


namespace larger_gate_width_is_10_l38_38599

-- Define the conditions as constants
def garden_length : ℝ := 225
def garden_width : ℝ := 125
def small_gate_width : ℝ := 3
def total_fencing_length : ℝ := 687

-- Define the perimeter function for a rectangle
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

-- Define the width of the larger gate
def large_gate_width : ℝ :=
  let total_perimeter := perimeter garden_length garden_width
  let remaining_fencing := total_perimeter - total_fencing_length
  remaining_fencing - small_gate_width

-- State the theorem
theorem larger_gate_width_is_10 : large_gate_width = 10 := by
  -- skipping proof part
  sorry

end larger_gate_width_is_10_l38_38599


namespace find_n_l38_38341

theorem find_n :
  ∃ n : ℕ, ∀ (a b c : ℕ), a + b + c = 200 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (n = a + b * c) ∧ (n = b + c * a) ∧ (n = c + a * b) → n = 199 :=
by {
  sorry
}

end find_n_l38_38341


namespace susan_spaces_to_win_l38_38973

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end susan_spaces_to_win_l38_38973


namespace range_of_a_l38_38622

open Set Real

theorem range_of_a (a : ℝ) (α : ℝ → Prop) (β : ℝ → Prop) (hα : ∀ x, α x ↔ x ≥ a) (hβ : ∀ x, β x ↔ |x - 1| < 1)
  (h : ∀ x, (β x → α x) ∧ (∃ x, α x ∧ ¬β x)) : a ≤ 0 :=
by
  sorry

end range_of_a_l38_38622


namespace a_8_value_l38_38162

variable {n : ℕ}
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_8_value : a 8 = 15 := by
  sorry

end a_8_value_l38_38162


namespace gnomes_in_fifth_house_l38_38698

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end gnomes_in_fifth_house_l38_38698


namespace denominator_of_first_fraction_l38_38642

theorem denominator_of_first_fraction (y x : ℝ) (h : y > 0) (h_eq : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 :=
by
  sorry

end denominator_of_first_fraction_l38_38642


namespace amount_of_bill_is_1575_l38_38260

noncomputable def time_in_years := (9 : ℝ) / 12

noncomputable def true_discount := 189
noncomputable def rate := 16

noncomputable def face_value (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * 100) / (R * T)

theorem amount_of_bill_is_1575 :
  face_value true_discount rate time_in_years = 1575 := by
  sorry

end amount_of_bill_is_1575_l38_38260


namespace remainder_of_square_l38_38955

variable (N X : Set ℤ)
variable (k : ℤ)

/-- Given any n in set N and any x in set X, where dividing n by x gives a remainder of 3,
prove that the remainder of n^2 divided by x is 9 mod x. -/
theorem remainder_of_square (n x : ℤ) (hn : n ∈ N) (hx : x ∈ X)
  (h : ∃ k, n = k * x + 3) : (n^2) % x = 9 % x :=
by
  sorry

end remainder_of_square_l38_38955


namespace solve_quadratic_l38_38541

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end solve_quadratic_l38_38541


namespace complement_union_eq_l38_38201

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l38_38201


namespace difference_of_squares_l38_38551

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end difference_of_squares_l38_38551


namespace committeeFormation_l38_38112

-- Establish the given problem conditions in Lean

open Classical

-- Noncomputable because we are working with combinations and products
noncomputable def numberOfWaysToFormCommittee (numSchools : ℕ) (membersPerSchool : ℕ) (hostSchools : ℕ) (hostReps : ℕ) (nonHostReps : ℕ) : ℕ :=
  let totalSchools := numSchools
  let chooseHostSchools := Nat.choose totalSchools hostSchools
  let chooseHostRepsPerSchool := Nat.choose membersPerSchool hostReps
  let allHostRepsChosen := chooseHostRepsPerSchool ^ hostSchools
  let chooseNonHostRepsPerSchool := Nat.choose membersPerSchool nonHostReps
  let allNonHostRepsChosen := chooseNonHostRepsPerSchool ^ (totalSchools - hostSchools)
  chooseHostSchools * allHostRepsChosen * allNonHostRepsChosen

-- We now state our theorem
theorem committeeFormation : numberOfWaysToFormCommittee 4 6 2 3 1 = 86400 :=
by
  -- This is the lemma we need to prove
  sorry

end committeeFormation_l38_38112


namespace scientific_notation_of_909_000_000_000_l38_38241

theorem scientific_notation_of_909_000_000_000 :
    ∃ (a : ℝ) (n : ℤ), 909000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 9.09 ∧ n = 11 := 
sorry

end scientific_notation_of_909_000_000_000_l38_38241


namespace abhay_speed_l38_38940

variables (A S : ℝ)

theorem abhay_speed (h1 : 24 / A = 24 / S + 2) (h2 : 24 / (2 * A) = 24 / S - 1) : A = 12 :=
by {
  sorry
}

end abhay_speed_l38_38940


namespace loss_per_metre_l38_38287

-- Definitions for given conditions
def TSP : ℕ := 15000           -- Total Selling Price
def CPM : ℕ := 40              -- Cost Price per Metre
def TMS : ℕ := 500             -- Total Metres Sold

-- Definition for the expected Loss Per Metre
def LPM : ℕ := 10

-- Statement to prove that the loss per metre is 10
theorem loss_per_metre :
  (CPM * TMS - TSP) / TMS = LPM :=
by
sorry

end loss_per_metre_l38_38287


namespace susan_remaining_spaces_to_win_l38_38976

/-- Susan's board game has 48 spaces. She makes three moves:
 1. She moves forward 8 spaces
 2. She moves forward 2 spaces and then back 5 spaces
 3. She moves forward 6 spaces
 Prove that the remaining spaces she has to move to reach the end is 37.
-/
theorem susan_remaining_spaces_to_win :
  let total_spaces := 48
  let first_turn := 8
  let second_turn := 2 - 5
  let third_turn := 6
  let total_moved := first_turn + second_turn + third_turn
  total_spaces - total_moved = 37 :=
by
  sorry

end susan_remaining_spaces_to_win_l38_38976


namespace find_Q_over_P_l38_38842

theorem find_Q_over_P (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -7 → x ≠ 0 → x ≠ 5 →
    (P / (x + 7 : ℝ) + Q / (x^2 - 6 * x) = (x^2 - 6 * x + 14) / (x^3 + x^2 - 30 * x))) :
  Q / P = 12 :=
  sorry

end find_Q_over_P_l38_38842


namespace deposit_percentage_correct_l38_38280

-- Define the conditions
def deposit_amount : ℕ := 50
def remaining_amount : ℕ := 950
def total_cost : ℕ := deposit_amount + remaining_amount

-- Define the proof problem statement
theorem deposit_percentage_correct :
  (deposit_amount / total_cost : ℚ) * 100 = 5 := 
by
  -- sorry is used to skip the proof
  sorry

end deposit_percentage_correct_l38_38280


namespace december_25_is_thursday_l38_38893

theorem december_25_is_thursday (thanksgiving : ℕ) (h : thanksgiving = 27) :
  (∀ n, n % 7 = 0 → n + thanksgiving = 25 → n / 7 = 4) :=
by
  sorry

end december_25_is_thursday_l38_38893


namespace value_of_expression_l38_38181

variables (a b c d m : ℝ)

theorem value_of_expression (h1: a + b = 0) (h2: c * d = 1) (h3: |m| = 3) :
  (a + b) / m + m^2 - c * d = 8 :=
by
  sorry

end value_of_expression_l38_38181


namespace cos_315_deg_l38_38440

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l38_38440


namespace danielle_money_for_supplies_l38_38299

-- Define the conditions
def cost_of_molds := 3
def cost_of_sticks_pack := 1
def sticks_in_pack := 100
def cost_of_juice_bottle := 2
def popsicles_per_bottle := 20
def remaining_sticks := 40
def used_sticks := sticks_in_pack - remaining_sticks

-- Define number of juice bottles used
def bottles_of_juice_used : ℕ := used_sticks / popsicles_per_bottle

-- Define the total cost
def total_cost : ℕ := cost_of_molds + cost_of_sticks_pack + bottles_of_juice_used * cost_of_juice_bottle

-- Prove that Danielle had $10 for supplies
theorem danielle_money_for_supplies : total_cost = 10 := by {
  sorry
}

end danielle_money_for_supplies_l38_38299


namespace charge_per_mile_l38_38669

def rental_fee : ℝ := 20.99
def total_amount_paid : ℝ := 95.74
def miles_driven : ℝ := 299

theorem charge_per_mile :
  (total_amount_paid - rental_fee) / miles_driven = 0.25 := 
sorry

end charge_per_mile_l38_38669


namespace train_cross_time_l38_38943

noncomputable def train_length : ℝ := 317.5
noncomputable def train_speed_kph : ℝ := 153.3
noncomputable def convert_speed_to_mps (speed_kph : ℝ) : ℝ :=
  (speed_kph * 1000) / 3600

noncomputable def train_speed_mps : ℝ := convert_speed_to_mps train_speed_kph
noncomputable def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time :
  time_to_cross_pole train_length train_speed_mps = 7.456 :=
by 
  -- This is where the proof would go
  sorry

end train_cross_time_l38_38943


namespace matching_pair_probability_l38_38455

theorem matching_pair_probability :
  let total_socks := 22
  let blue_socks := 12
  let red_socks := 10
  let total_ways := (total_socks * (total_socks - 1)) / 2
  let blue_ways := (blue_socks * (blue_socks - 1)) / 2
  let red_ways := (red_socks * (red_socks - 1)) / 2
  let matching_ways := blue_ways + red_ways
  total_ways = 231 →
  blue_ways = 66 →
  red_ways = 45 →
  matching_ways = 111 →
  (matching_ways : ℝ) / total_ways = 111 / 231 := by sorry

end matching_pair_probability_l38_38455


namespace remainder_when_200_divided_by_k_l38_38896

theorem remainder_when_200_divided_by_k (k : ℕ) (hk_pos : 0 < k)
  (h₁ : 125 % (k^3) = 5) : 200 % k = 0 :=
sorry

end remainder_when_200_divided_by_k_l38_38896


namespace probability_point_in_region_l38_38362

theorem probability_point_in_region (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 2010) 
  (h2 : 0 ≤ y ∧ y ≤ 2009) 
  (h3 : ∃ (u v : ℝ), (u, v) = (x, y) ∧ x > 2 * y ∧ y > 500) : 
  ∃ p : ℚ, p = 1505 / 4018 := 
sorry

end probability_point_in_region_l38_38362


namespace negation_of_existence_statement_l38_38010

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 3 * x + 2 = 0) = ∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existence_statement_l38_38010


namespace DF_is_5_point_5_l38_38900

variables {A B C D E F : Type}
variables (congruent : triangle A B C ≃ triangle D E F)
variables (ac_length : AC = 5.5)

theorem DF_is_5_point_5 : DF = 5.5 :=
by
  -- skipped proof
  sorry

end DF_is_5_point_5_l38_38900


namespace operation_B_correct_operation_C_correct_l38_38563

theorem operation_B_correct (x y : ℝ) : (-3 * x * y) ^ 2 = 9 * x ^ 2 * y ^ 2 :=
  sorry

theorem operation_C_correct (x y : ℝ) (h : x ≠ y) : 
  (x - y) / (2 * x * y - x ^ 2 - y ^ 2) = 1 / (y - x) :=
  sorry

end operation_B_correct_operation_C_correct_l38_38563


namespace system_of_equations_l38_38913

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end system_of_equations_l38_38913


namespace number_of_buses_proof_l38_38584

-- Define the conditions
def columns_per_bus : ℕ := 4
def rows_per_bus : ℕ := 10
def total_students : ℕ := 240
def seats_per_bus (c : ℕ) (r : ℕ) : ℕ := c * r
def number_of_buses (total : ℕ) (seats : ℕ) : ℕ := total / seats

-- State the theorem we want to prove
theorem number_of_buses_proof :
  number_of_buses total_students (seats_per_bus columns_per_bus rows_per_bus) = 6 := 
sorry

end number_of_buses_proof_l38_38584


namespace seconds_hand_revolution_l38_38985

theorem seconds_hand_revolution (revTimeSeconds revTimeMinutes : ℕ) : 
  (revTimeSeconds = 60) ∧ (revTimeMinutes = 1) :=
sorry

end seconds_hand_revolution_l38_38985


namespace find_second_number_l38_38566

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l38_38566


namespace cost_of_fencing_theorem_l38_38550

noncomputable def cost_of_fencing (area : ℝ) (ratio_length_width : ℝ) (cost_per_meter_paise : ℝ) : ℝ :=
  let width := (area / (ratio_length_width * 2 * ratio_length_width * 3)).sqrt
  let length := ratio_length_width * 3 * width
  let perimeter := 2 * (length + width)
  let cost_per_meter_rupees := cost_per_meter_paise / 100
  perimeter * cost_per_meter_rupees

theorem cost_of_fencing_theorem :
  cost_of_fencing 3750 3 50 = 125 :=
by
  sorry

end cost_of_fencing_theorem_l38_38550


namespace students_neither_football_nor_cricket_l38_38963

theorem students_neither_football_nor_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (cricket_players : ℕ) 
  (both_players : ℕ) 
  (H1 : total_students = 410) 
  (H2 : football_players = 325) 
  (H3 : cricket_players = 175) 
  (H4 : both_players = 140) :
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end students_neither_football_nor_cricket_l38_38963


namespace find_x_l38_38486

theorem find_x (x : ℝ) (h₁ : x > 0) (h₂ : x^4 = 390625) : x = 25 := 
by sorry

end find_x_l38_38486


namespace solution_set_of_inequality_l38_38384

theorem solution_set_of_inequality :
  { x : ℝ | (x - 3) * (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l38_38384


namespace power_mod_equiv_l38_38995

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end power_mod_equiv_l38_38995


namespace total_cost_of_products_l38_38595

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end total_cost_of_products_l38_38595


namespace cos_315_eq_sqrt2_div2_l38_38434

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l38_38434


namespace range_of_m_l38_38932

theorem range_of_m (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (hx : ∃ x < 0, a^x = 3 * m - 2) :
  1 < m :=
sorry

end range_of_m_l38_38932


namespace goose_eggs_count_l38_38040

theorem goose_eggs_count (E : ℕ) 
  (h1 : (1/2 : ℝ) * E = E/2)
  (h2 : (3/4 : ℝ) * (E/2) = (3 * E) / 8)
  (h3 : (2/5 : ℝ) * ((3 * E) / 8) = (3 * E) / 20)
  (h4 : (3 * E) / 20 = 120) :
  E = 400 :=
sorry

end goose_eggs_count_l38_38040


namespace distinct_possible_lunches_l38_38293

def main_dishes := 3
def beverages := 3
def snacks := 3

theorem distinct_possible_lunches : main_dishes * beverages * snacks = 27 := by
  sorry

end distinct_possible_lunches_l38_38293


namespace shark_feed_l38_38425

theorem shark_feed (S : ℝ) (h1 : S + S/2 + 5 * S = 26) : S = 4 := 
by sorry

end shark_feed_l38_38425


namespace inequlity_for_k_one_smallest_k_l38_38815

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

theorem inequlity_for_k_one (a b c : ℝ) (h : triangle_sides a b c) :
  a^3 + b^3 + c^3 < (a + b + c) * (a * b + b * c + c * a) :=
sorry

theorem smallest_k (a b c k : ℝ) (h : triangle_sides a b c) (hk : k = 1) :
  a^3 + b^3 + c^3 < k * (a + b + c) * (a * b + b * c + c * a) :=
sorry

end inequlity_for_k_one_smallest_k_l38_38815


namespace remainder_of_sum_of_primes_is_eight_l38_38562

-- Define the first eight primes and their sum
def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sumFirstEightPrimes : ℕ := 77

-- Define the ninth prime
def ninthPrime : ℕ := 23

-- Theorem stating the equivalence
theorem remainder_of_sum_of_primes_is_eight :
  (sumFirstEightPrimes % ninthPrime) = 8 := by
  sorry

end remainder_of_sum_of_primes_is_eight_l38_38562


namespace remainder_base12_div_9_l38_38398

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l38_38398


namespace solve_x4_minus_16_eq_0_l38_38464

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l38_38464


namespace arithmetic_mean_of_integers_from_neg3_to_6_l38_38071

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  let nums := list.range' (-3) 10 in
  (∑ i in nums, i) / (nums.length : ℝ) = 1.5 :=
by
  let nums := list.range' (-3) 10
  have h_sum : (∑ i in nums, i) = 15 := sorry
  have h_length : nums.length = 10 := sorry
  rw [h_sum, h_length]
  norm_num
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l38_38071


namespace min_value_x_squared_y_cubed_z_l38_38518

theorem min_value_x_squared_y_cubed_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(h : 1 / x + 1 / y + 1 / z = 9) : x^2 * y^3 * z ≥ 729 / 6912 :=
sorry

end min_value_x_squared_y_cubed_z_l38_38518


namespace discount_price_l38_38751

theorem discount_price (original_price : ℝ) (discount_percent : ℝ) (final_price : ℝ) :
  original_price = 800 ∧ discount_percent = 15 → final_price = 680 :=
by
  intros h
  cases' h with hp hd
  sorry

end discount_price_l38_38751


namespace number_of_girls_l38_38982

theorem number_of_girls (B G : ℕ) (ratio_condition : B = G / 2) (total_condition : B + G = 90) : 
  G = 60 := 
by
  -- This is the problem statement, with conditions and required result.
  sorry

end number_of_girls_l38_38982


namespace vertical_angles_congruent_l38_38681

-- Define what it means for two angles to be vertical angles
def areVerticalAngles (a b : ℝ) : Prop := -- placeholder definition
  sorry

-- Define what it means for two angles to be congruent
def areCongruentAngles (a b : ℝ) : Prop := a = b

-- State the problem in the form of a theorem
theorem vertical_angles_congruent (a b : ℝ) :
  areVerticalAngles a b → areCongruentAngles a b := by
  sorry

end vertical_angles_congruent_l38_38681


namespace certain_number_l38_38275

theorem certain_number (a b : ℕ) (n : ℕ) 
  (h1: a % n = 0) (h2: b % n = 0) 
  (h3: b = a + 9 * n)
  (h4: b = a + 126) : n = 14 :=
by
  sorry

end certain_number_l38_38275


namespace extreme_values_x_axis_l38_38353

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

theorem extreme_values_x_axis (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, f a b c x = x * (a * x^2 + b * x + c))
  (h3 : ∀ x, deriv (f a b c) x = 3 * a * x^2 + 2 * b * x + c)
  (h4 : deriv (f a b c) 1 = 0)
  (h5 : deriv (f a b c) (-1) = 0) :
  b = 0 :=
sorry

end extreme_values_x_axis_l38_38353


namespace arithmetic_mean_of_integers_from_neg3_to_6_l38_38072

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l38_38072


namespace units_digit_of_7_pow_3_l38_38082

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end units_digit_of_7_pow_3_l38_38082


namespace cos_315_deg_l38_38439

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l38_38439


namespace power_equation_l38_38869

theorem power_equation (m : ℤ) (h : 16 = 2 ^ 4) : (16 : ℝ) ^ (3 / 4) = (2 : ℝ) ^ (m : ℝ) → m = 3 := by
  intros
  sorry

end power_equation_l38_38869


namespace solve_x4_eq_16_l38_38468

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l38_38468


namespace solution_mixture_l38_38098

/-
  Let X be a solution that is 10% alcohol by volume.
  Let Y be a solution that is 30% alcohol by volume.
  We define the final solution to be 22% alcohol by volume.
  We need to prove that the amount of solution Y that needs
  to be added to 300 milliliters of solution X to achieve this 
  concentration is 450 milliliters.
-/

theorem solution_mixture (y : ℝ) : 
  (0.10 * 300) + (0.30 * y) = 0.22 * (300 + y) → 
  y = 450 :=
by {
  sorry
}

end solution_mixture_l38_38098


namespace possible_to_fill_grid_l38_38191

/-- Define the grid as a 2D array where each cell contains either 0 or 1. --/
def grid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), i < 5 → j < 5 → f i j = 0 ∨ f i j = 1

/-- Ensure the sum of every 2x2 subgrid is divisible by 3. --/
def divisible_by_3_in_subgrid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < 4 → j < 4 → (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 3 = 0

/-- Ensure both 0 and 1 are present in the grid. --/
def contains_0_and_1 (f : ℕ → ℕ → ℕ) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 0) ∧ (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 1)

/-- The main theorem stating the possibility of such a grid. --/
theorem possible_to_fill_grid :
  ∃ f, grid f ∧ divisible_by_3_in_subgrid f ∧ contains_0_and_1 f :=
sorry

end possible_to_fill_grid_l38_38191


namespace jonah_raisins_l38_38947

variable (y : ℝ)

theorem jonah_raisins :
  (y + 0.4 = 0.7) → (y = 0.3) :=
  by
  intro h
  sorry

end jonah_raisins_l38_38947


namespace eleven_place_unamed_racer_l38_38643

theorem eleven_place_unamed_racer
  (Rand Hikmet Jack Marta David Todd : ℕ)
  (positions : Fin 15)
  (C_1 : Rand = Hikmet + 6)
  (C_2 : Marta = Jack + 1)
  (C_3 : David = Hikmet + 3)
  (C_4 : Jack = Todd + 3)
  (C_5 : Todd = Rand + 1)
  (C_6 : Marta = 8) :
  ∃ (x : Fin 15), (x ≠ Rand) ∧ (x ≠ Hikmet) ∧ (x ≠ Jack) ∧ (x ≠ Marta) ∧ (x ≠ David) ∧ (x ≠ Todd) ∧ x = 11 := 
sorry

end eleven_place_unamed_racer_l38_38643


namespace photos_in_gallery_l38_38038

theorem photos_in_gallery (P : ℕ) 
  (h1 : P / 2 + (P / 2 + 120) + P = 920) : P = 400 :=
by
  sorry

end photos_in_gallery_l38_38038


namespace initial_soccer_balls_l38_38823

theorem initial_soccer_balls (x : ℝ) (h1 : 0.40 * x = y) (h2 : 0.20 * (0.60 * x) = z) (h3 : 0.80 * (0.60 * x) = 48) : x = 100 := by
  sorry

end initial_soccer_balls_l38_38823


namespace leila_total_expenditure_l38_38350

variable (cost_auto cost_market total : ℕ)
variable (h1 : cost_auto = 350)
variable (h2 : cost_auto = 3 * cost_market + 50)

theorem leila_total_expenditure : total = 450 :=
by
  have h3 : cost_market = 100 := by
    calc
      cost_market = (350 - 50) / 3 := by rw [← h2, ← h1]
      ... = 100 : by norm_num
  have h4 : total = cost_auto + cost_market := by norm_num
  calc
    total = 350 + 100 := by rw [h4, h1, h3]
    ... = 450 : by norm_num

end leila_total_expenditure_l38_38350


namespace cos_315_eq_l38_38437

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l38_38437


namespace units_digit_7_pow_3_l38_38081

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end units_digit_7_pow_3_l38_38081


namespace length_stationary_l38_38586

def speed : ℝ := 64.8
def time_pole : ℝ := 5
def time_stationary : ℝ := 25

def length_moving : ℝ := speed * time_pole
def length_combined : ℝ := speed * time_stationary

theorem length_stationary : length_combined - length_moving = 1296 :=
by
  sorry

end length_stationary_l38_38586


namespace final_bug_population_is_zero_l38_38735

def initial_population := 400
def spiders := 12
def spider_consumption := 7
def ladybugs := 5
def ladybug_consumption := 6
def mantises := 8
def mantis_consumption := 4

def day1_population := initial_population * 80 / 100

def predators_consumption_day := (spiders * spider_consumption) +
                                 (ladybugs * ladybug_consumption) +
                                 (mantises * mantis_consumption)

def day2_population := day1_population - predators_consumption_day
def day3_population := day2_population - predators_consumption_day
def day4_population := max 0 (day3_population - predators_consumption_day)
def day5_population := max 0 (day4_population - predators_consumption_day)
def day6_population := max 0 (day5_population - predators_consumption_day)

def day7_population := day6_population * 70 / 100

theorem final_bug_population_is_zero: 
  day7_population = 0 :=
  by
  sorry

end final_bug_population_is_zero_l38_38735


namespace black_and_blue_lines_l38_38570

-- Definition of given conditions
def grid_size : ℕ := 50
def total_points : ℕ := grid_size * grid_size
def blue_points : ℕ := 1510
def blue_edge_points : ℕ := 110
def red_segments : ℕ := 947
def corner_points : ℕ := 4

-- Calculations based on conditions
def red_points : ℕ := total_points - blue_points

def edge_points (size : ℕ) : ℕ := (size - 1) * 4
def non_corner_edge_points (edge : ℕ) : ℕ := edge - corner_points

-- Math translation
noncomputable def internal_red_points : ℕ := red_points - corner_points - (edge_points grid_size - blue_edge_points)
noncomputable def connections_from_red_points : ℕ :=
  corner_points * 2 + (non_corner_edge_points (edge_points grid_size) - blue_edge_points) * 3 + internal_red_points * 4

noncomputable def adjusted_red_lines : ℕ := red_segments * 2
noncomputable def black_lines : ℕ := connections_from_red_points - adjusted_red_lines

def total_lines (size : ℕ) : ℕ := (size - 1) * size + (size - 1) * size
noncomputable def blue_lines : ℕ := total_lines grid_size - red_segments - black_lines

-- The theorem to be proven
theorem black_and_blue_lines :
  (black_lines = 1972) ∧ (blue_lines = 1981) :=
by
  sorry

end black_and_blue_lines_l38_38570


namespace min_value_of_S_l38_38954

variable (x : ℝ)
def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S : ∀ x : ℝ, S x ≥ 112.5 :=
by
  sorry

end min_value_of_S_l38_38954


namespace find_solutions_l38_38465

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l38_38465


namespace gnomes_in_fifth_house_l38_38699

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end gnomes_in_fifth_house_l38_38699


namespace all_tell_truth_alice_bob_carol_truth_at_least_one_truth_l38_38450

noncomputable def alice_truth : ℝ := 0.70
noncomputable def bob_truth : ℝ := 0.60
noncomputable def carol_truth : ℝ := 0.80
noncomputable def david_truth : ℝ := 0.50
noncomputable def eric_truth : ℝ := 0.30

theorem all_tell_truth : alice_truth * bob_truth * carol_truth * david_truth * eric_truth = 0.042 := 
by sorry

theorem alice_bob_carol_truth : alice_truth * bob_truth * carol_truth = 0.336 :=
by sorry

theorem at_least_one_truth : 1 - (1 - alice_truth) * (1 - bob_truth) * (1 - carol_truth) * (1 - david_truth) * (1 - eric_truth) = 0.9916 :=
by sorry

end all_tell_truth_alice_bob_carol_truth_at_least_one_truth_l38_38450


namespace trees_in_park_l38_38263

variable (W O T : Nat)

theorem trees_in_park (h1 : W = 36) (h2 : O = W + 11) (h3 : T = W + O) : T = 83 := by
  sorry

end trees_in_park_l38_38263


namespace greatest_possible_value_of_x_l38_38076

theorem greatest_possible_value_of_x (x : ℝ) (h : ( (5 * x - 25) / (4 * x - 5) ) ^ 3 + ( (5 * x - 25) / (4 * x - 5) ) = 16):
  x = 5 :=
sorry

end greatest_possible_value_of_x_l38_38076


namespace problem_complement_intersection_l38_38914

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

def complement (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

theorem problem_complement_intersection :
  (complement U M) ∩ N = {3} :=
by
  sorry

end problem_complement_intersection_l38_38914


namespace lowest_possible_number_of_students_l38_38715

theorem lowest_possible_number_of_students :
  Nat.lcm 18 24 = 72 :=
by
  sorry

end lowest_possible_number_of_students_l38_38715


namespace rectangle_side_excess_l38_38941

theorem rectangle_side_excess
  (L W : ℝ)  -- length and width of the rectangle
  (x : ℝ)   -- percentage in excess for the first side
  (h1 : 0.95 * (L * (1 + x / 100) * W) = 1.102 * (L * W)) :
  x = 16 :=
by
  sorry

end rectangle_side_excess_l38_38941


namespace no_discount_profit_percentage_l38_38114

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 4 / 100  -- 4%
noncomputable def profit_percentage_with_discount : ℝ := 20 / 100  -- 20%

theorem no_discount_profit_percentage : 
  (1 + profit_percentage_with_discount) * cost_price / (1 - discount_percentage) / cost_price - 1 = 0.25 := by
  sorry

end no_discount_profit_percentage_l38_38114


namespace solution_l38_38153

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem solution (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by 
  -- Here we will skip the actual proof by using sorry
  sorry

end solution_l38_38153


namespace find_b_find_perimeter_b_plus_c_l38_38020

noncomputable def triangle_condition_1
  (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.cos B = (3 * c - b) * Real.cos A

noncomputable def triangle_condition_2
  (a b : ℝ) (C : ℝ) : Prop :=
  a * Real.sin C = 2 * Real.sqrt 2

noncomputable def triangle_condition_3
  (a b c : ℝ) (A : ℝ) : Prop :=
  (1 / 2) * b * c * Real.sin A = Real.sqrt 2

noncomputable def given_a
  (a : ℝ) : Prop :=
  a = 2 * Real.sqrt 2

theorem find_b
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b = 3 :=
sorry

theorem find_perimeter_b_plus_c
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b + c = 2 * Real.sqrt 3 :=
sorry

end find_b_find_perimeter_b_plus_c_l38_38020


namespace star_polygon_x_value_l38_38626

theorem star_polygon_x_value
  (a b c d e p q r s t : ℝ)
  (h1 : p + q + r + s + t = 500)
  (h2 : a + b + c + d + e = x)
  :
  x = 140 :=
sorry

end star_polygon_x_value_l38_38626


namespace problem_proof_l38_38351

theorem problem_proof (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
  (h2 : a + b + c + d = m^2) 
  (h3 : max (max a b) (max c d) = n^2) : 
  m = 9 ∧ n = 6 :=
by
  sorry

end problem_proof_l38_38351


namespace positive_quadratic_expression_l38_38300

theorem positive_quadratic_expression (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + 4 + m > 0) ↔ (- (Real.sqrt 55) / 2 < m ∧ m < (Real.sqrt 55) / 2) := 
sorry

end positive_quadratic_expression_l38_38300


namespace cos_315_proof_l38_38443

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l38_38443


namespace optimal_addition_amount_l38_38658

def optimal_material_range := {x : ℝ | 100 ≤ x ∧ x ≤ 200}

def second_trial_amounts := {x : ℝ | x = 138.2 ∨ x = 161.8}

theorem optimal_addition_amount (
  h1 : ∀ x ∈ optimal_material_range, x ∈ second_trial_amounts
  ) :
  138.2 ∈ second_trial_amounts ∧ 161.8 ∈ second_trial_amounts :=
by
  sorry

end optimal_addition_amount_l38_38658


namespace correct_assignment_l38_38132

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l38_38132


namespace lines_intersect_l38_38122

theorem lines_intersect (m : ℝ) : ∃ (x y : ℝ), 3 * x + 2 * y + m = 0 ∧ (m^2 + 1) * x - 3 * y - 3 * m = 0 := 
by {
  sorry
}

end lines_intersect_l38_38122


namespace algebraic_expression_value_l38_38641

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 :=
by
  sorry

end algebraic_expression_value_l38_38641


namespace vertical_angles_congruent_l38_38682

theorem vertical_angles_congruent (A B : Type) [angle A] [angle B] (h : vertical A B) : congruent A B :=
sorry

end vertical_angles_congruent_l38_38682


namespace imaginary_part_z1z2_l38_38491

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end imaginary_part_z1z2_l38_38491


namespace cos_315_eq_l38_38436

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l38_38436


namespace arctan_sum_l38_38505

theorem arctan_sum (a b : ℝ) : 
  Real.arctan (a / (a + 2 * b)) + Real.arctan (b / (2 * a + b)) = Real.arctan (1 / 2) :=
by {
  sorry
}

end arctan_sum_l38_38505


namespace problem_statement_l38_38226

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l38_38226


namespace vacation_cost_l38_38851

theorem vacation_cost (C : ℝ)
  (h1 : C / 5 - C / 8 = 60) :
  C = 800 :=
sorry

end vacation_cost_l38_38851


namespace ludwig_weekly_salary_is_55_l38_38524

noncomputable def daily_salary : ℝ := 10
noncomputable def full_days : ℕ := 4
noncomputable def half_days : ℕ := 3
noncomputable def half_day_salary := daily_salary / 2

theorem ludwig_weekly_salary_is_55 :
  (full_days * daily_salary + half_days * half_day_salary = 55) := by
  sorry

end ludwig_weekly_salary_is_55_l38_38524


namespace find_q_l38_38482

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l38_38482


namespace arithmetic_progression_product_l38_38619

theorem arithmetic_progression_product (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (b : ℕ), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) = b ^ 2008) :=
by
  sorry

end arithmetic_progression_product_l38_38619


namespace factorial_divisibility_l38_38232

theorem factorial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : a ≥ 2 * b + 1 :=
sorry

end factorial_divisibility_l38_38232


namespace distinct_integers_sum_l38_38355

theorem distinct_integers_sum {p q r s t : ℤ} 
    (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t) 
    (h9 : r ≠ s) (h10 : r ≠ t) (h11 : s ≠ t) : 
  p + q + r + s + t = 35 := 
sorry

end distinct_integers_sum_l38_38355


namespace remainder_7_pow_150_mod_12_l38_38997

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end remainder_7_pow_150_mod_12_l38_38997


namespace probability_at_least_one_diamond_or_ace_or_both_red_l38_38700

/-!
# Probability of Drawing Specific Cards

We want to prove that the probability of drawing at least one diamond or ace or both cards
are red in a two cards successive draw without replacement from a standard deck of 52 cards
is equal to 889/1326.
-/

theorem probability_at_least_one_diamond_or_ace_or_both_red :
  let total_cards := 52 in
  let diamond_cards := 13 in
  let ace_cards := 4 in
  let red_cards := 26 in
  let non_diamond_non_ace_cards := total_cards - diamond_cards - 1 in
  let non_red_cards := 24 in
  let p_complement := (non_diamond_non_ace_cards / total_cards) * 
                      ((non_diamond_non_ace_cards - non_red_cards) / (total_cards - 1)) in
  let p_event := 1 - p_complement in
  p_event = 889 / 1326 :=
by
  have p_complement := 19 / 26 * 23 / 51
  have p_event := 1 - p_complement
  have p_event := 889 / 1326
  sorry

end probability_at_least_one_diamond_or_ace_or_both_red_l38_38700


namespace contact_probability_l38_38761

-- Definition of the number of tourists in each group
def num_tourists_group1 : ℕ := 6
def num_tourists_group2 : ℕ := 7
def total_pairs : ℕ := num_tourists_group1 * num_tourists_group2

-- Definition of probability for no contact
def p : ℝ -- probability of contact
def prob_no_contact := (1 - p) ^ total_pairs

-- The theorem to be proven
theorem contact_probability : 1 - prob_no_contact = 1 - (1 - p) ^ total_pairs :=
by
  sorry

end contact_probability_l38_38761


namespace mean_marks_second_section_l38_38415

-- Definitions for the problem conditions
def num_students (section1 section2 section3 section4 : ℕ) : ℕ :=
  section1 + section2 + section3 + section4

def total_marks (section1 section2 section3 section4 : ℕ) (mean1 mean2 mean3 mean4 : ℝ) : ℝ :=
  section1 * mean1 + section2 * mean2 + section3 * mean3 + section4 * mean4

-- The final problem translated into a lean statement
theorem mean_marks_second_section :
  let section1 := 65
  let section2 := 35
  let section3 := 45
  let section4 := 42
  let mean1 := 50
  let mean3 := 55
  let mean4 := 45
  let overall_average := 51.95
  num_students section1 section2 section3 section4 = 187 →
  ((section1 : ℝ) * mean1 + (section2 : ℝ) * M + (section3 : ℝ) * mean3 + (section4 : ℝ) * mean4)
    = 187 * overall_average →
  M = 59.99 :=
by
  intros section1 section2 section3 section4 mean1 mean3 mean4 overall_average Hnum Htotal
  sorry

end mean_marks_second_section_l38_38415


namespace general_term_formula_l38_38690
-- Import the Mathlib library 

-- Define the conditions as given in the problem
/-- 
Define the sequence that represents the numerators. 
This is an arithmetic sequence of odd numbers starting from 1.
-/
def numerator (n : ℕ) : ℕ := 2 * n + 1

/-- 
Define the sequence that represents the denominators. 
This is a geometric sequence with the first term being 2 and common ratio being 2.
-/
def denominator (n : ℕ) : ℕ := 2^(n+1)

-- State the main theorem that we need to prove
theorem general_term_formula (n : ℕ) : (numerator n) / (denominator n) = (2 * n + 1) / 2^(n+1) :=
sorry

end general_term_formula_l38_38690


namespace find_m_n_l38_38632

theorem find_m_n (x m n : ℤ) : (x + 2) * (x + 3) = x^2 + m * x + n → m = 5 ∧ n = 6 :=
by {
    sorry
}

end find_m_n_l38_38632


namespace part_a_part_b_l38_38454

-- Define the players
inductive Player
| Ann | Bob | Con | Dot | Eve | Fay | Guy | Hal

open Player

-- No group of five players has all possible games played among them
def no_group_of_five (g : Graph Player) : Prop :=
  ∀ (s : Finset Player), s.card = 5 → ∃ v w, v ∈ s ∧ w ∈ s ∧ ¬g.adj v w

-- Part (a): Construct an arrangement of 24 games satisfying the conditions
theorem part_a : ∃ (g : Graph Player), g.edge_finset.card = 24 ∧ no_group_of_five g := 
sorry

-- Part (b): Show that it's impossible to have more than 24 games satisfying the conditions
theorem part_b : ∀ (g : Graph Player), no_group_of_five g → g.edge_finset.card ≤ 24 := 
sorry

end part_a_part_b_l38_38454


namespace max_possible_x_l38_38953

theorem max_possible_x (x y z : ℝ) 
  (h1 : 3 * x + 2 * y + z = 10)
  (h2 : x * y + x * z + y * z = 6) :
  x ≤ 2 * Real.sqrt 5 / 5 :=
sorry

end max_possible_x_l38_38953


namespace gnome_voting_l38_38840

theorem gnome_voting (n : ℕ) :
  (∀ g : ℕ, g < n →  
   (g % 3 = 0 → (∃ k : ℕ, k * 4 = n))
   ∧ (n ≠ 0 ∧ (∀ i : ℕ, i < n → (i + 1) % n ≠ (i + 2) % n) → (∃ k : ℕ, k * 4 = n))) := 
sorry

end gnome_voting_l38_38840


namespace Douglas_won_in_county_Y_l38_38023

def total_percentage (x y t r : ℝ) : Prop :=
  (0.74 * 2 + y * 1 = 0.66 * (2 + 1))

theorem Douglas_won_in_county_Y :
  ∀ (x y t r : ℝ), x = 0.74 → t = 0.66 → r = 2 →
  total_percentage x y t r → y = 0.50 := 
by
  intros x y t r hx ht hr H
  rw [hx, hr, ht] at H
  sorry

end Douglas_won_in_county_Y_l38_38023


namespace circumscribed_circle_radius_l38_38266

noncomputable def circumradius_of_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ :=
  c / 2

theorem circumscribed_circle_radius :
  circumradius_of_right_triangle 30 40 50 (by norm_num : 30^2 + 40^2 = 50^2) = 25 := by
norm_num /- correct answer confirmed -/
sorry

end circumscribed_circle_radius_l38_38266


namespace one_cow_one_bag_days_l38_38529

-- Definitions based on conditions in a)
def cows : ℕ := 60
def bags : ℕ := 75
def days_total : ℕ := 45

-- Main statement for the proof problem
theorem one_cow_one_bag_days : 
  (cows : ℝ) * (bags : ℝ) / (days_total : ℝ) = 1 / 36 := 
by
  sorry   -- Proof placeholder

end one_cow_one_bag_days_l38_38529


namespace arctan_sum_of_roots_l38_38258

theorem arctan_sum_of_roots (u v w : ℝ) (h1 : u + v + w = 0) (h2 : u * v + v * w + w * u = -10) (h3 : u * v * w = -11) :
  Real.arctan u + Real.arctan v + Real.arctan w = π / 4 :=
by
  sorry

end arctan_sum_of_roots_l38_38258


namespace total_wax_required_l38_38673

/-- Given conditions: -/
def wax_already_have : ℕ := 331
def wax_needed_more : ℕ := 22

/-- Prove the question (the total amount of wax required) -/
theorem total_wax_required :
  (wax_already_have + wax_needed_more) = 353 := by
  sorry

end total_wax_required_l38_38673


namespace inequality_half_l38_38925

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end inequality_half_l38_38925


namespace ordered_pair_solution_l38_38611

theorem ordered_pair_solution :
  ∃ (x y : ℤ), 
    (x + y = (7 - x) + (7 - y)) ∧ 
    (x - y = (x - 2) + (y - 2)) ∧ 
    (x = 5 ∧ y = 2) :=
by
  sorry

end ordered_pair_solution_l38_38611


namespace fraction_initially_filled_l38_38877

theorem fraction_initially_filled (x : ℕ) :
  ∃ (x : ℕ), 
    x + 15 + (15 + 5) = 100 ∧ 
    (x : ℚ) / 100 = 13 / 20 :=
by
  sorry

end fraction_initially_filled_l38_38877


namespace molecular_weight_BaCl2_l38_38077

def molecular_weight_one_mole (w_four_moles : ℕ) (n : ℕ) : ℕ := 
    w_four_moles / n

theorem molecular_weight_BaCl2 
    (w_four_moles : ℕ)
    (H : w_four_moles = 828) :
  molecular_weight_one_mole w_four_moles 4 = 207 :=
by
  -- sorry to skip the proof
  sorry

end molecular_weight_BaCl2_l38_38077


namespace total_stamps_l38_38032

-- Definitions based on conditions
def kylies_stamps : ℕ := 34
def nellys_stamps : ℕ := kylies_stamps + 44

-- Statement of the proof problem
theorem total_stamps : kylies_stamps + nellys_stamps = 112 :=
by
  -- Proof goes here
  sorry

end total_stamps_l38_38032


namespace ball_radius_l38_38106

theorem ball_radius (x r : ℝ) (h1 : x^2 + 256 = r^2) (h2 : r = x + 16) : r = 16 :=
by
  sorry

end ball_radius_l38_38106


namespace rectangle_area_l38_38372

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end rectangle_area_l38_38372


namespace stamp_problem_solution_l38_38307

theorem stamp_problem_solution : ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m ≥ 2 * n + 2 → ∃ a b : ℕ, m = n * a + (n + 2) * b) ∧ ∀ x : ℕ, 1 < x ∧ (∀ m : ℕ, m ≥ 2 * x + 2 → ∃ a b : ℕ, m = x * a + (x + 2) * b) → x ≥ 3 :=
by
  sorry

end stamp_problem_solution_l38_38307


namespace geometric_series_m_value_l38_38291

theorem geometric_series_m_value (m : ℝ) : 
    let a : ℝ := 20
    let r₁ : ℝ := 1 / 2  -- Common ratio for the first series
    let S₁ : ℝ := a / (1 - r₁)  -- Sum of the first series
    let b : ℝ := 1 / 2 + m / 20  -- Common ratio for the second series
    let S₂ : ℝ := a / (1 - b)  -- Sum of the second series
    S₁ = 40 ∧ S₂ = 120 → m = 20 / 3 :=
sorry

end geometric_series_m_value_l38_38291


namespace sara_total_spent_l38_38244

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end sara_total_spent_l38_38244


namespace find_q_l38_38483

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l38_38483


namespace math_problem_l38_38314

theorem math_problem (a b c : ℝ) (h₁ : a = 85) (h₂ : b = 32) (h₃ : c = 113) :
  (a + b / c) * c = 9637 :=
by
  rw [h₁, h₂, h₃]
  sorry

end math_problem_l38_38314


namespace complement_union_M_N_eq_ge_2_l38_38207

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l38_38207


namespace sum_of_squares_of_roots_l38_38885

theorem sum_of_squares_of_roots (a b : ℝ) (x₁ x₂ : ℝ)
  (h₁ : x₁^2 - (3 * a + b) * x₁ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0)
  (h₂ : x₂^2 - (3 * a + b) * x₂ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0) :
  x₁^2 + x₂^2 = 5 * (a^2 + b^2) := 
by
  sorry

end sum_of_squares_of_roots_l38_38885


namespace power_mod_equiv_l38_38996

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end power_mod_equiv_l38_38996


namespace cos_315_eq_sqrt2_div_2_l38_38431

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l38_38431


namespace total_students_are_45_l38_38420

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end total_students_are_45_l38_38420


namespace projectile_highest_point_l38_38285

noncomputable def highest_point (v w_h w_v θ g : ℝ) : ℝ × ℝ :=
  let t := (v * Real.sin θ + w_v) / g
  let x := (v * t + w_h * t) * Real.cos θ
  let y := (v * t + w_v * t) * Real.sin θ - (1/2) * g * t^2
  (x, y)

theorem projectile_highest_point : highest_point 100 10 (-2) (Real.pi / 4) 9.8 = (561.94, 236) :=
  sorry

end projectile_highest_point_l38_38285


namespace count_valid_numbers_is_31_l38_38331

def is_valid_digit (n : Nat) : Prop := n = 0 ∨ n = 2 ∨ n = 6 ∨ n = 8

def count_valid_numbers : Nat :=
  let valid_digits := [0, 2, 6, 8]
  let one_digit := valid_digits.filter (λ n => n % 4 = 0)
  let two_digits := valid_digits.product valid_digits |>.filter (λ (a, b) => (10*a + b) % 4 = 0)
  let three_digits := valid_digits.product two_digits |>.filter (λ (a, (b, c)) => (100*a + 10*b + c) % 4 = 0)
  one_digit.length + two_digits.length + three_digits.length

theorem count_valid_numbers_is_31 : count_valid_numbers = 31 := by
  sorry

end count_valid_numbers_is_31_l38_38331


namespace volume_tetrahedron_constant_l38_38157

theorem volume_tetrahedron_constant (m n h : ℝ) (ϕ : ℝ) :
  ∃ V : ℝ, V = (1 / 6) * m * n * h * Real.sin ϕ :=
by
  sorry

end volume_tetrahedron_constant_l38_38157


namespace polynomial_has_at_most_one_real_root_l38_38534

open Polynomial

noncomputable def P (n m : ℕ) : Polynomial ℝ :=
  ∑ i in Finset.range (m + 1), Polynomial.C (Nat.choose (n + i) n) * Polynomial.X ^ i

theorem polynomial_has_at_most_one_real_root (n m : ℕ) : 
  (P n m).real_roots.length ≤ 1 :=
sorry

end polynomial_has_at_most_one_real_root_l38_38534


namespace pears_left_l38_38196

theorem pears_left (keith_initial : ℕ) (keith_given : ℕ) (mike_initial : ℕ) 
  (hk : keith_initial = 47) (hg : keith_given = 46) (hm : mike_initial = 12) :
  (keith_initial - keith_given) + mike_initial = 13 := by
  sorry

end pears_left_l38_38196


namespace sqrt_calculation_l38_38121

theorem sqrt_calculation :
  Real.sqrt ((2:ℝ)^4 * 3^2 * 5^2) = 60 := 
by sorry

end sqrt_calculation_l38_38121


namespace sufficient_but_not_necessary_condition_l38_38158

noncomputable def are_parallel (a : ℝ) : Prop :=
  (2 + a) * a * 3 * a = 3 * a * (a - 2)

theorem sufficient_but_not_necessary_condition :
  (are_parallel 4) ∧ (∃ a ≠ 4, are_parallel a) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l38_38158


namespace worker_C_work_rate_worker_C_days_l38_38816

theorem worker_C_work_rate (A B C: ℚ) (hA: A = 1/10) (hB: B = 1/15) (hABC: A + B + C = 1/4) : C = 1/12 := 
by
  sorry

theorem worker_C_days (C: ℚ) (hC: C = 1/12) : 1 / C = 12 :=
by
  sorry

end worker_C_work_rate_worker_C_days_l38_38816


namespace enclosure_largest_side_l38_38522

theorem enclosure_largest_side (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 3600) : l = 60 :=
by
  sorry

end enclosure_largest_side_l38_38522


namespace who_wears_which_dress_l38_38149

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l38_38149


namespace optimal_hospital_location_l38_38394

-- Define the coordinates for points A, B, and C
def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

-- Define the distance function
def dist_sq (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the statement to be proved: minimizing sum of squares of distances
theorem optimal_hospital_location : ∃ y : ℝ, 
  (∀ (P : ℝ × ℝ), P = (0, y) → (dist_sq P A + dist_sq P B + dist_sq P C) = 146) ∧ y = 4 :=
by sorry

end optimal_hospital_location_l38_38394


namespace maximum_contribution_l38_38018

theorem maximum_contribution (total_contribution : ℕ) (num_people : ℕ) (individual_min_contribution : ℕ) :
  total_contribution = 20 → num_people = 10 → individual_min_contribution = 1 → 
  ∃ (max_contribution : ℕ), max_contribution = 11 := by
  intro h1 h2 h3
  existsi 11
  sorry

end maximum_contribution_l38_38018


namespace zack_traveled_to_18_countries_l38_38274

-- Defining the conditions
variables (countries_traveled_by_george countries_traveled_by_joseph 
           countries_traveled_by_patrick countries_traveled_by_zack : ℕ)

-- Set the conditions as per the problem statement
axiom george_traveled : countries_traveled_by_george = 6
axiom joseph_traveled : countries_traveled_by_joseph = countries_traveled_by_george / 2
axiom patrick_traveled : countries_traveled_by_patrick = 3 * countries_traveled_by_joseph
axiom zack_traveled : countries_traveled_by_zack = 2 * countries_traveled_by_patrick

-- The theorem to prove Zack traveled to 18 countries
theorem zack_traveled_to_18_countries : countries_traveled_by_zack = 18 :=
by
  -- Adding the proof here is unnecessary as per the instructions
  sorry

end zack_traveled_to_18_countries_l38_38274


namespace total_spent_l38_38349

-- Define the conditions
def cost_fix_automobile := 350
def cost_fix_formula (S : ℕ) := 3 * S + 50

-- Prove the total amount spent is $450
theorem total_spent (S : ℕ) (h : cost_fix_automobile = cost_fix_formula S) :
  S + cost_fix_automobile = 450 :=
by
  sorry

end total_spent_l38_38349


namespace problem_l38_38423

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def NotParallel (v1 v2 : Vector3D) : Prop := ¬ ∃ k : ℝ, v2 = ⟨k * v1.x, k * v1.y, k * v1.z⟩

def a : Vector3D := ⟨1, 2, -2⟩
def b : Vector3D := ⟨-2, -4, 4⟩
def c : Vector3D := ⟨1, 0, 0⟩
def d : Vector3D := ⟨-3, 0, 0⟩
def g : Vector3D := ⟨-2, 3, 5⟩
def h : Vector3D := ⟨16, 24, 40⟩
def e : Vector3D := ⟨2, 3, 0⟩
def f : Vector3D := ⟨0, 0, 0⟩

theorem problem : NotParallel g h := by
  sorry

end problem_l38_38423


namespace hannah_highest_score_l38_38790

theorem hannah_highest_score :
  ∀ (total_questions : ℕ) (wrong_answers_student1 : ℕ) (percentage_student2 : ℚ),
  total_questions = 40 →
  wrong_answers_student1 = 3 →
  percentage_student2 = 0.95 →
  ∃ (correct_answers_hannah : ℕ), correct_answers_hannah > 38 :=
by {
  intros,
  sorry,
}

end hannah_highest_score_l38_38790


namespace nonneg_real_inequality_l38_38972

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end nonneg_real_inequality_l38_38972


namespace cos_315_eq_sqrt2_div_2_l38_38447

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l38_38447


namespace simplify_rationalize_denominator_l38_38367

-- Definitions from the conditions
def fraction_term : ℝ := 1 / (sqrt 5 + 2)
def simplified_term : ℝ := sqrt 5 - 2
def main_expression : ℝ := 1 / (2 + fraction_term)

theorem simplify_rationalize_denominator :
  main_expression = sqrt 5 / 5 := by
  sorry

end simplify_rationalize_denominator_l38_38367


namespace divisible_by_5_l38_38322

theorem divisible_by_5 (x y : ℕ) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 5 ∣ x := sorry

end divisible_by_5_l38_38322


namespace complement_union_eq_ge_two_l38_38222

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l38_38222


namespace probability_of_prime_or_odd_is_half_l38_38994

-- Define the list of sections on the spinner
def sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  if n < 2 then false else List.foldr (λ p b => b && (n % p ≠ 0)) true (List.range (n - 2) |>.map (λ x => x + 2))

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Define the condition of being either prime or odd
def is_prime_or_odd (n : ℕ) : Bool := is_prime n || is_odd n

-- List of favorable outcomes where the number is either prime or odd
def favorable_outcomes : List ℕ := sections.filter is_prime_or_odd

-- Calculate the probability
def probability_prime_or_odd : ℚ := (favorable_outcomes.length : ℚ) / (sections.length : ℚ)

-- Statement to prove the probability is 1/2
theorem probability_of_prime_or_odd_is_half : probability_prime_or_odd = 1 / 2 := by
  sorry

end probability_of_prime_or_odd_is_half_l38_38994


namespace intersection_is_correct_l38_38168

def A : Set ℝ := {x | True}
def B : Set ℝ := {y | y ≥ 0}

theorem intersection_is_correct : A ∩ B = { x | x ≥ 0 } :=
by
  sorry

end intersection_is_correct_l38_38168


namespace num_pos_three_digit_div_by_seven_l38_38175

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l38_38175


namespace number_of_strikers_correct_l38_38728

-- Defining the initial conditions
def number_of_goalies := 3
def number_of_defenders := 10
def number_of_players := 40
def number_of_midfielders := 2 * number_of_defenders

-- Lean statement to prove
theorem number_of_strikers_correct : 
  let total_non_strikers := number_of_goalies + number_of_defenders + number_of_midfielders,
      number_of_strikers := number_of_players - total_non_strikers 
  in number_of_strikers = 7 :=
by
  sorry

end number_of_strikers_correct_l38_38728


namespace shopkeeper_loss_percent_l38_38109

theorem shopkeeper_loss_percent (cost_price goods_lost_percent profit_percent : ℝ)
    (h_cost_price : cost_price = 100)
    (h_goods_lost_percent : goods_lost_percent = 0.4)
    (h_profit_percent : profit_percent = 0.1) :
    let initial_revenue := cost_price * (1 + profit_percent)
    let goods_lost_value := cost_price * goods_lost_percent
    let remaining_goods_value := cost_price - goods_lost_value
    let remaining_revenue := remaining_goods_value * (1 + profit_percent)
    let loss_in_revenue := initial_revenue - remaining_revenue
    let loss_percent := (loss_in_revenue / initial_revenue) * 100
    loss_percent = 40 := sorry

end shopkeeper_loss_percent_l38_38109


namespace robin_hair_length_l38_38247

theorem robin_hair_length
  (l d g : ℕ)
  (h₁ : l = 16)
  (h₂ : d = 11)
  (h₃ : g = 12) :
  (l - d + g = 17) :=
by sorry

end robin_hair_length_l38_38247


namespace absolute_value_inequality_solution_set_l38_38389

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end absolute_value_inequality_solution_set_l38_38389


namespace base_representing_350_as_four_digit_number_with_even_final_digit_l38_38620

theorem base_representing_350_as_four_digit_number_with_even_final_digit {b : ℕ} :
  b ^ 3 ≤ 350 ∧ 350 < b ^ 4 ∧ (∃ d1 d2 d3 d4, 350 = d1 * b^3 + d2 * b^2 + d3 * b + d4 ∧ d4 % 2 = 0) ↔ b = 6 :=
by sorry

end base_representing_350_as_four_digit_number_with_even_final_digit_l38_38620


namespace hari_contribution_l38_38965

theorem hari_contribution 
    (P_investment : ℕ) (P_time : ℕ) (H_time : ℕ) (profit_ratio : ℚ)
    (investment_ratio : P_investment * P_time / (Hari_contribution * H_time) = profit_ratio) :
    Hari_contribution = 10080 :=
by
    have P_investment := 3920
    have P_time := 12
    have H_time := 7
    have profit_ratio := (2 : ℚ) / 3
    sorry

end hari_contribution_l38_38965


namespace find_triplets_l38_38471

theorem find_triplets (x y z : ℕ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h₁ : x ≤ y ∧ y ≤ z) (h₂ : 1 / x + 1 / y + 1 / z = 1) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
sorry

end find_triplets_l38_38471


namespace complement_intersection_l38_38169

-- Conditions
def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

-- Theorem statement (proof not included)
theorem complement_intersection :
  let C_UA : Set Int := U \ A
  (C_UA ∩ B) = {1, 2} := 
by
  sorry

end complement_intersection_l38_38169


namespace pascal_triangle_ratio_l38_38800

theorem pascal_triangle_ratio (n r : ℕ) 
  (h1 : (3 * r + 3 = 2 * n - 2 * r))
  (h2 : (4 * r + 8 = 3 * n - 3 * r - 3)) : 
  n = 34 :=
sorry

end pascal_triangle_ratio_l38_38800


namespace sum_of_solutions_eq_8_l38_38303

theorem sum_of_solutions_eq_8 :
    let a : ℝ := 1
    let b : ℝ := -8
    let c : ℝ := -26
    ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) →
      x1 + x2 = 8 :=
sorry

end sum_of_solutions_eq_8_l38_38303


namespace P_inequality_l38_38352

variable {α : Type*} [LinearOrderedField α]

def P (a b c : α) (x : α) : α := a * x^2 + b * x + c

theorem P_inequality (a b c x y : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (P a b c (x * y))^2 ≤ (P a b c (x^2)) * (P a b c (y^2)) :=
sorry

end P_inequality_l38_38352


namespace least_number_with_remainder_l38_38992

theorem least_number_with_remainder (n : ℕ) (d₁ d₂ d₃ d₄ r : ℕ) 
  (h₁ : d₁ = 5) (h₂ : d₂ = 6) (h₃ : d₃ = 9) (h₄ : d₄ = 12) (hr : r = 184) :
  (∀ d, d ∈ [d₁, d₂, d₃, d₄] → n % d = r % d) → n = 364 := 
sorry

end least_number_with_remainder_l38_38992


namespace jason_borrowed_amount_l38_38346

theorem jason_borrowed_amount :
  let cycle := [1, 3, 5, 7, 9, 11]
  let total_chores := 48
  let chores_per_cycle := cycle.length
  let earnings_one_cycle := cycle.sum
  let complete_cycles := total_chores / chores_per_cycle
  let total_earnings := complete_cycles * earnings_one_cycle
  total_earnings = 288 :=
by
  sorry

end jason_borrowed_amount_l38_38346


namespace avg_class_weight_l38_38697

def num_students_A : ℕ := 24
def num_students_B : ℕ := 16
def avg_weight_A : ℕ := 40
def avg_weight_B : ℕ := 35

/-- Theorem: The average weight of the whole class is 38 kg --/
theorem avg_class_weight :
  (num_students_A * avg_weight_A + num_students_B * avg_weight_B) / (num_students_A + num_students_B) = 38 :=
by
  -- Proof goes here
  sorry

end avg_class_weight_l38_38697


namespace decreasing_condition_l38_38781

noncomputable def f (a x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem decreasing_condition (a : ℝ) :
  (∀ x > 1, (Real.log x - 1) / (Real.log x)^2 + a ≤ 0) → a ≤ -1/4 := by
  sorry

end decreasing_condition_l38_38781


namespace stamps_ratio_l38_38988

theorem stamps_ratio (orig_stamps_P : ℕ) (addie_stamps : ℕ) (final_stamps_P : ℕ) 
  (h₁ : orig_stamps_P = 18) (h₂ : addie_stamps = 72) (h₃ : final_stamps_P = 36) :
  (final_stamps_P - orig_stamps_P) / addie_stamps = 1 / 4 :=
by {
  sorry
}

end stamps_ratio_l38_38988


namespace simplify_and_rationalize_l38_38366

theorem simplify_and_rationalize : (1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5) :=
by sorry

end simplify_and_rationalize_l38_38366


namespace perfect_square_trinomial_l38_38556

theorem perfect_square_trinomial (a b c : ℤ) (f : ℤ → ℤ) (h : ∀ x : ℤ, f x = a * x^2 + b * x + c) :
  ∃ d e : ℤ, ∀ x : ℤ, f x = (d * x + e) ^ 2 :=
sorry

end perfect_square_trinomial_l38_38556


namespace dress_assignment_l38_38147

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l38_38147


namespace card_area_after_shortening_l38_38528

theorem card_area_after_shortening 
  (length : ℕ) (width : ℕ) (area_after_shortening : ℕ) 
  (h_initial : length = 8) (h_initial_width : width = 3)
  (h_area_shortened_by_2 : area_after_shortening = 15) :
  (length - 2) * width = 8 :=
by
  -- Original dimensions
  let original_length := 8
  let original_width := 3
  -- Area after shortening one side by 2 inches
  let area_after_shortening_width := (original_length) * (original_width - 2)
  let area_after_shortening_length := (original_length - 2) * (original_width)
  sorry

end card_area_after_shortening_l38_38528


namespace more_visitors_that_day_l38_38731

def number_of_visitors_previous_day : ℕ := 100
def number_of_visitors_that_day : ℕ := 666

theorem more_visitors_that_day :
  number_of_visitors_that_day - number_of_visitors_previous_day = 566 :=
by
  sorry

end more_visitors_that_day_l38_38731


namespace find_k_value_l38_38338

noncomputable def solve_for_k (k : ℚ) : Prop :=
  ∃ x : ℚ, (x = 1) ∧ (3 * x + (2 * k - 1) = x - 6 * (3 * k + 2))

theorem find_k_value : solve_for_k (-13 / 20) :=
  sorry

end find_k_value_l38_38338


namespace find_solutions_of_x4_minus_16_l38_38469

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l38_38469


namespace dylan_trip_time_l38_38123

def total_time_of_trip (d1 d2 d3 v1 v2 v3 b : ℕ) : ℝ :=
  let t1 := d1 / v1
  let t2 := d2 / v2
  let t3 := d3 / v3
  let time_riding := t1 + t2 + t3
  let time_breaks := b * 25 / 60
  time_riding + time_breaks

theorem dylan_trip_time :
  total_time_of_trip 400 150 700 50 40 60 3 = 24.67 :=
by
  unfold total_time_of_trip
  sorry

end dylan_trip_time_l38_38123


namespace angle_measure_l38_38052

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end angle_measure_l38_38052


namespace jogger_ahead_of_train_l38_38720

noncomputable def distance_ahead_of_train (v_j v_t : ℕ) (L_t t : ℕ) : ℕ :=
  let relative_speed_kmh := v_t - v_j
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  let total_distance := relative_speed_ms * t
  total_distance - L_t

theorem jogger_ahead_of_train :
  distance_ahead_of_train 10 46 120 46 = 340 :=
by
  sorry

end jogger_ahead_of_train_l38_38720


namespace kristy_initial_cookies_l38_38806

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end kristy_initial_cookies_l38_38806


namespace lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l38_38171

/- Define lines l1 and l2 -/
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/- Prove intersection condition -/
theorem lines_intersect (a : ℝ) : (∃ x y, l1 a x y ∧ l2 a x y) ↔ (a ≠ -1 ∧ a ≠ 2) := 
sorry

/- Prove perpendicular condition -/
theorem lines_perpendicular (a : ℝ) : (∃ x1 y1 x2 y2, l1 a x1 y1 ∧ l2 a x2 y2 ∧ x1 * x2 + y1 * y2 = 0) ↔ (a = 2 / 3) :=
sorry

/- Prove coincident condition -/
theorem lines_coincide (a : ℝ) : (∀ x y, l1 a x y ↔ l2 a x y) ↔ (a = 2) := 
sorry

/- Prove parallel condition -/
theorem lines_parallel (a : ℝ) : (∀ x1 y1 x2 y2, l1 a x1 y1 → l2 a x2 y2 → (x1 * y2 - y1 * x2) = 0) ↔ (a = -1) := 
sorry

end lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l38_38171


namespace hyperbola_eccentricity_range_l38_38628

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (x y : ℝ) (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (h_A_B: ∃ A B : ℝ, x = -c ∧ |AF| = b^2 / a ∧ |CF| = a + c) :
  e > 2 :=
by
  sorry

end hyperbola_eccentricity_range_l38_38628


namespace parabola_vertex_coordinates_l38_38835

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), (y = -3 * (x - 1)^2 - 2) → (x, y) = (1, -2) := 
begin
  intros x y h,
  sorry
end

end parabola_vertex_coordinates_l38_38835


namespace no_real_solutions_l38_38302

theorem no_real_solutions :
  ∀ x : ℝ, (2 * x - 6) ^ 2 + 4 ≠ -(x - 3) :=
by
  intro x
  sorry

end no_real_solutions_l38_38302


namespace max_take_home_pay_at_5000_dollars_l38_38499

noncomputable def income_tax (x : ℕ) : ℕ :=
  if x ≤ 5000 then x * 5 / 100
  else 250 + 10 * ((x - 5000 / 1000) - 5) ^ 2

noncomputable def take_home_pay (y : ℕ) : ℕ :=
  y - income_tax y

theorem max_take_home_pay_at_5000_dollars : ∀ y : ℕ, take_home_pay y ≤ take_home_pay 5000 := by
  sorry

end max_take_home_pay_at_5000_dollars_l38_38499


namespace sectionBSeats_l38_38660

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end sectionBSeats_l38_38660


namespace max_edges_in_8_points_graph_no_square_l38_38113

open Finset

-- Define what a graph is and the properties needed for the problem
structure Graph (V : Type*) :=
  (edges : Finset (V × V))
  (sym : ∀ {x y : V}, (x, y) ∈ edges ↔ (y, x) ∈ edges)
  (irrefl : ∀ {x : V}, ¬ (x, x) ∈ edges)

-- Define the conditions of the problem
def no_square {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c d : V), 
    (a, b) ∈ G.edges → (b, c) ∈ G.edges → (c, d) ∈ G.edges → (d, a) ∈ G.edges →
    (a, c) ∈ G.edges → (b, d) ∈ G.edges → False

-- Define 8 vertices
inductive Vertices
| A | B | C | D | E | F | G | H

-- Define the number of edges
noncomputable def max_edges_no_square : ℕ :=
  11

-- Define the final theorem
theorem max_edges_in_8_points_graph_no_square :
  ∃ (G : Graph Vertices), 
    no_square G ∧ (G.edges.card = max_edges_no_square) :=
sorry

end max_edges_in_8_points_graph_no_square_l38_38113


namespace each_friend_eats_six_slices_l38_38577

-- Definitions
def slices_per_loaf : ℕ := 15
def loaves_bought : ℕ := 4
def friends : ℕ := 10
def total_slices : ℕ := loaves_bought * slices_per_loaf
def slices_per_friend : ℕ := total_slices / friends

-- Theorem to prove
theorem each_friend_eats_six_slices (h1 : slices_per_loaf = 15) (h2 : loaves_bought = 4) (h3 : friends = 10) : slices_per_friend = 6 :=
by
  sorry

end each_friend_eats_six_slices_l38_38577


namespace train_speed_fraction_l38_38115

theorem train_speed_fraction (T : ℝ) (hT : T = 3) : T / (T + 0.5) = 6 / 7 := by
  sorry

end train_speed_fraction_l38_38115


namespace problem_1_solution_set_problem_2_min_value_l38_38007

-- Problem (1)
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_1_solution_set :
  {x : ℝ | f (x + 3/2) ≥ 0} = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

-- Problem (2)
theorem problem_2_min_value (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 :=
by
  sorry

end problem_1_solution_set_problem_2_min_value_l38_38007


namespace bobby_toy_cars_in_5_years_l38_38600

noncomputable def toy_cars_after_n_years (initial_cars : ℕ) (percentage_increase : ℝ) (n : ℕ) : ℝ :=
initial_cars * (1 + percentage_increase)^n

theorem bobby_toy_cars_in_5_years :
  toy_cars_after_n_years 25 0.75 5 = 410 := by
  -- 25 * (1 + 0.75)^5 
  -- = 25 * (1.75)^5 
  -- ≈ 410.302734375
  -- After rounding
  sorry

end bobby_toy_cars_in_5_years_l38_38600


namespace problem_equivalent_l38_38218

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l38_38218


namespace difference_in_cm_l38_38055

def line_length : ℝ := 80  -- The length of the line is 80.0 centimeters
def diff_length_factor : ℝ := 0.35  -- The difference factor in the terms of the line's length

theorem difference_in_cm (l : ℝ) (d : ℝ) (h₀ : l = 80) (h₁ : d = 0.35 * l) : d = 28 :=
by
  sorry

end difference_in_cm_l38_38055


namespace minimal_q_for_fraction_l38_38665

theorem minimal_q_for_fraction :
  ∃ p q : ℕ, 0 < p ∧ 0 < q ∧ 
  (3/5 : ℚ) < p / q ∧ p / q < (5/8 : ℚ) ∧
  (∀ r : ℕ, 0 < r ∧ (3/5 : ℚ) < p / r ∧ p / r < (5/8 : ℚ) → q ≤ r) ∧
  p + q = 21 :=
by
  sorry

end minimal_q_for_fraction_l38_38665


namespace complement_union_eq_l38_38203

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l38_38203


namespace problem_statement_l38_38227

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l38_38227


namespace length_of_PQ_l38_38666

theorem length_of_PQ (R P Q : ℝ × ℝ) (hR : R = (10, 8))
(hP_line1 : ∃ p : ℝ, P = (p, 24 * p / 7))
(hQ_line2 : ∃ q : ℝ, Q = (q, 5 * q / 13))
(h_mid : ∃ (p q : ℝ), R = ((p + q) / 2, (24 * p / 14 + 5 * q / 26) / 2))
(answer_eq : ∃ (a b : ℕ), PQ_length = a / b ∧ a.gcd b = 1 ∧ a + b = 4925) : 
∃ a b : ℕ, a + b = 4925 := sorry

end length_of_PQ_l38_38666


namespace reflection_twice_is_identity_l38_38951

-- Define the reflection matrix R over the vector (1, 2)
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  -- Note: The specific definition of the reflection matrix over (1, 2) is skipped as we only need the final proof statement.
  sorry

-- Assign the reflection matrix R to variable R
def R := reflection_matrix

-- Prove that R^2 = I
theorem reflection_twice_is_identity : R * R = 1 := by
  sorry

end reflection_twice_is_identity_l38_38951


namespace system_of_equations_l38_38912

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end system_of_equations_l38_38912


namespace hank_route_distance_l38_38826

theorem hank_route_distance 
  (d : ℝ) 
  (h1 : ∃ t1 : ℝ, t1 = d / 70 ∧ t1 = d / 70 + 1 / 60) 
  (h2 : ∃ t2 : ℝ, t2 = d / 75 ∧ t2 = d / 75 - 1 / 60) 
  (time_diff : (d / 70 - d / 75) = 1 / 30) : 
  d = 35 :=
sorry

end hank_route_distance_l38_38826


namespace range_of_a_l38_38317

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | 5 < x}
  (A ∩ B = ∅) ↔ a ∈ {a : ℝ | a ≤ 2 ∨ a > 3} :=
by
  sorry

end range_of_a_l38_38317


namespace PointNegativeThreeTwo_l38_38043

def isInSecondQuadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem PointNegativeThreeTwo:
  isInSecondQuadrant (-3) 2 := by
  sorry

end PointNegativeThreeTwo_l38_38043


namespace speed_conversion_l38_38410

noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

theorem speed_conversion (h : kmh_to_ms 1 = 1000 / 3600) :
  kmh_to_ms 1.7 = 0.4722 :=
by sorry

end speed_conversion_l38_38410


namespace triangle_area_l38_38343

theorem triangle_area (AB CD : ℝ) (h₁ : 0 < AB) (h₂ : 0 < CD) (h₃ : CD = 3 * AB) :
    let trapezoid_area := 18
    let triangle_ABC_area := trapezoid_area / 4
    triangle_ABC_area = 4.5 := by
  sorry

end triangle_area_l38_38343


namespace problem_statement_l38_38750

-- Define operations "※" and "#"
def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

-- Define the proof statement
theorem problem_statement : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end problem_statement_l38_38750


namespace game_configurations_count_l38_38187

-- Definitions of the game conditions
def grid_size : ℕ × ℕ := (5, 7)

-- The number of unique paths in the grid game
def count_configurations (m n : ℕ) : ℕ := Nat.choose (m + n) m

-- The main theorem stating the number of different possible situations in the game
theorem game_configurations_count : count_configurations 7 5 = 792 := by
  unfold count_configurations
  simp [Nat.choose]
  norm_num
  sorry

end game_configurations_count_l38_38187


namespace find_cheesecake_price_l38_38938

def price_of_cheesecake (C : ℝ) (coffee_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  let original_price := coffee_price + C
  let discounted_price := discount_rate * original_price
  discounted_price = final_price

theorem find_cheesecake_price : ∃ C : ℝ,
  price_of_cheesecake C 6 0.75 12 ∧ C = 10 :=
by
  sorry

end find_cheesecake_price_l38_38938


namespace friendly_triangle_angle_l38_38889

theorem friendly_triangle_angle (α : ℝ) (β : ℝ) (γ : ℝ) (hα12β : α = 2 * β) (h_sum : α + β + γ = 180) :
    (α = 42 ∨ α = 84 ∨ α = 92) ∧ (42 = β ∨ 42 = γ) := 
sorry

end friendly_triangle_angle_l38_38889


namespace find_number_of_moles_of_CaCO3_formed_l38_38618

-- Define the molar ratios and the given condition in structures.
structure Reaction :=
  (moles_CaOH2 : ℕ)
  (moles_CO2 : ℕ)
  (moles_CaCO3 : ℕ)

-- Define a balanced reaction for Ca(OH)2 + CO2 -> CaCO3 + H2O with 1:1 molar ratio.
def balanced_reaction (r : Reaction) : Prop :=
  r.moles_CaOH2 = r.moles_CO2 ∧ r.moles_CaCO3 = r.moles_CO2

-- Define the given condition, which is we have 3 moles of CO2 and formed 3 moles of CaCO3.
def given_condition : Reaction :=
  { moles_CaOH2 := 3, moles_CO2 := 3, moles_CaCO3 := 3 }

-- Theorem: Given 3 moles of CO2, we need to prove 3 moles of CaCO3 are formed based on the balanced reaction.
theorem find_number_of_moles_of_CaCO3_formed :
  balanced_reaction given_condition :=
by {
  -- This part will contain the proof when implemented.
  sorry
}

end find_number_of_moles_of_CaCO3_formed_l38_38618


namespace complement_union_complement_l38_38234

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- The proof problem
theorem complement_union_complement : (U \ (M ∪ N)) = {1, 6} := by
  sorry

end complement_union_complement_l38_38234


namespace sum_of_variables_l38_38334

variables (a b c d : ℝ)

theorem sum_of_variables :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 → a + b + c + d = 16 :=
by
  intro h
  -- your proof goes here
  sorry

end sum_of_variables_l38_38334


namespace incorrect_statement_l38_38506

noncomputable def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem incorrect_statement
  (a b c : ℤ) (h₀ : a ≠ 0)
  (h₁ : 2 * a + b = 0)
  (h₂ : f a b c 1 = 3)
  (h₃ : f a b c 2 = 8) :
  ¬ (f a b c (-1) = 0) :=
sorry

end incorrect_statement_l38_38506


namespace exist_x_y_satisfy_condition_l38_38966

theorem exist_x_y_satisfy_condition (f g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 0) (h2 : ∀ y, 0 ≤ y ∧ y ≤ 1 → g y ≥ 0) :
  ∃ (x : ℝ), ∃ (y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |f x + g y - x * y| ≥ 1 / 4 :=
by
  sorry

end exist_x_y_satisfy_condition_l38_38966


namespace order_of_abc_l38_38319

theorem order_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a^2 + b^2 < a^2 + c^2) (h2 : a^2 + c^2 < b^2 + c^2) : a < b ∧ b < c := 
by
  sorry

end order_of_abc_l38_38319


namespace graph_shift_proof_l38_38265

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)
noncomputable def h (x : ℝ) : ℝ := g (x + Real.pi / 8)

theorem graph_shift_proof : ∀ x, h x = f x := by
  sorry

end graph_shift_proof_l38_38265


namespace solve_inequality_l38_38971

open Set

theorem solve_inequality (x : ℝ) (h : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0) : x ∈ Ioo 0 1 := by
  sorry

end solve_inequality_l38_38971


namespace binom_18_10_l38_38746

open Nat

-- Definitions of given binomial conditions
def binom_16_7 : ℕ := 11440
def binom_16_9 : ℕ := 11440

-- Define the equivalent Lean proof problem.
theorem binom_18_10 : Nat.choose 18 10 = 43758 :=
by
  -- Definitions of required intermediate binomial coefficients
  let binom_16_6 := 8008
  let binom_16_8 := 12870
  
  -- Stating the values provided in the problem
  have h1 : Nat.choose 16 7 = binom_16_7 := rfl
  have h2 : Nat.choose 16 9 = binom_16_9 := rfl

  -- Using these values to derive the final result
  have h3 : Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 := by
    -- Expand using Pascal's Rule
    sorry

  -- Combined result
  show Nat.choose 18 10 = 43758 from
  calc
    Nat.choose 18 10 = Nat.choose 16 9 + binom_16_6 + binom_16_8 + Nat.choose 16 9 : by rw <- h3
    ... = 11440 + 8008 + 12870 + 11440 : by rw [h2, h1]
    ... = 43758 : by norm_num

end binom_18_10_l38_38746


namespace min_value_of_expression_l38_38952

open Real

noncomputable def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / (x + 2) + 1 / (y + 2) = 1 / 4)

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 4) :
  2 * x + 3 * y = 5 + 4 * sqrt 3 :=
sorry

end min_value_of_expression_l38_38952


namespace gcd_2703_1113_l38_38379

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := 
by 
  sorry

end gcd_2703_1113_l38_38379


namespace min_additional_trains_needed_l38_38235

-- Definitions
def current_trains : ℕ := 31
def trains_per_row : ℕ := 8
def smallest_num_additional_trains (current : ℕ) (per_row : ℕ) : ℕ :=
  let next_multiple := ((current + per_row - 1) / per_row) * per_row
  next_multiple - current

-- Theorem
theorem min_additional_trains_needed :
  smallest_num_additional_trains current_trains trains_per_row = 1 :=
by
  sorry

end min_additional_trains_needed_l38_38235


namespace this_week_usage_less_next_week_usage_less_l38_38861

def last_week_usage : ℕ := 91

def usage_this_week : ℕ := (4 * 8) + (3 * 10)

def usage_next_week : ℕ := (5 * 5) + (2 * 12)

theorem this_week_usage_less : last_week_usage - usage_this_week = 29 := by
  -- proof goes here
  sorry

theorem next_week_usage_less : last_week_usage - usage_next_week = 42 := by
  -- proof goes here
  sorry

end this_week_usage_less_next_week_usage_less_l38_38861


namespace maurice_rides_l38_38820

theorem maurice_rides (M : ℕ) 
    (h1 : ∀ m_attended : ℕ, m_attended = 8)
    (h2 : ∀ matt_other : ℕ, matt_other = 16)
    (h3 : ∀ total_matt : ℕ, total_matt = matt_other + m_attended)
    (h4 : total_matt = 3 * M) : M = 8 :=
by 
  sorry

end maurice_rides_l38_38820


namespace discriminant_negative_of_positive_parabola_l38_38798

variable (a b c : ℝ)

theorem discriminant_negative_of_positive_parabola (h1 : ∀ x : ℝ, a * x^2 + b * x + c > 0) (h2 : a > 0) : b^2 - 4*a*c < 0 := 
sorry

end discriminant_negative_of_positive_parabola_l38_38798


namespace quadrilateral_area_is_114_5_l38_38888

noncomputable def area_of_quadrilateral_114_5 
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) : ℝ :=
  114.5

theorem quadrilateral_area_is_114_5
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) :
  area_of_quadrilateral_114_5 AB BC CD AD angle_ABC h1 h2 h3 h4 h5 = 114.5 :=
sorry

end quadrilateral_area_is_114_5_l38_38888


namespace expression_simplification_l38_38603

theorem expression_simplification :
  (4 * 6 / (12 * 8)) * ((5 * 12 * 8) / (4 * 5 * 5)) = 1 / 2 :=
by
  sorry

end expression_simplification_l38_38603


namespace shirts_needed_for_vacation_l38_38332

def vacation_days := 7
def same_shirt_days := 2
def different_shirts_per_day := 2
def different_shirt_days := vacation_days - same_shirt_days

theorem shirts_needed_for_vacation : different_shirt_days * different_shirts_per_day + same_shirt_days = 11 := by
  sorry

end shirts_needed_for_vacation_l38_38332


namespace cos_315_eq_sqrt2_div2_l38_38433

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l38_38433


namespace minimum_value_l38_38845

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.cos x)^2 - 2 * (Real.sin x) + 9 / 2

theorem minimum_value :
  ∃ (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)), f x = 2 :=
by
  use Real.pi / 6
  sorry

end minimum_value_l38_38845


namespace butterfat_in_final_mixture_l38_38173

noncomputable def final_butterfat_percentage (gallons_of_35_percentage : ℕ) 
                                             (percentage_of_35_butterfat : ℝ) 
                                             (total_gallons : ℕ)
                                             (percentage_of_10_butterfat : ℝ) : ℝ :=
  let gallons_of_10 := total_gallons - gallons_of_35_percentage
  let butterfat_35 := gallons_of_35_percentage * percentage_of_35_butterfat
  let butterfat_10 := gallons_of_10 * percentage_of_10_butterfat
  let total_butterfat := butterfat_35 + butterfat_10
  (total_butterfat / total_gallons) * 100

theorem butterfat_in_final_mixture : 
  final_butterfat_percentage 8 0.35 12 0.10 = 26.67 :=
sorry

end butterfat_in_final_mixture_l38_38173


namespace remainder_base12_2543_div_9_l38_38401

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l38_38401


namespace Gilda_marbles_left_l38_38766

theorem Gilda_marbles_left (M : ℝ) (h1 : M > 0) :
  let remaining_after_pedro := M - 0.30 * M
  let remaining_after_ebony := remaining_after_pedro - 0.40 * remaining_after_pedro
  remaining_after_ebony / M * 100 = 42 :=
by
  sorry

end Gilda_marbles_left_l38_38766


namespace central_angle_of_sector_l38_38183

theorem central_angle_of_sector 
  (r : ℝ) (s : ℝ) (c : ℝ)
  (h1 : r = 5)
  (h2 : s = 15)
  (h3 : c = 2 * π * r) :
  ∃ n : ℝ, (n * s * π / 180 = c) ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_sector_l38_38183


namespace intersection_M_N_l38_38630

open Set

def M : Set ℝ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} :=
by
  sorry

end intersection_M_N_l38_38630


namespace correct_value_l38_38707

theorem correct_value (x : ℕ) (h : 14 * x = 42) : 12 * x = 36 := by
  sorry

end correct_value_l38_38707


namespace at_least_one_angle_not_greater_than_60_l38_38363

theorem at_least_one_angle_not_greater_than_60 (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (hSum : A + B + C = 180) : false :=
by
  sorry

end at_least_one_angle_not_greater_than_60_l38_38363


namespace simplify_expression_l38_38539

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l38_38539


namespace second_group_product_number_l38_38261

theorem second_group_product_number (a₀ : ℕ) (h₀ : 0 ≤ a₀ ∧ a₀ < 20)
  (h₁ : 4 * 20 + a₀ = 94) : 1 * 20 + a₀ = 34 :=
by
  sorry

end second_group_product_number_l38_38261


namespace necessary_but_not_sufficient_for_x_gt_4_l38_38101

theorem necessary_but_not_sufficient_for_x_gt_4 (x : ℝ) : (x^2 > 16) → ¬ (x > 4) :=
by
  sorry

end necessary_but_not_sufficient_for_x_gt_4_l38_38101


namespace generalized_schur_inequality_l38_38100

theorem generalized_schur_inequality (t : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 :=
sorry

end generalized_schur_inequality_l38_38100


namespace ludwig_weekly_earnings_l38_38523

theorem ludwig_weekly_earnings :
  (7 = 7) ∧
  (∀ day : ℕ, day ∈ {5, 6, 7} → (1 / 2) = 1 / 2) ∧
  (daily_salary = 10) →
  (weekly_earnings = 55) :=
by
  sorry

end ludwig_weekly_earnings_l38_38523


namespace cost_per_book_l38_38829

theorem cost_per_book (initial_amount : ℤ) (remaining_amount : ℤ) (num_books : ℤ) (cost_per_book : ℤ) :
  initial_amount = 79 →
  remaining_amount = 16 →
  num_books = 9 →
  cost_per_book = (initial_amount - remaining_amount) / num_books →
  cost_per_book = 7 := 
by
  sorry

end cost_per_book_l38_38829


namespace quadratic_has_real_roots_b_3_c_1_l38_38771

theorem quadratic_has_real_roots_b_3_c_1 :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x * x + 3 * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  x₁ = (-3 + Real.sqrt 5) / 2 ∧
  x₂ = (-3 - Real.sqrt 5) / 2 :=
by
  sorry

end quadratic_has_real_roots_b_3_c_1_l38_38771


namespace spadesuit_eval_l38_38609

def spadesuit (a b : ℤ) := abs (a - b)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 3 (spadesuit 8 12)) = 4 := 
by
  sorry

end spadesuit_eval_l38_38609


namespace tied_in_runs_l38_38374

def aaron_runs : List ℕ := [4, 8, 15, 7, 4, 12, 11, 5]
def bonds_runs : List ℕ := [3, 5, 18, 9, 12, 14, 9, 0]

def total_runs (runs : List ℕ) : ℕ := runs.foldl (· + ·) 0

theorem tied_in_runs : total_runs aaron_runs = total_runs bonds_runs := by
  sorry

end tied_in_runs_l38_38374


namespace fruit_weight_sister_and_dad_l38_38530

-- Defining the problem statement and conditions
variable (strawberries_m blueberries_m raspberries_m : ℝ)
variable (strawberries_d blueberries_d raspberries_d : ℝ)
variable (strawberries_s blueberries_s raspberries_s : ℝ)
variable (total_weight : ℝ)

-- Given initial conditions
def conditions : Prop :=
  strawberries_m = 5 ∧
  blueberries_m = 3 ∧
  raspberries_m = 6 ∧
  strawberries_d = 2 * strawberries_m ∧
  blueberries_d = 2 * blueberries_m ∧
  raspberries_d = 2 * raspberries_m ∧
  strawberries_s = strawberries_m / 2 ∧
  blueberries_s = blueberries_m / 2 ∧
  raspberries_s = raspberries_m / 2 ∧
  total_weight = (strawberries_m + blueberries_m + raspberries_m) + 
                 (strawberries_d + blueberries_d + raspberries_d) + 
                 (strawberries_s + blueberries_s + raspberries_s)

-- Defining the property to prove
theorem fruit_weight_sister_and_dad :
  conditions strawberries_m blueberries_m raspberries_m strawberries_d blueberries_d raspberries_d strawberries_s blueberries_s raspberries_s total_weight →
  (strawberries_d + blueberries_d + raspberries_d) +
  (strawberries_s + blueberries_s + raspberries_s) = 35 := by
  sorry

end fruit_weight_sister_and_dad_l38_38530


namespace rooks_control_chosen_squares_l38_38186

theorem rooks_control_chosen_squares (n : Nat) 
  (chessboard : Fin (2 * n) × Fin (2 * n)) 
  (chosen_squares : Finset (Fin (2 * n) × Fin (2 * n))) 
  (h : chosen_squares.card = 3 * n) :
  ∃ rooks : Finset (Fin (2 * n) × Fin (2 * n)), rooks.card = n ∧
  ∀ (square : Fin (2 * n) × Fin (2 * n)), square ∈ chosen_squares → 
  (square ∈ rooks ∨ ∃ (rook : Fin (2 * n) × Fin (2 * n)) (hr : rook ∈ rooks), 
  rook.1 = square.1 ∨ rook.2 = square.2) :=
sorry

end rooks_control_chosen_squares_l38_38186


namespace find_a_of_even_function_l38_38956

-- Define the function f
def f (x a : ℝ) := (x + 1) * (x + a)

-- State the theorem to be proven
theorem find_a_of_even_function (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  -- The actual proof goes here
  sorry

end find_a_of_even_function_l38_38956


namespace more_chickens_than_chicks_l38_38558

-- Let's define the given conditions
def total : Nat := 821
def chicks : Nat := 267

-- The statement we need to prove
theorem more_chickens_than_chicks : (total - chicks) - chicks = 287 :=
by
  -- This is needed for the proof and not part of conditions
  -- Add sorry as a placeholder for proof steps 
  sorry

end more_chickens_than_chicks_l38_38558


namespace rate_of_current_l38_38413

variable (c : ℝ)

-- Define the given conditions
def speed_still_water : ℝ := 4.5
def time_ratio : ℝ := 2

-- Define the effective speeds
def speed_downstream : ℝ := speed_still_water + c
def speed_upstream : ℝ := speed_still_water - c

-- Define the condition that it takes twice as long to row upstream as downstream
def rowing_equation : Prop := 1 / speed_upstream = 2 * (1 / speed_downstream)

-- The Lean theorem stating the problem we need to prove
theorem rate_of_current (h : rowing_equation) : c = 1.5 := by
  sorry

end rate_of_current_l38_38413


namespace absolute_value_inequality_solution_set_l38_38388

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end absolute_value_inequality_solution_set_l38_38388


namespace sausages_placement_and_path_length_l38_38041

variables {a b x y : ℝ} (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
variables (h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y)

theorem sausages_placement_and_path_length (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
(h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y) : 
  x < y ∧ (x / y) = 1.4 :=
by {
  sorry
}

end sausages_placement_and_path_length_l38_38041


namespace question_proof_l38_38213

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l38_38213


namespace size_of_each_group_l38_38852

theorem size_of_each_group 
  (skittles : ℕ) (erasers : ℕ) (groups : ℕ)
  (h_skittles : skittles = 4502) (h_erasers : erasers = 4276) (h_groups : groups = 154) :
  (skittles + erasers) / groups = 57 :=
by
  sorry

end size_of_each_group_l38_38852


namespace quadratic_three_distinct_solutions_l38_38831

open Classical

variable (a b c : ℝ) (x1 x2 x3 : ℝ)

-- Conditions:
variables (hx1 : a * x1^2 + b * x1 + c = 0)
          (hx2 : a * x2^2 + b * x2 + c = 0)
          (hx3 : a * x3^2 + b * x3 + c = 0)
          (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

-- Proof problem
theorem quadratic_three_distinct_solutions : a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end quadratic_three_distinct_solutions_l38_38831


namespace simplify_expr_l38_38795

theorem simplify_expr (a b x : ℝ) (h₁ : x = a^3 / b^3) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) := 
by 
  sorry

end simplify_expr_l38_38795


namespace rational_inequalities_l38_38156

theorem rational_inequalities (a b c d : ℚ)
  (h : a^3 - 2005 = b^3 + 2027 ∧ b^3 + 2027 = c^3 - 2822 ∧ c^3 - 2822 = d^3 + 2820) :
  c > a ∧ a > b ∧ b > d :=
by
  sorry

end rational_inequalities_l38_38156


namespace solve_inequality_l38_38046

theorem solve_inequality (x : ℝ) :
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3 / 2 :=
sorry

end solve_inequality_l38_38046


namespace find_bags_l38_38063

theorem find_bags (x : ℕ) : 10 + x + 7 = 20 → x = 3 :=
by
  sorry

end find_bags_l38_38063


namespace remainder_base12_2543_div_9_l38_38402

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l38_38402


namespace binom_18_10_l38_38744

/-- Placeholder for combinatorics and factorial, both of which need to be defined properly in Lean. -/
noncomputable def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

theorem binom_18_10 :
  binom 16 7 = 11440 →
  binom 16 9 = 11440 →
  binom 18 10 = 45760 :=
by
  intros h1 h2
  sorry

end binom_18_10_l38_38744


namespace base12_division_remainder_l38_38399

theorem base12_division_remainder :
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3 in
  n % 9 = 8 :=
by
  let n := 2 * (12^3) + 5 * (12^2) + 4 * 12 + 3
  show n % 9 = 8
  sorry

end base12_division_remainder_l38_38399


namespace original_triangle_area_l38_38056

-- Define the scaling factor and given areas
def scaling_factor : ℕ := 2
def new_triangle_area : ℕ := 32

-- State that if the dimensions of the original triangle are doubled, the area becomes 32 square feet
theorem original_triangle_area (original_area : ℕ) : (scaling_factor * scaling_factor) * original_area = new_triangle_area → original_area = 8 := 
by
  intros h
  sorry

end original_triangle_area_l38_38056


namespace negation_of_proposition_l38_38257

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l38_38257


namespace cost_to_cover_wall_with_tiles_l38_38560

/--
There is a wall in the shape of a rectangle with a width of 36 centimeters (cm) and a height of 72 centimeters (cm).
On this wall, you want to attach tiles that are 3 centimeters (cm) and 4 centimeters (cm) in length and width, respectively,
without any empty space. If it costs 2500 won per tile, prove that the total cost to cover the wall is 540,000 won.

Conditions:
- width_wall = 36
- height_wall = 72
- width_tile = 3
- height_tile = 4
- cost_per_tile = 2500

Target:
- Total_cost = 540,000 won
-/
theorem cost_to_cover_wall_with_tiles :
  let width_wall := 36
  let height_wall := 72
  let width_tile := 3
  let height_tile := 4
  let cost_per_tile := 2500
  let area_wall := width_wall * height_wall
  let area_tile := width_tile * height_tile
  let number_of_tiles := area_wall / area_tile
  let total_cost := number_of_tiles * cost_per_tile
  total_cost = 540000 := by
  sorry

end cost_to_cover_wall_with_tiles_l38_38560


namespace proof_problem_l38_38160

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := (x + 1) * f x

axiom domain_f : ∀ x : ℝ, true
axiom even_f : ∀ x : ℝ, f (2 * x - 1) = f (-(2 * x - 1))
axiom mono_g_neg_inf_minus_1 : ∀ x y : ℝ, x ≤ y → x ≤ -1 → y ≤ -1 → g x ≤ g y

-- Proof Problem Statement
theorem proof_problem :
  (∀ x y : ℝ, x ≤ y → -1 ≤ x → -1 ≤ y → g x ≤ g y) ∧
  (∀ a b : ℝ, g a + g b > 0 → a + b + 2 > 0) :=
by
  sorry

end proof_problem_l38_38160


namespace necklaces_caught_l38_38613

theorem necklaces_caught
  (LatchNecklaces RhondaNecklaces BoudreauxNecklaces: ℕ)
  (h1 : LatchNecklaces = 3 * RhondaNecklaces - 4)
  (h2 : RhondaNecklaces = BoudreauxNecklaces / 2)
  (h3 : BoudreauxNecklaces = 12) :
  LatchNecklaces = 14 := by
  sorry

end necklaces_caught_l38_38613


namespace combined_rate_l38_38305

theorem combined_rate
  (earl_rate : ℕ)
  (ellen_time : ℚ)
  (total_envelopes : ℕ)
  (total_time : ℕ)
  (combined_total_envelopes : ℕ)
  (combined_total_time : ℕ) :
  earl_rate = 36 →
  ellen_time = 1.5 →
  total_envelopes = 36 →
  total_time = 1 →
  combined_total_envelopes = 180 →
  combined_total_time = 3 →
  (earl_rate + (total_envelopes / ellen_time)) = 60 :=
by
  sorry

end combined_rate_l38_38305


namespace inequality_holds_l38_38923

theorem inequality_holds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : a^2 + b^2 ≥ 2 :=
sorry

end inequality_holds_l38_38923


namespace emily_euros_contribution_l38_38426

-- Declare the conditions as a definition
def conditions : Prop :=
  ∃ (cost_of_pie : ℝ) (emily_usd : ℝ) (berengere_euros : ℝ) (exchange_rate : ℝ),
    cost_of_pie = 15 ∧
    emily_usd = 10 ∧
    berengere_euros = 3 ∧
    exchange_rate = 1.1

-- Define the proof problem based on the conditions and required contribution
theorem emily_euros_contribution : conditions → (∃ emily_euros_more : ℝ, emily_euros_more = 3) :=
by
  intro h
  sorry

end emily_euros_contribution_l38_38426


namespace total_units_l38_38960

theorem total_units (A B C: ℕ) (hA: A = 2 + 4 + 6 + 8 + 10 + 12) (hB: B = A) (hC: C = 3 + 5 + 7 + 9) : 
  A + B + C = 108 := 
sorry

end total_units_l38_38960


namespace rectangle_area_error_percentage_l38_38650

theorem rectangle_area_error_percentage (L W : ℝ) :
  let L' := 1.10 * L
  let W' := 0.95 * W
  let A := L * W 
  let A' := L' * W'
  let error := A' - A
  let error_percentage := (error / A) * 100
  error_percentage = 4.5 := by
  sorry

end rectangle_area_error_percentage_l38_38650


namespace abc_sum_l38_38377

theorem abc_sum : ∃ a b c : ℤ, 
  (∀ x : ℤ, x^2 + 13 * x + 30 = (x + a) * (x + b)) ∧ 
  (∀ x : ℤ, x^2 + 5 * x - 50 = (x + b) * (x - c)) ∧
  a + b + c = 18 := by
  sorry

end abc_sum_l38_38377


namespace power_modulus_l38_38704

theorem power_modulus (n : ℕ) : (2 : ℕ) ^ 345 % 5 = 2 :=
by sorry

end power_modulus_l38_38704


namespace b_alone_completion_days_l38_38276

theorem b_alone_completion_days (Rab : ℝ) (w_12_days : (1 / (Rab + 4 * Rab)) = 12⁻¹) : 
    (1 / Rab = 60) :=
sorry

end b_alone_completion_days_l38_38276


namespace minimum_number_of_guests_l38_38252

theorem minimum_number_of_guests :
  ∀ (total_food : ℝ) (max_food_per_guest : ℝ), total_food = 411 → max_food_per_guest = 2.5 →
  ⌈total_food / max_food_per_guest⌉ = 165 :=
by
  intros total_food max_food_per_guest h1 h2
  rw [h1, h2]
  norm_num
  sorry

end minimum_number_of_guests_l38_38252


namespace total_students_count_l38_38939

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) : Prop := g * 4 = b * 3
def boys_count : ℕ := 28

-- Theorem to prove the total number of students
theorem total_students_count {g : ℕ} (h : ratio_girls_to_boys g boys_count) : g + boys_count = 49 :=
sorry

end total_students_count_l38_38939


namespace travel_time_l38_38414

theorem travel_time (speed distance : ℕ) (h_speed : speed = 100) (h_distance : distance = 500) :
  distance / speed = 5 := by
  sorry

end travel_time_l38_38414


namespace maximum_teams_tied_for_most_wins_l38_38342

/-- In a round-robin tournament with 8 teams, each team plays one game
    against each other team, and each game results in one team winning
    and one team losing. -/
theorem maximum_teams_tied_for_most_wins :
  ∀ (teams games wins : ℕ), 
    teams = 8 → 
    games = (teams * (teams - 1)) / 2 →
    wins = 28 →
    ∃ (max_tied_teams : ℕ), max_tied_teams = 5 :=
by
  sorry

end maximum_teams_tied_for_most_wins_l38_38342


namespace periodic_function_l38_38812

noncomputable theory
open Function

theorem periodic_function {f : ℝ → ℝ} (h : ∀ x : ℝ, f(x + 1) + f(x - 1) = Real.sqrt 2 * f x) : ∃ T : ℝ, T ≠ 0 ∧ (∀ x : ℝ, f(x + T) = f x) := 
begin
  use 8,
  split,
  { norm_num,
  },
  { intro x,
    sorry,
  },
end

end periodic_function_l38_38812


namespace ratio_in_range_l38_38786

theorem ratio_in_range {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end ratio_in_range_l38_38786


namespace smallest_four_digit_number_l38_38282

theorem smallest_four_digit_number : 
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (∃ (AB CD : ℕ), 
      n = 1000 * (AB / 10) + 100 * (AB % 10) + CD ∧
      ((AB / 10) * 10 + (AB % 10) + 2) * CD = 100 ∧ 
      n / CD = ((AB / 10) * 10 + (AB % 10) + 1)^2) ∧
    n = 1805 :=
by
  sorry

end smallest_four_digit_number_l38_38282


namespace remainder_base12_div_9_l38_38397

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l38_38397


namespace inv_sum_eq_six_l38_38017

theorem inv_sum_eq_six (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := 
by 
  sorry

end inv_sum_eq_six_l38_38017


namespace polygon_sides_l38_38634

theorem polygon_sides (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  (sum_exterior_angles / exterior_angle) = 12 :=
by
  have h3 : ∀ (n : ℝ), n = 360 / 30, from sorry
  rw [h1, h2] at h3
  exact h3

end polygon_sides_l38_38634


namespace john_using_three_colors_l38_38663

theorem john_using_three_colors {total_paint liters_per_color : ℕ} 
    (h1 : total_paint = 15) 
    (h2 : liters_per_color = 5) :
    total_ppaint / liters_per_color = 3 := 
by
  sorry

end john_using_three_colors_l38_38663


namespace greatest_possible_employees_take_subway_l38_38096

variable (P F : ℕ)

def part_time_employees_take_subway : ℕ := P / 3
def full_time_employees_take_subway : ℕ := F / 4

theorem greatest_possible_employees_take_subway 
  (h1 : P + F = 48) : part_time_employees_take_subway P + full_time_employees_take_subway F ≤ 15 := 
sorry

end greatest_possible_employees_take_subway_l38_38096


namespace power_of_sqrt2_minus_1_l38_38537

noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 - 1) ^ n
noncomputable def b (n : ℕ) : ℝ := (Real.sqrt 2 + 1) ^ n
noncomputable def c (n : ℕ) : ℝ := (b n + a n) / 2
noncomputable def d (n : ℕ) : ℝ := (b n - a n) / 2

theorem power_of_sqrt2_minus_1 (n : ℕ) : a n = Real.sqrt (d n ^ 2 + 1) - Real.sqrt (d n ^ 2) :=
by
  sorry

end power_of_sqrt2_minus_1_l38_38537


namespace john_friends_count_l38_38957

-- Define the initial conditions
def initial_amount : ℚ := 7.10
def cost_of_sweets : ℚ := 1.05
def amount_per_friend : ℚ := 1.00
def remaining_amount : ℚ := 4.05

-- Define the intermediate values
def after_sweets : ℚ := initial_amount - cost_of_sweets
def given_away : ℚ := after_sweets - remaining_amount

-- Define the final proof statement
theorem john_friends_count : given_away / amount_per_friend = 2 :=
by
  sorry

end john_friends_count_l38_38957


namespace liu_xiang_hurdles_l38_38418

theorem liu_xiang_hurdles :
  let total_distance := 110
  let first_hurdle_distance := 13.72
  let last_hurdle_distance := 14.02
  let best_time_first_segment := 2.5
  let best_time_last_segment := 1.4
  let hurdle_cycle_time := 0.96
  let num_hurdles := 10
  (total_distance - first_hurdle_distance - last_hurdle_distance) / num_hurdles = 8.28 ∧
  best_time_first_segment + num_hurdles * hurdle_cycle_time + best_time_last_segment  = 12.1 :=
by
  sorry

end liu_xiang_hurdles_l38_38418


namespace trajectory_eq_l38_38060

theorem trajectory_eq {x y : ℝ} (h₁ : (x-2)^2 + y^2 = 1) (h₂ : ∃ r, (x+1)^2 = (x-2)^2 + y^2 - r^2) :
  y^2 = 6 * x - 3 :=
by
  sorry

end trajectory_eq_l38_38060


namespace find_f_of_9_l38_38477

theorem find_f_of_9 (α : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x ^ α)
  (h2 : f 2 = Real.sqrt 2) :
  f 9 = 3 :=
sorry

end find_f_of_9_l38_38477


namespace binom_18_10_l38_38747

theorem binom_18_10 (h1 : nat.choose 16 7 = 11440) (h2 : nat.choose 16 9 = 11440) : nat.choose 18 10 = 45760 := 
by
  sorry

end binom_18_10_l38_38747


namespace number_of_strikers_l38_38724

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l38_38724


namespace complement_union_eq_ge2_l38_38210

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l38_38210


namespace haley_initial_shirts_l38_38013

-- Defining the conditions
def returned_shirts := 6
def endup_shirts := 5

-- The theorem statement
theorem haley_initial_shirts : returned_shirts + endup_shirts = 11 := by 
  sorry

end haley_initial_shirts_l38_38013


namespace statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l38_38089

-- Statement A: Proving the solution set of the inequality
theorem statement_A_solution_set (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ (-1 / 2) < x ∧ x < 1 :=
sorry

-- Statement B: "ab > 1" is not a sufficient condition for "a > 1, b > 1"
theorem statement_B_insufficient_condition (a b : ℝ) :
  (a * b > 1) → ¬(a > 1 ∧ b > 1) :=
sorry

-- Statement C: The negation of p: ∀ x ∈ ℝ, x² > 0 is true
theorem statement_C_negation (x0 : ℝ) : 
  (∀ x : ℝ, x^2 > 0) → ¬ (∃ x0 : ℝ, x0^2 ≤ 0) :=
sorry

-- Statement D: "a < 2" is not a necessary condition for "a < 6"
theorem statement_D_not_necessary_condition (a : ℝ) :
  (a < 2) → ¬(a < 6) :=
sorry

end statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l38_38089


namespace ratio_of_triangle_and_hexagon_l38_38364

variable {n m : ℝ}

-- Conditions:
def is_regular_hexagon (ABCDEF : Type) : Prop := sorry
def area_of_hexagon (ABCDEF : Type) (n : ℝ) : Prop := sorry
def area_of_triangle_ACE (ABCDEF : Type) (m : ℝ) : Prop := sorry
  
theorem ratio_of_triangle_and_hexagon
  (ABCDEF : Type)
  (H1 : is_regular_hexagon ABCDEF)
  (H2 : area_of_hexagon ABCDEF n)
  (H3 : area_of_triangle_ACE ABCDEF m) :
  m / n = 2 / 3 := 
  sorry

end ratio_of_triangle_and_hexagon_l38_38364


namespace winning_percentage_l38_38503

theorem winning_percentage (total_votes winner_votes : ℕ) 
  (h1 : winner_votes = 1344) 
  (h2 : winner_votes - 288 = total_votes - winner_votes) : 
  (winner_votes * 100 / total_votes = 56) :=
sorry

end winning_percentage_l38_38503


namespace password_encryption_l38_38092

variables (a b x : ℝ)

theorem password_encryption :
  3 * a * (x^2 - 1) - 3 * b * (x^2 - 1) = 3 * (x + 1) * (x - 1) * (a - b) :=
by sorry

end password_encryption_l38_38092


namespace book_cost_l38_38830

theorem book_cost 
  (initial_money : ℕ) 
  (num_books : ℕ) 
  (money_left : ℕ) 
  (h_init : initial_money = 79) 
  (h_books : num_books = 9) 
  (h_left : money_left = 16) : 
  (initial_money - money_left) / num_books = 7 :=
by
  rw [h_init, h_books, h_left] 
  norm_num
  sorry

end book_cost_l38_38830


namespace complement_union_eq_ge2_l38_38211

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l38_38211


namespace female_muscovy_ducks_l38_38393

theorem female_muscovy_ducks :
  let total_ducks := 40
  let muscovy_percentage := 0.5
  let female_muscovy_percentage := 0.3
  let muscovy_ducks := total_ducks * muscovy_percentage
  let female_muscovy_ducks := muscovy_ducks * female_muscovy_percentage
  female_muscovy_ducks = 6 :=
by
  sorry

end female_muscovy_ducks_l38_38393


namespace avg_decrease_by_one_l38_38689

noncomputable def average_decrease (obs : Fin 7 → ℕ) : ℕ :=
  let sum6 := 90
  let seventh := 8
  let new_sum := sum6 + seventh
  let new_avg := new_sum / 7
  let old_avg := 15
  old_avg - new_avg

theorem avg_decrease_by_one :
  (average_decrease (fun _ => 0)) = 1 :=
by
  sorry

end avg_decrease_by_one_l38_38689


namespace sitio_proof_l38_38891

theorem sitio_proof :
  (∃ t : ℝ, t = 4 + 7 + 12 ∧ 
    (∃ f : ℝ, 
      (∃ s : ℝ, s = 6 + 5 + 10 ∧ t = 23 ∧ f = 23 - s) ∧
      f = 2) ∧
    (∃ cost_per_hectare : ℝ, cost_per_hectare = 2420 / (4 + 12) ∧ 
      (∃ saci_spent : ℝ, saci_spent = 6 * cost_per_hectare ∧ saci_spent = 1320))) :=
by sorry

end sitio_proof_l38_38891


namespace cos_315_proof_l38_38441

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l38_38441


namespace no_real_solution_l38_38754

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end no_real_solution_l38_38754


namespace probability_at_least_one_defective_item_l38_38005

def total_products : ℕ := 10
def defective_items : ℕ := 3
def selected_items : ℕ := 3
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_least_one_defective_item :
    let total_combinations := comb total_products selected_items
    let non_defective_combinations := comb (total_products - defective_items) selected_items
    let opposite_probability := (non_defective_combinations : ℚ) / (total_combinations : ℚ)
    let probability := 1 - opposite_probability
    probability = 17 / 24 :=
by
  sorry

end probability_at_least_one_defective_item_l38_38005


namespace find_angle_B_l38_38358

-- Define the parallel lines and angles
variables (l m : ℝ) -- Representing the lines as real numbers for simplicity
variables (A C B : ℝ) -- Representing the angles as real numbers

-- The conditions
def parallel_lines (l m : ℝ) : Prop := l = m
def angle_A (A : ℝ) : Prop := A = 100
def angle_C (C : ℝ) : Prop := C = 60

-- The theorem stating that, given the conditions, the angle B is 120 degrees
theorem find_angle_B (l m : ℝ) (A C B : ℝ) 
  (h1 : parallel_lines l m) 
  (h2 : angle_A A) 
  (h3 : angle_C C) : B = 120 :=
sorry

end find_angle_B_l38_38358


namespace triangle_area_l38_38730

theorem triangle_area (x : ℝ) (h1 : 6 * x = 6) (h2 : 8 * x = 8) (h3 : 10 * x = 2 * 5) : 
  1 / 2 * 6 * 8 = 24 := 
sorry

end triangle_area_l38_38730


namespace number_of_strikers_l38_38723

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l38_38723


namespace minimum_value_l38_38908

theorem minimum_value (x : ℝ) (hx : 0 ≤ x) : ∃ y : ℝ, y = x^2 - 6 * x + 8 ∧ (∀ t : ℝ, 0 ≤ t → y ≤ t^2 - 6 * t + 8) :=
sorry

end minimum_value_l38_38908


namespace domain_of_f_l38_38376

def domain_condition1 (x : ℝ) : Prop := 1 - |x - 1| > 0
def domain_condition2 (x : ℝ) : Prop := x - 1 ≠ 0

theorem domain_of_f :
  (∀ x : ℝ, domain_condition1 x ∧ domain_condition2 x → 0 < x ∧ x < 2 ∧ x ≠ 1) ↔
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2)) :=
by
  sorry

end domain_of_f_l38_38376


namespace probability_eta_geq_2_l38_38519

noncomputable def binomialPGeq (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  1 - (Finset.range k).sum (λ i, (nat.choose n i) * (p^i) * ((1 - p)^(n - i)))

theorem probability_eta_geq_2 (p : ℝ) (η ξ : ℕ → ℕ → ℝ) (hξ : ∀ a b, ξ a b = (nat.choose 2 a) * (p^a) * ((1 - p)^(2 - a)))
  (hη : ∀ a b, η a b = (nat.choose 3 a) * (p^a) * ((1 - p)^(3 - a))) 
  (h5_9 : binomialPGeq 2 p 1 = 5/9) :
  binomialPGeq 3 p 2 = 7/27 :=
by sorry

end probability_eta_geq_2_l38_38519


namespace negation_of_every_planet_orbits_the_sun_l38_38693

variables (Planet : Type) (orbits_sun : Planet → Prop)

theorem negation_of_every_planet_orbits_the_sun :
  (¬ ∀ x : Planet, (¬ (¬ (exists x : Planet, true)) → orbits_sun x)) ↔
  ∃ x : Planet, ¬ orbits_sun x :=
by sorry

end negation_of_every_planet_orbits_the_sun_l38_38693


namespace cos_315_is_sqrt2_div_2_l38_38446

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l38_38446


namespace differential_savings_l38_38639

-- Defining conditions given in the problem
def initial_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

-- Statement of the theorem to prove the differential savings
theorem differential_savings : (annual_income * initial_tax_rate) - (annual_income * new_tax_rate) = 7200 := by
  sorry  -- providing the proof is not required

end differential_savings_l38_38639


namespace percentage_increase_l38_38099

def originalPrice : ℝ := 300
def newPrice : ℝ := 390

theorem percentage_increase :
  ((newPrice - originalPrice) / originalPrice) * 100 = 30 := by
  sorry

end percentage_increase_l38_38099


namespace solve_problem_l38_38480

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end solve_problem_l38_38480


namespace remainder_problem_l38_38668

theorem remainder_problem (x y : ℤ) (k m : ℤ) 
  (hx : x = 126 * k + 11) 
  (hy : y = 126 * m + 25) :
  (x + y + 23) % 63 = 59 := 
by
  sorry

end remainder_problem_l38_38668


namespace oil_leak_before_fix_l38_38881

theorem oil_leak_before_fix (total_leak : ℕ) (leak_during_fix : ℕ) 
    (total_leak_eq : total_leak = 11687) (leak_during_fix_eq : leak_during_fix = 5165) :
    total_leak - leak_during_fix = 6522 :=
by 
  rw [total_leak_eq, leak_during_fix_eq]
  simp
  sorry

end oil_leak_before_fix_l38_38881


namespace cos_315_eq_sqrt2_div_2_l38_38448

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l38_38448


namespace simplified_form_l38_38540

def simplify_expression (x : ℝ) : ℝ :=
  (3 * x - 2) * (6 * x ^ 8 + 3 * x ^ 7 - 2 * x ^ 3 + x)

theorem simplified_form (x : ℝ) : 
  simplify_expression x = 18 * x ^ 9 - 3 * x ^ 8 - 6 * x ^ 7 - 6 * x ^ 4 - 4 * x ^ 3 + x :=
by
  sorry

end simplified_form_l38_38540


namespace third_smallest_is_five_l38_38125

noncomputable def probability_third_smallest_is_five : ℚ :=
  let total_ways := (Nat.choose 15 8) in
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 10 5) in
  favorable_ways / total_ways

theorem third_smallest_is_five :
  probability_third_smallest_is_five = 4 / 17 := sorry

end third_smallest_is_five_l38_38125


namespace circle_radius_l38_38866

theorem circle_radius {r : ℤ} (center: ℝ × ℝ) (inside_pt: ℝ × ℝ) (outside_pt: ℝ × ℝ)
  (h_center: center = (2, 1))
  (h_inside: dist center inside_pt < r)
  (h_outside: dist center outside_pt > r)
  (h_inside_pt: inside_pt = (-2, 1))
  (h_outside_pt: outside_pt = (2, -5))
  (h_integer: r > 0) :
  r = 5 :=
by
  sorry

end circle_radius_l38_38866


namespace correct_answer_l38_38022

theorem correct_answer (x : ℤ) (h : (x - 11) / 5 = 31) : (x - 5) / 11 = 15 :=
by
  sorry

end correct_answer_l38_38022


namespace train_length_from_speed_l38_38729

-- Definitions based on conditions
def seconds_to_cross_post : ℕ := 40
def seconds_to_cross_bridge : ℕ := 480
def bridge_length_meters : ℕ := 7200

-- Theorem statement to be proven
theorem train_length_from_speed :
  (bridge_length_meters / seconds_to_cross_bridge) * seconds_to_cross_post = 600 :=
sorry -- Proof is not provided

end train_length_from_speed_l38_38729


namespace ellipse_and_triangle_properties_l38_38636

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  1/2 * a * b

theorem ellipse_and_triangle_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y ↔ (x, y) = (1, 3/2) ∨ (x, y) = (1, -3/2)) ∧
  area_triangle 2 3 = 3 :=
by
  sorry

end ellipse_and_triangle_properties_l38_38636


namespace solve_quadratic_l38_38542

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end solve_quadratic_l38_38542


namespace num_possible_y_l38_38844

theorem num_possible_y : 
  (∃ (count : ℕ), count = (54 - 26 + 1) ∧ 
  ∀ (y : ℤ), 25 < y ∧ y < 55 ↔ (26 ≤ y ∧ y ≤ 54)) :=
by {
  sorry 
}

end num_possible_y_l38_38844


namespace find_positive_root_l38_38623

open Real

theorem find_positive_root 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (x : ℝ) :
  sqrt (a * b * x * (a + b + x)) + sqrt (b * c * x * (b + c + x)) + sqrt (c * a * x * (c + a + x)) = sqrt (a * b * c * (a + b + c)) →
  x = (a * b * c) / (a * b + b * c + c * a + 2 * sqrt (a * b * c * (a + b + c))) := 
sorry

end find_positive_root_l38_38623


namespace coats_from_high_schools_l38_38688

-- Define the total number of coats collected.
def total_coats_collected : ℕ := 9437

-- Define the number of coats collected from elementary schools.
def coats_from_elementary : ℕ := 2515

-- Goal: Prove that the number of coats collected from high schools is 6922.
theorem coats_from_high_schools : (total_coats_collected - coats_from_elementary) = 6922 := by
  sorry

end coats_from_high_schools_l38_38688


namespace ducks_and_chickens_l38_38557

theorem ducks_and_chickens : 
  (∃ ducks chickens : ℕ, ducks = 7 ∧ chickens = 6 ∧ ducks + chickens = 13) :=
by
  sorry

end ducks_and_chickens_l38_38557


namespace y1_gt_y2_for_line_through_points_l38_38476

theorem y1_gt_y2_for_line_through_points (x1 y1 x2 y2 k b : ℝ) 
  (h_line_A : y1 = k * x1 + b) 
  (h_line_B : y2 = k * x2 + b) 
  (h_k_neq_0 : k ≠ 0)
  (h_k_pos : k > 0)
  (h_b_nonneg : b ≥ 0)
  (h_x1_gt_x2 : x1 > x2) : 
  y1 > y2 := 
  sorry

end y1_gt_y2_for_line_through_points_l38_38476


namespace num_of_positive_divisors_l38_38756

-- Given conditions
variables {x y z : ℕ}
variables (p1 p2 p3 : ℕ) -- primes
variables (h1 : x = p1 ^ 3) (h2 : y = p2 ^ 3) (h3 : z = p3 ^ 3)
variables (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)

-- Lean statement to prove
theorem num_of_positive_divisors (hx3 : x = p1 ^ 3) (hy3 : y = p2 ^ 3) (hz3 : z = p3 ^ 3) 
    (Hdist : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
    ∃ n : ℕ, n = 10 * 13 * 7 ∧ n = (x^3 * y^4 * z^2).factors.length :=
sorry

end num_of_positive_divisors_l38_38756


namespace find_d_l38_38460

noncomputable def d : ℝ := 3.44

theorem find_d :
  (∃ x : ℝ, (3 * x^2 + 19 * x - 84 = 0) ∧ x = ⌊d⌋) ∧
  (∃ y : ℝ, (5 * y^2 - 26 * y + 12 = 0) ∧ y = d - ⌊d⌋) →
  d = 3.44 :=
by
  sorry

end find_d_l38_38460


namespace total_points_l38_38644

def jon_points (sam_points : ℕ) : ℕ := 2 * sam_points + 3
def sam_points (alex_points : ℕ) : ℕ := alex_points / 2
def jack_points (jon_points : ℕ) : ℕ := jon_points + 5
def tom_points (jon_points jack_points : ℕ) : ℕ := jon_points + jack_points - 4
def alex_points : ℕ := 18

theorem total_points : jon_points (sam_points alex_points) + 
                       jack_points (jon_points (sam_points alex_points)) + 
                       tom_points (jon_points (sam_points alex_points)) 
                       (jack_points (jon_points (sam_points alex_points))) + 
                       sam_points alex_points + 
                       alex_points = 117 :=
by sorry

end total_points_l38_38644


namespace Ted_has_15_bags_l38_38368

-- Define the parameters
def total_candy_bars : ℕ := 75
def candy_per_bag : ℝ := 5.0

-- Define the assertion to be proved
theorem Ted_has_15_bags : total_candy_bars / candy_per_bag = 15 := 
by
  sorry

end Ted_has_15_bags_l38_38368


namespace no_x2_term_a_eq_1_l38_38472

theorem no_x2_term_a_eq_1 (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a * x + 1) * (x^2 - 3 * a + 2) = x^4 + bx^3 + cx + d) →
  c = 0 →
  a = 1 :=
sorry

end no_x2_term_a_eq_1_l38_38472


namespace necklace_cost_l38_38248

theorem necklace_cost (total_savings earrings_cost remaining_savings: ℕ) 
                      (h1: total_savings = 80) 
                      (h2: earrings_cost = 23) 
                      (h3: remaining_savings = 9) : 
   total_savings - earrings_cost - remaining_savings = 48 :=
by
  sorry

end necklace_cost_l38_38248


namespace min_frac_sum_min_frac_sum_achieved_l38_38901

theorem min_frac_sum (a b : ℝ) (h₁ : 2 * a + 3 * b = 6) (h₂ : 0 < a) (h₃ : 0 < b) :
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

theorem min_frac_sum_achieved :
  (2 / (6 / 5) + 3 / (6 / 5)) = 25 / 6 :=
by sorry


end min_frac_sum_min_frac_sum_achieved_l38_38901


namespace weight_of_bowling_ball_l38_38980

-- We define the given conditions
def bowlingBalls := 10
def canoes := 4
def weightOfCanoe := 35

-- We state the theorem we want to prove
theorem weight_of_bowling_ball:
    (canoes * weightOfCanoe) / bowlingBalls = 14 :=
by
  -- Additional needed definitions
  let weightOfCanoes := canoes * weightOfCanoe
  have weightEquality : weightOfCanoes = 140 := by sorry  -- Calculating the total weight of the canoes
  -- Final division to find the weight of one bowling ball
  have weightOfOneBall := weightEquality / bowlingBalls
  show weightOfOneBall = 14 from sorry
  sorry

end weight_of_bowling_ball_l38_38980


namespace prob_heads_removed_correct_l38_38880

noncomputable def prob_heads_removed
  (initial_heads : 1 = 1)
  (bill_first_flip_heads : Prop)
  (bill_second_flip_heads : Prop)
  (carl_removes_coin : Prop)
  (alice_sees_two_heads : Prop)
  : ℚ := 
  (3 : ℚ) / 5

theorem prob_heads_removed_correct : 
  ∀ (initial_heads : 1 = 1) 
    (bill_first_flip_heads bill_second_flip_heads 
     carl_removes_coin alice_sees_two_heads : Prop),
  alice_sees_two_heads →
  prob_heads_removed initial_heads bill_first_flip_heads bill_second_flip_heads carl_removes_coin alice_sees_two_heads = 3 / 5 :=
by
  intros
  sorry

end prob_heads_removed_correct_l38_38880


namespace bases_for_final_digit_one_l38_38890

noncomputable def numberOfBases : ℕ :=
  (Finset.filter (λ b => ((625 - 1) % b = 0)) (Finset.range 11)).card - 
  (Finset.filter (λ b => b ≤ 2) (Finset.range 11)).card

theorem bases_for_final_digit_one : numberOfBases = 4 :=
by sorry

end bases_for_final_digit_one_l38_38890


namespace max_expression_value_l38_38993

open Real

theorem max_expression_value : 
  ∃ q : ℝ, ∀ q : ℝ, -3 * q ^ 2 + 18 * q + 5 ≤ 32 ∧ (-3 * (3 ^ 2) + 18 * 3 + 5 = 32) :=
by
  sorry

end max_expression_value_l38_38993


namespace xy_plus_four_is_square_l38_38326

theorem xy_plus_four_is_square (x y : ℕ) (h : ((1 / (x : ℝ)) + (1 / (y : ℝ)) + 1 / (x * y : ℝ)) = (1 / (x + 4 : ℝ) + 1 / (y - 4 : ℝ) + 1 / ((x + 4) * (y - 4) : ℝ))) : 
  ∃ (k : ℕ), xy + 4 = k^2 :=
by
  sorry

end xy_plus_four_is_square_l38_38326


namespace principal_sum_l38_38545

noncomputable def diff_simple_compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
(P * ((1 + r / 100)^t) - P) - (P * r * t / 100)

theorem principal_sum (P : ℝ) (r : ℝ) (t : ℝ) (h : diff_simple_compound_interest P r t = 631) (hr : r = 10) (ht : t = 2) :
    P = 63100 := by
  sorry

end principal_sum_l38_38545


namespace prove_solution_l38_38832

noncomputable def problem_statement : Prop := ∀ x : ℝ, (16 : ℝ)^(2 * x - 3) = (4 : ℝ)^(3 - x) → x = 9 / 5

theorem prove_solution : problem_statement :=
by
  intro x h
  -- The proof would go here
  sorry

end prove_solution_l38_38832


namespace probability_product_positive_correct_l38_38990

noncomputable def probability_product_positive : ℚ :=
  let length_total := 45
  let length_negative := 30
  let length_positive := 15
  let prob_negative := (length_negative : ℚ) / length_total
  let prob_positive := (length_positive : ℚ) / length_total
  let prob_product_positive := prob_negative^2 + prob_positive^2
  prob_product_positive

theorem probability_product_positive_correct :
  probability_product_positive = 5 / 9 :=
by
  sorry

end probability_product_positive_correct_l38_38990


namespace angle_with_same_terminal_side_l38_38290

-- Given conditions in the problem: angles to choose from
def angles : List ℕ := [60, 70, 100, 130]

-- Definition of the equivalence relation (angles having the same terminal side)
def same_terminal_side (θ α : ℕ) : Prop :=
  ∃ k : ℤ, θ = α + k * 360

-- Proof goal: 420° has the same terminal side as one of the angles in the list
theorem angle_with_same_terminal_side :
  ∃ α ∈ angles, same_terminal_side 420 α :=
sorry  -- proof not required

end angle_with_same_terminal_side_l38_38290


namespace height_of_each_step_l38_38946

-- Define the number of steps in each staircase
def first_staircase_steps : ℕ := 20
def second_staircase_steps : ℕ := 2 * first_staircase_steps
def third_staircase_steps : ℕ := second_staircase_steps - 10

-- Define the total steps climbed
def total_steps_climbed : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

-- Define the total height climbed
def total_height_climbed : ℝ := 45

-- Prove the height of each step
theorem height_of_each_step : (total_height_climbed / total_steps_climbed) = 0.5 := by
  sorry

end height_of_each_step_l38_38946


namespace Sidney_JumpJacks_Tuesday_l38_38685

variable (JumpJacksMonday JumpJacksTuesday JumpJacksWednesday JumpJacksThursday : ℕ)
variable (SidneyTotalJumpJacks BrookeTotalJumpJacks : ℕ)

-- Given conditions
axiom H1 : JumpJacksMonday = 20
axiom H2 : JumpJacksWednesday = 40
axiom H3 : JumpJacksThursday = 50
axiom H4 : BrookeTotalJumpJacks = 3 * SidneyTotalJumpJacks
axiom H5 : BrookeTotalJumpJacks = 438

-- Prove Sidney's JumpJacks on Tuesday
theorem Sidney_JumpJacks_Tuesday : JumpJacksTuesday = 36 :=
by
  sorry

end Sidney_JumpJacks_Tuesday_l38_38685


namespace sufficient_but_not_necessary_l38_38711

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

def z (a : ℝ) : ℂ := ⟨a^2 - 4, a + 1⟩

theorem sufficient_but_not_necessary (a : ℝ) (h : a = -2) : 
  is_purely_imaginary (z a) ∧ ¬(∀ a, is_purely_imaginary (z a) → a = -2) :=
by
  sorry

end sufficient_but_not_necessary_l38_38711


namespace fraction_of_grid_covered_l38_38587

open Real EuclideanGeometry

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem fraction_of_grid_covered :
  let A := (2, 2)
  let B := (6, 2)
  let C := (4, 5)
  let grid_area := 7 * 7
  let triangle_area := area_of_triangle A B C
  triangle_area / grid_area = 6 / 49 := by
  sorry

end fraction_of_grid_covered_l38_38587


namespace solve_problem_l38_38481

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end solve_problem_l38_38481


namespace rational_solutions_of_quadratic_l38_38764

theorem rational_solutions_of_quadratic (k : ℕ) (hk : 0 < k ∧ k ≤ 10) :
  ∃ (x : ℚ), k * x^2 + 20 * x + k = 0 ↔ (k = 6 ∨ k = 8 ∨ k = 10) :=
by sorry

end rational_solutions_of_quadratic_l38_38764


namespace probability_third_smallest_five_l38_38126

theorem probability_third_smallest_five :
  let S := finset.Icc 1 15 in
  let total_ways := S.card.choose 8 in
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2 in
  (favorable_ways : ℚ) / total_ways = 4 / 21 :=
by {
  let S := finset.Icc 1 15,
  let total_ways := S.card.choose 8,
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2,
  have h : favorable_ways = 1260 := rfl,
  have h2 : total_ways = 6435 := rfl,
  calc
    (favorable_ways : ℚ) / total_ways
        = (1260 : ℚ) / 6435 : by rw [h, h2]
    ... = 4 / 21 : by norm_num,
  sorry
}

end probability_third_smallest_five_l38_38126


namespace range_of_4x_plus_2y_l38_38627

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h₁ : 1 ≤ x + y ∧ x + y ≤ 3)
  (h₂ : -1 ≤ x - y ∧ x - y ≤ 1) : 
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 :=
sorry

end range_of_4x_plus_2y_l38_38627


namespace coordinates_with_respect_to_origin_l38_38375

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (2, -6)) : (x, y) = (2, -6) :=
by
  sorry

end coordinates_with_respect_to_origin_l38_38375


namespace arithmetic_mean_of_range_neg3_to_6_l38_38073

theorem arithmetic_mean_of_range_neg3_to_6 :
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  (sum : Float) / (count : Float) = 1.5 := by
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  have h_sum : sum = 15 := by sorry
  have h_count : count = 10 := by sorry
  calc
    (sum : Float) / (count : Float)
        = (15 : Float) / (10 : Float) : by rw [h_sum, h_count]
    ... = 1.5 : by norm_num

end arithmetic_mean_of_range_neg3_to_6_l38_38073


namespace dress_assignments_l38_38135

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l38_38135


namespace lead_to_tin_ratio_l38_38572

variable (L T T_B : ℝ)

def mix_alloys (L T T_B : ℝ) : Prop :=
  (L + T = 120) ∧
  (T_B = 67.5) ∧
  (T + T_B = 139.5) ∧
  let ratio : ℚ := (L / (T : ℝ)).toRat in ratio = 2 / 3

theorem lead_to_tin_ratio (h : mix_alloys L T T_B) : L / T = 2 / 3 :=
by sorry

end lead_to_tin_ratio_l38_38572


namespace find_yellow_shells_l38_38977

-- Define the conditions
def total_shells : ℕ := 65
def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

-- Define the result as the proof goal
theorem find_yellow_shells (total_shells purple_shells pink_shells blue_shells orange_shells : ℕ) : 
  total_shells = 65 →
  purple_shells = 13 →
  pink_shells = 8 →
  blue_shells = 12 →
  orange_shells = 14 →
  65 - (13 + 8 + 12 + 14) = 18 :=
by
  intros
  sorry

end find_yellow_shells_l38_38977


namespace find_A_l38_38838

-- Define the four-digit number being a multiple of 9 and the sum of its digits condition
def digit_sum_multiple_of_9 (A : ℤ) : Prop :=
  (3 + A + A + 1) % 9 = 0

-- The Lean statement for the proof problem
theorem find_A (A : ℤ) (h : digit_sum_multiple_of_9 A) : A = 7 :=
sorry

end find_A_l38_38838


namespace nate_age_when_ember_is_14_l38_38614

theorem nate_age_when_ember_is_14 (nate_age : ℕ) (ember_age : ℕ) 
  (h1 : ember_age = nate_age / 2) (h2 : nate_age = 14) :
  ∃ (years_later : ℕ), ember_age + years_later = 14 ∧ nate_age + years_later = 21 :=
by
  -- sorry to skip the proof, adhering to the instructions
  sorry

end nate_age_when_ember_is_14_l38_38614


namespace maximize_area_playground_l38_38590

noncomputable def maxAreaPlayground : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area_playground : ∀ (l w : ℝ),
  (2 * l + 2 * w = 400) ∧ (l ≥ 100) ∧ (w ≥ 60) → l * w ≤ maxAreaPlayground :=
by
  intros l w h
  sorry

end maximize_area_playground_l38_38590


namespace arcsin_one_half_eq_pi_six_l38_38886

theorem arcsin_one_half_eq_pi_six :
  Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l38_38886


namespace cos_A_value_compare_angles_l38_38344

variable (A B C : ℝ) (a b c : ℝ)

-- Given conditions
variable (h1 : a = 3) (h2 : b = 2 * Real.sqrt 6) (h3 : B = 2 * A)

-- Problem (I) statement
theorem cos_A_value (hcosA : Real.cos A = Real.sqrt 6 / 3) : 
  Real.cos A = Real.sqrt 6 / 3 :=
by 
  sorry

-- Problem (II) statement
theorem compare_angles (hcosA : Real.cos A = Real.sqrt 6 / 3) (hcosC : Real.cos C = Real.sqrt 6 / 9) :
  B < C :=
by
  sorry

end cos_A_value_compare_angles_l38_38344


namespace Brad_pumpkin_weight_l38_38647

theorem Brad_pumpkin_weight (B : ℝ)
  (h1 : ∃ J : ℝ, J = B / 2)
  (h2 : ∃ Be : ℝ, Be = 4 * (B / 2))
  (h3 : ∃ Be J : ℝ, Be - J = 81) : B = 54 := by
  obtain ⟨J, hJ⟩ := h1
  obtain ⟨Be, hBe⟩ := h2
  obtain ⟨_, hBeJ⟩ := h3
  sorry

end Brad_pumpkin_weight_l38_38647


namespace calvin_winning_strategy_l38_38091

theorem calvin_winning_strategy :
  ∃ (n : ℤ), ∃ (p : ℤ), ∃ (q : ℤ),
  (∀ k : ℕ, k > 0 → p = 0 ∧ (q = 2014 + k ∨ q = 2014 - k) → ∃ x : ℤ, (x^2 + p * x + q = 0)) :=
sorry

end calvin_winning_strategy_l38_38091


namespace problem_l38_38323

theorem problem (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : f 1 = f 3) 
  (h2 : f 1 > f 4) 
  (hf : ∀ x, f x = a * x ^ 2 + b * x + c) :
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l38_38323


namespace latest_time_to_reach_80_degrees_l38_38799

theorem latest_time_to_reach_80_degrees :
  ∀ (t : ℝ), (-t^2 + 14 * t + 40 = 80) → t ≤ 10 :=
by
  sorry

end latest_time_to_reach_80_degrees_l38_38799


namespace positive_difference_of_squares_l38_38554

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l38_38554


namespace incorrect_conclusion_C_l38_38009

noncomputable def f (x : ℝ) := (x - 1)^2 * Real.exp x

theorem incorrect_conclusion_C : 
  ¬(∀ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) ≥ ε) :=
by
  sorry

end incorrect_conclusion_C_l38_38009


namespace train_crossing_time_l38_38865

noncomputable def time_to_cross_bridge (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_crossing_time :
  time_to_cross_bridge 100 145 65 = 13.57 :=
by
  sorry

end train_crossing_time_l38_38865


namespace sum_of_zeros_of_even_function_is_zero_l38_38778

open Function

theorem sum_of_zeros_of_even_function_is_zero (f : ℝ → ℝ) (hf: Even f) (hx: ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) :
  x1 + x2 + x3 + x4 = 0 := by
  sorry

end sum_of_zeros_of_even_function_is_zero_l38_38778


namespace fraction_meaningful_l38_38549

theorem fraction_meaningful (x : ℝ) : (∃ z, z = 3 / (x - 4)) ↔ x ≠ 4 :=
by
  sorry

end fraction_meaningful_l38_38549


namespace tan_of_alpha_l38_38498

noncomputable def point_P : ℝ × ℝ := (1, -2)

theorem tan_of_alpha (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = point_P ∧ P.2 / P.1 = -2) :
  Real.tan α = -2 :=
sorry

end tan_of_alpha_l38_38498


namespace sara_total_spent_l38_38243

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end sara_total_spent_l38_38243


namespace kristy_initial_cookies_l38_38807

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end kristy_initial_cookies_l38_38807


namespace problem_a_problem_b_problem_c_problem_d_problem_e_l38_38970

section problem_a
  -- Conditions
  def rainbow_russian_first_letters_sequence := ["к", "о", "ж", "з", "г", "с", "ф"]
  
  -- Theorem (question == answer)
  theorem problem_a : rainbow_russian_first_letters_sequence[4] = "г" ∧
                      rainbow_russian_first_letters_sequence[5] = "с" ∧
                      rainbow_russian_first_letters_sequence[6] = "ф" :=
  by
    -- Skip proof: sorry
    sorry
end problem_a

section problem_b
  -- Conditions
  def russian_alphabet_alternating_sequence := ["а", "в", "г", "ё", "ж", "з", "л", "м", "н", "о", "п", "т", "у"]
 
  -- Theorem (question == answer)
  theorem problem_b : russian_alphabet_alternating_sequence[10] = "п" ∧
                      russian_alphabet_alternating_sequence[11] = "т" ∧
                      russian_alphabet_alternating_sequence[12] = "у" :=
  by
    -- Skip proof: sorry
    sorry
end problem_b

section problem_c
  -- Conditions
  def russian_number_of_letters_sequence := ["один", "четыре", "шесть", "пять", "семь", "восемь"]
  
  -- Theorem (question == answer)
  theorem problem_c : russian_number_of_letters_sequence[4] = "семь" ∧
                      russian_number_of_letters_sequence[5] = "восемь" :=
  by
    -- Skip proof: sorry
    sorry
end problem_c

section problem_d
  -- Conditions
  def approximate_symmetry_letters_sequence := ["Ф", "Х", "Ш", "В"]

  -- Theorem (question == answer)
  theorem problem_d : approximate_symmetry_letters_sequence[3] = "В" :=
  by
    -- Skip proof: sorry
    sorry
end problem_d

section problem_e
  -- Conditions
  def russian_loops_in_digit_sequence := ["0", "д", "т", "ч", "п", "ш", "с", "в", "д"]

  -- Theorem (question == answer)
  theorem problem_e : russian_loops_in_digit_sequence[7] = "в" ∧
                      russian_loops_in_digit_sequence[8] = "д" :=
  by
    -- Skip proof: sorry
    sorry
end problem_e

end problem_a_problem_b_problem_c_problem_d_problem_e_l38_38970


namespace SummitAcademy_Contestants_l38_38292

theorem SummitAcademy_Contestants (s j : ℕ)
  (h1 : s > 0)
  (h2 : j > 0)
  (hs : (1 / 3 : ℚ) * s = (3 / 4 : ℚ) * j) :
  s = (9 / 4 : ℚ) * j :=
sorry

end SummitAcademy_Contestants_l38_38292


namespace no_real_solution_l38_38753

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end no_real_solution_l38_38753


namespace num_ordered_triples_l38_38581

theorem num_ordered_triples (b : ℕ) (h : b = 1681) : 
  {t : ℕ × ℕ // t.fst ≤ b ∧ b ≤ t.snd ∧ (t.fst * t.snd = 1681 ^ 2)}.to_finset.card = 2 :=
by
  sorry

end num_ordered_triples_l38_38581


namespace bill_amount_is_correct_l38_38259

-- Define the given conditions
def true_discount : ℝ := 189
def rate : ℝ := 0.16
def time : ℝ := 9 / 12

-- Define the true discount formula
def true_discount_formula (FV : ℝ) (R : ℝ) (T : ℝ) : ℝ := 
  (FV * R * T) / (100 + (R * T))

-- State that we want to prove that the Face Value is Rs. 1764
theorem bill_amount_is_correct : ∃ (FV : ℝ), FV = 1764 ∧ true_discount = true_discount_formula FV rate time :=
sorry

end bill_amount_is_correct_l38_38259


namespace integer_solutions_of_prime_equation_l38_38569

theorem integer_solutions_of_prime_equation (p : ℕ) (hp : Prime p) :
  ∃ x y : ℤ, (p * (x + y) = x * y) ↔ 
    (x = (p * (p + 1)) ∧ y = (p + 1)) ∨ 
    (x = 2 * p ∧ y = 2 * p) ∨ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p * (1 - p) ∧ y = (p - 1)) := 
sorry

end integer_solutions_of_prime_equation_l38_38569


namespace find_t_given_conditions_l38_38021

variables (p t j x y : ℝ)

theorem find_t_given_conditions
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p * (1 - t / 100))
  (h4 : x = 0.10 * t)
  (h5 : y = 0.50 * j)
  (h6 : x + y = 12) :
  t = 24 :=
by sorry

end find_t_given_conditions_l38_38021


namespace weight_of_3_moles_of_CaI2_is_881_64_l38_38268

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
noncomputable def weight_3_moles_CaI2 : ℝ := 3 * molar_mass_CaI2

theorem weight_of_3_moles_of_CaI2_is_881_64 :
  weight_3_moles_CaI2 = 881.64 :=
by sorry

end weight_of_3_moles_of_CaI2_is_881_64_l38_38268


namespace max_value_sqrt_expression_l38_38390

theorem max_value_sqrt_expression
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (4 : ℝ)) * (Real.sqrt (2 : ℝ)) ≤ sqrt (4 : ℝ) * sqrt (2 : ℝ ) :=
begin
  sorry,
end

end max_value_sqrt_expression_l38_38390


namespace find_digits_l38_38653

theorem find_digits (A B D E C : ℕ) 
  (hC : C = 9) 
  (hA : 2 < A ∧ A < 4)
  (hB : B = 5)
  (hE : E = 6)
  (hD : D = 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) :
  (A, B, D, E) = (3, 5, 0, 6) := by
  sorry

end find_digits_l38_38653


namespace eval_g_at_3_l38_38485

def g (x : ℤ) : ℤ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem eval_g_at_3 : g 3 = 183 := by
  sorry

end eval_g_at_3_l38_38485


namespace amanda_needs_how_many_bags_of_grass_seeds_l38_38591

theorem amanda_needs_how_many_bags_of_grass_seeds
    (lot_length : ℕ := 120)
    (lot_width : ℕ := 60)
    (concrete_length : ℕ := 40)
    (concrete_width : ℕ := 40)
    (bag_coverage : ℕ := 56) :
    (lot_length * lot_width - concrete_length * concrete_width) / bag_coverage = 100 := by
  sorry

end amanda_needs_how_many_bags_of_grass_seeds_l38_38591


namespace dividend_is_217_l38_38640

-- Given conditions
def r : ℕ := 1
def q : ℕ := 54
def d : ℕ := 4

-- Define the problem as a theorem in Lean 4
theorem dividend_is_217 : (d * q) + r = 217 := by
  -- proof is omitted
  sorry

end dividend_is_217_l38_38640


namespace mr_greene_probability_l38_38525

noncomputable def probability_more_sons_or_daughters
  (children : ℕ) (twins : ℕ) (independent : ℕ) (p : ℚ) :=
  let total_combinations := 2 ^ independent
  let twins_combinations := 2
  let total_scenarios := total_combinations * twins_combinations
  let even_distribution := (choose independent (independent/2).to_nat) * 2
  let unequal_distribution := total_scenarios - even_distribution
  unequal_distribution / total_scenarios 

theorem mr_greene_probability :
  probability_more_sons_or_daughters 8 2 6 (49 / 64) = (49 : ℚ) / 64 :=
by
  unfold probability_more_sons_or_daughters
  sorry

end mr_greene_probability_l38_38525


namespace length_proof_l38_38381

noncomputable def length_of_plot 
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ) -- cost of fencing per meter on flat ground
  (height_rise : ℝ) -- total height rise in meters
  (total_cost: ℝ) -- total cost of fencing
  (length_increase : ℝ) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ) -- scaling factor for cost increase on breadth
  (increased_breadth_cost_rate : ℝ) -- actual increased cost rate per meter for breadth
: ℝ :=
2 * (b + length_increase) * fence_cost_flat + 
2 * b * (fence_cost_flat + fence_cost_flat * (height_rise * cost_increase_rate))

theorem length_proof
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ := 26.50) -- cost of fencing per meter on flat ground
  (height_rise : ℝ := 5) -- total height rise in meters
  (total_cost: ℝ := 5300) -- total cost of fencing
  (length_increase : ℝ := 20) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ := 0.10) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ := fence_cost_flat * 0.5) -- increased cost factor
  (increased_breadth_cost_rate : ℝ := 39.75) -- recalculated cost rate per meter for breadth
  (length: ℝ := b + length_increase)
  (proof_step : total_cost = length_of_plot b fence_cost_flat height_rise total_cost length_increase cost_increase_rate breadth_cost_increase_factor increased_breadth_cost_rate)
: length = 52 :=
by
  sorry -- Proof omitted

end length_proof_l38_38381


namespace cos_315_eq_sqrt2_div2_l38_38432

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l38_38432


namespace cos_315_deg_l38_38438

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l38_38438


namespace number_of_terms_in_expansion_l38_38255

theorem number_of_terms_in_expansion :
  (∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 c1 c2 c3 : ℕ), (a1 + a2 + a3 + a4 + a5) * (b1 + b2 + b3 + b4) * (c1 + c2 + c3) = 60) :=
by
  sorry

end number_of_terms_in_expansion_l38_38255


namespace greatest_possible_perimeter_l38_38648

def triangle_side_lengths (x : ℤ) : Prop :=
  (x > 0) ∧ (5 * x > 18) ∧ (x < 6)

def perimeter (x : ℤ) : ℤ :=
  x + 4 * x + 18

theorem greatest_possible_perimeter :
  ∃ x : ℤ, triangle_side_lengths x ∧ (perimeter x = 38) :=
by
  sorry

end greatest_possible_perimeter_l38_38648


namespace sara_total_spent_l38_38242

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end sara_total_spent_l38_38242


namespace Kristy_baked_cookies_l38_38809

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end Kristy_baked_cookies_l38_38809


namespace infinite_grid_rectangles_l38_38676

theorem infinite_grid_rectangles (m : ℕ) (hm : m > 12) : 
  ∃ (x y : ℕ), x * y > m ∧ x * (y - 1) < m := 
  sorry

end infinite_grid_rectangles_l38_38676


namespace fraction_multiplication_l38_38335

theorem fraction_multiplication :
  ((2 / 5) * (5 / 7) * (7 / 3) * (3 / 8) = 1 / 4) :=
sorry

end fraction_multiplication_l38_38335


namespace largest_among_four_l38_38318

theorem largest_among_four (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  max (max a (max (a + b) (a - b))) (ab) = a - b :=
by {
  sorry
}

end largest_among_four_l38_38318


namespace blocks_needed_for_enclosure_l38_38127

noncomputable def volume_of_rectangular_prism (length: ℝ) (width: ℝ) (height: ℝ) : ℝ :=
  length * width * height

theorem blocks_needed_for_enclosure 
  (length width height thickness : ℝ)
  (H_length : length = 15)
  (H_width : width = 12)
  (H_height : height = 6)
  (H_thickness : thickness = 1.5) :
  volume_of_rectangular_prism length width height - 
  volume_of_rectangular_prism (length - 2 * thickness) (width - 2 * thickness) (height - thickness) = 594 :=
by
  sorry

end blocks_needed_for_enclosure_l38_38127


namespace find_k_and_a_range_l38_38002

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^2 + Real.exp x - k * Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem find_k_and_a_range (k a : ℝ) (h_even : ∀ x : ℝ, f x k = f (-x) k) :
  k = -1 ∧ 2 ≤ a := by
    sorry

end find_k_and_a_range_l38_38002


namespace two_R_theta_bounds_l38_38598

variables {R : ℝ} (θ : ℝ)
variables (h_pos : 0 < R) (h_triangle : (R + 1 + (R + 1/2)) > 2 *R)

-- Define that θ is the angle between sides R and R + 1/2
-- Here we assume θ is defined via the cosine rule for simplicity

noncomputable def angle_between_sides (R : ℝ) := 
  Real.arccos ((R^2 + (R + 1/2)^2 - 1^2) / (2 * R * (R + 1/2)))

-- State the theorem
theorem two_R_theta_bounds (h : θ = angle_between_sides R) : 
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
by
  sorry

end two_R_theta_bounds_l38_38598


namespace find_a_l38_38329

-- Definitions and conditions from the problem
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}
def intersection_is_three (a : ℝ) : Prop := M a ∩ N a = {3}

-- The theorem we want to prove
theorem find_a (a : ℝ) (h : intersection_is_three a) : a = 4 :=
by
  sorry

end find_a_l38_38329


namespace number_of_strikers_l38_38726

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l38_38726


namespace boat_speed_in_still_water_l38_38709

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := by
  sorry

end boat_speed_in_still_water_l38_38709


namespace sectionB_seats_correct_l38_38662

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end sectionB_seats_correct_l38_38662


namespace exist_end_2015_l38_38154

def in_sequence (n : Nat) : Nat :=
  90 * n + 75

theorem exist_end_2015 :
  ∃ n : Nat, in_sequence n % 10000 = 2015 :=
by
  sorry

end exist_end_2015_l38_38154


namespace expression_simplification_l38_38119

theorem expression_simplification :
  (2 ^ 2 / 3 + (-(3 ^ 2) + 5) + (-(3) ^ 2) * ((2 / 3) ^ 2)) = 4 / 3 :=
sorry

end expression_simplification_l38_38119


namespace Kristy_baked_cookies_l38_38808

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end Kristy_baked_cookies_l38_38808


namespace base12_division_remainder_l38_38400

theorem base12_division_remainder :
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3 in
  n % 9 = 8 :=
by
  let n := 2 * (12^3) + 5 * (12^2) + 4 * 12 + 3
  show n % 9 = 8
  sorry

end base12_division_remainder_l38_38400


namespace ab_range_l38_38931

theorem ab_range (a b : ℝ) : (a + b = 1/2) → ab ≤ 1/16 :=
by
  sorry

end ab_range_l38_38931


namespace no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l38_38863

-- Part (a): Prove that it is impossible to arrange five distinct-sized squares to form a rectangle.
theorem no_rectangle_with_five_distinct_squares (s1 s2 s3 s4 s5 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s4 ≠ s5) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5)) :=
by
  -- Proof placeholder
  sorry

-- Part (b): Prove that it is impossible to arrange six distinct-sized squares to form a rectangle.
theorem no_rectangle_with_six_distinct_squares (s1 s2 s3 s4 s5 s6 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s1 ≠ s6 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s2 ≠ s6 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s3 ≠ s6 ∧ s4 ≠ s5 ∧ s4 ≠ s6 ∧ s5 ≠ s6) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧ (s6 ≤ l ∧ s6 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5 + s6)) :=
by
  -- Proof placeholder
  sorry

end no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l38_38863


namespace oxygen_atom_count_l38_38412

-- Definitions and conditions
def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def molecular_weight_O : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def total_molecular_weight : ℝ := 65.0

-- Theorem statement
theorem oxygen_atom_count : 
  ∃ (num_oxygen_atoms : ℕ), 
  num_oxygen_atoms * molecular_weight_O = total_molecular_weight - (num_carbon_atoms * molecular_weight_C + num_hydrogen_atoms * molecular_weight_H) 
  ∧ num_oxygen_atoms = 1 :=
by
  sorry

end oxygen_atom_count_l38_38412


namespace units_digit_of_7_pow_3_l38_38083

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end units_digit_of_7_pow_3_l38_38083


namespace each_tree_takes_one_square_foot_l38_38500

theorem each_tree_takes_one_square_foot (total_length : ℝ) (num_trees : ℕ) (gap_length : ℝ)
    (total_length_eq : total_length = 166) (num_trees_eq : num_trees = 16) (gap_length_eq : gap_length = 10) :
    (total_length - (((num_trees - 1) : ℝ) * gap_length)) / (num_trees : ℝ) = 1 :=
by
  rw [total_length_eq, num_trees_eq, gap_length_eq]
  sorry

end each_tree_takes_one_square_foot_l38_38500


namespace base5_division_l38_38458

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end base5_division_l38_38458


namespace part1_real_roots_part2_distinct_positive_integer_roots_l38_38166

noncomputable def equation := λ (m x : ℝ), m * x^2 - (m + 2) * x + 2 = 0

theorem part1_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, equation m x₁ ∧ equation m x₂ := by
  sorry

theorem part2_distinct_positive_integer_roots (m : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ equation m (x₁ : ℝ) ∧ equation m (x₂ : ℝ)) ↔ (m = 1) := by
  sorry

end part1_real_roots_part2_distinct_positive_integer_roots_l38_38166


namespace inequality_div_half_l38_38926

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end inequality_div_half_l38_38926


namespace range_of_a_l38_38782

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 - x ^ 2 + x - 5

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ x_max x_min : ℝ, x_max ≠ x_min ∧
  f a x_max = max (f a x_max) (f a x_min) ∧ f a x_min = min (f a x_max) (f a x_min)) → 
  a < 1 / 3 ∧ a ≠ 0 := sorry

end range_of_a_l38_38782


namespace count_three_digit_numbers_divisible_by_7_l38_38174

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

def count_three_digit_divisible_by_7 : ℕ :=
  (list.range' 105 890).countp (λ x, is_divisible_by_7 (x * 7))

theorem count_three_digit_numbers_divisible_by_7 :
  count_three_digit_divisible_by_7 = 128 :=
sorry

end count_three_digit_numbers_divisible_by_7_l38_38174


namespace second_valve_emits_more_l38_38404

noncomputable def V1 : ℝ := 12000 / 120 -- Rate of first valve (100 cubic meters/minute)
noncomputable def V2 : ℝ := 12000 / 48 - V1 -- Rate of second valve

theorem second_valve_emits_more : V2 - V1 = 50 :=
by
  sorry

end second_valve_emits_more_l38_38404


namespace Tim_marble_count_l38_38473

theorem Tim_marble_count (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 := 
sorry

end Tim_marble_count_l38_38473


namespace boat_avg_speed_ratio_l38_38575

/--
A boat moves at a speed of 20 mph in still water. When traveling in a river with a current of 3 mph, it travels 24 miles downstream and then returns upstream to the starting point. Prove that the ratio of the average speed for the entire round trip to the boat's speed in still water is 97765 / 100000.
-/
theorem boat_avg_speed_ratio :
  let boat_speed := 20 -- mph in still water
  let current_speed := 3 -- mph river current
  let distance := 24 -- miles downstream and upstream
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  let total_time := time_downstream + time_upstream
  let total_distance := distance * 2
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 97765 / 100000 :=
by
  sorry

end boat_avg_speed_ratio_l38_38575


namespace probability_at_least_one_white_ball_l38_38801

def balls : Finset ℕ := {0, 1, 2, 3, 4}  -- represent the 5 balls
def white_balls : Finset ℕ := {1, 2}  -- represent the white balls

def all_pairs : Finset (ℕ × ℕ) := balls.product balls
def valid_pairs : Finset (ℕ × ℕ) := all_pairs.filter (λ p, p.1 ≠ p.2 ∧ (p.1 ∈ white_balls ∨ p.2 ∈ white_balls))

theorem probability_at_least_one_white_ball :
  ((valid_pairs.card : ℚ) / (all_pairs.card : ℚ)) = 7 / 10 :=
by
  sorry

end probability_at_least_one_white_ball_l38_38801


namespace price_of_each_tomato_l38_38854

theorem price_of_each_tomato
  (customers_per_month : ℕ)
  (lettuce_per_customer : ℕ)
  (lettuce_price : ℕ)
  (tomatoes_per_customer : ℕ)
  (total_monthly_sales : ℕ)
  (total_lettuce_sales : ℕ)
  (total_tomato_sales : ℕ)
  (price_per_tomato : ℝ)
  (h1 : customers_per_month = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomatoes_per_customer = 4)
  (h5 : total_monthly_sales = 2000)
  (h6 : total_lettuce_sales = customers_per_month * lettuce_per_customer * lettuce_price)
  (h7 : total_tomato_sales = total_monthly_sales - total_lettuce_sales)
  (h8 : total_lettuce_sales = 1000)
  (h9 : total_tomato_sales = 1000)
  (total_tomatoes_sold : ℕ := customers_per_month * tomatoes_per_customer)
  (h10 : total_tomatoes_sold = 2000) :
  price_per_tomato = 0.50 :=
by
  sorry

end price_of_each_tomato_l38_38854


namespace problem_statement_l38_38228

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l38_38228


namespace functional_expression_selling_price_for_profit_l38_38817

-- Define the initial conditions
def cost_price : ℚ := 8
def initial_selling_price : ℚ := 10
def initial_sales_volume : ℚ := 200
def sales_decrement_per_yuan_increase : ℚ := 20

-- Functional expression between y (items) and x (yuan)
theorem functional_expression (x : ℚ) : 
  (200 - 20 * (x - 10) = -20 * x + 400) :=
sorry

-- Determine the selling price to achieve a daily profit of 640 yuan
theorem selling_price_for_profit (x : ℚ) (h1 : 8 ≤ x) (h2 : x ≤ 15) : 
  ((x - 8) * (400 - 20 * x) = 640) → (x = 12) :=
sorry

end functional_expression_selling_price_for_profit_l38_38817


namespace fraction_simplify_l38_38605

theorem fraction_simplify :
  (3 + 9 - 27 + 81 - 243 + 729) / (9 + 27 - 81 + 243 - 729 + 2187) = 1 / 3 :=
by
  sorry

end fraction_simplify_l38_38605


namespace min_value_of_reciprocals_l38_38199

theorem min_value_of_reciprocals {x y a b : ℝ} 
  (h1 : 8 * x - y - 4 ≤ 0)
  (h2 : x + y + 1 ≥ 0)
  (h3 : y - 4 * x ≤ 0)
  (h4 : 2 = a * (1 / 2) + b * 1)
  (ha : a > 0)
  (hb : b > 0) :
  (1 / a) + (1 / b) = 9 / 2 :=
sorry

end min_value_of_reciprocals_l38_38199


namespace g_is_correct_l38_38664

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2

axiom g_functional_eq : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2

theorem g_is_correct : ∀ x : ℝ, g x = 2 - 2 * x := 
by 
  sorry

end g_is_correct_l38_38664


namespace binomial_coeff_18_10_l38_38743

theorem binomial_coeff_18_10 :
  ∀ (binom : ℕ → ℕ → ℕ), 
    binom 16 7 = 11440 → 
    binom 16 9 = 11440 → 
    binom 18 10 = 43858 :=
by
  intros binom h1 h2
  have h3 : binom 16 6 = binom 16 10, sorry
  have h4 : binom 16 8 = 12870, sorry
  sorry

end binomial_coeff_18_10_l38_38743


namespace coffee_bags_per_week_l38_38892

def bags_morning : Nat := 3
def bags_afternoon : Nat := 3 * bags_morning
def bags_evening : Nat := 2 * bags_morning
def bags_per_day : Nat := bags_morning + bags_afternoon + bags_evening
def days_per_week : Nat := 7

theorem coffee_bags_per_week : bags_per_day * days_per_week = 126 := by
  sorry

end coffee_bags_per_week_l38_38892


namespace find_solutions_of_x4_minus_16_l38_38470

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l38_38470


namespace solve_x4_eq_16_l38_38467

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l38_38467


namespace find_eccentricity_l38_38904

noncomputable def ellipse_eccentricity (m : ℝ) (c : ℝ) (a : ℝ) : ℝ :=
  c / a

theorem find_eccentricity
  (m : ℝ) (c := Real.sqrt 2) (a := 3 * Real.sqrt 2 / 2)
  (h1 : 2 * m^2 - (m + 1) = 2)
  (h2 : m > 0) :
  ellipse_eccentricity m c a = 2 / 3 :=
by sorry

end find_eccentricity_l38_38904


namespace cube_minus_self_divisible_by_10_l38_38129

theorem cube_minus_self_divisible_by_10 (k : ℤ) : 10 ∣ ((5 * k) ^ 3 - 5 * k) :=
by sorry

end cube_minus_self_divisible_by_10_l38_38129


namespace candies_per_child_rounded_l38_38986

/-- There are 15 pieces of candy divided equally among 7 children. The number of candies per child, rounded to the nearest tenth, is 2.1. -/
theorem candies_per_child_rounded :
  let candies := 15
  let children := 7
  Float.round (candies / children * 10) / 10 = 2.1 :=
by
  sorry

end candies_per_child_rounded_l38_38986


namespace division_in_base_5_l38_38457

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end division_in_base_5_l38_38457


namespace remainder_of_polynomial_division_l38_38999

-- Definitions based on conditions in the problem
def polynomial (x : ℝ) : ℝ := 8 * x^4 - 22 * x^3 + 9 * x^2 + 10 * x - 45

def divisor (x : ℝ) : ℝ := 4 * x - 8

-- Proof statement as per the problem equivalence
theorem remainder_of_polynomial_division : polynomial 2 = -37 := by
  sorry

end remainder_of_polynomial_division_l38_38999


namespace fraction_power_multiplication_l38_38749

theorem fraction_power_multiplication :
  ( (1 / 3) ^ 4 * (1 / 5) = 1 / 405 ) :=
by
  sorry

end fraction_power_multiplication_l38_38749


namespace patsy_needs_more_appetizers_l38_38532

def appetizers_per_guest := 6
def number_of_guests := 30
def deviled_eggs := 3 -- dozens
def pigs_in_a_blanket := 2 -- dozens
def kebabs := 2 -- dozens

theorem patsy_needs_more_appetizers :
  let total_required := appetizers_per_guest * number_of_guests,
      total_made := (deviled_eggs + pigs_in_a_blanket + kebabs) * 12,
      total_needed := total_required - total_made
  in total_needed / 12 = 8 := sorry

end patsy_needs_more_appetizers_l38_38532


namespace cos_45_deg_l38_38118

theorem cos_45_deg : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_45_deg_l38_38118


namespace combination_indices_l38_38921
open Nat

theorem combination_indices (x : ℕ) (h : choose 18 x = choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end combination_indices_l38_38921


namespace exists_range_of_real_numbers_l38_38629

theorem exists_range_of_real_numbers (x : ℝ) :
  (x^2 - 5 * x + 7 ≠ 1) ↔ (x ≠ 3 ∧ x ≠ 2) := 
sorry

end exists_range_of_real_numbers_l38_38629


namespace dragons_at_meeting_l38_38067

def dragon_meeting : Prop :=
  ∃ (x y : ℕ), 
    (2 * x + 7 * y = 26) ∧ 
    (x + y = 8)

theorem dragons_at_meeting : dragon_meeting :=
by
  sorry

end dragons_at_meeting_l38_38067


namespace father_l38_38934

theorem father's_age :
  ∃ (S F : ℕ), 2 * S + F = 70 ∧ S + 2 * F = 95 ∧ F = 40 :=
by
  sorry

end father_l38_38934


namespace simplify_trig_expression_l38_38686

theorem simplify_trig_expression (A : ℝ) :
  (2 - (Real.cos A / Real.sin A) + (1 / Real.sin A)) * (3 - (Real.sin A / Real.cos A) - (1 / Real.cos A)) = 
  7 * Real.sin A * Real.cos A - 2 * Real.cos A ^ 2 - 3 * Real.sin A ^ 2 - 3 * Real.cos A + Real.sin A + 1 :=
by
  sorry

end simplify_trig_expression_l38_38686


namespace units_digit_k_squared_plus_2_k_is_7_l38_38354

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_k_squared_plus_2_k_is_7 : (k^2 + 2^k) % 10 = 7 :=
by sorry

end units_digit_k_squared_plus_2_k_is_7_l38_38354


namespace hexagon_diagonals_l38_38862

theorem hexagon_diagonals : (6 * (6 - 3)) / 2 = 9 := 
by 
  sorry

end hexagon_diagonals_l38_38862


namespace subset_condition_l38_38621

noncomputable def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : (B m ⊆ A) ↔ m ≤ 3 :=
sorry

end subset_condition_l38_38621


namespace yaw_yaw_age_in_2016_l38_38093

def is_lucky_double_year (y : Nat) : Prop :=
  let d₁ := y / 1000 % 10
  let d₂ := y / 100 % 10
  let d₃ := y / 10 % 10
  let last_digit := y % 10
  last_digit = 2 * (d₁ + d₂ + d₃)

theorem yaw_yaw_age_in_2016 (next_lucky_year : Nat) (yaw_yaw_age_in_next_lucky_year : Nat)
  (h1 : is_lucky_double_year 2016)
  (h2 : ∀ y, y > 2016 → is_lucky_double_year y → y = next_lucky_year)
  (h3 : yaw_yaw_age_in_next_lucky_year = 17) :
  (17 - (next_lucky_year - 2016)) = 5 := sorry

end yaw_yaw_age_in_2016_l38_38093


namespace dress_assignments_l38_38134

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l38_38134


namespace trigonometric_inequality_C_trigonometric_inequality_D_l38_38087

theorem trigonometric_inequality_C (x : Real) : Real.cos (3*Real.pi/5) > Real.cos (-4*Real.pi/5) :=
by
  sorry

theorem trigonometric_inequality_D (y : Real) : Real.sin (Real.pi/10) < Real.cos (Real.pi/10) :=
by
  sorry

end trigonometric_inequality_C_trigonometric_inequality_D_l38_38087


namespace number_of_strikers_correct_l38_38727

-- Defining the initial conditions
def number_of_goalies := 3
def number_of_defenders := 10
def number_of_players := 40
def number_of_midfielders := 2 * number_of_defenders

-- Lean statement to prove
theorem number_of_strikers_correct : 
  let total_non_strikers := number_of_goalies + number_of_defenders + number_of_midfielders,
      number_of_strikers := number_of_players - total_non_strikers 
  in number_of_strikers = 7 :=
by
  sorry

end number_of_strikers_correct_l38_38727
