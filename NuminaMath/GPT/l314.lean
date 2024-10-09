import Mathlib

namespace number_of_ways_to_purchase_magazines_l314_31445

/-
Conditions:
1. The bookstore sells 11 different magazines.
2. 8 of these magazines are priced at 2 yuan each.
3. 3 of these magazines are priced at 1 yuan each.
4. Xiao Zhang has 10 yuan to buy magazines.
5. Xiao Zhang can buy at most one copy of each magazine.
6. Xiao Zhang wants to spend all 10 yuan.

Question:
The number of different ways Xiao Zhang can purchase magazines with 10 yuan.

Answer:
266
-/

theorem number_of_ways_to_purchase_magazines : ∀ (magazines_1_yuan magazines_2_yuan : ℕ),
  magazines_1_yuan = 3 →
  magazines_2_yuan = 8 →
  (∃ (ways : ℕ), ways = 266) :=
by
  intros
  sorry

end number_of_ways_to_purchase_magazines_l314_31445


namespace part1_part2_l314_31402

-- Step 1: Define the problem for a triangle with specific side length conditions and perimeter
theorem part1 (x : ℝ) (h1 : 2 * x + 2 * (2 * x) = 18) : 
  x = 18 / 5 ∧ 2 * x = 36 / 5 :=
by
  sorry

-- Step 2: Verify if an isosceles triangle with a side length of 4 cm can be formed
theorem part2 (a b c : ℝ) (h2 : a = 4 ∨ b = 4 ∨ c = 4) (h3 : a + b + c = 18) : 
  (a = 4 ∧ b = 7 ∧ c = 7 ∨ b = 4 ∧ a = 7 ∧ c = 7 ∨ c = 4 ∧ a = 7 ∧ b = 7) ∨
  (¬(a = 4 ∧ b + c <= a ∨ b = 4 ∧ a + c <= b ∨ c = 4 ∧ a + b <= c)) :=
by
  sorry

end part1_part2_l314_31402


namespace find_solutions_l314_31410

theorem find_solutions (x y z : ℝ) :
    (x^2 + y^2 - z * (x + y) = 2 ∧ y^2 + z^2 - x * (y + z) = 4 ∧ z^2 + x^2 - y * (z + x) = 8) ↔
    (x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2) := sorry

end find_solutions_l314_31410


namespace three_digit_numbers_with_square_ending_in_them_l314_31468

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l314_31468


namespace div_pow_sub_one_l314_31482

theorem div_pow_sub_one (n : ℕ) (h : n > 1) : (n - 1) ^ 2 ∣ n ^ (n - 1) - 1 :=
sorry

end div_pow_sub_one_l314_31482


namespace find_y_l314_31408

theorem find_y (x y : Int) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := 
by 
  sorry

end find_y_l314_31408


namespace bus_capacities_rental_plan_l314_31494

variable (x y : ℕ)
variable (m n : ℕ)

theorem bus_capacities :
  3 * x + 2 * y = 195 ∧ 2 * x + 4 * y = 210 → x = 45 ∧ y = 30 :=
by
  sorry

theorem rental_plan :
  7 * m + 3 * n = 20 ∧ m + n ≤ 7 ∧ 65 * m + 45 * n + 30 * (7 - m - n) = 310 →
  m = 2 ∧ n = 2 ∧ 7 - m - n = 3 :=
by
  sorry

end bus_capacities_rental_plan_l314_31494


namespace larger_number_is_23_l314_31428

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l314_31428


namespace set_intersection_example_l314_31415

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l314_31415


namespace overall_average_commission_rate_l314_31448

-- Define conditions for the commissions and transaction amounts
def C₁ := 0.25 / 100 * 100 + 0.25 / 100 * 105.25
def C₂ := 0.35 / 100 * 150 + 0.45 / 100 * 155.50
def C₃ := 0.30 / 100 * 80 + 0.40 / 100 * 83
def total_commission := C₁ + C₂ + C₃
def TA := 100 + 105.25 + 150 + 155.50 + 80 + 83

-- The proposition to prove
theorem overall_average_commission_rate : (total_commission / TA) * 100 = 0.3429 :=
  by
  sorry

end overall_average_commission_rate_l314_31448


namespace cotton_needed_l314_31438

noncomputable def feet_of_cotton_per_teeshirt := 4
noncomputable def number_of_teeshirts := 15

theorem cotton_needed : feet_of_cotton_per_teeshirt * number_of_teeshirts = 60 := 
by 
  sorry

end cotton_needed_l314_31438


namespace triangle_angle_B_l314_31406

theorem triangle_angle_B (A B C : ℕ) (h₁ : B + C = 110) (h₂ : A + B + C = 180) (h₃ : A = 70) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end triangle_angle_B_l314_31406


namespace complex_frac_eq_l314_31432

theorem complex_frac_eq (a b : ℝ) (i : ℂ) (h : i^2 = -1)
  (h1 : (1 - i) / (1 + i) = a + b * i) : a - b = 1 :=
by
  sorry

end complex_frac_eq_l314_31432


namespace evaluate_expression_l314_31434

def a : ℕ := 3^1
def b : ℕ := 3^2
def c : ℕ := 3^3
def d : ℕ := 3^4
def e : ℕ := 3^10
def S : ℕ := a + b + c + d

theorem evaluate_expression : e - S = 58929 := 
by
  sorry

end evaluate_expression_l314_31434


namespace books_read_l314_31460

theorem books_read (total_books remaining_books read_books : ℕ)
  (h_total : total_books = 14)
  (h_remaining : remaining_books = 6)
  (h_eq : read_books = total_books - remaining_books) : read_books = 8 := 
by 
  sorry

end books_read_l314_31460


namespace bags_of_cookies_l314_31458

theorem bags_of_cookies (bags : ℕ) (cookies_total candies_total : ℕ) 
    (h1 : bags = 14) (h2 : cookies_total = 28) (h3 : candies_total = 86) :
    bags = 14 :=
by
  exact h1

end bags_of_cookies_l314_31458


namespace solve_system_of_equations_l314_31437

/-- Definition representing our system of linear equations. --/
def system_of_equations (x1 x2 : ℚ) : Prop :=
  (3 * x1 - 5 * x2 = 2) ∧ (2 * x1 + 4 * x2 = 5)

/-- The main theorem stating the solution to our system of equations. --/
theorem solve_system_of_equations : 
  ∃ x1 x2 : ℚ, system_of_equations x1 x2 ∧ x1 = 3/2 ∧ x2 = 1/2 :=
by
  sorry

end solve_system_of_equations_l314_31437


namespace product_of_invertible_function_labels_l314_31470

noncomputable def Function6 (x : ℝ) : ℝ := x^3 - 3 * x
def points7 : List (ℝ × ℝ) := [(-6, 3), (-5, 1), (-4, 2), (-3, -1), (-2, 0), (-1, -2), (0, 4), (1, 5)]
noncomputable def Function8 (x : ℝ) : ℝ := Real.sin x
noncomputable def Function9 (x : ℝ) : ℝ := 3 / x

def is_invertible6 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function6 x1 = y ∧ Function6 x2 = y ∧ (-2 ≤ x1 ∧ x1 ≤ 2) ∧ (-2 ≤ x2 ∧ x2 ≤ 2)
def is_invertible7 : Prop := ∀ (y : ℝ), ∃! x : ℝ, (x, y) ∈ points7
def is_invertible8 : Prop := ∀ (x1 x2 : ℝ), Function8 x1 = Function8 x2 → x1 = x2 ∧ (-Real.pi/2 ≤ x1 ∧ x1 ≤ Real.pi/2) ∧ (-Real.pi/2 ≤ x2 ∧ x2 ≤ Real.pi/2)
def is_invertible9 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function9 x1 = y ∧ Function9 x2 = y ∧ (-4 ≤ x1 ∧ x1 ≤ 4 ∧ x1 ≠ 0) ∧ (-4 ≤ x2 ∧ x2 ≤ 4 ∧ x2 ≠ 0)

theorem product_of_invertible_function_labels :
  (is_invertible6 = false) →
  (is_invertible7 = true) →
  (is_invertible8 = true) →
  (is_invertible9 = true) →
  7 * 8 * 9 = 504
:= by
  intros h6 h7 h8 h9
  sorry

end product_of_invertible_function_labels_l314_31470


namespace larger_number_of_hcf_lcm_is_322_l314_31451

theorem larger_number_of_hcf_lcm_is_322
  (A B : ℕ)
  (hcf: ℕ := 23)
  (factor1 : ℕ := 13)
  (factor2 : ℕ := 14)
  (hcf_condition : ∀ d, d ∣ A → d ∣ B → d ≤ hcf)
  (lcm_condition : ∀ m n, m * n = A * B → m = factor1 * hcf ∨ m = factor2 * hcf) :
  max A B = 322 :=
by sorry

end larger_number_of_hcf_lcm_is_322_l314_31451


namespace apples_to_pears_l314_31497

theorem apples_to_pears :
  (∀ (apples oranges pears : ℕ),
  12 * apples = 6 * oranges →
  3 * oranges = 5 * pears →
  24 * apples = 20 * pears) :=
by
  intros apples oranges pears h₁ h₂
  sorry

end apples_to_pears_l314_31497


namespace distinct_sequences_ten_flips_l314_31447

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l314_31447


namespace polygon_sides_l314_31492

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 := sorry

end polygon_sides_l314_31492


namespace average_visitors_30_day_month_l314_31409

def visitors_per_day (total_visitors : ℕ) (days : ℕ) : ℕ := total_visitors / days

theorem average_visitors_30_day_month (visitors_sunday : ℕ) (visitors_other_days : ℕ) 
  (total_days : ℕ) (sundays : ℕ) (other_days : ℕ) :
  visitors_sunday = 510 →
  visitors_other_days = 240 →
  total_days = 30 →
  sundays = 4 →
  other_days = 26 →
  visitors_per_day (sundays * visitors_sunday + other_days * visitors_other_days) total_days = 276 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end average_visitors_30_day_month_l314_31409


namespace matrix_A_pow_100_eq_l314_31411

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1], ![-9, -2]]

theorem matrix_A_pow_100_eq : matrix_A ^ 100 = ![![301, 100], ![-900, -299]] :=
  sorry

end matrix_A_pow_100_eq_l314_31411


namespace total_handshakes_is_316_l314_31433

def number_of_couples : ℕ := 15
def number_of_people : ℕ := number_of_couples * 2

def handshakes_among_men (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)
def handshakes_between_women : ℕ := 1
def total_handshakes (n : ℕ) : ℕ := handshakes_among_men n + handshakes_men_women n + handshakes_between_women

theorem total_handshakes_is_316 : total_handshakes number_of_couples = 316 :=
by
  sorry

end total_handshakes_is_316_l314_31433


namespace saving_percentage_l314_31423

variable (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ)

-- Conditions from problem
def condition1 := saved_percent_last_year = 0.06
def condition2 := made_more = 1.20
def condition3 := saved_percent_this_year = 0.05 * made_more

-- The problem statement to prove
theorem saving_percentage (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ) :
  condition1 saved_percent_last_year →
  condition2 made_more →
  condition3 saved_percent_this_year made_more →
  (saved_percent_this_year * made_more = saved_percent_last_year * S * 1) :=
by 
  intros h1 h2 h3
  sorry

end saving_percentage_l314_31423


namespace total_money_made_l314_31443

-- Define the conditions
def dollars_per_day : Int := 144
def number_of_days : Int := 22

-- State the proof problem
theorem total_money_made : (dollars_per_day * number_of_days = 3168) :=
by
  sorry

end total_money_made_l314_31443


namespace kite_height_30_sqrt_43_l314_31476

theorem kite_height_30_sqrt_43
  (c d h : ℝ)
  (h1 : h^2 + c^2 = 170^2)
  (h2 : h^2 + d^2 = 150^2)
  (h3 : c^2 + d^2 = 160^2) :
  h = 30 * Real.sqrt 43 := by
  sorry

end kite_height_30_sqrt_43_l314_31476


namespace soldiers_movement_l314_31401

theorem soldiers_movement (n : ℕ) 
  (initial_positions : Fin (n+3) × Fin (n+1) → Prop) 
  (moves_to_adjacent : ∀ p : Fin (n+3) × Fin (n+1), initial_positions p → initial_positions (p.1 + 1, p.2) ∨ initial_positions (p.1 - 1, p.2) ∨ initial_positions (p.1, p.2 + 1) ∨ initial_positions (p.1, p.2 - 1))
  (final_positions : Fin (n+1) × Fin (n+3) → Prop) : Even n := 
sorry

end soldiers_movement_l314_31401


namespace arithmetic_sequence_8th_term_l314_31413

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end arithmetic_sequence_8th_term_l314_31413


namespace remainder_of_sum_l314_31453

theorem remainder_of_sum (p q : ℤ) (c d : ℤ) 
  (hc : c = 100 * p + 78)
  (hd : d = 150 * q + 123) :
  (c + d) % 50 = 1 :=
sorry

end remainder_of_sum_l314_31453


namespace larger_solution_quadratic_l314_31450

theorem larger_solution_quadratic : 
  ∀ x1 x2 : ℝ, (x^2 - 13 * x - 48 = 0) → x1 ≠ x2 → (x1 = 16 ∨ x2 = 16) → max x1 x2 = 16 :=
by
  sorry

end larger_solution_quadratic_l314_31450


namespace toby_photo_shoot_l314_31419

theorem toby_photo_shoot (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_pictures : ℕ) (deleted_post_editing : ℕ) (final_photos : ℕ) (photo_shoot_photos : ℕ) :
  initial_photos = 63 →
  deleted_bad_shots = 7 →
  cat_pictures = 15 →
  deleted_post_editing = 3 →
  final_photos = 84 →
  final_photos = initial_photos - deleted_bad_shots + cat_pictures + photo_shoot_photos - deleted_post_editing →
  photo_shoot_photos = 16 :=
by
  intros
  sorry

end toby_photo_shoot_l314_31419


namespace natural_number_sets_solution_l314_31479

theorem natural_number_sets_solution (x y n : ℕ) (h : (x! + y!) / n! = 3^n) : (x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end natural_number_sets_solution_l314_31479


namespace import_tax_percentage_l314_31430

theorem import_tax_percentage
  (total_value : ℝ)
  (non_taxable_portion : ℝ)
  (import_tax_paid : ℝ)
  (h_total_value : total_value = 2610)
  (h_non_taxable_portion : non_taxable_portion = 1000)
  (h_import_tax_paid : import_tax_paid = 112.70) :
  ((import_tax_paid / (total_value - non_taxable_portion)) * 100) = 7 :=
by
  sorry

end import_tax_percentage_l314_31430


namespace pavan_distance_travelled_l314_31412

theorem pavan_distance_travelled (D : ℝ) (h1 : D / 60 + D / 50 = 11) : D = 300 :=
sorry

end pavan_distance_travelled_l314_31412


namespace price_per_unit_max_profit_l314_31449

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end price_per_unit_max_profit_l314_31449


namespace binomial_12_10_eq_66_l314_31481

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l314_31481


namespace arithmetic_sequence_general_term_l314_31457

theorem arithmetic_sequence_general_term:
  ∃ (a : ℕ → ℕ), 
    (∀ n, a n + 1 > a n) ∧
    (a 1 = 2) ∧ 
    ((a 2) ^ 2 = a 5 + 6) ∧ 
    (∀ n, a n = 2 * n) :=
by
  sorry

end arithmetic_sequence_general_term_l314_31457


namespace license_plate_count_l314_31444

theorem license_plate_count : 
  let consonants := 20
  let vowels := 6
  let digits := 10
  4 * consonants * vowels * consonants * digits = 24000 :=
by
  sorry

end license_plate_count_l314_31444


namespace matrix_power_A_2023_l314_31478

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 1]
  ]

theorem matrix_power_A_2023 :
  A ^ 2023 = ![
    ![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]
  ] :=
sorry

end matrix_power_A_2023_l314_31478


namespace number_of_subsets_of_M_l314_31471

def M : Set ℝ := { x | x^2 - 2 * x + 1 = 0 }

theorem number_of_subsets_of_M : M = {1} → ∃ n, n = 2 := by
  sorry

end number_of_subsets_of_M_l314_31471


namespace original_players_count_l314_31456

theorem original_players_count (n : ℕ) (W : ℕ) :
  (W = n * 103) →
  ((W + 110 + 60) = (n + 2) * 99) →
  n = 7 :=
by sorry

end original_players_count_l314_31456


namespace smallest_positive_integer_l314_31473

theorem smallest_positive_integer :
  ∃ (n a b m : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ n = 153846 ∧
  (n = 10^m * a + b) ∧
  (7 * n = 2 * (10 * b + a)) :=
by
  sorry

end smallest_positive_integer_l314_31473


namespace dodecahedron_interior_diagonals_l314_31446

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l314_31446


namespace power_mod_l314_31462

theorem power_mod (n : ℕ) : 3^100 % 7 = 4 := by
  sorry

end power_mod_l314_31462


namespace simplify_expression_l314_31485

theorem simplify_expression (x : ℤ) (h1 : 2 * (x - 1) < x + 1) (h2 : 5 * x + 3 ≥ 2 * x) :
  (x = 2) → (2 / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1 / 2) :=
by
  sorry

end simplify_expression_l314_31485


namespace intersection_eq_l314_31487

theorem intersection_eq {A : Set ℕ} {B : Set ℕ} 
  (hA : A = {0, 1, 2, 3, 4, 5, 6}) 
  (hB : B = {x | ∃ n ∈ A, x = 2 * n}) : 
  A ∩ B = {0, 2, 4, 6} := by
  sorry

end intersection_eq_l314_31487


namespace zeros_in_expansion_l314_31440

def num_zeros_expansion (n : ℕ) : ℕ :=
-- This function counts the number of trailing zeros in the decimal representation of n.
sorry

theorem zeros_in_expansion : num_zeros_expansion ((10^12 - 3)^2) = 11 :=
sorry

end zeros_in_expansion_l314_31440


namespace tom_gave_jessica_some_seashells_l314_31405

theorem tom_gave_jessica_some_seashells
  (original_seashells : ℕ := 5)
  (current_seashells : ℕ := 3) :
  original_seashells - current_seashells = 2 :=
by
  sorry

end tom_gave_jessica_some_seashells_l314_31405


namespace interest_is_less_by_1940_l314_31442

noncomputable def principal : ℕ := 2000
noncomputable def rate : ℕ := 3
noncomputable def time : ℕ := 3

noncomputable def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

noncomputable def difference (sum_lent interest : ℕ) : ℕ :=
  sum_lent - interest

theorem interest_is_less_by_1940 :
  difference principal (simple_interest principal rate time) = 1940 :=
by
  sorry

end interest_is_less_by_1940_l314_31442


namespace certain_number_is_3_l314_31407

theorem certain_number_is_3 (x : ℚ) (h : (x / 11) * ((121 : ℚ) / 3) = 11) : x = 3 := 
sorry

end certain_number_is_3_l314_31407


namespace probability_of_pink_gumball_l314_31417

theorem probability_of_pink_gumball (P_B P_P : ℝ)
    (h1 : P_B ^ 2 = 25 / 49)
    (h2 : P_B + P_P = 1) :
    P_P = 2 / 7 := 
    sorry

end probability_of_pink_gumball_l314_31417


namespace find_c_value_l314_31436

theorem find_c_value (b c : ℝ) 
  (h1 : 1 + b + c = 4) 
  (h2 : 25 + 5 * b + c = 4) : 
  c = 9 :=
by
  sorry

end find_c_value_l314_31436


namespace positive_real_numbers_l314_31477

theorem positive_real_numbers
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : b * c + c * a + a * b > 0)
  (h3 : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end positive_real_numbers_l314_31477


namespace calculate_expression_l314_31452

theorem calculate_expression : 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 :=
by 
  sorry

end calculate_expression_l314_31452


namespace circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l314_31421

variable {α β γ : ℝ}

/-- The equation of the circumscribed Steiner ellipse in barycentric coordinates -/
theorem circumscribed_steiner_ellipse (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 :=
sorry

/-- The equation of the inscribed Steiner ellipse in barycentric coordinates -/
theorem inscribed_steiner_ellipse (h : α + β + γ = 1) :
  2 * β * γ + 2 * α * γ + 2 * α * β = α^2 + β^2 + γ^2 :=
sorry

end circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l314_31421


namespace sequence_2019_value_l314_31491

theorem sequence_2019_value :
  ∃ a : ℕ → ℤ, (∀ n ≥ 4, a n = a (n-1) * a (n-3)) ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ a 2019 = -1 :=
by
  sorry

end sequence_2019_value_l314_31491


namespace trapezoid_perimeter_area_sum_l314_31404

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

noncomputable def perimeter (vertices : List (Real × Real)) : Real :=
  match vertices with
  | [a, b, c, d] => (distance a b) + (distance b c) + (distance c d) + (distance d a)
  | _ => 0

noncomputable def area_trapezoid (b1 b2 h : Real) : Real :=
  0.5 * (b1 + b2) * h

theorem trapezoid_perimeter_area_sum
  (A B C D : Real × Real)
  (h_AB : A = (2, 3))
  (h_BC : B = (7, 3))
  (h_CD : C = (9, 7))
  (h_DA : D = (0, 7)) :
  let perimeter := perimeter [A, B, C, D]
  let area := area_trapezoid (distance C D) (distance A B) (C.2 - B.2)
  perimeter + area = 42 + 4 * Real.sqrt 5 :=
by
  sorry

end trapezoid_perimeter_area_sum_l314_31404


namespace number_of_chickens_l314_31484

def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12
def full_cartons : ℕ := 10

theorem number_of_chickens :
  (full_cartons * eggs_per_carton) / eggs_per_chicken = 20 :=
by
  sorry

end number_of_chickens_l314_31484


namespace find_B_share_l314_31490

theorem find_B_share (x : ℕ) (x_pos : 0 < x) (C_share_difference : 5 * x = 4 * x + 1000) (B_share_eq : 3 * x = B) : B = 3000 :=
by
  sorry

end find_B_share_l314_31490


namespace arithmetic_sequence_properties_l314_31496

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a n = a_1 + d * (n - 1)

def sum_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def sum_b (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = n^2 + n + (3^(n+1) - 3)/2

theorem arithmetic_sequence_properties :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (arithmetic_seq a) →
    a 5 = 10 →
    S 7 = 56 →
    (∀ n, a n = 2 * n) ∧
    ∃ (b T : ℕ → ℕ), (∀ n, b n = a n + 3^n) ∧ sum_b b T :=
by
  intros a S ha h5 hS7
  sorry

end arithmetic_sequence_properties_l314_31496


namespace mean_temperature_correct_l314_31414

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

def mean_temperature (temps : List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

theorem mean_temperature_correct :
  mean_temperature temperatures = -9 / 7 := 
by
  sorry

end mean_temperature_correct_l314_31414


namespace find_number_l314_31469

theorem find_number (x : ℝ) (h : 4 * (x - 220) = 320) : (5 * x) / 3 = 500 :=
by
  sorry

end find_number_l314_31469


namespace M_union_N_eq_M_l314_31422

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs (p.1 * p.2) = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end M_union_N_eq_M_l314_31422


namespace LindseyMinimumSavings_l314_31425
-- Import the library to bring in the necessary definitions and notations

-- Definitions from the problem conditions
def SeptemberSavings : ℕ := 50
def OctoberSavings : ℕ := 37
def NovemberSavings : ℕ := 11
def MomContribution : ℕ := 25
def VideoGameCost : ℕ := 87
def RemainingMoney : ℕ := 36

-- Problem statement as a Lean theorem
theorem LindseyMinimumSavings : 
  (SeptemberSavings + OctoberSavings + NovemberSavings) > 98 :=
  sorry

end LindseyMinimumSavings_l314_31425


namespace profit_calculation_l314_31475

open Nat

-- Define the conditions 
def cost_of_actors : Nat := 1200 
def number_of_people : Nat := 50
def cost_per_person_food : Nat := 3
def sale_price : Nat := 10000

-- Define the derived costs
def total_food_cost : Nat := number_of_people * cost_per_person_food
def total_combined_cost : Nat := cost_of_actors + total_food_cost
def equipment_rental_cost : Nat := 2 * total_combined_cost
def total_cost : Nat := cost_of_actors + total_food_cost + equipment_rental_cost
def expected_profit : Nat := 5950 

-- Define the profit calculation
def profit : Nat := sale_price - total_cost 

-- The theorem to be proved
theorem profit_calculation : profit = expected_profit := by
  -- Proof is omitted
  sorry

end profit_calculation_l314_31475


namespace has_two_distinct_real_roots_parabola_equation_l314_31418

open Real

-- Define the quadratic polynomial
def quad_poly (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * m * x + m^2 - 4

-- Question 1: Prove that the quadratic equation has two distinct real roots
theorem has_two_distinct_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (quad_poly m x₁ = 0) ∧ (quad_poly m x₂ = 0) := by
  sorry

-- Question 2: Prove the equation of the parabola given certain conditions
theorem parabola_equation (m : ℝ) (hx : quad_poly m 0 = 0) : 
  m = 0 ∧ ∀ x : ℝ, quad_poly m x = x^2 - 4 := by
  sorry

end has_two_distinct_real_roots_parabola_equation_l314_31418


namespace difference_of_squares_example_l314_31466

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 305) (h2 : b = 295) :
  (a^2 - b^2) / 10 = 600 :=
by
  sorry

end difference_of_squares_example_l314_31466


namespace smallest_product_of_set_l314_31435

noncomputable def smallest_product_set : Set ℤ := { -10, -3, 0, 4, 6 }

theorem smallest_product_of_set :
  ∃ (a b : ℤ), a ∈ smallest_product_set ∧ b ∈ smallest_product_set ∧ a ≠ b ∧ a * b = -60 ∧
  ∀ (x y : ℤ), x ∈ smallest_product_set ∧ y ∈ smallest_product_set ∧ x ≠ y → x * y ≥ -60 := 
sorry

end smallest_product_of_set_l314_31435


namespace qualified_flour_l314_31427

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end qualified_flour_l314_31427


namespace minor_premise_of_syllogism_l314_31467

theorem minor_premise_of_syllogism (P Q : Prop)
  (h1 : ¬ (P ∧ ¬ Q))
  (h2 : Q) :
  Q :=
by
  sorry

end minor_premise_of_syllogism_l314_31467


namespace xy_ratio_l314_31459

variables (x y z t : ℝ)
variables (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t))

theorem xy_ratio (x y : ℝ) (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t)) :
  x / y = 25 :=
sorry

end xy_ratio_l314_31459


namespace naomi_total_wheels_l314_31464

theorem naomi_total_wheels 
  (regular_bikes : ℕ) (children_bikes : ℕ) (tandem_bikes_4_wheels : ℕ) (tandem_bikes_6_wheels : ℕ)
  (wheels_per_regular_bike : ℕ) (wheels_per_children_bike : ℕ) (wheels_per_tandem_4wheel : ℕ) (wheels_per_tandem_6wheel : ℕ) :
  regular_bikes = 7 →
  children_bikes = 11 →
  tandem_bikes_4_wheels = 5 →
  tandem_bikes_6_wheels = 3 →
  wheels_per_regular_bike = 2 →
  wheels_per_children_bike = 4 →
  wheels_per_tandem_4wheel = 4 →
  wheels_per_tandem_6wheel = 6 →
  (regular_bikes * wheels_per_regular_bike) + 
  (children_bikes * wheels_per_children_bike) + 
  (tandem_bikes_4_wheels * wheels_per_tandem_4wheel) + 
  (tandem_bikes_6_wheels * wheels_per_tandem_6wheel) = 96 := 
by
  intros; sorry

end naomi_total_wheels_l314_31464


namespace effect_of_dimension_changes_on_area_l314_31455

variable {L B : ℝ}  -- Original length and breadth

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.15 * L

def new_breadth (B : ℝ) : ℝ := 0.90 * B

def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem effect_of_dimension_changes_on_area (L B : ℝ) :
  new_area L B = 1.035 * original_area L B :=
by
  sorry

end effect_of_dimension_changes_on_area_l314_31455


namespace complement_of_supplement_of_35_degree_l314_31480

def angle : ℝ := 35
def supplement (x : ℝ) : ℝ := 180 - x
def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_supplement_of_35_degree :
  complement (supplement angle) = -55 := by
  sorry

end complement_of_supplement_of_35_degree_l314_31480


namespace greatest_possible_avg_speed_l314_31454

theorem greatest_possible_avg_speed (initial_odometer : ℕ) (max_speed : ℕ) (time_hours : ℕ) (max_distance : ℕ) (target_palindrome : ℕ) :
  initial_odometer = 12321 →
  max_speed = 80 →
  time_hours = 4 →
  (target_palindrome = 12421 ∨ target_palindrome = 12521 ∨ target_palindrome = 12621 ∨ target_palindrome = 12721 ∨ target_palindrome = 12821 ∨ target_palindrome = 12921 ∨ target_palindrome = 13031) →
  target_palindrome - initial_odometer ≤ max_distance →
  max_distance = 300 →
  target_palindrome = 12621 →
  time_hours = 4 →
  target_palindrome - initial_odometer = 300 →
  (target_palindrome - initial_odometer) / time_hours = 75 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end greatest_possible_avg_speed_l314_31454


namespace average_rate_of_change_l314_31493

noncomputable def f (x : ℝ) := 2 * x + 1

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_l314_31493


namespace power_of_product_l314_31463

theorem power_of_product (x : ℝ) : (-x^4)^3 = -x^12 := 
by sorry

end power_of_product_l314_31463


namespace measure_angle_R_l314_31483

theorem measure_angle_R (P Q R : ℝ) (h1 : P + Q = 60) : R = 120 :=
by
  have sum_of_angles_in_triangle : P + Q + R = 180 := sorry
  rw [h1] at sum_of_angles_in_triangle
  linarith

end measure_angle_R_l314_31483


namespace gasoline_tank_capacity_l314_31431

theorem gasoline_tank_capacity :
  ∀ (x : ℕ), (5 / 6 * (x : ℚ) - 18 = 1 / 3 * (x : ℚ)) → x = 36 :=
by
  sorry

end gasoline_tank_capacity_l314_31431


namespace rectangle_dimensions_l314_31441

theorem rectangle_dimensions (a1 a2 : ℝ) (h1 : a1 * a2 = 216) (h2 : a1 + a2 = 30 - 6)
  (h3 : 6 * 6 = 36) : (a1 = 12 ∧ a2 = 18) ∨ (a1 = 18 ∧ a2 = 12) :=
by
  -- The conditions are set; now we need the proof, which we'll replace with sorry for now.
  sorry

end rectangle_dimensions_l314_31441


namespace solution_set_of_inequality_l314_31400

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2*x + 15 ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 5} := 
sorry

end solution_set_of_inequality_l314_31400


namespace integer_solutions_l314_31472

theorem integer_solutions (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intros h
  sorry

end integer_solutions_l314_31472


namespace wire_cut_problem_l314_31499

theorem wire_cut_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq_area : (a / 4) ^ 2 = π * (b / (2 * π)) ^ 2) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_problem_l314_31499


namespace conic_section_union_l314_31486

theorem conic_section_union : 
  ∀ (y x : ℝ), y^4 - 6*x^4 = 3*y^2 - 2 → 
  ( ( y^2 - 3*x^2 = 1 ∨ y^2 - 2*x^2 = 1 ) ∧ 
    ( y^2 - 2*x^2 = 2 ∨ y^2 - 3*x^2 = 2 ) ) :=
by
  sorry

end conic_section_union_l314_31486


namespace number_of_tables_l314_31416

-- Define the total number of customers the waiter is serving
def total_customers := 90

-- Define the number of women per table
def women_per_table := 7

-- Define the number of men per table
def men_per_table := 3

-- Define the total number of people per table
def people_per_table : ℕ := women_per_table + men_per_table

-- Statement to prove the number of tables
theorem number_of_tables (T : ℕ) (h : T * people_per_table = total_customers) : T = 9 := by
  sorry

end number_of_tables_l314_31416


namespace reciprocal_of_fraction_diff_l314_31465

theorem reciprocal_of_fraction_diff : 
  (∃ (a b : ℚ), a = 1/4 ∧ b = 1/5 ∧ (1 / (a - b)) = 20) :=
sorry

end reciprocal_of_fraction_diff_l314_31465


namespace ratio_sea_horses_penguins_l314_31403

def sea_horses := 70
def penguins := sea_horses + 85

theorem ratio_sea_horses_penguins : (70 : ℚ) / (sea_horses + 85) = 14 / 31 :=
by
  -- Proof omitted
  sorry

end ratio_sea_horses_penguins_l314_31403


namespace question_inequality_l314_31498

theorem question_inequality
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (cond : a + b ≤ 4) :
  (1 / a + 1 / b) ≥ 1 := 
sorry

end question_inequality_l314_31498


namespace yoongi_class_combination_l314_31439

theorem yoongi_class_combination : (Nat.choose 10 3 = 120) := by
  sorry

end yoongi_class_combination_l314_31439


namespace problem_statement_l314_31420

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l314_31420


namespace fraction_calculation_l314_31461

theorem fraction_calculation : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 :=
by sorry

end fraction_calculation_l314_31461


namespace basket_weight_l314_31424

def weight_of_basket_alone (n_pears : ℕ) (weight_per_pear total_weight : ℚ) : ℚ :=
  total_weight - (n_pears * weight_per_pear)

theorem basket_weight :
  weight_of_basket_alone 30 0.36 11.26 = 0.46 := by
  sorry

end basket_weight_l314_31424


namespace draw_points_value_l314_31429

theorem draw_points_value
  (D : ℕ) -- Let D be the number of points for a draw
  (victory_points : ℕ := 3) -- points for a victory
  (defeat_points : ℕ := 0) -- points for a defeat
  (total_matches : ℕ := 20) -- total matches
  (points_after_5_games : ℕ := 8) -- points scored in the first 5 games
  (minimum_wins_remaining : ℕ := 9) -- at least 9 matches should be won in the remaining matches
  (target_points : ℕ := 40) : -- target points by the end of the tournament
  D = 1 := 
by 
  sorry


end draw_points_value_l314_31429


namespace minimum_spending_l314_31489

noncomputable def box_volume (length width height : ℕ) : ℕ := length * width * height
noncomputable def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
noncomputable def total_cost (num_boxes : ℕ) (price_per_box : ℝ) : ℝ := num_boxes * price_per_box

theorem minimum_spending
  (box_length box_width box_height : ℕ)
  (price_per_box : ℝ)
  (total_collection_volume : ℕ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : price_per_box = 0.90)
  (h5 : total_collection_volume = 3060000) :
  total_cost (total_boxes_needed total_collection_volume (box_volume box_length box_width box_height)) price_per_box = 459 :=
by
  rw [h1, h2, h3, h4, h5]
  have box_vol : box_volume 20 20 15 = 6000 := by norm_num [box_volume]
  have boxes_needed : total_boxes_needed 3060000 6000 = 510 := by norm_num [total_boxes_needed, box_volume, *]
  have cost : total_cost 510 0.90 = 459 := by norm_num [total_cost]
  exact cost

end minimum_spending_l314_31489


namespace number_of_red_cars_l314_31488

theorem number_of_red_cars (B R : ℕ) (h1 : R / B = 3 / 8) (h2 : B = 70) : R = 26 :=
by
  sorry

end number_of_red_cars_l314_31488


namespace arithmetic_series_sum_l314_31474

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 50
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  S = 442 := by
  sorry

end arithmetic_series_sum_l314_31474


namespace mindmaster_code_count_l314_31426

theorem mindmaster_code_count :
  let colors := 7
  let slots := 5
  (colors ^ slots) = 16807 :=
by
  -- Define the given conditions
  let colors := 7
  let slots := 5
  -- Proof statement to be inserted here
  sorry

end mindmaster_code_count_l314_31426


namespace time_to_write_all_rearrangements_l314_31495

-- Define the problem conditions
def sophie_name_length := 6
def rearrangements_per_minute := 18

-- Define the factorial function for calculating permutations
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the total number of rearrangements of Sophie's name
noncomputable def total_rearrangements := factorial sophie_name_length

-- Define the time in minutes to write all rearrangements
noncomputable def time_in_minutes := total_rearrangements / rearrangements_per_minute

-- Convert the time to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Prove the time in hours to write all the rearrangements
theorem time_to_write_all_rearrangements : minutes_to_hours time_in_minutes = (2 : ℚ) / 3 := 
  sorry

end time_to_write_all_rearrangements_l314_31495
