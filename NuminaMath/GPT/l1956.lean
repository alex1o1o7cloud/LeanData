import Mathlib

namespace albums_in_either_but_not_both_l1956_195678

-- Defining the conditions
def shared_albums : ℕ := 9
def total_albums_andrew : ℕ := 17
def unique_albums_john : ℕ := 6

-- Stating the theorem to prove
theorem albums_in_either_but_not_both :
  (total_albums_andrew - shared_albums) + unique_albums_john = 14 :=
sorry

end albums_in_either_but_not_both_l1956_195678


namespace part_a_l1956_195631

theorem part_a (a b : ℤ) (x : ℤ) :
  (x % 5 = a) ∧ (x % 6 = b) → x = 6 * a + 25 * b :=
by
  sorry

end part_a_l1956_195631


namespace total_books_l1956_195630

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

end total_books_l1956_195630


namespace total_cookies_eaten_l1956_195674

theorem total_cookies_eaten :
  let charlie := 15
  let father := 10
  let mother := 5
  let grandmother := 12 / 2
  let dog := 3 * 0.75
  charlie + father + mother + grandmother + dog = 38.25 :=
by
  sorry

end total_cookies_eaten_l1956_195674


namespace supplement_of_complement_of_30_degrees_l1956_195637

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α
def α : ℝ := 30

theorem supplement_of_complement_of_30_degrees : supplement (complement α) = 120 := 
by
  sorry

end supplement_of_complement_of_30_degrees_l1956_195637


namespace jenny_hours_left_l1956_195612

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

end jenny_hours_left_l1956_195612


namespace rotated_angle_new_measure_l1956_195694

theorem rotated_angle_new_measure (initial_angle : ℝ) (rotation : ℝ) (final_angle : ℝ) :
  initial_angle = 60 ∧ rotation = 300 → final_angle = 120 :=
by
  intros h
  sorry

end rotated_angle_new_measure_l1956_195694


namespace kevin_sold_13_crates_of_grapes_l1956_195666

-- Define the conditions
def total_crates : ℕ := 50
def crates_of_mangoes : ℕ := 20
def crates_of_passion_fruits : ℕ := 17

-- Define the question and expected answer
def crates_of_grapes : ℕ := total_crates - (crates_of_mangoes + crates_of_passion_fruits)

-- Prove that the crates of grapes equals to 13
theorem kevin_sold_13_crates_of_grapes :
  crates_of_grapes = 13 :=
by
  -- The proof steps are omitted as per instructions
  sorry

end kevin_sold_13_crates_of_grapes_l1956_195666


namespace second_and_fourth_rows_identical_l1956_195660

def count_occurrences (lst : List ℕ) (a : ℕ) (i : ℕ) : ℕ :=
  (lst.take (i + 1)).count a

def fill_next_row (current_row : List ℕ) : List ℕ :=
  current_row.enum.map (λ ⟨i, a⟩ => count_occurrences current_row a i)

theorem second_and_fourth_rows_identical (first_row : List ℕ) :
  let second_row := fill_next_row first_row 
  let third_row := fill_next_row second_row 
  let fourth_row := fill_next_row third_row 
  second_row = fourth_row :=
by
  sorry

end second_and_fourth_rows_identical_l1956_195660


namespace ratio_black_white_extended_pattern_l1956_195604

def originalBlackTiles : ℕ := 8
def originalWhiteTiles : ℕ := 17
def originalSquareSide : ℕ := 5
def extendedSquareSide : ℕ := 7
def newBlackTiles : ℕ := (extendedSquareSide * extendedSquareSide) - (originalSquareSide * originalSquareSide)
def totalBlackTiles : ℕ := originalBlackTiles + newBlackTiles
def totalWhiteTiles : ℕ := originalWhiteTiles

theorem ratio_black_white_extended_pattern : totalBlackTiles / totalWhiteTiles = 32 / 17 := sorry

end ratio_black_white_extended_pattern_l1956_195604


namespace smallest_number_to_add_quotient_of_resulting_number_l1956_195680

theorem smallest_number_to_add (k : ℕ) : 456 ∣ (897326 + k) → k = 242 := 
sorry

theorem quotient_of_resulting_number : (897326 + 242) / 456 = 1968 := 
sorry

end smallest_number_to_add_quotient_of_resulting_number_l1956_195680


namespace roots_polynomial_pq_sum_l1956_195632

theorem roots_polynomial_pq_sum :
  ∀ p q : ℝ, 
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) * (x - 4) = x^4 - 10 * x^3 + p * x^2 - q * x + 24) 
  → p + q = 85 :=
by 
  sorry

end roots_polynomial_pq_sum_l1956_195632


namespace tricia_age_is_5_l1956_195667

theorem tricia_age_is_5 :
  (∀ Amilia Yorick Eugene Khloe Rupert Vincent : ℕ,
    Tricia = 5 ∧
    (3 * Tricia = Amilia) ∧
    (4 * Amilia = Yorick) ∧
    (2 * Eugene = Yorick) ∧
    (Eugene / 3 = Khloe) ∧
    (Khloe + 10 = Rupert) ∧
    (Vincent = 22)) → 
  Tricia = 5 :=
by
  sorry

end tricia_age_is_5_l1956_195667


namespace rational_roots_of_quadratic_l1956_195690

theorem rational_roots_of_quadratic (r : ℚ) :
  (∃ a b : ℤ, a ≠ b ∧ (r * a^2 + (r + 1) * a + r = 1 ∧ r * b^2 + (r + 1) * b + r = 1)) ↔ (r = 1 ∨ r = -1 / 7) :=
by
  sorry

end rational_roots_of_quadratic_l1956_195690


namespace find_share_of_A_l1956_195650

variable (A B C : ℝ)
variable (h1 : A = (2/3) * B)
variable (h2 : B = (1/4) * C)
variable (h3 : A + B + C = 510)

theorem find_share_of_A : A = 60 :=
by
  sorry

end find_share_of_A_l1956_195650


namespace find_x_l1956_195611

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end find_x_l1956_195611


namespace correct_operation_l1956_195649

variable (a b : ℝ)

theorem correct_operation :
  ¬ (a^2 + a^3 = a^5) ∧
  ¬ ((a^2)^3 = a^5) ∧
  ¬ (a^2 * a^3 = a^6) ∧
  ((-a * b)^5 / (-a * b)^3 = a^2 * b^2) :=
by
  sorry

end correct_operation_l1956_195649


namespace probability_of_head_l1956_195636

def events : Type := {e // e = "H" ∨ e = "T"}

def equallyLikely (e : events) : Prop :=
  e = ⟨"H", Or.inl rfl⟩ ∨ e = ⟨"T", Or.inr rfl⟩

def totalOutcomes := 2

def probOfHead : ℚ := 1 / totalOutcomes

theorem probability_of_head : probOfHead = 1 / 2 :=
by
  sorry

end probability_of_head_l1956_195636


namespace accommodation_ways_l1956_195675

-- Definition of the problem
def triple_room_count : ℕ := 1
def double_room_count : ℕ := 2
def adults_count : ℕ := 3
def children_count : ℕ := 2
def total_ways : ℕ := 60

-- Main statement to be proved
theorem accommodation_ways :
  (triple_room_count = 1) →
  (double_room_count = 2) →
  (adults_count = 3) →
  (children_count = 2) →
  -- Children must be accompanied by adults, and not all rooms need to be occupied.
  -- We are to prove that the number of valid ways to assign the rooms is 60
  total_ways = 60 :=
by sorry

end accommodation_ways_l1956_195675


namespace length_of_plot_l1956_195645

theorem length_of_plot (W P C r : ℝ) (hW : W = 65) (hP : P = 2.5) (hC : C = 340) (hr : r = 0.4) :
  let L := (C / r - (W + 2 * P) * P) / (W - 2 * P)
  L = 100 :=
by
  sorry

end length_of_plot_l1956_195645


namespace first_ring_time_l1956_195646

-- Define the properties of the clock
def rings_every_three_hours : Prop := ∀ n : ℕ, 3 * n < 24
def rings_eight_times_a_day : Prop := ∀ n : ℕ, n = 8 → 3 * n = 24

-- The theorem statement
theorem first_ring_time : rings_every_three_hours → rings_eight_times_a_day → (∀ n : ℕ, n = 1 → 3 * n = 3) := 
    sorry

end first_ring_time_l1956_195646


namespace john_shots_l1956_195621

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

end john_shots_l1956_195621


namespace factorization_correct_l1956_195692

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l1956_195692


namespace evaluate_expression_l1956_195657

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end evaluate_expression_l1956_195657


namespace peanuts_difference_is_correct_l1956_195672

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- Define the difference in the number of peanuts between Kenya and Jose
def peanuts_difference : ℕ := Kenya_peanuts - Jose_peanuts

-- Prove that the number of peanuts Kenya has minus the number of peanuts Jose has is equal to 48
theorem peanuts_difference_is_correct : peanuts_difference = 48 := by
  sorry

end peanuts_difference_is_correct_l1956_195672


namespace teacher_age_l1956_195669

theorem teacher_age {student_count : ℕ} (avg_age_students : ℕ) (avg_age_with_teacher : ℕ)
    (h1 : student_count = 25) (h2 : avg_age_students = 26) (h3 : avg_age_with_teacher = 27) :
    ∃ (teacher_age : ℕ), teacher_age = 52 :=
by
  sorry

end teacher_age_l1956_195669


namespace typing_time_in_hours_l1956_195620

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

end typing_time_in_hours_l1956_195620


namespace gary_money_left_l1956_195691

theorem gary_money_left (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 73)
  (h2 : spent_amount = 55)
  (h3 : remaining_amount = 18) : initial_amount - spent_amount = remaining_amount := 
by 
  sorry

end gary_money_left_l1956_195691


namespace square_area_l1956_195619

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

end square_area_l1956_195619


namespace arithmetic_sequence_sum_square_l1956_195624

theorem arithmetic_sequence_sum_square (a d : ℕ) :
  (∀ n : ℕ, ∃ k : ℕ, n * (a + (n-1) * d / 2) = k * k) ↔ (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) := 
by
  sorry

end arithmetic_sequence_sum_square_l1956_195624


namespace smallest_product_not_factor_of_48_exists_l1956_195698

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l1956_195698


namespace convert_base4_to_base10_l1956_195671

-- Define a function to convert a base 4 number to base 10
def base4_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

-- Assert the proof problem
theorem convert_base4_to_base10 : base4_to_base10 3201 = 225 :=
by
  -- The proof script goes here; for now, we use 'sorry' as a placeholder
  sorry

end convert_base4_to_base10_l1956_195671


namespace find_a_l1956_195659

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

end find_a_l1956_195659


namespace time_after_hours_l1956_195697

-- Definitions based on conditions
def current_time : ℕ := 3
def hours_later : ℕ := 2517
def clock_cycle : ℕ := 12

-- Statement to prove
theorem time_after_hours :
  (current_time + hours_later) % clock_cycle = 12 := 
sorry

end time_after_hours_l1956_195697


namespace find_x_l1956_195601

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end find_x_l1956_195601


namespace divisible_by_6_l1956_195685

theorem divisible_by_6 {n : ℕ} (h2 : 2 ∣ n) (h3 : 3 ∣ n) : 6 ∣ n :=
sorry

end divisible_by_6_l1956_195685


namespace weights_problem_l1956_195603

theorem weights_problem (n : ℕ) (x : ℝ) (h_avg : ∀ (i : ℕ), i < n → ∃ (w : ℝ), w = x) 
  (h_heaviest : ∃ (w_max : ℝ), w_max = 5 * x) : n > 5 :=
by
  sorry

end weights_problem_l1956_195603


namespace common_area_of_rectangle_and_circle_eqn_l1956_195615

theorem common_area_of_rectangle_and_circle_eqn :
  let rect_length := 8
  let rect_width := 4
  let circle_radius := 3
  let common_area := (3^2 * 2 * Real.pi / 4) - 2 * Real.sqrt 5  
  common_area = (9 * Real.pi / 2) - 2 * Real.sqrt 5 := 
sorry

end common_area_of_rectangle_and_circle_eqn_l1956_195615


namespace find_YZ_l1956_195661

noncomputable def triangle_YZ (angle_Y : ℝ) (XY : ℝ) (XZ : ℝ) : ℝ :=
  if angle_Y = 45 ∧ XY = 100 ∧ XZ = 50 * Real.sqrt 2 then
    50 * Real.sqrt 6
  else
    0

theorem find_YZ :
  triangle_YZ 45 100 (50 * Real.sqrt 2) = 50 * Real.sqrt 6 :=
by
  sorry

end find_YZ_l1956_195661


namespace mass_percentage_H3BO3_l1956_195681

theorem mass_percentage_H3BO3 :
  ∃ (element : String) (mass_percent : ℝ), 
    element ∈ ["H", "B", "O"] ∧ 
    mass_percent = 4.84 ∧ 
    mass_percent = 4.84 :=
sorry

end mass_percentage_H3BO3_l1956_195681


namespace n_is_one_sixth_sum_of_list_l1956_195609

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

end n_is_one_sixth_sum_of_list_l1956_195609


namespace positive_slope_asymptote_l1956_195689

def hyperbola (x y : ℝ) :=
  Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 6) ^ 2 + (y + 2) ^ 2) = 4

theorem positive_slope_asymptote :
  ∃ (m : ℝ), m = 0.75 ∧ (∃ x y, hyperbola x y) :=
sorry

end positive_slope_asymptote_l1956_195689


namespace no_odd_tens_digit_in_square_l1956_195647

theorem no_odd_tens_digit_in_square (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n > 0) (h₃ : n < 100) : 
  (n * n / 10) % 10 % 2 = 0 := 
sorry

end no_odd_tens_digit_in_square_l1956_195647


namespace positive_difference_between_two_numbers_l1956_195665

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l1956_195665


namespace exists_four_digit_number_divisible_by_101_l1956_195683

theorem exists_four_digit_number_divisible_by_101 :
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
    b ≠ c ∧ b ≠ d ∧
    c ≠ d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) % 101 = 0 := 
by
  -- To be proven
  sorry

end exists_four_digit_number_divisible_by_101_l1956_195683


namespace range_of_a_l1956_195643

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end range_of_a_l1956_195643


namespace four_thirds_of_nine_halves_l1956_195642

theorem four_thirds_of_nine_halves :
  (4 / 3) * (9 / 2) = 6 := 
sorry

end four_thirds_of_nine_halves_l1956_195642


namespace triangle_BC_length_l1956_195629

theorem triangle_BC_length (A B C X : Type) 
  (AB AC : ℕ) (BX CX BC : ℕ)
  (h1 : AB = 100)
  (h2 : AC = 121)
  (h3 : ∃ x y : ℕ, x = BX ∧ y = CX ∧ AB = 100 ∧ x + y = BC)
  (h4 : x * y = 31 * 149 ∧ x + y = 149) :
  BC = 149 := 
by
  sorry

end triangle_BC_length_l1956_195629


namespace geometric_to_arithmetic_common_ratio_greater_than_1_9_l1956_195608

theorem geometric_to_arithmetic (q : ℝ) (h : q = (1 + Real.sqrt 5) / 2) :
  ∃ (a b c : ℝ), b - a = c - b ∧ a / b = b / c := 
sorry

theorem common_ratio_greater_than_1_9 (q : ℝ) (h_pos : q > 1.9 ∧ q < 2) :
  ∃ (n : ℕ), q^(n+1) - 2 * q^n + 1 = 0 :=
sorry

end geometric_to_arithmetic_common_ratio_greater_than_1_9_l1956_195608


namespace standard_eq_minimal_circle_l1956_195695

-- Definitions
variables {x y : ℝ}
variables (h₀ : 0 < x) (h₁ : 0 < y)
variables (h₂ : 3 / (2 + x) + 3 / (2 + y) = 1)

-- Theorem statement
theorem standard_eq_minimal_circle : (x - 4)^2 + (y - 4)^2 = 16^2 :=
sorry

end standard_eq_minimal_circle_l1956_195695


namespace average_weight_14_children_l1956_195699

theorem average_weight_14_children 
  (average_weight_boys : ℕ → ℤ → ℤ)
  (average_weight_girls : ℕ → ℤ → ℤ)
  (total_children : ℕ)
  (total_weight : ℤ)
  (total_average_weight : ℤ)
  (boys_count : ℕ)
  (girls_count : ℕ)
  (boys_average : ℤ)
  (girls_average : ℤ) :
  boys_count = 8 →
  girls_count = 6 →
  boys_average = 160 →
  girls_average = 130 →
  total_children = boys_count + girls_count →
  total_weight = average_weight_boys boys_count boys_average + average_weight_girls girls_count girls_average →
  average_weight_boys boys_count boys_average = boys_count * boys_average →
  average_weight_girls girls_count girls_average = girls_count * girls_average →
  total_average_weight = total_weight / total_children →
  total_average_weight = 147 :=
by
  sorry

end average_weight_14_children_l1956_195699


namespace max_value_abs_cube_sum_l1956_195613

theorem max_value_abs_cube_sum (x : Fin 5 → ℝ) (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (|x 0 - x 1|^3 + |x 1 - x 2|^3 + |x 2 - x 3|^3 + |x 3 - x 4|^3 + |x 4 - x 0|^3) ≤ 4 :=
sorry

end max_value_abs_cube_sum_l1956_195613


namespace cos_alpha_value_l1956_195686

variable (α : ℝ)
variable (x y r : ℝ)

-- Conditions
def point_condition : Prop := (x = 1 ∧ y = -Real.sqrt 3 ∧ r = 2 ∧ r = Real.sqrt (x^2 + y^2))

-- Question/Proof Statement
theorem cos_alpha_value (h : point_condition x y r) : Real.cos α = 1 / 2 :=
sorry

end cos_alpha_value_l1956_195686


namespace solve_equation1_solve_equation2_l1956_195664

open Real

theorem solve_equation1 (x : ℝ) : (x - 2)^2 = 9 → (x = 5 ∨ x = -1) :=
by
  intro h
  sorry -- Proof would go here

theorem solve_equation2 (x : ℝ) : (2 * x^2 - 3 * x - 1 = 0) → (x = (3 + sqrt 17) / 4 ∨ x = (3 - sqrt 17) / 4) :=
by
  intro h
  sorry -- Proof would go here

end solve_equation1_solve_equation2_l1956_195664


namespace turquoise_beads_count_l1956_195687

-- Define the conditions
def num_beads_total : ℕ := 40
def num_amethyst : ℕ := 7
def num_amber : ℕ := 2 * num_amethyst

-- Define the main theorem to prove
theorem turquoise_beads_count :
  num_beads_total - (num_amethyst + num_amber) = 19 :=
by
  sorry

end turquoise_beads_count_l1956_195687


namespace onewaynia_road_closure_l1956_195633

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

end onewaynia_road_closure_l1956_195633


namespace find_d_l1956_195648

theorem find_d (d : ℚ) (int_part frac_part : ℚ) 
  (h1 : 3 * int_part^2 + 19 * int_part - 28 = 0)
  (h2 : 4 * frac_part^2 - 11 * frac_part + 3 = 0)
  (h3 : frac_part ≥ 0 ∧ frac_part < 1)
  (h4 : d = int_part + frac_part) :
  d = -29 / 4 :=
by
  sorry

end find_d_l1956_195648


namespace part1_part2_l1956_195635

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x + a - 1) + abs (x - 2 * a)

-- Part (1) of the proof problem
theorem part1 (a : ℝ) : f 1 a < 3 → - (2 : ℝ)/3 < a ∧ a < 4 / 3 := sorry

-- Part (2) of the proof problem
theorem part2 (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := sorry

end part1_part2_l1956_195635


namespace max_value_expression_l1956_195606

theorem max_value_expression : ∃ (max_val : ℝ), max_val = (1 / 16) ∧ ∀ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → (a - b^2) * (b - a^2) ≤ max_val :=
by
  sorry

end max_value_expression_l1956_195606


namespace product_is_correct_l1956_195673

theorem product_is_correct :
  50 * 29.96 * 2.996 * 500 = 2244004 :=
by
  sorry

end product_is_correct_l1956_195673


namespace max_pies_without_ingredients_l1956_195658

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

end max_pies_without_ingredients_l1956_195658


namespace h_at_3_eq_3_l1956_195688

-- Define the function h(x) based on the given condition
noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * 
    (x^32 + 1) * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) - 1) / 
  (x^(2^10 - 1) - 1)

-- State the required theorem
theorem h_at_3_eq_3 : h 3 = 3 := by
  sorry

end h_at_3_eq_3_l1956_195688


namespace find_value_of_c_l1956_195602

variable (a b c : ℚ)
variable (x : ℚ)

-- Conditions converted to Lean statements
def condition1 := a = 2 * x ∧ b = 3 * x ∧ c = 7 * x
def condition2 := a - b + 3 = c - 2 * b

theorem find_value_of_c : condition1 x a b c ∧ condition2 a b c → c = 21 / 2 :=
by 
  sorry

end find_value_of_c_l1956_195602


namespace base_s_computation_l1956_195622

theorem base_s_computation (s : ℕ) (h : 550 * s + 420 * s = 1100 * s) : s = 7 := by
  sorry

end base_s_computation_l1956_195622


namespace age_problem_l1956_195682

theorem age_problem
    (D X : ℕ) 
    (h1 : D = 4 * X) 
    (h2 : D = X + 30) : D = 40 ∧ X = 10 := by
  sorry

end age_problem_l1956_195682


namespace markup_is_correct_l1956_195663

noncomputable def profit (S : ℝ) : ℝ := 0.12 * S
noncomputable def expenses (S : ℝ) : ℝ := 0.10 * S
noncomputable def cost (S : ℝ) : ℝ := S - (profit S + expenses S)
noncomputable def markup (S : ℝ) : ℝ :=
  ((S - cost S) / (cost S)) * 100

theorem markup_is_correct:
  markup 10 = 28.21 :=
by
  sorry

end markup_is_correct_l1956_195663


namespace lorelai_jellybeans_correct_l1956_195634

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

end lorelai_jellybeans_correct_l1956_195634


namespace value_of_clothing_piece_eq_l1956_195627

def annual_remuneration := 10
def work_months := 7
def received_silver_coins := 2

theorem value_of_clothing_piece_eq : 
  ∃ x : ℝ, (x + received_silver_coins) * 12 = (x + annual_remuneration) * work_months → x = 9.2 :=
by
  sorry

end value_of_clothing_piece_eq_l1956_195627


namespace car_transport_distance_l1956_195651

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

end car_transport_distance_l1956_195651


namespace rr_sr_sum_le_one_l1956_195652

noncomputable def rr_sr_le_one (r s : ℝ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : Prop :=
  r^r * s^s + r^s * s^r ≤ 1

theorem rr_sr_sum_le_one {r s : ℝ} (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : rr_sr_le_one r s h_pos_r h_pos_s h_sum :=
  sorry

end rr_sr_sum_le_one_l1956_195652


namespace coins_in_distinct_colors_l1956_195617

theorem coins_in_distinct_colors 
  (n : ℕ)  (h1 : 1 < n) (h2 : n < 2010) : (∃ k : ℕ, 2010 = n * k) ↔ 
  ∀ i : ℕ, i < 2010 → (∃ f : ℕ → ℕ, ∀ j : ℕ, j < n → f (j + i) % n = j % n) :=
sorry

end coins_in_distinct_colors_l1956_195617


namespace select_from_companyA_l1956_195618

noncomputable def companyA_representatives : ℕ := 40
noncomputable def companyB_representatives : ℕ := 60
noncomputable def total_representatives : ℕ := companyA_representatives + companyB_representatives
noncomputable def sample_size : ℕ := 10
noncomputable def sampling_ratio : ℚ := sample_size / total_representatives
noncomputable def selected_from_companyA : ℚ := companyA_representatives * sampling_ratio

theorem select_from_companyA : selected_from_companyA = 4 := by
  sorry


end select_from_companyA_l1956_195618


namespace median_of_first_ten_positive_integers_l1956_195641

def first_ten_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem median_of_first_ten_positive_integers : 
  ∃ median : ℝ, median = 5.5 := by
  sorry

end median_of_first_ten_positive_integers_l1956_195641


namespace airline_passenger_capacity_l1956_195684

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end airline_passenger_capacity_l1956_195684


namespace jane_received_change_l1956_195644

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l1956_195644


namespace remainder_3005_98_l1956_195616

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end remainder_3005_98_l1956_195616


namespace bella_truck_stamps_more_l1956_195614

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end bella_truck_stamps_more_l1956_195614


namespace expenditure_proof_l1956_195696

namespace OreoCookieProblem

variables (O C : ℕ) (CO CC : ℕ → ℕ) (total_items cost_difference : ℤ)

def oreo_count_eq : Prop := O = (4 * (65 : ℤ) / 13)
def cookie_count_eq : Prop := C = (9 * (65 : ℤ) / 13)
def oreo_cost (o : ℕ) : ℕ := o * 2
def cookie_cost (c : ℕ) : ℕ := c * 3
def total_item_condition : Prop := O + C = 65
def ratio_condition : Prop := 9 * O = 4 * C
def cost_difference_condition (o_cost c_cost : ℕ) : Prop := cost_difference = (c_cost - o_cost)

theorem expenditure_proof :
  (O + C = 65) →
  (9 * O = 4 * C) →
  (O = 20) →
  (C = 45) →
  cost_difference = (45 * 3 - 20 * 2) →
  cost_difference = 95 :=
by sorry

end OreoCookieProblem

end expenditure_proof_l1956_195696


namespace area_of_roof_l1956_195600

def roof_area (w l : ℕ) : ℕ := l * w

theorem area_of_roof :
  ∃ (w l : ℕ), l = 4 * w ∧ l - w = 45 ∧ roof_area w l = 900 :=
by
  -- Defining witnesses for width and length
  use 15, 60
  -- Splitting the goals for clarity
  apply And.intro
  -- Proving the first condition: l = 4 * w
  · show 60 = 4 * 15
    rfl
  apply And.intro
  -- Proving the second condition: l - w = 45
  · show 60 - 15 = 45
    rfl
  -- Proving the area calculation: roof_area w l = 900
  · show roof_area 15 60 = 900
    rfl

end area_of_roof_l1956_195600


namespace x_coordinate_incenter_eq_l1956_195662

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end x_coordinate_incenter_eq_l1956_195662


namespace sonnets_not_read_l1956_195610

-- Define the conditions in the original problem
def sonnet_lines := 14
def unheard_lines := 70

-- Define a statement that needs to be proven
-- Prove that the number of sonnets not read is 5
theorem sonnets_not_read : unheard_lines / sonnet_lines = 5 := by
  sorry

end sonnets_not_read_l1956_195610


namespace multiply_binomials_l1956_195653

theorem multiply_binomials :
  ∀ (x : ℝ), 
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 :=
by
  sorry

end multiply_binomials_l1956_195653


namespace dubblefud_red_balls_l1956_195625

theorem dubblefud_red_balls (R B : ℕ) 
  (h1 : 2 ^ R * 4 ^ B * 5 ^ B = 16000)
  (h2 : B = G) : R = 6 :=
by
  -- Skipping the actual proof
  sorry

end dubblefud_red_balls_l1956_195625


namespace average_marks_l1956_195679

variable (P C M : ℕ)

theorem average_marks :
  P = 140 →
  (P + M) / 2 = 90 →
  (P + C) / 2 = 70 →
  (P + C + M) / 3 = 60 :=
by
  intros hP hM hC
  sorry

end average_marks_l1956_195679


namespace sixth_power_sum_l1956_195605

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

end sixth_power_sum_l1956_195605


namespace arithmetic_sequence_a1_a7_a3_a5_l1956_195623

noncomputable def arithmetic_sequence_property (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_a1_a7_a3_a5 (a : ℕ → ℝ) (h_arith : arithmetic_sequence_property a)
  (h_cond : a 1 + a 7 = 10) : a 3 + a 5 = 10 :=
by
  sorry

end arithmetic_sequence_a1_a7_a3_a5_l1956_195623


namespace sum_of_integers_with_product_neg13_l1956_195670

theorem sum_of_integers_with_product_neg13 (a b c : ℤ) (h : a * b * c = -13) : 
  a + b + c = 13 ∨ a + b + c = -11 := 
sorry

end sum_of_integers_with_product_neg13_l1956_195670


namespace range_of_m_l1956_195654

noncomputable def inequality_solutions (x m : ℝ) := |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) : (∃ x : ℝ, inequality_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l1956_195654


namespace carol_rectangle_length_l1956_195640

theorem carol_rectangle_length :
  let j_length := 6
  let j_width := 30
  let c_width := 15
  let c_length := j_length * j_width / c_width
  c_length = 12 := by
  sorry

end carol_rectangle_length_l1956_195640


namespace geometric_sequence_seventh_term_l1956_195677

theorem geometric_sequence_seventh_term (r : ℕ) (r_pos : 0 < r) 
  (h1 : 3 * r^4 = 243) : 
  3 * r^6 = 2187 :=
by
  sorry

end geometric_sequence_seventh_term_l1956_195677


namespace count_oddly_powerful_integers_l1956_195676

def is_oddly_powerful (m : ℕ) : Prop :=
  ∃ (c d : ℕ), d > 1 ∧ d % 2 = 1 ∧ c^d = m

theorem count_oddly_powerful_integers :
  ∃ (S : Finset ℕ), 
  (∀ m, m ∈ S ↔ (m < 1500 ∧ is_oddly_powerful m)) ∧ S.card = 13 :=
by
  sorry

end count_oddly_powerful_integers_l1956_195676


namespace chess_tournament_l1956_195626

theorem chess_tournament (n games : ℕ) 
  (h_games : games = 81)
  (h_equation : (n - 2) * (n - 3) = 156) :
  n = 15 :=
sorry

end chess_tournament_l1956_195626


namespace geom_sequence_a1_l1956_195656

noncomputable def a_n (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n-1)

theorem geom_sequence_a1 {a1 q : ℝ} 
  (h1 : 0 < q)
  (h2 : a_n a1 q 4 * a_n a1 q 8 = 2 * (a_n a1 q 5)^2)
  (h3 : a_n a1 q 2 = 1) :
  a1 = (Real.sqrt 2) / 2 :=
sorry

end geom_sequence_a1_l1956_195656


namespace revision_cost_per_page_is_4_l1956_195639

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

end revision_cost_per_page_is_4_l1956_195639


namespace quarters_in_school_year_l1956_195668

variable (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ)

def number_of_quarters (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ) : ℕ :=
  (total_artworks / (students * artworks_per_student_per_quarter * school_years))

theorem quarters_in_school_year :
  number_of_quarters 15 2 240 2 = 4 :=
by sorry

end quarters_in_school_year_l1956_195668


namespace base_unit_digit_l1956_195693

def unit_digit (n : ℕ) : ℕ := n % 10

theorem base_unit_digit (x : ℕ) :
  unit_digit ((x^41) * (41^14) * (14^87) * (87^76)) = 4 →
  unit_digit x = 1 :=
by
  sorry

end base_unit_digit_l1956_195693


namespace solve_B_share_l1956_195607

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

end solve_B_share_l1956_195607


namespace simplify_expression_l1956_195655

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l1956_195655


namespace time_for_six_visits_l1956_195638

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end time_for_six_visits_l1956_195638


namespace badge_counts_l1956_195628

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

end badge_counts_l1956_195628
