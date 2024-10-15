import Mathlib

namespace NUMINAMATH_CALUDE_residue_mod_17_l3677_367751

theorem residue_mod_17 : (207 * 13 - 22 * 8 + 5) % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l3677_367751


namespace NUMINAMATH_CALUDE_anthony_lunch_money_l3677_367771

theorem anthony_lunch_money (initial_money juice_cost cupcake_cost : ℕ) 
  (h1 : initial_money = 75)
  (h2 : juice_cost = 27)
  (h3 : cupcake_cost = 40) :
  initial_money - (juice_cost + cupcake_cost) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_anthony_lunch_money_l3677_367771


namespace NUMINAMATH_CALUDE_billboard_problem_l3677_367795

/-- The number of billboards to be erected -/
def num_billboards : ℕ := 200

/-- The length of the road in meters -/
def road_length : ℕ := 1100

/-- The spacing between billboards in the first scenario (in meters) -/
def spacing1 : ℚ := 5

/-- The spacing between billboards in the second scenario (in meters) -/
def spacing2 : ℚ := 11/2

/-- The number of missing billboards in the first scenario -/
def missing1 : ℕ := 21

/-- The number of missing billboards in the second scenario -/
def missing2 : ℕ := 1

theorem billboard_problem :
  (spacing1 * (num_billboards + missing1 - 1 : ℚ) = road_length) ∧
  (spacing2 * (num_billboards + missing2 - 1 : ℚ) = road_length) := by
  sorry

end NUMINAMATH_CALUDE_billboard_problem_l3677_367795


namespace NUMINAMATH_CALUDE_smallest_fraction_above_four_fifths_l3677_367775

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_above_four_fifths :
  ∀ (a b : ℕ), is_two_digit a → is_two_digit b → (a : ℚ) / b > 4 / 5 → Nat.gcd a b = 1 →
  (77 : ℚ) / 96 ≤ (a : ℚ) / b :=
sorry

end NUMINAMATH_CALUDE_smallest_fraction_above_four_fifths_l3677_367775


namespace NUMINAMATH_CALUDE_smallest_club_size_club_size_exists_l3677_367738

theorem smallest_club_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 3) → n ≥ 343 :=
by sorry

theorem club_size_exists : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 3) ∧ n = 343 :=
by sorry

end NUMINAMATH_CALUDE_smallest_club_size_club_size_exists_l3677_367738


namespace NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l3677_367783

structure School where
  male_count : Nat
  female_count : Nat

def total_teachers (s : School) : Nat :=
  s.male_count + s.female_count

def school_A : School :=
  { male_count := 2, female_count := 1 }

def school_B : School :=
  { male_count := 1, female_count := 2 }

def total_schools : Nat := 2

def total_all_teachers : Nat :=
  total_teachers school_A + total_teachers school_B

theorem same_gender_probability :
  (school_A.male_count * school_B.male_count + school_A.female_count * school_B.female_count) /
  (total_teachers school_A * total_teachers school_B) = 4 / 9 := by
  sorry

theorem same_school_probability :
  (Nat.choose (total_teachers school_A) 2 + Nat.choose (total_teachers school_B) 2) /
  Nat.choose total_all_teachers 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l3677_367783


namespace NUMINAMATH_CALUDE_equation_solution_l3677_367744

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/3 + 1/x + 1/(x^2)
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 33) / 4 ∧ 
              x₂ = (3 - Real.sqrt 33) / 4 ∧ 
              f x₁ = 1 ∧ 
              f x₂ = 1 ∧ 
              ∀ x : ℝ, f x = 1 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3677_367744


namespace NUMINAMATH_CALUDE_amy_initial_amount_l3677_367702

/-- The amount of money Amy had when she got to the fair -/
def initial_amount : ℕ := sorry

/-- The amount of money Amy had when she left the fair -/
def final_amount : ℕ := 11

/-- The amount of money Amy spent at the fair -/
def spent_amount : ℕ := 4

/-- Theorem: Amy had $15 when she got to the fair -/
theorem amy_initial_amount : initial_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_amy_initial_amount_l3677_367702


namespace NUMINAMATH_CALUDE_total_heads_count_l3677_367707

/-- Proves that the total number of heads is 48 given the conditions of the problem -/
theorem total_heads_count (hens cows : ℕ) : 
  hens = 28 →
  2 * hens + 4 * cows = 136 →
  hens + cows = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_count_l3677_367707


namespace NUMINAMATH_CALUDE_cosine_amplitude_l3677_367716

theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cos (b * x) ≤ 3) ∧ (∃ x, a * Real.cos (b * x) = 3) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l3677_367716


namespace NUMINAMATH_CALUDE_power_equality_l3677_367728

theorem power_equality (q : ℕ) : 16^10 = 4^q → q = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3677_367728


namespace NUMINAMATH_CALUDE_fraction_product_square_l3677_367708

theorem fraction_product_square : (8 / 9) ^ 2 * (1 / 3) ^ 2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_square_l3677_367708


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3677_367700

theorem no_infinite_prime_sequence : 
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧ 
    (∀ n, p n < p (n + 1)) ∧
    (∀ k, p (k + 1) = 2 * p k + 1 ∨ p (k + 1) = 2 * p k - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3677_367700


namespace NUMINAMATH_CALUDE_modified_sum_theorem_l3677_367794

theorem modified_sum_theorem (S a b : ℝ) (h : a + b = S) :
  (3 * a + 4) + (2 * b + 5) = 3 * S + 9 := by
  sorry

end NUMINAMATH_CALUDE_modified_sum_theorem_l3677_367794


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_ratio_l3677_367760

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem folded_rectangle_perimeter_ratio :
  let original := Rectangle.mk 8 4
  let folded := Rectangle.mk 4 2
  (perimeter folded) / (perimeter original) = 1/2 := by sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_ratio_l3677_367760


namespace NUMINAMATH_CALUDE_units_digit_characteristic_l3677_367773

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate to check if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_characteristic (p : ℕ) 
  (h1 : p > 0) 
  (h2 : isEven p) 
  (h3 : unitsDigit (p^3) - unitsDigit (p^2) = 0)
  (h4 : unitsDigit (p + 4) = 0) : 
  unitsDigit p = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_characteristic_l3677_367773


namespace NUMINAMATH_CALUDE_matches_for_128_teams_l3677_367721

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  num_teams : ℕ
  num_teams_positive : 0 < num_teams

/-- Calculates the number of matches required to determine a champion. -/
def matches_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.num_teams - 1

/-- Theorem: In a single-elimination tournament with 128 teams, 127 matches are required. -/
theorem matches_for_128_teams :
  let tournament := SingleEliminationTournament.mk 128 (by norm_num)
  matches_required tournament = 127 := by
  sorry

end NUMINAMATH_CALUDE_matches_for_128_teams_l3677_367721


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3677_367799

-- Define the edge lengths in inches
def edge_length_1 : ℚ := 9
def edge_length_2 : ℚ := 3 * 12

-- Define the volume ratio function
def volume_ratio (a b : ℚ) : ℚ := (a / b) ^ 3

-- Theorem statement
theorem cube_volume_ratio : volume_ratio edge_length_1 edge_length_2 = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3677_367799


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3677_367705

theorem imaginary_part_of_z (z : ℂ) : z = -2 * Complex.I * (-1 + Real.sqrt 3 * Complex.I) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3677_367705


namespace NUMINAMATH_CALUDE_sum_a_b_equals_21_over_8_l3677_367755

/-- Operation ⊕ defined for real numbers -/
def circle_plus (x y : ℝ) : ℝ := x + 2*y + 3

/-- Theorem stating the result of a + b given the conditions -/
theorem sum_a_b_equals_21_over_8 (a b : ℝ) 
  (h : (circle_plus (circle_plus (a^3) (a^2)) a) = (circle_plus (a^3) (circle_plus (a^2) a)) ∧ 
       (circle_plus (circle_plus (a^3) (a^2)) a) = b) : 
  a + b = 21/8 := by sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_21_over_8_l3677_367755


namespace NUMINAMATH_CALUDE_marker_selection_combinations_l3677_367765

theorem marker_selection_combinations : ∀ n r : ℕ, 
  n = 15 → r = 5 → (n.choose r) = 3003 := by
  sorry

end NUMINAMATH_CALUDE_marker_selection_combinations_l3677_367765


namespace NUMINAMATH_CALUDE_pet_shop_legs_l3677_367754

/-- The number of legs for each animal type --/
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def snake_legs : ℕ := 0
def spider_legs : ℕ := 8

/-- The number of each animal type --/
def num_birds : ℕ := 3
def num_dogs : ℕ := 5
def num_snakes : ℕ := 4
def num_spiders : ℕ := 1

/-- The total number of legs in the pet shop --/
def total_legs : ℕ := 
  num_birds * bird_legs + 
  num_dogs * dog_legs + 
  num_snakes * snake_legs + 
  num_spiders * spider_legs

theorem pet_shop_legs : total_legs = 34 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_legs_l3677_367754


namespace NUMINAMATH_CALUDE_distance_city_AC_l3677_367742

/-- The distance between two cities given travel times and speeds -/
theorem distance_city_AC (time_eddy time_freddy : ℝ) (distance_AB : ℝ) (speed_ratio : ℝ) 
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_AB = 480)
  (h4 : speed_ratio = 2.1333333333333333)
  (h5 : speed_ratio = (distance_AB / time_eddy) / ((distance_AB / time_eddy) / speed_ratio)) :
  (distance_AB / time_eddy) / speed_ratio * time_freddy = 300 := by
  sorry

#eval (480 / 3) / 2.1333333333333333 * 4

end NUMINAMATH_CALUDE_distance_city_AC_l3677_367742


namespace NUMINAMATH_CALUDE_shelves_needed_l3677_367782

theorem shelves_needed (initial_books : ℝ) (added_books : ℝ) (books_per_shelf : ℝ) :
  initial_books = 46.0 →
  added_books = 10.0 →
  books_per_shelf = 4.0 →
  ((initial_books + added_books) / books_per_shelf) = 14.0 := by
  sorry

end NUMINAMATH_CALUDE_shelves_needed_l3677_367782


namespace NUMINAMATH_CALUDE_changsha_tourism_l3677_367732

/-- The number of visitors (in millions) to Changsha during May Day holiday in 2021 -/
def visitors_2021 : ℝ := 2

/-- The number of visitors (in millions) to Changsha during May Day holiday in 2023 -/
def visitors_2023 : ℝ := 2.88

/-- The amount spent on Youlan Latte -/
def spent_youlan : ℝ := 216

/-- The amount spent on Shengsheng Oolong -/
def spent_oolong : ℝ := 96

/-- The price difference between Youlan Latte and Shengsheng Oolong -/
def price_difference : ℝ := 2

theorem changsha_tourism (r x : ℝ) : 
  ((1 + r)^2 = visitors_2023 / visitors_2021) ∧ 
  (spent_youlan / x = 2 * spent_oolong / (x - price_difference)) → 
  (r = 0.2 ∧ x = 18) := by sorry

end NUMINAMATH_CALUDE_changsha_tourism_l3677_367732


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_13_l3677_367740

def numbers : List Nat := [45, 63, 98, 121, 169]

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def prime_factors (n : Nat) : Set Nat :=
  {p : Nat | is_prime p ∧ n % p = 0}

theorem largest_prime_factor_is_13 :
  ∃ (n : Nat), n ∈ numbers ∧ 13 ∈ prime_factors n ∧
  ∀ (m : Nat), m ∈ numbers → ∀ (p : Nat), p ∈ prime_factors m → p ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_is_13_l3677_367740


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l3677_367767

theorem largest_number_divisible_by_88_has_4_digits :
  let n : ℕ := 9944
  (∀ m : ℕ, m > n → m % 88 ≠ 0 ∨ (String.length (toString m) > String.length (toString n))) →
  n % 88 = 0 →
  String.length (toString n) = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l3677_367767


namespace NUMINAMATH_CALUDE_product_a_b_equals_27_over_8_l3677_367796

theorem product_a_b_equals_27_over_8 
  (a b c : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : c = 3 → a = b^2) 
  (h3 : b + c = 2*a) 
  (h4 : c = 3) 
  (h5 : b + c = b * c) : 
  a * b = 27/8 := by
sorry

end NUMINAMATH_CALUDE_product_a_b_equals_27_over_8_l3677_367796


namespace NUMINAMATH_CALUDE_final_color_is_yellow_l3677_367746

/-- Represents the color of an elf -/
inductive ElfColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of elves on the island -/
structure ElfState where
  blue : Nat
  red : Nat
  yellow : Nat
  total : Nat
  h_total : blue + red + yellow = total

/-- The score assigned to each color -/
def colorScore (c : ElfColor) : Nat :=
  match c with
  | ElfColor.Blue => 1
  | ElfColor.Red => 2
  | ElfColor.Yellow => 3

/-- The total score of all elves -/
def totalScore (state : ElfState) : Nat :=
  state.blue * colorScore ElfColor.Blue +
  state.red * colorScore ElfColor.Red +
  state.yellow * colorScore ElfColor.Yellow

/-- Theorem: The final color of all elves is yellow -/
theorem final_color_is_yellow (initial_state : ElfState)
  (h_initial : initial_state.blue = 7 ∧ initial_state.red = 10 ∧ initial_state.yellow = 17 ∧ initial_state.total = 34)
  (h_change : ∀ (state : ElfState), totalScore state % 3 = totalScore initial_state % 3)
  (h_final : ∃ (final_state : ElfState), (final_state.blue = final_state.total ∨ final_state.red = final_state.total ∨ final_state.yellow = final_state.total) ∧
              totalScore final_state % 3 = totalScore initial_state % 3) :
  ∃ (final_state : ElfState), final_state.yellow = final_state.total :=
sorry

end NUMINAMATH_CALUDE_final_color_is_yellow_l3677_367746


namespace NUMINAMATH_CALUDE_student_count_proof_l3677_367768

theorem student_count_proof (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 := by
sorry

end NUMINAMATH_CALUDE_student_count_proof_l3677_367768


namespace NUMINAMATH_CALUDE_cassidy_grounding_l3677_367785

/-- Calculates the number of extra days grounded per grade below B -/
def extraDaysPerGrade (totalDays : ℕ) (baseDays : ℕ) (gradesBelowB : ℕ) : ℕ :=
  if gradesBelowB = 0 then 0 else (totalDays - baseDays) / gradesBelowB

theorem cassidy_grounding (totalDays : ℕ) (baseDays : ℕ) (gradesBelowB : ℕ) 
  (h1 : totalDays = 26)
  (h2 : baseDays = 14)
  (h3 : gradesBelowB = 4) :
  extraDaysPerGrade totalDays baseDays gradesBelowB = 3 := by
  sorry

#eval extraDaysPerGrade 26 14 4

end NUMINAMATH_CALUDE_cassidy_grounding_l3677_367785


namespace NUMINAMATH_CALUDE_solution_product_l3677_367772

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27

-- State the theorem
theorem solution_product (a b : ℝ) : 
  a ≠ b ∧ equation a ∧ equation b → (a + 2) * (b + 2) = -30 := by
  sorry

end NUMINAMATH_CALUDE_solution_product_l3677_367772


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_result_l3677_367717

/-- Given a geometric sequence {a_n} with the specified conditions, 
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tan_result (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 2 * a 3 * a 4 = -a 7^2 →                        -- given condition
  a 7^2 = 64 →                                      -- given condition
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_result_l3677_367717


namespace NUMINAMATH_CALUDE_f_three_pow_ge_f_two_pow_l3677_367724

/-- A quadratic function f(x) = ax^2 + bx + c with a > 0 and symmetric about x = 1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) ≥ f(2^x) for all x ∈ ℝ -/
theorem f_three_pow_ge_f_two_pow (a b c : ℝ) (h_a : a > 0) 
  (h_sym : ∀ x, f a b c (1 - x) = f a b c (1 + x)) :
  ∀ x, f a b c (3^x) ≥ f a b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_f_three_pow_ge_f_two_pow_l3677_367724


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l3677_367733

theorem quadratic_inequality_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l3677_367733


namespace NUMINAMATH_CALUDE_perpendicular_vectors_result_l3677_367710

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (4, m)

theorem perpendicular_vectors_result (m : ℝ) 
  (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) : 
  (5 : ℝ) • a - (3 : ℝ) • (b m) = (-7, -16) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_result_l3677_367710


namespace NUMINAMATH_CALUDE_problem_solution_l3677_367737

theorem problem_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  Real.log b / Real.log a = 3 → b - a = 1000 → a + b = 1010 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3677_367737


namespace NUMINAMATH_CALUDE_hostel_expenditure_l3677_367757

/-- Calculates the new total expenditure of a hostel after accommodating additional students --/
def new_total_expenditure (initial_students : ℕ) (additional_students : ℕ) (average_decrease : ℕ) (total_increase : ℕ) : ℕ :=
  let new_students := initial_students + additional_students
  let original_average := (total_increase + new_students * average_decrease) / (new_students - initial_students)
  new_students * (original_average - average_decrease)

/-- Theorem stating that the new total expenditure is 5400 rupees --/
theorem hostel_expenditure :
  new_total_expenditure 100 20 5 400 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_l3677_367757


namespace NUMINAMATH_CALUDE_power_sum_value_l3677_367786

theorem power_sum_value (a : ℝ) (x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(x+y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l3677_367786


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l3677_367774

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n > 0 ∧ 
    is_perfect_square n ∧
    is_divisible_by n 2 ∧
    is_divisible_by n 3 ∧
    is_divisible_by n 5 ∧
    (∀ m : ℕ, m > 0 ∧ 
      is_perfect_square m ∧
      is_divisible_by m 2 ∧
      is_divisible_by m 3 ∧
      is_divisible_by m 5 →
      n ≤ m) ∧
    n = 900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l3677_367774


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l3677_367706

/-- The equation of a line passing through two points -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ x y : ℝ) : 
  (x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁) ↔ 
  (x₁ = x₂ ∧ y₁ = y₂) ∨ 
  (∃ (t : ℝ), x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l3677_367706


namespace NUMINAMATH_CALUDE_complete_square_formula_not_complete_square_A_not_complete_square_B_not_complete_square_C_l3677_367703

theorem complete_square_formula (a b : ℝ) : 
  (a - b) * (b - a) = -(a - b)^2 :=
sorry

theorem not_complete_square_A (a b : ℝ) :
  (a - b) * (a + b) = a^2 - b^2 :=
sorry

theorem not_complete_square_B (a b : ℝ) :
  -(a + b) * (b - a) = a^2 - b^2 :=
sorry

theorem not_complete_square_C (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
sorry

end NUMINAMATH_CALUDE_complete_square_formula_not_complete_square_A_not_complete_square_B_not_complete_square_C_l3677_367703


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3677_367789

def a (n : ℕ) : ℤ := 3 * n - 5

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, a (n + 1) - a n = 3) ∧
  (a 1 = -2) ∧
  (∀ n : ℕ, a (n + 1) - a n = 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3677_367789


namespace NUMINAMATH_CALUDE_f_positive_iff_l3677_367739

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff (x : ℝ) : f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_l3677_367739


namespace NUMINAMATH_CALUDE_third_fraction_is_two_ninths_l3677_367790

-- Define a fraction type
structure Fraction where
  numerator : ℤ
  denominator : ℕ
  denominator_nonzero : denominator ≠ 0

-- Define the HCF function for fractions
def hcf_fractions (f1 f2 f3 : Fraction) : ℚ :=
  sorry

-- Theorem statement
theorem third_fraction_is_two_ninths
  (f1 : Fraction)
  (f2 : Fraction)
  (f3 : Fraction)
  (h1 : f1 = ⟨2, 3, sorry⟩)
  (h2 : f2 = ⟨4, 9, sorry⟩)
  (h3 : hcf_fractions f1 f2 f3 = 1 / 9) :
  f3 = ⟨2, 9, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_third_fraction_is_two_ninths_l3677_367790


namespace NUMINAMATH_CALUDE_sqrt_5_simplest_l3677_367720

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℚ), y = a / b ∧ b ≠ 1

theorem sqrt_5_simplest :
  is_simplest_sqrt (Real.sqrt 5) ∧
  ¬is_simplest_sqrt (Real.sqrt 2.5) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/3)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_simplest_l3677_367720


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3677_367736

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | x < -3 ∨ x > 4}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, f a b c x > 0 ↔ x ∈ solution_set a b c) :
  a > 0 ∧
  (∀ x, c * x^2 - b * x + a < 0 ↔ x < -1/4 ∨ x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3677_367736


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l3677_367712

/-- The number of players in the basketball team -/
def total_players : ℕ := 18

/-- The number of players in a lineup excluding the point guard -/
def lineup_size : ℕ := 7

/-- The number of different lineups that can be chosen -/
def number_of_lineups : ℕ := total_players * (Nat.choose (total_players - 1) lineup_size)

/-- Theorem stating the number of different lineups -/
theorem basketball_lineup_count : number_of_lineups = 349464 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l3677_367712


namespace NUMINAMATH_CALUDE_blocks_differing_in_three_ways_l3677_367734

/-- Represents the number of options for each attribute of a block -/
structure BlockOptions :=
  (materials : Nat)
  (sizes : Nat)
  (colors : Nat)
  (shapes : Nat)

/-- Calculates the number of blocks that differ in exactly k ways from a specific block -/
def countDifferingBlocks (options : BlockOptions) (k : Nat) : Nat :=
  sorry

/-- The specific block options for our problem -/
def ourBlockOptions : BlockOptions :=
  { materials := 2, sizes := 4, colors := 4, shapes := 4 }

/-- The main theorem: 45 blocks differ in exactly 3 ways from a specific block -/
theorem blocks_differing_in_three_ways :
  countDifferingBlocks ourBlockOptions 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_blocks_differing_in_three_ways_l3677_367734


namespace NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l3677_367753

theorem negative_third_greater_than_negative_half : -1/3 > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l3677_367753


namespace NUMINAMATH_CALUDE_trajectory_and_equilateral_triangle_l3677_367748

-- Define the points
def H : ℝ × ℝ := (-3, 0)
def T : ℝ × ℝ := (-1, 0)

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {(x, y) | y^2 = 4*x ∧ x > 0}

-- Define the conditions
def on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0
def on_positive_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0 ∧ Q.1 > 0
def on_line (P Q M : ℝ × ℝ) : Prop := ∃ t : ℝ, M = (1 - t) • P + t • Q

def orthogonal (HP PM : ℝ × ℝ) : Prop := HP.1 * PM.1 + HP.2 * PM.2 = 0
def vector_ratio (PM MQ : ℝ × ℝ) : Prop := PM = (-3/2) • MQ

-- Main theorem
theorem trajectory_and_equilateral_triangle 
  (P Q M : ℝ × ℝ) 
  (hP : on_y_axis P) 
  (hQ : on_positive_x_axis Q) 
  (hM : on_line P Q M) 
  (hOrth : orthogonal (H.1 - P.1, H.2 - P.2) (M.1 - P.1, M.2 - P.2))
  (hRatio : vector_ratio (M.1 - P.1, M.2 - P.2) (Q.1 - M.1, Q.2 - M.2)) :
  (M ∈ C) ∧ 
  (∀ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) (E : ℝ × ℝ),
    (A ∈ C ∧ B ∈ C ∧ T ∈ l ∧ A ∈ l ∧ B ∈ l ∧ E.2 = 0) →
    (∃ (x₀ : ℝ), E.1 = x₀ ∧ 
      (norm (A - E) = norm (B - E) ∧ norm (A - E) = norm (A - B)) →
      x₀ = 11/3)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_equilateral_triangle_l3677_367748


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l3677_367718

theorem floor_expression_equals_eight :
  ⌊(2021^3 : ℝ) / (2019 * 2020) - (2019^3 : ℝ) / (2020 * 2021)⌋ = 8 := by
  sorry

#check floor_expression_equals_eight

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l3677_367718


namespace NUMINAMATH_CALUDE_sec_330_deg_l3677_367726

/-- Prove that sec 330° = 2√3 / 3 -/
theorem sec_330_deg : 
  let sec : Real → Real := λ θ ↦ 1 / Real.cos θ
  let θ : Real := 330 * Real.pi / 180
  sec θ = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sec_330_deg_l3677_367726


namespace NUMINAMATH_CALUDE_logarithm_inequality_l3677_367769

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log a ^ 2 / Real.log (b + c) + Real.log b ^ 2 / Real.log (c + a) + Real.log c ^ 2 / Real.log (a + b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l3677_367769


namespace NUMINAMATH_CALUDE_inconsistent_pricing_problem_l3677_367770

theorem inconsistent_pricing_problem (shirt trouser tie : ℕ → ℚ) :
  (∃ x : ℕ, 6 * shirt 1 + 4 * trouser 1 + x * tie 1 = 80) →
  (4 * shirt 1 + 2 * trouser 1 + 2 * tie 1 = 140) →
  (5 * shirt 1 + 3 * trouser 1 + 2 * tie 1 = 110) →
  False :=
by
  sorry

end NUMINAMATH_CALUDE_inconsistent_pricing_problem_l3677_367770


namespace NUMINAMATH_CALUDE_intersection_points_on_horizontal_line_l3677_367701

/-- Given two lines parameterized by a real number s, 
    prove that their intersection points lie on a horizontal line -/
theorem intersection_points_on_horizontal_line :
  ∀ (s : ℝ), 
  ∃ (x y : ℝ), 
  (2 * x + 3 * y = 6 * s + 4) ∧ 
  (x + 2 * y = 3 * s - 1) → 
  y = -6 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_on_horizontal_line_l3677_367701


namespace NUMINAMATH_CALUDE_student_claim_incorrect_l3677_367729

theorem student_claim_incorrect (m n : ℤ) (hn : 0 < n) (hn_bound : n ≤ 100) :
  ¬ (167 * n ≤ 1000 * m ∧ 1000 * m < 168 * n) := by
  sorry

end NUMINAMATH_CALUDE_student_claim_incorrect_l3677_367729


namespace NUMINAMATH_CALUDE_travel_theorem_l3677_367784

def travel_problem (total_time : ℝ) (foot_speed : ℝ) (bike_speed : ℝ) (foot_distance : ℝ) : Prop :=
  let foot_time : ℝ := foot_distance / foot_speed
  let bike_time : ℝ := total_time - foot_time
  let bike_distance : ℝ := bike_speed * bike_time
  let total_distance : ℝ := foot_distance + bike_distance
  total_distance = 80

theorem travel_theorem :
  travel_problem 7 8 16 32 := by
  sorry

end NUMINAMATH_CALUDE_travel_theorem_l3677_367784


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l3677_367752

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), x - 2*y = 1 ∧ 4*x + 3*y = 26 ∧ x = 5 ∧ y = 2 := by
  sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), 2*x + 3*y = 3 ∧ 5*x - 3*y = 18 ∧ x = 3 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l3677_367752


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3677_367715

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := n

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define T_n as the sum of the first n terms of {1/S_n}
def T (n : ℕ) : ℚ := 2 * (1 - 1 / (n + 1))

theorem arithmetic_geometric_sequence_properties :
  -- The sequence {a_n} is arithmetic with common difference 1
  (∀ n : ℕ, a (n + 1) - a n = 1) ∧
  -- a_1, a_3, a_9 form a geometric sequence
  (a 3)^2 = a 1 * a 9 →
  -- Prove the following:
  (-- 1. General term formula
   (∀ n : ℕ, n ≥ 1 → a n = n) ∧
   -- 2. Sum of first n terms
   (∀ n : ℕ, n ≥ 1 → S n = n * (n + 1) / 2) ∧
   -- 3. T_n < 2
   (∀ n : ℕ, n ≥ 1 → T n < 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3677_367715


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3677_367787

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, 4), and x-intercepts at (1, 0) and (5, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 4
  intercept1_x : ℝ := 1
  intercept2_x : ℝ := 5

/-- The parabola satisfies its vertex condition -/
axiom vertex_condition (p : Parabola) : p.vertex_y = p.a * p.vertex_x^2 + p.b * p.vertex_x + p.c

/-- The parabola satisfies its first x-intercept condition -/
axiom intercept1_condition (p : Parabola) : 0 = p.a * p.intercept1_x^2 + p.b * p.intercept1_x + p.c

/-- The parabola satisfies its second x-intercept condition -/
axiom intercept2_condition (p : Parabola) : 0 = p.a * p.intercept2_x^2 + p.b * p.intercept2_x + p.c

/-- The sum of coefficients a, b, and c is zero for a parabola satisfying the given conditions -/
theorem sum_of_coefficients_zero (p : Parabola) : p.a + p.b + p.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3677_367787


namespace NUMINAMATH_CALUDE_sarah_hair_products_usage_l3677_367735

/-- Given Sarah's daily shampoo and conditioner usage, calculate the total volume used in 14 days -/
theorem sarah_hair_products_usage 
  (shampoo_daily : ℝ) 
  (conditioner_daily : ℝ) 
  (h1 : shampoo_daily = 1) 
  (h2 : conditioner_daily = shampoo_daily / 2) 
  (days : ℕ) 
  (h3 : days = 14) : 
  shampoo_daily * days + conditioner_daily * days = 21 := by
  sorry


end NUMINAMATH_CALUDE_sarah_hair_products_usage_l3677_367735


namespace NUMINAMATH_CALUDE_root_properties_l3677_367747

theorem root_properties (a b : ℝ) :
  (a - b)^3 + 3*a*b*(a - b) + b^3 - a^3 = 0 ∧
  (∀ a : ℝ, (a - 1)^3 - a*(a - 1)^2 + 1 = 0 ↔ a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_root_properties_l3677_367747


namespace NUMINAMATH_CALUDE_smallest_sum_B_d_l3677_367792

theorem smallest_sum_B_d : 
  ∃ (B d : ℕ), 
    B < 5 ∧ 
    d > 6 ∧ 
    125 * B + 25 * B + B = 4 * d + 4 ∧
    (∀ (B' d' : ℕ), 
      B' < 5 → 
      d' > 6 → 
      125 * B' + 25 * B' + B' = 4 * d' + 4 → 
      B + d ≤ B' + d') ∧
    B + d = 77 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_B_d_l3677_367792


namespace NUMINAMATH_CALUDE_jake_shooting_improvement_l3677_367730

theorem jake_shooting_improvement (initial_shots : ℕ) (additional_shots : ℕ) 
  (initial_percentage : ℚ) (final_percentage : ℚ) :
  initial_shots = 30 →
  additional_shots = 10 →
  initial_percentage = 60 / 100 →
  final_percentage = 62 / 100 →
  ∃ (last_successful_shots : ℕ),
    last_successful_shots = 7 ∧
    (initial_percentage * initial_shots).floor + last_successful_shots = 
      (final_percentage * (initial_shots + additional_shots)).floor :=
by sorry

end NUMINAMATH_CALUDE_jake_shooting_improvement_l3677_367730


namespace NUMINAMATH_CALUDE_unique_f_3_l3677_367779

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 2 = 3 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- The main theorem -/
theorem unique_f_3 (f : ℝ → ℝ) (hf : special_function f) : f 3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_3_l3677_367779


namespace NUMINAMATH_CALUDE_integer_solutions_l3677_367791

theorem integer_solutions (a : ℤ) : 
  (∃ b c : ℤ, ∀ x : ℤ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)) ↔ 
  (a = 10 ∨ a = 14) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_l3677_367791


namespace NUMINAMATH_CALUDE_bill_sunday_miles_l3677_367745

/-- Proves that Bill ran 9 miles on Sunday given the problem conditions --/
theorem bill_sunday_miles : ℕ → ℕ → Prop :=
  fun (bill_saturday : ℕ) (bill_sunday : ℕ) =>
    let julia_sunday := 2 * bill_sunday
    bill_saturday + bill_sunday + julia_sunday = 32 ∧
    bill_sunday = bill_saturday + 4 →
    bill_sunday = 9

/-- Proof of the theorem --/
lemma bill_sunday_miles_proof : ∃ (bill_saturday : ℕ), bill_sunday_miles bill_saturday (bill_saturday + 4) :=
  sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_l3677_367745


namespace NUMINAMATH_CALUDE_base7_addition_l3677_367711

/-- Addition of numbers in base 7 -/
def base7_add (a b c : ℕ) : ℕ :=
  (a + b + c) % 7^3

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 7^2) * 7^2 + ((n / 7) % 7) * 7 + (n % 7)

theorem base7_addition :
  base7_add (base7_to_decimal 26) (base7_to_decimal 64) (base7_to_decimal 135) = base7_to_decimal 261 :=
sorry

end NUMINAMATH_CALUDE_base7_addition_l3677_367711


namespace NUMINAMATH_CALUDE_money_left_after_debts_l3677_367761

def lottery_winnings : ℕ := 100
def payment_to_colin : ℕ := 20

def payment_to_helen (colin_payment : ℕ) : ℕ := 2 * colin_payment

def payment_to_benedict (helen_payment : ℕ) : ℕ := helen_payment / 2

def total_payments (colin : ℕ) (helen : ℕ) (benedict : ℕ) : ℕ := colin + helen + benedict

theorem money_left_after_debts :
  lottery_winnings - total_payments payment_to_colin (payment_to_helen payment_to_colin) (payment_to_benedict (payment_to_helen payment_to_colin)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_debts_l3677_367761


namespace NUMINAMATH_CALUDE_intersection_M_N_l3677_367722

open Set

def M : Set ℝ := {x | x < 2017}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3677_367722


namespace NUMINAMATH_CALUDE_second_place_limit_l3677_367781

/-- Represents an election with five candidates -/
structure Election where
  totalVoters : ℕ
  nonParticipationRate : ℚ
  invalidVotes : ℕ
  winnerVoteShare : ℚ
  winnerMargin : ℕ

/-- Conditions for a valid election -/
def validElection (e : Election) : Prop :=
  e.nonParticipationRate = 15/100 ∧
  e.invalidVotes = 250 ∧
  e.winnerVoteShare = 38/100 ∧
  e.winnerMargin = 300

/-- Calculate the percentage of valid votes for the second-place candidate -/
def secondPlacePercentage (e : Election) : ℚ :=
  let validVotes := e.totalVoters * (1 - e.nonParticipationRate) - e.invalidVotes
  let secondPlaceVotes := e.totalVoters * e.winnerVoteShare - e.winnerMargin
  secondPlaceVotes / validVotes * 100

/-- Theorem stating that as the number of voters approaches infinity, 
    the percentage of valid votes for the second-place candidate approaches 44.71% -/
theorem second_place_limit (ε : ℚ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ e : Election, validElection e → e.totalVoters ≥ N → 
    |secondPlacePercentage e - 4471/100| < ε :=
sorry

end NUMINAMATH_CALUDE_second_place_limit_l3677_367781


namespace NUMINAMATH_CALUDE_average_daily_sales_l3677_367725

/-- Represents the sales data for a baker's pastry shop over a week. -/
structure BakerSales where
  weekdayPrice : ℕ
  weekendPrice : ℕ
  mondaySales : ℕ
  weekdayIncrease : ℕ
  weekendIncrease : ℕ

/-- Calculates the total pastries sold in a week based on the given sales data. -/
def totalWeeklySales (sales : BakerSales) : ℕ :=
  let tue := sales.mondaySales + sales.weekdayIncrease
  let wed := tue + sales.weekdayIncrease
  let thu := wed + sales.weekdayIncrease
  let fri := thu + sales.weekdayIncrease
  let sat := fri + sales.weekendIncrease
  let sun := sat + sales.weekendIncrease
  sales.mondaySales + tue + wed + thu + fri + sat + sun

/-- Theorem stating that the average daily sales for the given conditions is 59/7. -/
theorem average_daily_sales (sales : BakerSales)
    (h1 : sales.weekdayPrice = 5)
    (h2 : sales.weekendPrice = 6)
    (h3 : sales.mondaySales = 2)
    (h4 : sales.weekdayIncrease = 2)
    (h5 : sales.weekendIncrease = 3) :
    (totalWeeklySales sales : ℚ) / 7 = 59 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_sales_l3677_367725


namespace NUMINAMATH_CALUDE_problem_solution_l3677_367762

theorem problem_solution : 
  ((-1/2 - 1/3 + 3/4) * (-60) = 5) ∧ 
  ((-1)^4 - 1/6 * (3 - (-3)^2) = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3677_367762


namespace NUMINAMATH_CALUDE_simplify_expression_l3677_367727

theorem simplify_expression (x : ℝ) : (2*x)^5 + (4*x)*(x^4) + 5*x^3 = 36*x^5 + 5*x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3677_367727


namespace NUMINAMATH_CALUDE_candy_pencils_count_l3677_367704

/-- The number of pencils Candy has -/
def candy_pencils : ℕ := 9

/-- The number of pencils Caleb has -/
def caleb_pencils : ℕ := 2 * candy_pencils - 3

/-- The original number of pencils Calen had -/
def calen_original_pencils : ℕ := caleb_pencils + 5

/-- The number of pencils Calen lost -/
def calen_lost_pencils : ℕ := 10

/-- The number of pencils Calen has now -/
def calen_current_pencils : ℕ := 10

theorem candy_pencils_count :
  calen_original_pencils - calen_lost_pencils = calen_current_pencils :=
by sorry

end NUMINAMATH_CALUDE_candy_pencils_count_l3677_367704


namespace NUMINAMATH_CALUDE_infinite_triples_with_coprime_c_l3677_367793

theorem infinite_triples_with_coprime_c : ∃ (a b c : ℕ → ℕ+), 
  (∀ n, (a n)^2 + (b n)^2 = (c n)^4) ∧ 
  (∀ n, Nat.gcd (c n) (c (n + 1)) = 1) := by
  sorry

end NUMINAMATH_CALUDE_infinite_triples_with_coprime_c_l3677_367793


namespace NUMINAMATH_CALUDE_star_calculation_l3677_367749

def star (a b : ℝ) : ℝ := a * b + a + b

theorem star_calculation : star 1 2 + star 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3677_367749


namespace NUMINAMATH_CALUDE_prize_probabilities_l3677_367780

/-- Represents the outcome of drawing a ball from a box -/
inductive BallColor
| Red
| White

/-- Represents a box with red and white balls -/
structure Box where
  red : Nat
  white : Nat

/-- Probability of drawing a red ball from a box -/
def probRed (box : Box) : Rat :=
  box.red / (box.red + box.white)

/-- Probability of winning first prize in one draw -/
def probFirstPrize (boxA boxB : Box) : Rat :=
  probRed boxA * probRed boxB

/-- Probability of winning second prize in one draw -/
def probSecondPrize (boxA boxB : Box) : Rat :=
  probRed boxA * (1 - probRed boxB) + (1 - probRed boxA) * probRed boxB

/-- Probability of winning a prize in one draw -/
def probWinPrize (boxA boxB : Box) : Rat :=
  probFirstPrize boxA boxB + probSecondPrize boxA boxB

/-- Expected number of first prizes in n draws -/
def expectedFirstPrizes (boxA boxB : Box) (n : Nat) : Rat :=
  n * probFirstPrize boxA boxB

theorem prize_probabilities (boxA boxB : Box) :
  boxA.red = 4 ∧ boxA.white = 6 ∧ boxB.red = 5 ∧ boxB.white = 5 →
  probWinPrize boxA boxB = 7/10 ∧ expectedFirstPrizes boxA boxB 3 = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_prize_probabilities_l3677_367780


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3677_367797

theorem quadratic_inequality_solution_set (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x - b - 3/4 > 0) ↔ -3 < b ∧ b < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3677_367797


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l3677_367709

theorem sqrt_18_div_sqrt_2_equals_3 : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l3677_367709


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3677_367756

/-- Two lines are parallel if and only if their slopes are equal and they are not the same line -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ (m₁, n₁, c₁) ≠ (m₂, n₂, c₂)

/-- The theorem states that a = 3 is a necessary and sufficient condition for the given lines to be parallel -/
theorem parallel_lines_condition (a : ℝ) :
  are_parallel a 2 (3*a) 3 (a-1) (a-7) ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3677_367756


namespace NUMINAMATH_CALUDE_nineteen_only_vegetarian_l3677_367764

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat only vegetarian -/
def only_vegetarian (f : FamilyDiet) : ℕ :=
  f.total_veg - f.both_veg_and_non_veg

/-- Theorem stating that 19 people eat only vegetarian in the given family -/
theorem nineteen_only_vegetarian (f : FamilyDiet) 
  (h1 : f.only_non_veg = 9)
  (h2 : f.both_veg_and_non_veg = 12)
  (h3 : f.total_veg = 31) :
  only_vegetarian f = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_only_vegetarian_l3677_367764


namespace NUMINAMATH_CALUDE_circumcircle_radius_is_13_l3677_367778

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Ratio of the shorter base to the longer base -/
  base_ratio : ℚ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- The midline of the trapezoid equals its height -/
  midline_eq_height : True

/-- Calculate the radius of the circumcircle of an isosceles trapezoid -/
def circumcircle_radius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with given properties, its circumcircle radius is 13 -/
theorem circumcircle_radius_is_13 (t : IsoscelesTrapezoid) 
  (h1 : t.base_ratio = 5 / 12)
  (h2 : t.height = 17) : 
  circumcircle_radius t = 13 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_is_13_l3677_367778


namespace NUMINAMATH_CALUDE_dog_food_preferences_l3677_367763

theorem dog_food_preferences (total : ℕ) (carrot : ℕ) (chicken : ℕ) (both : ℕ) 
  (h1 : total = 85)
  (h2 : carrot = 12)
  (h3 : chicken = 62)
  (h4 : both = 8) :
  total - (carrot + chicken - both) = 19 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_preferences_l3677_367763


namespace NUMINAMATH_CALUDE_exactly_one_correct_proposition_l3677_367723

open Real

theorem exactly_one_correct_proposition : ∃! n : Nat, n = 1 ∧
  (¬ (∀ x : ℝ, (x^2 < 1 → -1 < x ∧ x < 1) ↔ ((x > 1 ∨ x < -1) → x^2 > 1))) ∧
  (¬ ((∀ x : ℝ, sin x ≤ 1) ∧ (∀ a b : ℝ, a < b → a^2 < b^2))) ∧
  ((∀ x : ℝ, ¬(x^2 - x > 0)) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  (¬ (∀ x : ℝ, x^2 > 4 → x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_correct_proposition_l3677_367723


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3677_367719

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 + x₁ - 3 = 0 ∧ x₁ = 1) ∧
                (2 * x₂^2 + x₂ - 3 = 0 ∧ x₂ = -3/2)) ∧
  (∃ y₁ y₂ : ℝ, ((y₁ - 3)^2 = 2 * y₁ * (3 - y₁) ∧ y₁ = 3) ∧
                ((y₂ - 3)^2 = 2 * y₂ * (3 - y₂) ∧ y₂ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3677_367719


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l3677_367750

/-- The total amount Tom spent on video games -/
def total_spent : ℝ := 35.52

/-- The cost of the football game -/
def football_cost : ℝ := 14.02

/-- The cost of the strategy game -/
def strategy_cost : ℝ := 9.46

/-- The cost of the Batman game -/
def batman_cost : ℝ := 12.04

/-- Theorem: The total amount Tom spent on video games is equal to the sum of the costs of the football game, strategy game, and Batman game -/
theorem total_spent_equals_sum_of_games : 
  total_spent = football_cost + strategy_cost + batman_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l3677_367750


namespace NUMINAMATH_CALUDE_divide_by_repeating_decimal_l3677_367798

theorem divide_by_repeating_decimal :
  ∃ (x : ℚ), (∀ (n : ℕ), x = (3 * 10^n - 3) / (9 * 10^n)) ∧ (8 / x = 24) := by
  sorry

end NUMINAMATH_CALUDE_divide_by_repeating_decimal_l3677_367798


namespace NUMINAMATH_CALUDE_parabola_directrix_l3677_367743

/-- The directrix of a parabola given by y = -3x^2 + 6x - 5 -/
theorem parabola_directrix : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = -3 * x^2 + 6 * x - 5 ↔ 4 * a * y = (x - b)^2 + c) ∧ 
    (a = -1/12 ∧ b = 1 ∧ c = -23/3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3677_367743


namespace NUMINAMATH_CALUDE_prob_four_genuine_given_equal_weights_l3677_367741

-- Define the total number of coins
def total_coins : ℕ := 20

-- Define the number of genuine coins
def genuine_coins : ℕ := 12

-- Define the number of counterfeit coins
def counterfeit_coins : ℕ := 8

-- Define a function to calculate the probability of selecting genuine coins
def prob_genuine_selection (selected : ℕ) (remaining : ℕ) : ℚ :=
  (genuine_coins.choose selected) / (total_coins.choose selected)

-- Define the probability of selecting four genuine coins
def prob_four_genuine : ℚ :=
  (prob_genuine_selection 2 total_coins) * (prob_genuine_selection 2 (total_coins - 2))

-- Define the probability of equal weights (approximation)
def prob_equal_weights : ℚ :=
  prob_four_genuine + (counterfeit_coins / total_coins) * ((counterfeit_coins - 1) / (total_coins - 1)) *
  ((counterfeit_coins - 2) / (total_coins - 2)) * (1 / (total_coins - 3))

-- State the theorem
theorem prob_four_genuine_given_equal_weights :
  prob_four_genuine / prob_equal_weights = 550 / 703 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_genuine_given_equal_weights_l3677_367741


namespace NUMINAMATH_CALUDE_exists_rational_with_prime_multiples_l3677_367758

theorem exists_rational_with_prime_multiples : ∃ x : ℚ, 
  (Nat.Prime (Int.natAbs (Int.floor (10 * x)))) ∧ 
  (Nat.Prime (Int.natAbs (Int.floor (15 * x)))) := by
  sorry

end NUMINAMATH_CALUDE_exists_rational_with_prime_multiples_l3677_367758


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3677_367759

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_eq_one : a + b = 1 ∧ c + d = 1) 
  (product_gt_one : a * c + b * d > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3677_367759


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l3677_367777

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon : num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l3677_367777


namespace NUMINAMATH_CALUDE_min_voters_to_win_is_24_l3677_367776

/-- Represents the voting structure and outcome of a giraffe beauty contest. -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  (total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section)
  (num_districts_eq : num_districts = 5)
  (sections_per_district_eq : sections_per_district = 7)
  (voters_per_section_eq : voters_per_section = 3)

/-- Calculates the minimum number of voters required to win the contest. -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win := contest.sections_per_district / 2 + 1
  let voters_to_win_section := contest.voters_per_section / 2 + 1
  districts_to_win * sections_to_win * voters_to_win_section

/-- Theorem stating that the minimum number of voters required to win the contest is 24. -/
theorem min_voters_to_win_is_24 (contest : GiraffeContest) :
  min_voters_to_win contest = 24 := by
  sorry

#eval min_voters_to_win {
  total_voters := 105,
  num_districts := 5,
  sections_per_district := 7,
  voters_per_section := 3,
  total_voters_eq := rfl,
  num_districts_eq := rfl,
  sections_per_district_eq := rfl,
  voters_per_section_eq := rfl
}

end NUMINAMATH_CALUDE_min_voters_to_win_is_24_l3677_367776


namespace NUMINAMATH_CALUDE_min_value_of_f_l3677_367766

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := ∫ t, (2 * t - 4)

-- State the theorem
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧ ∀ x ∈ Set.Icc (-1) 3, f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3677_367766


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3677_367731

/-- The minimum distance from the origin (0,0) to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt (p.1^2 + p.2^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3677_367731


namespace NUMINAMATH_CALUDE_lens_break_probability_l3677_367714

def prob_break_first : ℝ := 0.3
def prob_break_second_given_not_first : ℝ := 0.4
def prob_break_third_given_not_first_two : ℝ := 0.9

theorem lens_break_probability :
  let prob_break_second := (1 - prob_break_first) * prob_break_second_given_not_first
  let prob_break_third := (1 - prob_break_first) * (1 - prob_break_second_given_not_first) * prob_break_third_given_not_first_two
  prob_break_first + prob_break_second + prob_break_third = 0.958 := by
  sorry

end NUMINAMATH_CALUDE_lens_break_probability_l3677_367714


namespace NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l3677_367713

theorem gcf_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l3677_367713


namespace NUMINAMATH_CALUDE_smallest_high_efficiency_l3677_367788

def efficiency (n : ℕ) : ℚ :=
  (n - (Nat.totient n)) / n

theorem smallest_high_efficiency : 
  ∀ m : ℕ, m < 30030 → efficiency m ≤ 4/5 ∧ efficiency 30030 > 4/5 :=
sorry

end NUMINAMATH_CALUDE_smallest_high_efficiency_l3677_367788
