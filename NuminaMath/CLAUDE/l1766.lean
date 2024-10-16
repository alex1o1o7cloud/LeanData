import Mathlib

namespace NUMINAMATH_CALUDE_mistaken_addition_correction_l1766_176653

theorem mistaken_addition_correction (x : ℤ) : x + 16 = 64 → x - 16 = 32 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_addition_correction_l1766_176653


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1766_176689

theorem triangle_perimeter (a b x : ℝ) : 
  a = 7 → 
  b = 11 → 
  x^2 - 25 = 2*(x - 5)^2 → 
  x > 0 →
  a + b > x →
  x + b > a →
  x + a > b →
  (a + b + x = 23 ∨ a + b + x = 33) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1766_176689


namespace NUMINAMATH_CALUDE_range_of_a_l1766_176665

-- Define p as a predicate on m
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ (∀ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 → y^2 ≤ c^2)

-- Define q as a predicate on m and a
def q (m a : ℝ) : Prop := m^2 - (2*a + 1)*m + a^2 + a < 0

-- State the theorem
theorem range_of_a : 
  (∀ m : ℝ, p m → ∃ a : ℝ, q m a) → 
  ∃ a : ℝ, 1/2 ≤ a ∧ a ≤ 1 ∧ 
    (∀ b : ℝ, (∀ m : ℝ, p m → q m b) → 1/2 ≤ b ∧ b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1766_176665


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l1766_176670

theorem largest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 7 + Nat.factorial 8) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 7 + Nat.factorial 8) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l1766_176670


namespace NUMINAMATH_CALUDE_ben_basketball_boxes_ben_basketball_boxes_correct_l1766_176645

theorem ben_basketball_boxes (basketball_cards_per_box : ℕ) 
                              (baseball_boxes : ℕ) 
                              (baseball_cards_per_box : ℕ) 
                              (cards_given_away : ℕ) 
                              (cards_left : ℕ) : ℕ :=
  let total_baseball_cards := baseball_boxes * baseball_cards_per_box
  let total_cards_before := cards_given_away + cards_left
  let basketball_boxes := (total_cards_before - total_baseball_cards) / basketball_cards_per_box
  basketball_boxes

#check ben_basketball_boxes 10 5 8 58 22 = 4

theorem ben_basketball_boxes_correct : ben_basketball_boxes 10 5 8 58 22 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ben_basketball_boxes_ben_basketball_boxes_correct_l1766_176645


namespace NUMINAMATH_CALUDE_cubic_sum_equals_27_l1766_176697

theorem cubic_sum_equals_27 (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9*a*b = 27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_27_l1766_176697


namespace NUMINAMATH_CALUDE_number_of_dimes_l1766_176649

/-- Proves the number of dimes given the number of pennies, nickels, and total value -/
theorem number_of_dimes 
  (num_pennies : ℕ) 
  (num_nickels : ℕ) 
  (total_value : ℚ) 
  (h_num_pennies : num_pennies = 9)
  (h_num_nickels : num_nickels = 4)
  (h_total_value : total_value = 59 / 100)
  (h_penny_value : ∀ n : ℕ, n * (1 / 100 : ℚ) = (n : ℚ) / 100)
  (h_nickel_value : ∀ n : ℕ, n * (5 / 100 : ℚ) = (5 * n : ℚ) / 100)
  (h_dime_value : ∀ n : ℕ, n * (10 / 100 : ℚ) = (10 * n : ℚ) / 100) :
  ∃ num_dimes : ℕ, 
    num_dimes = 3 ∧ 
    total_value = 
      num_pennies * (1 / 100 : ℚ) + 
      num_nickels * (5 / 100 : ℚ) + 
      num_dimes * (10 / 100 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_number_of_dimes_l1766_176649


namespace NUMINAMATH_CALUDE_amulet_seller_profit_l1766_176629

/-- Calculates the profit for an amulet seller at a Ren Faire --/
theorem amulet_seller_profit
  (days : ℕ)
  (amulets_per_day : ℕ)
  (selling_price : ℕ)
  (cost_price : ℕ)
  (faire_cut_percent : ℕ)
  (h1 : days = 2)
  (h2 : amulets_per_day = 25)
  (h3 : selling_price = 40)
  (h4 : cost_price = 30)
  (h5 : faire_cut_percent = 10)
  : (days * amulets_per_day * selling_price) - 
    (days * amulets_per_day * cost_price) - 
    (days * amulets_per_day * selling_price * faire_cut_percent / 100) = 300 :=
by sorry

end NUMINAMATH_CALUDE_amulet_seller_profit_l1766_176629


namespace NUMINAMATH_CALUDE_second_race_outcome_l1766_176678

/-- Represents the speeds of Katie and Sarah -/
structure RunnerSpeeds where
  katie : ℝ
  sarah : ℝ

/-- The problem setup -/
def race_problem (speeds : RunnerSpeeds) : Prop :=
  speeds.katie > 0 ∧ 
  speeds.sarah > 0 ∧
  speeds.katie * 95 = speeds.sarah * 100

/-- The theorem to prove -/
theorem second_race_outcome (speeds : RunnerSpeeds) 
  (h : race_problem speeds) : 
  speeds.katie * 105 = speeds.sarah * 99.75 := by
  sorry

#check second_race_outcome

end NUMINAMATH_CALUDE_second_race_outcome_l1766_176678


namespace NUMINAMATH_CALUDE_probability_of_drawing_two_l1766_176656

/-- Represents a card with a number -/
structure Card where
  number : ℕ

/-- Represents the set of cards -/
def cardSet : Finset Card := sorry

/-- The total number of cards -/
def totalCards : ℕ := 5

/-- The number of cards with the number 2 -/
def cardsWithTwo : ℕ := 2

/-- The probability of drawing a card with the number 2 -/
def probabilityOfTwo : ℚ := cardsWithTwo / totalCards

theorem probability_of_drawing_two :
  probabilityOfTwo = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_two_l1766_176656


namespace NUMINAMATH_CALUDE_leg_length_in_45_45_90_triangle_l1766_176683

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = side * Real.sqrt 2

/-- The length of a leg in a 45-45-90 triangle with hypotenuse 9 is 9 -/
theorem leg_length_in_45_45_90_triangle (t : RightIsoscelesTriangle) 
  (h : t.hypotenuse = 9) : t.side = 9 := by
  sorry

end NUMINAMATH_CALUDE_leg_length_in_45_45_90_triangle_l1766_176683


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1766_176642

theorem quadratic_factorization_sum (p q r : ℤ) : 
  (∀ x, x^2 + 19*x + 88 = (x + p) * (x + q)) →
  (∀ x, x^2 - 23*x + 132 = (x - q) * (x - r)) →
  p + q + r = 31 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1766_176642


namespace NUMINAMATH_CALUDE_min_value_line_circle_l1766_176607

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_line_circle (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x y : ℝ, a * x + b * y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x' y' : ℝ, a' * x' + b' * y' + 1 = 0 ∧ x'^2 + y'^2 + 8*x' + 2*y' + 1 = 0) →
    1/a + 4/b ≤ 1/a' + 4/b') →
  1/a + 4/b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_line_circle_l1766_176607


namespace NUMINAMATH_CALUDE_scaled_roots_polynomial_l1766_176655

theorem scaled_roots_polynomial (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 10 = 0) → 
  (r₂^3 - 4*r₂^2 + 10 = 0) → 
  (r₃^3 - 4*r₃^2 + 10 = 0) → 
  (∀ x : ℂ, x^3 - 12*x^2 + 270 = (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃)) := by
sorry

end NUMINAMATH_CALUDE_scaled_roots_polynomial_l1766_176655


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l1766_176650

theorem factorial_ratio_equals_seven_and_half :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 8) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l1766_176650


namespace NUMINAMATH_CALUDE_shirt_sale_price_l1766_176669

theorem shirt_sale_price (original_price : ℝ) (initial_sale_price : ℝ) 
  (h1 : initial_sale_price > 0)
  (h2 : original_price > 0)
  (h3 : initial_sale_price * 0.8 = original_price * 0.64) :
  initial_sale_price / original_price = 0.8 := by
sorry

end NUMINAMATH_CALUDE_shirt_sale_price_l1766_176669


namespace NUMINAMATH_CALUDE_book_pages_theorem_l1766_176614

/-- Calculates the total number of pages in a book given the number of chapters and pages per chapter -/
def totalPages (chapters : ℕ) (pagesPerChapter : ℕ) : ℕ :=
  chapters * pagesPerChapter

/-- Theorem stating that a book with 31 chapters, each 61 pages long, has 1891 pages in total -/
theorem book_pages_theorem :
  totalPages 31 61 = 1891 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l1766_176614


namespace NUMINAMATH_CALUDE_square_product_eq_sum_squares_solution_l1766_176693

theorem square_product_eq_sum_squares_solution (a b : ℤ) :
  a^2 * b^2 = a^2 + b^2 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_product_eq_sum_squares_solution_l1766_176693


namespace NUMINAMATH_CALUDE_baseball_tickets_sold_l1766_176648

theorem baseball_tickets_sold (fair_tickets : ℕ) (baseball_tickets : ℕ) : 
  fair_tickets = 25 → 
  fair_tickets = 2 * baseball_tickets + 6 → 
  baseball_tickets = 9 := by
sorry

end NUMINAMATH_CALUDE_baseball_tickets_sold_l1766_176648


namespace NUMINAMATH_CALUDE_remaining_garlic_cloves_l1766_176681

-- Define the initial number of garlic cloves
def initial_cloves : ℕ := 93

-- Define the number of cloves used
def used_cloves : ℕ := 86

-- Theorem stating that the remaining cloves is 7
theorem remaining_garlic_cloves : initial_cloves - used_cloves = 7 := by
  sorry

end NUMINAMATH_CALUDE_remaining_garlic_cloves_l1766_176681


namespace NUMINAMATH_CALUDE_pennys_initial_money_l1766_176608

/-- Penny's initial amount of money given her purchases and remaining balance -/
theorem pennys_initial_money :
  ∀ (sock_pairs : ℕ) (sock_price hat_price remaining : ℚ),
    sock_pairs = 4 →
    sock_price = 2 →
    hat_price = 7 →
    remaining = 5 →
    (sock_pairs : ℚ) * sock_price + hat_price + remaining = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_pennys_initial_money_l1766_176608


namespace NUMINAMATH_CALUDE_seating_arrangements_l1766_176698

def n : ℕ := 8

def numArrangements : ℕ := n.factorial - (n-1).factorial * 2

theorem seating_arrangements (n : ℕ) (h : n = 8) : 
  numArrangements = 30240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1766_176698


namespace NUMINAMATH_CALUDE_max_sum_cubes_max_sum_cubes_tight_l1766_176658

theorem max_sum_cubes (p q r s t : ℝ) (h : p^2 + q^2 + r^2 + s^2 + t^2 = 5) :
  p^3 + q^3 + r^3 + s^3 + t^3 ≤ 5 * Real.sqrt 5 :=
by sorry

theorem max_sum_cubes_tight : ∃ (p q r s t : ℝ),
  p^2 + q^2 + r^2 + s^2 + t^2 = 5 ∧ p^3 + q^3 + r^3 + s^3 + t^3 = 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_max_sum_cubes_tight_l1766_176658


namespace NUMINAMATH_CALUDE_age_difference_l1766_176609

theorem age_difference : ∀ (a b : ℕ), 
  (a < 10 ∧ b < 10) →  -- a and b are single digits
  (10 * a + b + 5 = 3 * (10 * b + a + 5)) →  -- In 5 years, Rachel's age will be three times Sam's age
  ((10 * a + b) - (10 * b + a) = 63) :=  -- The difference in their current ages is 63
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1766_176609


namespace NUMINAMATH_CALUDE_nth_number_in_set_l1766_176610

theorem nth_number_in_set (n : ℕ) : 
  (n + 1) * 19 + 13 = (499 * 19 + 13) → n = 498 := by
  sorry

#check nth_number_in_set

end NUMINAMATH_CALUDE_nth_number_in_set_l1766_176610


namespace NUMINAMATH_CALUDE_area_of_region_l1766_176635

-- Define the region
def region (x y : ℝ) : Prop := 
  |x - 2*y^2| + x + 2*y^2 ≤ 8 - 4*y

-- Define symmetry about y-axis
def symmetric_about_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem area_of_region : 
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ region x y) ∧ 
    symmetric_about_y_axis S ∧
    MeasureTheory.volume S = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l1766_176635


namespace NUMINAMATH_CALUDE_smallest_angle_3_4_5_triangle_l1766_176694

theorem smallest_angle_3_4_5_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a^2 + b^2 = c^2 →
    a / c = 3 / 5 ∧ b / c = 4 / 5 →
    min (Real.arctan (a / b)) (Real.arctan (b / a)) = Real.arctan (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_3_4_5_triangle_l1766_176694


namespace NUMINAMATH_CALUDE_set_problem_l1766_176618

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3*a}

theorem set_problem (a : ℝ) :
  (A ⊆ B a → 4/3 ≤ a ∧ a ≤ 2) ∧
  (A ∩ B a ≠ ∅ → 2/3 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l1766_176618


namespace NUMINAMATH_CALUDE_sexagesimal_cubes_correct_l1766_176680

/-- Converts a sexagesimal number to decimal -/
def sexagesimal_to_decimal (whole : ℕ) (frac : ℕ) : ℕ :=
  whole * 60 + frac

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- Represents a sexagesimal number as a pair of natural numbers -/
structure Sexagesimal :=
  (whole : ℕ)
  (frac : ℕ)

/-- Theorem stating that the sexagesimal representation of cubes is correct for numbers from 1 to 32 -/
theorem sexagesimal_cubes_correct :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 32 →
    ∃ s : Sexagesimal, 
      sexagesimal_to_decimal s.whole s.frac = n^3 ∧
      is_perfect_cube (sexagesimal_to_decimal s.whole s.frac) :=
by sorry

end NUMINAMATH_CALUDE_sexagesimal_cubes_correct_l1766_176680


namespace NUMINAMATH_CALUDE_zero_of_f_floor_l1766_176638

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem zero_of_f_floor (x : ℝ) (hx : f x = 0) : Int.floor x = 2 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_floor_l1766_176638


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1766_176641

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 120) (hac : Nat.gcd a c = 360) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 120 ∧ Nat.gcd a c' = 360 ∧ Nat.gcd b' c' = 120 ∧
  ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 120 → Nat.gcd a c'' = 360 → Nat.gcd b'' c'' ≥ 120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1766_176641


namespace NUMINAMATH_CALUDE_problem_solution_l1766_176623

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ)
  (h_xavier : p_xavier = 1/3)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1766_176623


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1766_176666

theorem inequality_and_equality_condition (a b c : ℝ) :
  (5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c) ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1766_176666


namespace NUMINAMATH_CALUDE_proportional_division_l1766_176664

theorem proportional_division (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 →
  (a : ℚ) = 2 →
  (b : ℚ) = 1/2 →
  (c : ℚ) = 1/4 →
  ∃ (x : ℚ), a * x + b * x + c * x = total ∧ b * x = 208/11 := by
  sorry

end NUMINAMATH_CALUDE_proportional_division_l1766_176664


namespace NUMINAMATH_CALUDE_haleys_marbles_l1766_176687

/-- The number of boys in Haley's class who love to play marbles. -/
def num_boys : ℕ := 5

/-- The number of marbles each boy would receive. -/
def marbles_per_boy : ℕ := 7

/-- The total number of marbles Haley has. -/
def total_marbles : ℕ := num_boys * marbles_per_boy

/-- Theorem stating that the total number of marbles Haley has is equal to
    the product of the number of boys and the number of marbles each boy would receive. -/
theorem haleys_marbles : total_marbles = num_boys * marbles_per_boy := by
  sorry

end NUMINAMATH_CALUDE_haleys_marbles_l1766_176687


namespace NUMINAMATH_CALUDE_inequality_proof_l1766_176603

theorem inequality_proof (a b c : ℝ) : 
  a = (1/2) * Real.cos (6 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * π / 180) →
  b = (2 * Real.tan (13 * π / 180)) / (1 - Real.tan (13 * π / 180)^2) →
  c = Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2) →
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1766_176603


namespace NUMINAMATH_CALUDE_inequality_solution_l1766_176657

theorem inequality_solution (x : ℝ) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -5/3 ∧ x ≠ -4/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1766_176657


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1766_176676

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 576) 
  (h2 : height = 18) : 
  area / height = 32 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1766_176676


namespace NUMINAMATH_CALUDE_intersection_probability_l1766_176647

/-- Given probabilities for events a and b, prove their intersection probability -/
theorem intersection_probability (a b : Set α) (p : Set α → ℝ) 
  (ha : p a = 0.18)
  (hb : p b = 0.5)
  (hba : p (b ∩ a) / p a = 0.2) :
  p (a ∩ b) = 0.036 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_l1766_176647


namespace NUMINAMATH_CALUDE_existence_condition_range_l1766_176677

theorem existence_condition_range (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, -x₀^2 + 3*x₀ + a > 0) ↔ a > -2 :=
sorry

end NUMINAMATH_CALUDE_existence_condition_range_l1766_176677


namespace NUMINAMATH_CALUDE_fortieth_day_from_tuesday_is_sunday_l1766_176643

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem fortieth_day_from_tuesday_is_sunday :
  advanceDay DayOfWeek.Tuesday 40 = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_fortieth_day_from_tuesday_is_sunday_l1766_176643


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1766_176671

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ) (X : ℕ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt 11 + E * Real.sqrt X) / F ∧
    F > 0 ∧
    A + B + C + D + E + F = 20 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1766_176671


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l1766_176644

theorem triangle_perimeter_from_inradius_and_area :
  ∀ (r A p : ℝ),
  r > 0 →
  A > 0 →
  r = 2.5 →
  A = 60 →
  A = r * (p / 2) →
  p = 48 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l1766_176644


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1766_176634

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1766_176634


namespace NUMINAMATH_CALUDE_billys_age_l1766_176659

theorem billys_age (B J A : ℕ) 
  (h1 : B = 3 * J)
  (h2 : J = A / 2)
  (h3 : B + J + A = 90) :
  B = 45 := by sorry

end NUMINAMATH_CALUDE_billys_age_l1766_176659


namespace NUMINAMATH_CALUDE_f_2018_l1766_176621

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom periodic : ∀ x : ℝ, f (x + 4) = -f x
axiom symmetric : ∀ x : ℝ, f (1 - (x - 1)) = f (x - 1)
axiom f_2 : f 2 = 2

-- The theorem to prove
theorem f_2018 : f 2018 = 2 := by sorry

end NUMINAMATH_CALUDE_f_2018_l1766_176621


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1766_176686

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  b = 2 * a →        -- one leg is twice the other
  a^2 + b^2 + c^2 = 1450 →  -- sum of squares of sides
  c = 5 * Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1766_176686


namespace NUMINAMATH_CALUDE_range_a_theorem_l1766_176662

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 2, x^2 - a ≥ 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, 2*x^2 + a*x + 1 > 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2*Real.sqrt 2 ∨ (0 < a ∧ a < 2*Real.sqrt 2)

-- State the theorem
theorem range_a_theorem (a : ℝ) : 
  (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1766_176662


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l1766_176624

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Divisibility relation -/
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem fibonacci_divisibility (m n : ℕ) (h : m > 2) :
  divides (fib m) (fib n) ↔ divides m n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l1766_176624


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1766_176601

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1766_176601


namespace NUMINAMATH_CALUDE_fifth_root_unity_sum_l1766_176661

theorem fifth_root_unity_sum (x : ℂ) : x^5 = 1 → 1 + x^4 + x^8 + x^12 + x^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_sum_l1766_176661


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l1766_176660

/-- The cost of building a fence around a rectangular plot -/
def fence_cost (length width price_length price_width : ℕ) : ℕ :=
  2 * (length * price_length + width * price_width)

/-- Theorem: The cost of the fence for the given dimensions and prices is 5408 -/
theorem fence_cost_calculation :
  fence_cost 17 21 59 81 = 5408 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_calculation_l1766_176660


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1766_176640

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (area : ℝ)
  (valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem
theorem triangle_area_inequality (ABC : Triangle) :
  ∃ (A₁B₁C₁ : Triangle),
    A₁B₁C₁.a = Real.sqrt ABC.a ∧
    A₁B₁C₁.b = Real.sqrt ABC.b ∧
    A₁B₁C₁.c = Real.sqrt ABC.c ∧
    A₁B₁C₁.area ^ 2 ≥ (ABC.area * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1766_176640


namespace NUMINAMATH_CALUDE_candy_box_price_increase_candy_box_price_after_increase_l1766_176674

theorem candy_box_price_increase (initial_soda_price : ℝ) 
  (candy_increase_rate : ℝ) (soda_increase_rate : ℝ) 
  (initial_total_price : ℝ) : ℝ :=
  let initial_candy_price := initial_total_price - initial_soda_price
  let final_candy_price := initial_candy_price * (1 + candy_increase_rate)
  final_candy_price

theorem candy_box_price_after_increase :
  candy_box_price_increase 12 0.25 0.5 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_candy_box_price_after_increase_l1766_176674


namespace NUMINAMATH_CALUDE_probability_one_male_one_female_l1766_176605

-- Define the total number of students
def total_students : ℕ := 4

-- Define the number of male students
def male_students : ℕ := 1

-- Define the number of female students
def female_students : ℕ := 3

-- Define the number of students to be selected
def selected_students : ℕ := 2

-- Theorem statement
theorem probability_one_male_one_female :
  (Nat.choose male_students 1 * Nat.choose female_students 1) / Nat.choose total_students selected_students = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_probability_one_male_one_female_l1766_176605


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1766_176602

theorem geometric_progression_fourth_term (a : ℝ) (r : ℝ) :
  (∃ (b c : ℝ), a = 3^(3/4) ∧ b = 3^(2/4) ∧ c = 3^(1/4) ∧ 
   b / a = c / b) → 
  c^2 / b = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1766_176602


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l1766_176699

theorem sum_of_roots_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7)
  (∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) →
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = -2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l1766_176699


namespace NUMINAMATH_CALUDE_polynomial_product_simplification_l1766_176688

theorem polynomial_product_simplification (x y : ℝ) :
  (3 * x^2 - 7 * y^3) * (9 * x^4 + 21 * x^2 * y^3 + 49 * y^6) = 27 * x^6 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_simplification_l1766_176688


namespace NUMINAMATH_CALUDE_exists_good_placement_l1766_176617

def RegularPolygon (n : ℕ) := Fin n → ℕ

def IsGoodPlacement (p : RegularPolygon 1983) : Prop :=
  ∀ (axis : Fin 1983), 
    ∀ (i : Fin 991), 
      p ((axis + i) % 1983) > p ((axis - i) % 1983)

theorem exists_good_placement : 
  ∃ (p : RegularPolygon 1983), IsGoodPlacement p :=
sorry

end NUMINAMATH_CALUDE_exists_good_placement_l1766_176617


namespace NUMINAMATH_CALUDE_tangent_line_m_range_l1766_176632

/-- The range of m for a line mx - y - 5m + 4 = 0 tangent to a circle (x+1)^2 + y^2 = 4 -/
theorem tangent_line_m_range :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), (x + 1)^2 + y^2 = 4 ∧ m*x - y - 5*m + 4 = 0) →
  (∃ (Q : ℝ × ℝ), (Q.1 + 1)^2 + Q.2^2 = 4 ∧ 
    ∃ (P : ℝ × ℝ), m*P.1 - P.2 - 5*m + 4 = 0 ∧
    Real.cos (30 * π / 180) = (Q.1 - P.1) / (4 * ((Q.1 - P.1)^2 + (Q.2 - P.2)^2).sqrt)) →
  0 ≤ m ∧ m ≤ 12/5 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_m_range_l1766_176632


namespace NUMINAMATH_CALUDE_previous_painting_price_l1766_176636

/-- Proves the price of a previous painting given the price of the most recent painting and the relationship between the two prices. -/
theorem previous_painting_price (recent_price : ℝ) (h1 : recent_price = 49000) 
  (h2 : recent_price = 3.5 * previous_price - 1000) : previous_price = 14285.71 := by
  sorry

end NUMINAMATH_CALUDE_previous_painting_price_l1766_176636


namespace NUMINAMATH_CALUDE_johns_shower_water_usage_rate_l1766_176663

/-- Calculates the water usage rate of John's shower -/
theorem johns_shower_water_usage_rate :
  let weeks : ℕ := 4
  let days_per_week : ℕ := 7
  let shower_frequency : ℕ := 2  -- every other day
  let shower_duration : ℕ := 10  -- minutes
  let total_water_usage : ℕ := 280  -- gallons
  
  let total_days : ℕ := weeks * days_per_week
  let number_of_showers : ℕ := total_days / shower_frequency
  let total_shower_time : ℕ := number_of_showers * shower_duration
  
  (total_water_usage : ℚ) / total_shower_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_shower_water_usage_rate_l1766_176663


namespace NUMINAMATH_CALUDE_four_number_sequence_l1766_176604

theorem four_number_sequence :
  ∃ (a b c d : ℝ),
    (b / c = c / a) ∧
    (a + b + c = 19) ∧
    (b - c = c - d) ∧
    (b + c + d = 12) ∧
    ((a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨
     (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_four_number_sequence_l1766_176604


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l1766_176679

theorem sphere_volume_surface_area_relation (r₁ r₂ : ℝ) (h : r₁ > 0) :
  (4 / 3 * Real.pi * r₂^3) = 8 * (4 / 3 * Real.pi * r₁^3) →
  (4 * Real.pi * r₂^2) = 4 * (4 * Real.pi * r₁^2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l1766_176679


namespace NUMINAMATH_CALUDE_count_equal_S_consecutive_l1766_176668

def S (n : ℕ) : ℕ := (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8)

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem count_equal_S_consecutive : 
  ∃ (A B : ℕ), A ≠ B ∧ 
    is_three_digit A ∧ is_three_digit B ∧
    S A = S (A + 1) ∧ S B = S (B + 1) ∧
    ∀ (n : ℕ), is_three_digit n ∧ S n = S (n + 1) → n = A ∨ n = B :=
by sorry

end NUMINAMATH_CALUDE_count_equal_S_consecutive_l1766_176668


namespace NUMINAMATH_CALUDE_a_work_days_l1766_176672

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 8

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 2

/-- The total amount of work to be done -/
def total_work : ℝ := 1

theorem a_work_days : ∃ (a : ℝ), 
  a > 0 ∧ 
  together_days * (1/a + 1/b_days) + b_alone_days * (1/b_days) = total_work ∧ 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_work_days_l1766_176672


namespace NUMINAMATH_CALUDE_veranda_width_l1766_176630

/-- Proves that the width of a veranda surrounding a rectangular room is 2 meters -/
theorem veranda_width (room_length room_width veranda_area : ℝ) : 
  room_length = 18 → 
  room_width = 12 → 
  veranda_area = 136 → 
  ∃ w : ℝ, w = 2 ∧ 
    (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width = veranda_area :=
by sorry

end NUMINAMATH_CALUDE_veranda_width_l1766_176630


namespace NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l1766_176637

theorem sum_positive_implies_at_least_one_positive (x y : ℝ) : x + y > 0 → x > 0 ∨ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l1766_176637


namespace NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l1766_176611

theorem consecutive_product_plus_one_is_square : ∃ m : ℕ, 
  2017 * 2018 * 2019 * 2020 + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l1766_176611


namespace NUMINAMATH_CALUDE_origami_stars_per_bottle_l1766_176613

/-- Represents the problem of determining the number of origami stars per bottle -/
theorem origami_stars_per_bottle
  (total_bottles : ℕ)
  (total_stars : ℕ)
  (h1 : total_bottles = 5)
  (h2 : total_stars = 75) :
  total_stars / total_bottles = 15 := by
  sorry

end NUMINAMATH_CALUDE_origami_stars_per_bottle_l1766_176613


namespace NUMINAMATH_CALUDE_jake_final_bitcoins_l1766_176625

/-- Represents the number of bitcoins Jake has at each step -/
def bitcoin_transactions (initial : ℕ) (donation1 : ℕ) (donation2 : ℕ) : ℕ :=
  let after_donation1 := initial - donation1
  let after_brother := after_donation1 / 2
  let after_triple := after_brother * 3
  after_triple - donation2

/-- Theorem stating that Jake ends up with 80 bitcoins after all transactions -/
theorem jake_final_bitcoins :
  bitcoin_transactions 80 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_bitcoins_l1766_176625


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1766_176685

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 13*x₁ + 40 = 0) ∧
  (x₂^2 - 13*x₂ + 40 = 0) ∧
  x₁ = 5 ∧
  x₂ = 8 ∧
  x₁ > 0 ∧
  x₂ > 0 ∧
  x₂ > x₁ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1766_176685


namespace NUMINAMATH_CALUDE_at_least_two_inequalities_hold_l1766_176695

theorem at_least_two_inequalities_hold (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∧ 2 / a + 3 / b + 6 / c ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_inequalities_hold_l1766_176695


namespace NUMINAMATH_CALUDE_smallest_odd_probability_l1766_176619

/-- The probability that the smallest number in a lottery draw is odd -/
theorem smallest_odd_probability (n : ℕ) (k : ℕ) (h1 : n = 90) (h2 : k = 5) :
  let prob := (1 : ℚ) / 2 + (44 : ℚ) * (Nat.choose 45 3 : ℚ) / (2 * (Nat.choose n k : ℚ))
  ∃ (ε : ℚ), abs (prob - 0.5142) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_odd_probability_l1766_176619


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l1766_176633

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure any additional person must sit next to someone -/
def minOccupiedSeats (totalSeats : ℕ) : ℕ :=
  totalSeats / 3

theorem min_occupied_seats_for_150 :
  minOccupiedSeats 150 = 50 := by
  sorry

#eval minOccupiedSeats 150

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l1766_176633


namespace NUMINAMATH_CALUDE_problem_statement_l1766_176691

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (0 < a * b ∧ a * b ≤ 1) ∧ (a^2 + b^2 ≥ 2) ∧ (0 < b ∧ b < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1766_176691


namespace NUMINAMATH_CALUDE_line_parameterization_specific_line_parameterization_l1766_176612

/-- A parameterization of a line is valid if it satisfies the line equation and has a correct direction vector -/
def IsValidParameterization (a b : ℝ) (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  (∀ t, (p.1 + t * v.1, p.2 + t * v.2) ∈ {(x, y) | y = a * x + b}) ∧
  ∃ k ≠ 0, v = (k, a * k)

theorem line_parameterization (a b : ℝ) :
  let line := fun x y ↦ y = a * x + b
  IsValidParameterization a b (1, 8) (1, 3) ∧
  IsValidParameterization a b (2, 11) (-1/3, -1) ∧
  IsValidParameterization a b (-1.5, 0.5) (1, 3) ∧
  ¬IsValidParameterization a b (-5/3, 0) (3, 9) ∧
  ¬IsValidParameterization a b (0, 5) (6, 2) :=
by sorry

/-- The specific line y = 3x + 5 has valid parameterizations A, D, and E, but not B and C -/
theorem specific_line_parameterization :
  let line := fun x y ↦ y = 3 * x + 5
  IsValidParameterization 3 5 (1, 8) (1, 3) ∧
  IsValidParameterization 3 5 (2, 11) (-1/3, -1) ∧
  IsValidParameterization 3 5 (-1.5, 0.5) (1, 3) ∧
  ¬IsValidParameterization 3 5 (-5/3, 0) (3, 9) ∧
  ¬IsValidParameterization 3 5 (0, 5) (6, 2) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_specific_line_parameterization_l1766_176612


namespace NUMINAMATH_CALUDE_percentage_increase_l1766_176690

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 80 → final = 120 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1766_176690


namespace NUMINAMATH_CALUDE_factor_theorem_cubic_l1766_176628

/-- A cubic polynomial P(x) with a given factor x - 3 -/
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4*x^2 + c*x + 20

/-- The theorem stating the value of c for which x - 3 is a factor of P(x) -/
theorem factor_theorem_cubic (c : ℚ) : 
  (∀ x, P c x = 0 ↔ x = 3) → c = -83/3 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_cubic_l1766_176628


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l1766_176682

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is (4/5, 38/5, 59/5) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 4, 6)
  let B : ℝ × ℝ × ℝ := (6, 5, 3)
  let C : ℝ × ℝ × ℝ := (4, 6, 7)
  orthocenter A B C = (4/5, 38/5, 59/5) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l1766_176682


namespace NUMINAMATH_CALUDE_intersection_points_sum_l1766_176652

def f (x : ℝ) : ℝ := (x + 2) * (x - 4)

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (2 * x)

def intersection_points_fg : ℕ := 2

def intersection_points_fh : ℕ := 2

theorem intersection_points_sum : 
  10 * intersection_points_fg + intersection_points_fh = 22 := by sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l1766_176652


namespace NUMINAMATH_CALUDE_function_properties_l1766_176620

/-- Given a function f(x) = 2x - a/x where f(1) = 3, this theorem proves
    properties about the value of a, the parity of f, and its monotonicity. -/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = 2*x - a/x)
    (h_f1 : f 1 = 3) :
  (a = -1) ∧ 
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, 1 < x₂ ∧ x₂ < x₁ → f x₂ < f x₁) :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l1766_176620


namespace NUMINAMATH_CALUDE_pen_price_before_discount_l1766_176615

-- Define the problem parameters
def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def total_cost : ℚ := 570
def pen_discount : ℚ := 0.1
def pencil_tax : ℚ := 0.05
def avg_pencil_price : ℚ := 2

-- Define the theorem
theorem pen_price_before_discount :
  let pencil_cost := num_pencils * avg_pencil_price
  let pencil_cost_with_tax := pencil_cost * (1 + pencil_tax)
  let pen_cost_with_discount := total_cost - pencil_cost_with_tax
  let pen_cost_before_discount := pen_cost_with_discount / (1 - pen_discount)
  let avg_pen_price := pen_cost_before_discount / num_pens
  ∃ (x : ℚ), abs (x - avg_pen_price) < 0.005 ∧ x = 15.28 :=
by sorry


end NUMINAMATH_CALUDE_pen_price_before_discount_l1766_176615


namespace NUMINAMATH_CALUDE_pythagorean_triple_even_l1766_176606

theorem pythagorean_triple_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : Even x ∨ Even y := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_even_l1766_176606


namespace NUMINAMATH_CALUDE_election_winning_percentage_bound_l1766_176684

def total_votes_sept30 : ℕ := 15000
def total_votes_oct10 : ℕ := 22000
def geoff_votes_sept30 : ℕ := 150
def additional_votes_needed_sept30 : ℕ := 5000
def additional_votes_needed_oct10 : ℕ := 2000

def winning_percentage : ℚ :=
  (geoff_votes_sept30 + additional_votes_needed_sept30 + additional_votes_needed_oct10) / total_votes_oct10

theorem election_winning_percentage_bound :
  winning_percentage < 325/1000 := by sorry

end NUMINAMATH_CALUDE_election_winning_percentage_bound_l1766_176684


namespace NUMINAMATH_CALUDE_three_times_work_days_l1766_176626

/-- The number of days Aarti needs to complete one piece of work -/
def base_work_days : ℕ := 9

/-- The number of times the work is multiplied -/
def work_multiplier : ℕ := 3

/-- Theorem: The time required to complete three times the work is 27 days -/
theorem three_times_work_days : base_work_days * work_multiplier = 27 := by
  sorry

end NUMINAMATH_CALUDE_three_times_work_days_l1766_176626


namespace NUMINAMATH_CALUDE_B_2_2_equals_9_l1766_176673

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2_equals_9 : B 2 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_B_2_2_equals_9_l1766_176673


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l1766_176631

theorem log_inequality_solution_set :
  ∀ x : ℝ, (Real.log (x - 1) < 1) ↔ (1 < x ∧ x < 11) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l1766_176631


namespace NUMINAMATH_CALUDE_stating_rectangular_box_area_diagonal_product_l1766_176692

/-- Represents a rectangular box with dimensions a, b, and c -/
structure RectangularBox (a b c : ℝ) where
  bottom_area : ℝ := a * b
  side_area : ℝ := b * c
  front_area : ℝ := c * a
  diagonal_squared : ℝ := a^2 + b^2 + c^2

/-- 
Theorem stating that for a rectangular box, the product of its face areas 
multiplied by the square of its diagonal equals a²b²c² · (a² + b² + c²)
-/
theorem rectangular_box_area_diagonal_product 
  (a b c : ℝ) (box : RectangularBox a b c) : 
  box.bottom_area * box.side_area * box.front_area * box.diagonal_squared = 
  a^2 * b^2 * c^2 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_stating_rectangular_box_area_diagonal_product_l1766_176692


namespace NUMINAMATH_CALUDE_sum_of_roots_l1766_176696

/-- The function f(x) = x^3 - 6x^2 + 17x - 5 -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 17*x - 5

/-- Theorem: If f(a) = 3 and f(b) = 23, then a + b = 4 -/
theorem sum_of_roots (a b : ℝ) (ha : f a = 3) (hb : f b = 23) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1766_176696


namespace NUMINAMATH_CALUDE_product_equality_l1766_176646

theorem product_equality : 50 * 29.96 * 2.996 * 500 = 2244004 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1766_176646


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l1766_176675

theorem solution_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, 2 * x + k = 3) → 
  (2 * 1 + k = 3) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l1766_176675


namespace NUMINAMATH_CALUDE_original_group_size_l1766_176654

/-- Proves that the original number of men in a group is 12, given the conditions of the problem -/
theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 8 →
  absent_men = 3 →
  final_days = 10 →
  ∃ (original_men : ℕ),
    original_men > 0 ∧
    (original_men : ℚ) / initial_days = (original_men - absent_men : ℚ) / final_days ∧
    original_men = 12 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l1766_176654


namespace NUMINAMATH_CALUDE_tree_height_when_boy_grows_l1766_176651

-- Define the problem parameters
def initial_tree_height : ℝ := 16
def initial_boy_height : ℝ := 24
def final_boy_height : ℝ := 36

-- Define the growth rate relationship
def tree_growth_rate (boy_growth : ℝ) : ℝ := 2 * boy_growth

-- Theorem statement
theorem tree_height_when_boy_grows (boy_growth : ℝ) 
  (h : final_boy_height = initial_boy_height + boy_growth) :
  initial_tree_height + tree_growth_rate boy_growth = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_tree_height_when_boy_grows_l1766_176651


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1766_176622

theorem inequality_solution_set :
  ∀ x : ℝ, (((2*x - 1) / (x + 1) ≤ 1 ∧ x + 1 ≠ 0) ↔ x ∈ Set.Ioo (-1 : ℝ) 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1766_176622


namespace NUMINAMATH_CALUDE_cube_plus_n_minus_two_power_of_two_l1766_176667

theorem cube_plus_n_minus_two_power_of_two (n : ℕ+) :
  (∃ k : ℕ, (n : ℕ)^3 + n - 2 = 2^k) ↔ n = 2 ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_n_minus_two_power_of_two_l1766_176667


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1766_176600

/-- Given that the solution set of ax^2 + 5x - 2 > 0 is {x | 1/2 < x < 2},
    prove that a = -2 and the solution set of ax^2 - 5x + a^2 - 1 > 0 is {x | -3 < x < 1/2} -/
theorem quadratic_inequality_problem (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (a = -2 ∧ 
   ∀ x : ℝ, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1766_176600


namespace NUMINAMATH_CALUDE_division_problem_l1766_176627

theorem division_problem : ∃ (D : ℕ+) (N : ℤ), 
  N = 5 * D.val ∧ N % 11 = 2 ∧ D = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1766_176627


namespace NUMINAMATH_CALUDE_S_formula_l1766_176616

/-- g(k) is the largest odd factor of k -/
def g (k : ℕ+) : ℕ+ :=
  sorry

/-- Sn is the sum of g(k) for k from 1 to 2^n -/
def S (n : ℕ) : ℚ :=
  sorry

/-- The main theorem: Sn = (1/3)(4^n + 2) for all natural numbers n -/
theorem S_formula (n : ℕ) : S n = (1/3) * (4^n + 2) :=
  sorry

end NUMINAMATH_CALUDE_S_formula_l1766_176616


namespace NUMINAMATH_CALUDE_multiplicative_inverse_of_3_mod_47_l1766_176639

theorem multiplicative_inverse_of_3_mod_47 : ∃ x : ℕ, x < 47 ∧ (3 * x) % 47 = 1 :=
by
  use 16
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_of_3_mod_47_l1766_176639
