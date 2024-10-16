import Mathlib

namespace NUMINAMATH_CALUDE_sin_239_deg_l3575_357560

theorem sin_239_deg (a : ℝ) (h : Real.cos (31 * π / 180) = a) : 
  Real.sin (239 * π / 180) = -a := by
  sorry

end NUMINAMATH_CALUDE_sin_239_deg_l3575_357560


namespace NUMINAMATH_CALUDE_average_equals_one_l3575_357517

theorem average_equals_one (x : ℝ) : 
  (5 + (-1) + (-2) + x) / 4 = 1 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_average_equals_one_l3575_357517


namespace NUMINAMATH_CALUDE_extremum_condition_l3575_357580

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_condition (a b : ℝ) : 
  (f a b 1 = 10) ∧ (f_derivative a b 1 = 0) → (a = 4 ∧ b = -11) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l3575_357580


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l3575_357514

theorem cubic_expansion_coefficient (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l3575_357514


namespace NUMINAMATH_CALUDE_rahim_average_price_per_book_l3575_357561

/-- The average price per book given two purchases -/
def average_price_per_book (books1 : ℕ) (cost1 : ℕ) (books2 : ℕ) (cost2 : ℕ) : ℚ :=
  (cost1 + cost2) / (books1 + books2)

/-- Theorem: The average price per book for Rahim's purchases is 85 -/
theorem rahim_average_price_per_book :
  average_price_per_book 65 6500 35 2000 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rahim_average_price_per_book_l3575_357561


namespace NUMINAMATH_CALUDE_immigrants_calculation_l3575_357528

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := 106491

/-- The number of people who immigrated to the country last year -/
def immigrants : ℕ := total_new_people - people_born

theorem immigrants_calculation :
  immigrants = 16320 := by
  sorry

end NUMINAMATH_CALUDE_immigrants_calculation_l3575_357528


namespace NUMINAMATH_CALUDE_circles_intersect_l3575_357557

/-- Two circles are intersecting if the distance between their centers is greater than the absolute 
    difference of their radii and less than the sum of their radii. -/
def are_intersecting (r1 r2 d : ℝ) : Prop :=
  d > |r1 - r2| ∧ d < r1 + r2

/-- Given two circles with radii 5 and 8, and distance between centers 8, 
    prove that they are intersecting. -/
theorem circles_intersect : are_intersecting 5 8 8 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3575_357557


namespace NUMINAMATH_CALUDE_cost_to_fill_displays_l3575_357525

/-- Represents the inventory and pricing of a jewelry store -/
structure JewelryStore where
  necklace_capacity : ℕ
  current_necklaces : ℕ
  ring_capacity : ℕ
  current_rings : ℕ
  bracelet_capacity : ℕ
  current_bracelets : ℕ
  necklace_price : ℕ
  ring_price : ℕ
  bracelet_price : ℕ

/-- Calculates the total cost to fill all displays in the jewelry store -/
def total_cost_to_fill (store : JewelryStore) : ℕ :=
  ((store.necklace_capacity - store.current_necklaces) * store.necklace_price) +
  ((store.ring_capacity - store.current_rings) * store.ring_price) +
  ((store.bracelet_capacity - store.current_bracelets) * store.bracelet_price)

/-- Theorem stating that the total cost to fill all displays is $183 -/
theorem cost_to_fill_displays (store : JewelryStore) 
  (h1 : store.necklace_capacity = 12)
  (h2 : store.current_necklaces = 5)
  (h3 : store.ring_capacity = 30)
  (h4 : store.current_rings = 18)
  (h5 : store.bracelet_capacity = 15)
  (h6 : store.current_bracelets = 8)
  (h7 : store.necklace_price = 4)
  (h8 : store.ring_price = 10)
  (h9 : store.bracelet_price = 5) :
  total_cost_to_fill store = 183 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_fill_displays_l3575_357525


namespace NUMINAMATH_CALUDE_green_candy_pieces_l3575_357562

theorem green_candy_pieces (total red blue : ℝ) (h1 : total = 3409.7) (h2 : red = 145.5) (h3 : blue = 785.2) :
  total - red - blue = 2479 := by
  sorry

end NUMINAMATH_CALUDE_green_candy_pieces_l3575_357562


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l3575_357559

theorem polynomial_factorization_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 - 1)) :
  a₁*d₁ + a₂*d₂ + a₃*d₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l3575_357559


namespace NUMINAMATH_CALUDE_intersection_perpendicular_line_max_distance_to_origin_l3575_357553

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y - 3 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line3 (x y : ℝ) : Prop := 3*x + y - 1 = 0

-- Define the general form of line l
def line_l (m x y : ℝ) : Prop := m*x + y - 2*(m+1) = 0

-- Part I
theorem intersection_perpendicular_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, line1 x y ∧ line2 x y → a*x + b*y + c = 0) ∧
    (∀ x y : ℝ, (a*x + b*y + c = 0) → (3*a + b = 0)) ∧
    (a = 5 ∧ b = -15 ∧ c = -18) :=
sorry

-- Part II
theorem max_distance_to_origin :
  ∃ (d : ℝ), 
    (∀ m x y : ℝ, line_l m x y → (x^2 + y^2 ≤ d^2)) ∧
    (∃ m x y : ℝ, line_l m x y ∧ x^2 + y^2 = d^2) ∧
    (d = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_line_max_distance_to_origin_l3575_357553


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l3575_357513

/-- Given a committee meeting with only associate and assistant professors, where:
    - Each associate professor brings 2 pencils and 1 chart
    - Each assistant professor brings 1 pencil and 2 charts
    - A total of 10 pencils and 11 charts are brought to the meeting
    Prove that the total number of people present is 7. -/
theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 10 →
    associate_profs + 2 * assistant_profs = 11 →
    associate_profs + assistant_profs = 7 :=
by sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l3575_357513


namespace NUMINAMATH_CALUDE_part1_part2_l3575_357595

-- Define the points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-2, 3)
def C : ℝ × ℝ := (8, -5)

-- Define vectors
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Part 1
theorem part1 (x y : ℝ) : 
  OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) → x = 2 ∧ y = -3 := by sorry

-- Part 2
theorem part2 (m : ℝ) :
  ∃ (k : ℝ), k ≠ 0 ∧ AB = (k * (m * OA.1 + OC.1), k * (m * OA.2 + OC.2)) → m = 1 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3575_357595


namespace NUMINAMATH_CALUDE_fraction_calculation_l3575_357541

theorem fraction_calculation : 
  (((4 : ℚ) / 9 + (1 : ℚ) / 9) / ((5 : ℚ) / 8 - (1 : ℚ) / 8)) = (10 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3575_357541


namespace NUMINAMATH_CALUDE_nate_cooking_for_eight_l3575_357502

/-- The number of scallops per pound -/
def scallops_per_pound : ℕ := 8

/-- The cost of scallops per pound in cents -/
def cost_per_pound : ℕ := 2400

/-- The number of scallops per person -/
def scallops_per_person : ℕ := 2

/-- The total cost of scallops Nate is spending in cents -/
def total_cost : ℕ := 4800

/-- The number of people Nate is cooking for -/
def number_of_people : ℕ := total_cost / cost_per_pound * scallops_per_pound / scallops_per_person

theorem nate_cooking_for_eight : number_of_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_nate_cooking_for_eight_l3575_357502


namespace NUMINAMATH_CALUDE_min_value_is_four_l3575_357570

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  d : ℚ
  hd : d ≠ 0
  ha1 : a 1 = 1
  hGeometric : (a 3) ^ 2 = (a 1) * (a 13)
  hArithmetic : ∀ n : ℕ+, a n = a 1 + (n - 1) * d

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The expression to be minimized -/
def f (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (2 * S seq n + 16) / (seq.a n + 3)

/-- Theorem stating the minimum value of the expression -/
theorem min_value_is_four (seq : ArithmeticSequence) :
  ∃ n₀ : ℕ+, ∀ n : ℕ+, f seq n ≥ f seq n₀ ∧ f seq n₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_four_l3575_357570


namespace NUMINAMATH_CALUDE_number_difference_l3575_357563

theorem number_difference (large small : ℕ) (x : ℕ) : 
  large = 2 * small + x →
  large + small = 27 →
  large = 19 →
  large - 2 * small = 3 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3575_357563


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l3575_357544

theorem algebraic_expression_simplification (x y : ℝ) :
  3 * (x^2 - 2*x*y + y^2) - 3 * (x^2 - 2*x*y + y^2 - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l3575_357544


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l3575_357518

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.sqrt x > x

-- Theorem to prove
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l3575_357518


namespace NUMINAMATH_CALUDE_vins_bike_trips_l3575_357542

theorem vins_bike_trips (distance_to_school : ℕ) (distance_from_school : ℕ) (total_distance : ℕ) :
  distance_to_school = 6 →
  distance_from_school = 7 →
  total_distance = 65 →
  total_distance / (distance_to_school + distance_from_school) = 5 := by
sorry

end NUMINAMATH_CALUDE_vins_bike_trips_l3575_357542


namespace NUMINAMATH_CALUDE_perpendicular_vector_implies_y_coord_l3575_357524

/-- Given two points A and B, and a vector a, if AB is perpendicular to a, 
    then the y-coordinate of B is -4. -/
theorem perpendicular_vector_implies_y_coord (A B : ℝ × ℝ) (a : ℝ × ℝ) : 
  A = (-1, 2) → 
  B.1 = 2 → 
  a = (2, 1) → 
  (B.1 - A.1, B.2 - A.2) • a = 0 → 
  B.2 = -4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vector_implies_y_coord_l3575_357524


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3575_357589

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 4 = (x + a)^2) → 
  (m = 3 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3575_357589


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l3575_357520

/-- The number to be multiplied -/
def n : ℕ := 54

/-- The incorrect multiplier -/
def incorrect_multiplier : ℚ := 2.35

/-- The difference between correct and incorrect results -/
def difference : ℚ := 1.8

/-- The two-digit number formed by the repeating digits -/
def repeating_digits : ℕ := 35

/-- The correct multiplier as a rational number -/
def correct_multiplier : ℚ := 2 + repeating_digits / 99

theorem repeating_decimal_problem :
  n * correct_multiplier - n * incorrect_multiplier = difference :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l3575_357520


namespace NUMINAMATH_CALUDE_coffee_packages_solution_l3575_357568

/-- Represents the number of 10-ounce packages -/
def num_10oz : ℕ := 4

/-- Represents the number of 5-ounce packages -/
def num_5oz : ℕ := num_10oz + 2

/-- Total ounces of coffee -/
def total_ounces : ℕ := 115

/-- Cost of a 5-ounce package in cents -/
def cost_5oz : ℕ := 150

/-- Cost of a 10-ounce package in cents -/
def cost_10oz : ℕ := 250

/-- Maximum total cost in cents -/
def max_cost : ℕ := 2000

theorem coffee_packages_solution :
  (num_10oz * 10 + num_5oz * 5 = total_ounces) ∧
  (num_10oz * cost_10oz + num_5oz * cost_5oz ≤ max_cost) :=
by sorry

end NUMINAMATH_CALUDE_coffee_packages_solution_l3575_357568


namespace NUMINAMATH_CALUDE_charlie_golden_delicious_l3575_357591

/-- The number of bags of Golden Delicious apples Charlie picked -/
def golden_delicious : ℝ :=
  0.67 - (0.17 + 0.33)

theorem charlie_golden_delicious :
  golden_delicious = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_charlie_golden_delicious_l3575_357591


namespace NUMINAMATH_CALUDE_program_output_l3575_357547

def program (a b : ℤ) : ℤ :=
  if a > b then a else b

theorem program_output : program 2 3 = 3 := by sorry

end NUMINAMATH_CALUDE_program_output_l3575_357547


namespace NUMINAMATH_CALUDE_tamil_speakers_l3575_357508

theorem tamil_speakers (total_population : ℕ) (english_speakers : ℕ) (both_speakers : ℕ) (hindi_probability : ℚ) : 
  total_population = 1024 →
  english_speakers = 562 →
  both_speakers = 346 →
  hindi_probability = 0.0859375 →
  ∃ tamil_speakers : ℕ, tamil_speakers = 720 ∧ 
    tamil_speakers = total_population - (english_speakers + (total_population * hindi_probability).floor - both_speakers) :=
by
  sorry

end NUMINAMATH_CALUDE_tamil_speakers_l3575_357508


namespace NUMINAMATH_CALUDE_min_value_abc_l3575_357535

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ 
  (a + 3 * b + 9 * c = 27 ↔ a = 9 ∧ b = 3 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l3575_357535


namespace NUMINAMATH_CALUDE_cook_selection_ways_l3575_357556

theorem cook_selection_ways (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_cook_selection_ways_l3575_357556


namespace NUMINAMATH_CALUDE_nearest_integer_to_cube_root_five_sixth_power_l3575_357533

theorem nearest_integer_to_cube_root_five_sixth_power :
  ∃ (n : ℕ), n = 74608 ∧ ∀ (m : ℕ), |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_cube_root_five_sixth_power_l3575_357533


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3575_357567

theorem cos_2alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = -Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3575_357567


namespace NUMINAMATH_CALUDE_men_working_count_l3575_357581

/-- Represents the amount of work done by one person in one hour -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  size : ℕ
  work_rate : WorkRate
  days : ℕ
  hours_per_day : ℕ

/-- The total work done by a group is the product of their size, work rate, days, and hours per day -/
def total_work (group : WorkGroup) : ℝ :=
  group.size * group.work_rate.rate * group.days * group.hours_per_day

/-- Given the conditions of the problem, prove that the number of men working is 15 -/
theorem men_working_count (men_group women_group : WorkGroup) :
  men_group.days = 21 →
  men_group.hours_per_day = 8 →
  women_group.size = 21 →
  women_group.days = 60 →
  women_group.hours_per_day = 3 →
  3 * women_group.work_rate.rate = 2 * men_group.work_rate.rate →
  total_work men_group = total_work women_group →
  men_group.size = 15 := by
  sorry

end NUMINAMATH_CALUDE_men_working_count_l3575_357581


namespace NUMINAMATH_CALUDE_alex_friends_cookout_l3575_357550

theorem alex_friends_cookout (burgers_per_guest : ℕ) (buns_per_pack : ℕ) (packs_of_buns : ℕ) 
  (h1 : burgers_per_guest = 3)
  (h2 : buns_per_pack = 8)
  (h3 : packs_of_buns = 3) :
  ∃ (friends : ℕ), friends = 9 ∧ 
    (packs_of_buns * buns_per_pack) / burgers_per_guest + 1 = friends :=
by
  sorry

end NUMINAMATH_CALUDE_alex_friends_cookout_l3575_357550


namespace NUMINAMATH_CALUDE_polycarp_kolka_numbers_l3575_357526

/-- The smallest 5-digit number composed of distinct even digits -/
def polycarp_number : ℕ := 20468

/-- Kolka's incorrect 5-digit number -/
def kolka_number : ℕ := 20486

/-- Checks if a number is a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Checks if a number is composed of distinct even digits -/
def has_distinct_even_digits (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    Even a ∧ Even b ∧ Even c ∧ Even d ∧ Even e

theorem polycarp_kolka_numbers :
  (is_five_digit polycarp_number) ∧
  (has_distinct_even_digits polycarp_number) ∧
  (∀ n : ℕ, is_five_digit n → has_distinct_even_digits n → n ≥ polycarp_number) ∧
  (is_five_digit kolka_number) ∧
  (has_distinct_even_digits kolka_number) ∧
  (kolka_number - polycarp_number < 100) ∧
  (kolka_number ≠ polycarp_number) →
  kolka_number = 20486 :=
by sorry

end NUMINAMATH_CALUDE_polycarp_kolka_numbers_l3575_357526


namespace NUMINAMATH_CALUDE_sequence_properties_l3575_357522

/-- Given a sequence and its partial sum satisfying certain conditions, 
    prove that it's geometric and find the range of t when the sum converges to 1 -/
theorem sequence_properties (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (t : ℝ) 
    (h1 : ∀ n : ℕ+, S n = 1 + t * a n) 
    (h2 : t ≠ 1) (h3 : t ≠ 0) :
  (∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n) ∧ 
  (∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |S n - 1| < ε) → 
  (t < 1/2 ∧ t ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3575_357522


namespace NUMINAMATH_CALUDE_domain_intersection_subset_l3575_357566

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def C (m : ℝ) : Set ℝ := {x | 3*x < 2*m - 1}

-- State the theorem
theorem domain_intersection_subset (m : ℝ) : 
  (A ∩ B) ⊆ C m → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_domain_intersection_subset_l3575_357566


namespace NUMINAMATH_CALUDE_square_difference_l3575_357578

theorem square_difference (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3575_357578


namespace NUMINAMATH_CALUDE_opposite_point_exists_l3575_357501

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define diametrically opposite points
def DiametricallyOpposite (c : Circle) (p q : ℝ × ℝ) : Prop :=
  PointOnCircle c p ∧ PointOnCircle c q ∧
  (p.1 - c.center.1) = -(q.1 - c.center.1) ∧
  (p.2 - c.center.2) = -(q.2 - c.center.2)

-- Theorem statement
theorem opposite_point_exists (c : Circle) (A₁ : ℝ × ℝ) 
  (h : PointOnCircle c A₁) : 
  ∃ B₂ : ℝ × ℝ, DiametricallyOpposite c A₁ B₂ := by
  sorry

end NUMINAMATH_CALUDE_opposite_point_exists_l3575_357501


namespace NUMINAMATH_CALUDE_pencil_count_original_pencils_count_l3575_357594

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := sorry

/-- The number of pencils Tim added to the drawer -/
def added_pencils : ℕ := 3

/-- The total number of pencils in the drawer after Tim added some -/
def total_pencils : ℕ := 5

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by sorry

/-- Theorem proving that the original number of pencils in the drawer was 2 -/
theorem original_pencils_count : original_pencils = 2 := by sorry

end NUMINAMATH_CALUDE_pencil_count_original_pencils_count_l3575_357594


namespace NUMINAMATH_CALUDE_binomial_2000_3_l3575_357564

theorem binomial_2000_3 : Nat.choose 2000 3 = 1331000333 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2000_3_l3575_357564


namespace NUMINAMATH_CALUDE_hexagon_smallest_angle_l3575_357529

-- Define a hexagon with angles in arithmetic progression
def hexagon_angles (x : ℝ) : List ℝ := [x, x + 10, x + 20, x + 30, x + 40, x + 50]

-- Theorem statement
theorem hexagon_smallest_angle :
  ∃ (x : ℝ), 
    (List.sum (hexagon_angles x) = 720) ∧ 
    (∀ (angle : ℝ), angle ∈ hexagon_angles x → angle ≥ x) ∧
    x = 95 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_smallest_angle_l3575_357529


namespace NUMINAMATH_CALUDE_quadratic_trinomial_existence_l3575_357592

theorem quadratic_trinomial_existence : ∃ f : ℝ → ℝ, 
  (∀ x, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧ 
  f 2014 = 2015 ∧ 
  f 2015 = 0 ∧ 
  f 2016 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_existence_l3575_357592


namespace NUMINAMATH_CALUDE_rational_cube_equality_l3575_357593

theorem rational_cube_equality (a b c : ℚ) 
  (eq1 : (a^2 + 1)^3 = b + 1)
  (eq2 : (b^2 + 1)^3 = c + 1)
  (eq3 : (c^2 + 1)^3 = a + 1) :
  a = 0 ∧ b = 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_rational_cube_equality_l3575_357593


namespace NUMINAMATH_CALUDE_morning_campers_l3575_357510

theorem morning_campers (total : ℕ) (afternoon : ℕ) (morning : ℕ) : 
  total = 62 → afternoon = 27 → morning = total - afternoon → morning = 35 := by
  sorry

end NUMINAMATH_CALUDE_morning_campers_l3575_357510


namespace NUMINAMATH_CALUDE_c_investment_amount_l3575_357583

/-- A business partnership between C and D -/
structure Business where
  c_investment : ℕ
  d_investment : ℕ
  total_profit : ℕ
  d_profit_share : ℕ

/-- The business scenario as described in the problem -/
def scenario : Business where
  c_investment := 0  -- Unknown, to be proved
  d_investment := 1500
  total_profit := 500
  d_profit_share := 100

/-- Theorem stating C's investment amount -/
theorem c_investment_amount (b : Business) (h1 : b = scenario) :
  b.c_investment = 6000 := by
  sorry

end NUMINAMATH_CALUDE_c_investment_amount_l3575_357583


namespace NUMINAMATH_CALUDE_solution_product_l3575_357527

theorem solution_product (p q : ℝ) : 
  (p - 3) * (3 * p + 8) = p^2 - 5*p + 6 →
  (q - 3) * (3 * q + 8) = q^2 - 5*q + 6 →
  (p + 4) * (q + 4) = 7 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l3575_357527


namespace NUMINAMATH_CALUDE_exactly_three_statements_true_l3575_357539

-- Define the polyline distance function
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define points A, B, M, and N
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Statement 1
def statement_1 : Prop :=
  polyline_distance A.1 A.2 B.1 B.2 = 5

-- Statement 2
def statement_2 : Prop :=
  ∃ (S : Set (ℝ × ℝ)), S = {p : ℝ × ℝ | polyline_distance p.1 p.2 0 0 = 1} ∧
  ¬(∃ (center : ℝ × ℝ) (radius : ℝ), S = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2})

-- Statement 3
def statement_3 : Prop :=
  ∀ (C : ℝ × ℝ), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →
    polyline_distance A.1 A.2 C.1 C.2 + polyline_distance C.1 C.2 B.1 B.2 = polyline_distance A.1 A.2 B.1 B.2

-- Statement 4
def statement_4 : Prop :=
  {p : ℝ × ℝ | polyline_distance p.1 p.2 M.1 M.2 = polyline_distance p.1 p.2 N.1 N.2} =
  {p : ℝ × ℝ | p.1 = 0}

-- Main theorem
theorem exactly_three_statements_true :
  (statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ statement_4) := by sorry

end NUMINAMATH_CALUDE_exactly_three_statements_true_l3575_357539


namespace NUMINAMATH_CALUDE_comparison_theorem_l3575_357548

theorem comparison_theorem :
  (-3/4 : ℚ) > -4/5 ∧ (3 : ℝ) > Real.rpow 9 (1/3) := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l3575_357548


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3575_357519

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₁ + a₃ + a₅ = -121 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3575_357519


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_8_11_24_l3575_357534

theorem smallest_number_divisible_by_8_11_24 :
  ∃ (k : ℕ), 255 + k > 255 ∧ (255 + k) % 8 = 0 ∧ (255 + k) % 11 = 0 ∧ (255 + k) % 24 = 0 ∧
  ∀ (n : ℕ), n < 255 → ¬∃ (m : ℕ), m > 0 ∧ (n + m) % 8 = 0 ∧ (n + m) % 11 = 0 ∧ (n + m) % 24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_8_11_24_l3575_357534


namespace NUMINAMATH_CALUDE_pages_ratio_is_one_to_two_l3575_357573

-- Define the total number of pages in the book
def total_pages : ℕ := 120

-- Define the number of pages read yesterday
def pages_read_yesterday : ℕ := 12

-- Define the number of pages read today
def pages_read_today : ℕ := 2 * pages_read_yesterday

-- Define the number of pages to be read tomorrow
def pages_to_read_tomorrow : ℕ := 42

-- Theorem statement
theorem pages_ratio_is_one_to_two :
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  (pages_to_read_tomorrow : ℚ) / remaining_pages = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_pages_ratio_is_one_to_two_l3575_357573


namespace NUMINAMATH_CALUDE_function_behavior_l3575_357586

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_increasing : ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
variable (h_f7 : f 7 = 6)

-- State the theorem
theorem function_behavior :
  (∀ x y, -7 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f y ≤ f x) ∧
  (∀ x, -7 ≤ x ∧ x ≤ 7 → f x ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_function_behavior_l3575_357586


namespace NUMINAMATH_CALUDE_consecutive_integers_base_equation_l3575_357558

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

theorem consecutive_integers_base_equation :
  ∀ C D : ℕ,
  C > 0 →
  D = C + 1 →
  toBase10 154 C + toBase10 52 D = toBase10 76 (C + D) →
  C + D = 11 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_equation_l3575_357558


namespace NUMINAMATH_CALUDE_investment_balance_l3575_357571

/-- Proves that given an initial investment of 1800 at 7% interest, an additional investment of 1800 at 10% interest will result in a total annual income equal to 8.5% of the entire investment. -/
theorem investment_balance (initial_investment : ℝ) (additional_investment : ℝ) 
  (initial_rate : ℝ) (additional_rate : ℝ) (total_rate : ℝ) : 
  initial_investment = 1800 →
  additional_investment = 1800 →
  initial_rate = 0.07 →
  additional_rate = 0.10 →
  total_rate = 0.085 →
  initial_rate * initial_investment + additional_rate * additional_investment = 
    total_rate * (initial_investment + additional_investment) :=
by sorry

end NUMINAMATH_CALUDE_investment_balance_l3575_357571


namespace NUMINAMATH_CALUDE_no_family_of_lines_exist_l3575_357515

theorem no_family_of_lines_exist :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧ 
    (∀ n, k n * k (n + 1) ≥ 0) ∧ 
    (∀ n, k n ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_family_of_lines_exist_l3575_357515


namespace NUMINAMATH_CALUDE_train_speed_train_speed_problem_l3575_357565

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed - man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- Theorem stating that the train's speed is approximately 67 km/h given the problem conditions -/
theorem train_speed_problem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_speed 120 6 5 - 67| < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_problem_l3575_357565


namespace NUMINAMATH_CALUDE_greatest_negative_root_of_equation_l3575_357512

open Real

theorem greatest_negative_root_of_equation :
  ∃ (x : ℝ), x = -7/6 ∧ 
  (sin (π * x) - cos (2 * π * x)) / ((sin (π * x) + 1)^2 + cos (π * x)^2) = 0 ∧
  (∀ y < 0, y > x → 
    (sin (π * y) - cos (2 * π * y)) / ((sin (π * y) + 1)^2 + cos (π * y)^2) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_negative_root_of_equation_l3575_357512


namespace NUMINAMATH_CALUDE_jake_peaches_l3575_357579

theorem jake_peaches (steven_peaches jill_peaches jake_peaches : ℕ) : 
  jake_peaches + 7 = steven_peaches → 
  steven_peaches = jill_peaches + 14 → 
  steven_peaches = 15 → 
  jake_peaches = 8 := by
sorry

end NUMINAMATH_CALUDE_jake_peaches_l3575_357579


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l3575_357552

/-- The probability that all three quitters are from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (n : ℕ) (k : ℕ) (q : ℕ) : 
  n = 20 → -- Total number of contestants
  k = 10 → -- Number of contestants in each tribe
  q = 3 →  -- Number of quitters
  (n = 2 * k) → -- Two equally sized tribes
  (Fintype.card {s : Finset (Fin n) // s.card = q ∧ (∀ i ∈ s, i < k) } +
   Fintype.card {s : Finset (Fin n) // s.card = q ∧ (∀ i ∈ s, k ≤ i) }) /
  Fintype.card {s : Finset (Fin n) // s.card = q} = 20 / 95 :=
by sorry


end NUMINAMATH_CALUDE_survivor_quitters_probability_l3575_357552


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3575_357530

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3575_357530


namespace NUMINAMATH_CALUDE_no_base_441_cube_and_fourth_power_l3575_357584

theorem no_base_441_cube_and_fourth_power :
  ¬ ∃ (a : ℕ), a > 4 ∧
    (∃ (n : ℕ), 4 * a^2 + 4 * a + 1 = n^3) ∧
    (∃ (m : ℕ), 4 * a^2 + 4 * a + 1 = m^4) := by
  sorry

end NUMINAMATH_CALUDE_no_base_441_cube_and_fourth_power_l3575_357584


namespace NUMINAMATH_CALUDE_f_of_three_equals_nine_l3575_357572

theorem f_of_three_equals_nine (f : ℝ → ℝ) (h : ∀ x, f x = x^2) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_nine_l3575_357572


namespace NUMINAMATH_CALUDE_decagon_interior_exterior_angle_sum_l3575_357554

theorem decagon_interior_exterior_angle_sum (n : ℕ) : 
  (n - 2) * 180 = 4 * 360 ↔ n = 10 :=
sorry

end NUMINAMATH_CALUDE_decagon_interior_exterior_angle_sum_l3575_357554


namespace NUMINAMATH_CALUDE_circle_area_after_folding_l3575_357587

theorem circle_area_after_folding (original_area : ℝ) (sector_area : ℝ) : 
  sector_area = 5 → original_area / 64 = sector_area → original_area = 320 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_after_folding_l3575_357587


namespace NUMINAMATH_CALUDE_vector_collinearity_l3575_357585

/-- Given vectors a and b, prove that k makes k*a + b collinear with a - 3*b -/
theorem vector_collinearity (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 2) →
  b = (-3, 2) →
  k = -1/3 →
  ∃ (t : ℝ), t • (k • a + b) = a - 3 • b := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3575_357585


namespace NUMINAMATH_CALUDE_custom_product_equals_interval_l3575_357545

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - x^2)}
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define the custom Cartesian product
def custom_product (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem custom_product_equals_interval :
  custom_product A B = {x : ℝ | 0 ≤ x ∧ x ≤ 1 ∨ x > 2} :=
sorry

end NUMINAMATH_CALUDE_custom_product_equals_interval_l3575_357545


namespace NUMINAMATH_CALUDE_degree_of_sum_polynomials_l3575_357588

-- Define the polynomials f and g
def f (z : ℂ) (c₃ c₂ c₁ c₀ : ℂ) : ℂ := c₃ * z^3 + c₂ * z^2 + c₁ * z + c₀
def g (z : ℂ) (d₂ d₁ d₀ : ℂ) : ℂ := d₂ * z^2 + d₁ * z + d₀

-- Define the degree of a polynomial
def degree (p : ℂ → ℂ) : ℕ := sorry

-- Theorem statement
theorem degree_of_sum_polynomials 
  (c₃ c₂ c₁ c₀ d₂ d₁ d₀ : ℂ) 
  (h₁ : c₃ ≠ 0) 
  (h₂ : d₂ ≠ 0) 
  (h₃ : c₃ + d₂ ≠ 0) : 
  degree (fun z ↦ f z c₃ c₂ c₁ c₀ + g z d₂ d₁ d₀) = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_sum_polynomials_l3575_357588


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3575_357521

-- Define the function g
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (2 / (x + 3))
       else if x < -3 then Int.floor (2 / (x + 3))
       else 0  -- arbitrary value for x = -3, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3575_357521


namespace NUMINAMATH_CALUDE_bus_capacity_l3575_357509

theorem bus_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let regular_seat_capacity : ℕ := 3
  let back_seat_capacity : ℕ := 12
  let total_regular_seats : ℕ := left_seats + right_seats
  let regular_seats_capacity : ℕ := total_regular_seats * regular_seat_capacity
  let total_capacity : ℕ := regular_seats_capacity + back_seat_capacity
  total_capacity = 93 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l3575_357509


namespace NUMINAMATH_CALUDE_laptop_repair_cost_laptop_repair_cost_proof_l3575_357511

/-- The cost of a laptop repair given the following conditions:
  * Phone repair costs $11
  * Computer repair costs $18
  * 5 phone repairs, 2 laptop repairs, and 2 computer repairs were performed
  * Total earnings were $121
-/
theorem laptop_repair_cost : ℕ :=
  let phone_cost : ℕ := 11
  let computer_cost : ℕ := 18
  let phone_repairs : ℕ := 5
  let laptop_repairs : ℕ := 2
  let computer_repairs : ℕ := 2
  let total_earnings : ℕ := 121
  15

theorem laptop_repair_cost_proof :
  (let phone_cost : ℕ := 11
   let computer_cost : ℕ := 18
   let phone_repairs : ℕ := 5
   let laptop_repairs : ℕ := 2
   let computer_repairs : ℕ := 2
   let total_earnings : ℕ := 121
   laptop_repair_cost = 15) :=
by sorry

end NUMINAMATH_CALUDE_laptop_repair_cost_laptop_repair_cost_proof_l3575_357511


namespace NUMINAMATH_CALUDE_min_group_size_repunit_sum_l3575_357504

def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ n = (10^k - 1) / 9

theorem min_group_size_repunit_sum :
  ∃ m : ℕ, m > 1 ∧
    (∀ m' : ℕ, m' > 1 → m' < m →
      ¬∃ n k : ℕ, n > k ∧ k > 1 ∧
        is_repunit n ∧ is_repunit k ∧ n = k * m') ∧
    (∃ n k : ℕ, n > k ∧ k > 1 ∧
      is_repunit n ∧ is_repunit k ∧ n = k * m) ∧
  m = 101 :=
sorry

end NUMINAMATH_CALUDE_min_group_size_repunit_sum_l3575_357504


namespace NUMINAMATH_CALUDE_largest_abab_divisible_by_14_l3575_357516

/-- Represents a four-digit number of the form abab -/
def IsAbabForm (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * a + b

/-- Checks if a number is the product of a two-digit and a three-digit number -/
def IsProductOfTwoAndThreeDigit (n : ℕ) : Prop :=
  ∃ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ n = x * y

/-- The main theorem stating the largest four-digit number of the form abab
    that is divisible by 14 and a product of two-digit and three-digit numbers -/
theorem largest_abab_divisible_by_14 :
  ∀ A : ℕ,
  IsAbabForm A →
  IsProductOfTwoAndThreeDigit A →
  A % 14 = 0 →
  A ≤ 9898 :=
by sorry

end NUMINAMATH_CALUDE_largest_abab_divisible_by_14_l3575_357516


namespace NUMINAMATH_CALUDE_immigrant_count_l3575_357575

/-- The number of people born in the country last year -/
def births : ℕ := 90171

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := 106491

/-- The number of immigrants to the country last year -/
def immigrants : ℕ := total_new_people - births

theorem immigrant_count : immigrants = 16320 := by
  sorry

end NUMINAMATH_CALUDE_immigrant_count_l3575_357575


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3575_357555

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 6 / Real.sqrt 13) = 
  (3 * Real.sqrt 10010) / 1001 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3575_357555


namespace NUMINAMATH_CALUDE_andrew_total_hours_l3575_357537

/-- Andrew's work on his Science report -/
def andrew_work : ℝ → ℝ → ℝ := fun days hours_per_day => days * hours_per_day

/-- The theorem stating the total hours Andrew worked -/
theorem andrew_total_hours : andrew_work 3 2.5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_total_hours_l3575_357537


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_three_sqrt_ten_l3575_357549

theorem complex_magnitude_equals_three_sqrt_ten (x : ℝ) :
  x > 0 → Complex.abs (-3 + x * Complex.I) = 3 * Real.sqrt 10 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_three_sqrt_ten_l3575_357549


namespace NUMINAMATH_CALUDE_digit_count_proof_l3575_357569

/-- The number of valid digits for each position after the first -/
def valid_digits : ℕ := 4

/-- The number of valid digits for the first position -/
def valid_first_digits : ℕ := 3

/-- The total count of numbers with the given properties -/
def total_count : ℕ := 192

/-- The number of digits in the numbers -/
def n : ℕ := 4

theorem digit_count_proof :
  valid_first_digits * valid_digits^(n - 1) = total_count :=
sorry

end NUMINAMATH_CALUDE_digit_count_proof_l3575_357569


namespace NUMINAMATH_CALUDE_sqrt_D_always_irrational_l3575_357503

-- Define the relationship between a and b as consecutive integers
def consecutive (a b : ℤ) : Prop := b = a + 1

-- Define D in terms of a and b
def D (a b : ℤ) : ℤ := a^2 + b^2 + (a + b)^2

-- Theorem statement
theorem sqrt_D_always_irrational (a b : ℤ) (h : consecutive a b) :
  Irrational (Real.sqrt (D a b)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_always_irrational_l3575_357503


namespace NUMINAMATH_CALUDE_triangle_uniqueness_l3575_357596

/-- Given two excircle radii and an altitude of a triangle, 
    the triangle is uniquely determined iff the altitude is not 
    equal to the harmonic mean of the two radii. -/
theorem triangle_uniqueness (ρa ρb mc : ℝ) (h_pos : ρa > 0 ∧ ρb > 0 ∧ mc > 0) :
  ∃! (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a + b > c ∧ b + c > a ∧ c + a > b) ∧
    (ρa = (a + b + c) / (2 * (b + c))) ∧
    (ρb = (a + b + c) / (2 * (c + a))) ∧
    (mc = 2 * (a * b * c) / ((a + b + c) * c)) ↔ 
  mc ≠ 2 * ρa * ρb / (ρa + ρb) := by
sorry

end NUMINAMATH_CALUDE_triangle_uniqueness_l3575_357596


namespace NUMINAMATH_CALUDE_tan_beta_calculation_l3575_357523

open Real

theorem tan_beta_calculation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : sin α = 4/5) (h4 : tan (α - β) = 2/3) : tan β = 6/17 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_calculation_l3575_357523


namespace NUMINAMATH_CALUDE_min_value_of_g_l3575_357599

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 4*x + 9

-- State the theorem
theorem min_value_of_g :
  ∀ x ∈ Set.Icc (-2 : ℝ) 0, g x ≥ 9 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 0, g y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3575_357599


namespace NUMINAMATH_CALUDE_geometric_sequence_converse_l3575_357577

/-- The converse of a proposition "If P, then Q" is "If Q, then P" -/
def converse_of (P Q : Prop) : Prop :=
  Q → P

/-- Three real numbers form a geometric sequence if the middle term 
    is the geometric mean of the other two -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- The proposition "If a, b, c form a geometric sequence, then b^2 = ac" 
    and its converse -/
theorem geometric_sequence_converse :
  converse_of (is_geometric_sequence a b c) (b^2 = a * c) =
  (b^2 = a * c → is_geometric_sequence a b c) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_converse_l3575_357577


namespace NUMINAMATH_CALUDE_base_c_sum_theorem_l3575_357546

/-- Represents a number in base c --/
structure BaseC (c : ℕ) where
  value : ℕ

/-- Multiplication in base c --/
def mul_base_c (c : ℕ) (x y : BaseC c) : BaseC c :=
  ⟨(x.value * y.value) % c⟩

/-- Addition in base c --/
def add_base_c (c : ℕ) (x y : BaseC c) : BaseC c :=
  ⟨(x.value + y.value) % c⟩

theorem base_c_sum_theorem (c : ℕ) 
  (h : mul_base_c c (mul_base_c c ⟨13⟩ ⟨18⟩) ⟨17⟩ = ⟨4357⟩) :
  add_base_c c (add_base_c c ⟨13⟩ ⟨18⟩) ⟨17⟩ = ⟨47⟩ := by
  sorry

end NUMINAMATH_CALUDE_base_c_sum_theorem_l3575_357546


namespace NUMINAMATH_CALUDE_power_sum_equality_l3575_357590

theorem power_sum_equality : (-1)^49 + 2^(4^3 + 3^2 - 7^2) = 16777215 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3575_357590


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3575_357598

theorem simplify_sqrt_expression :
  Real.sqrt 300 / Real.sqrt 75 - Real.sqrt 98 / Real.sqrt 49 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3575_357598


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3575_357540

-- Define a cube with 8 vertices
structure Cube :=
  (vertices : Fin 8 → ℝ)

-- Define the sum of numbers on a face
def face_sum (c : Cube) (v1 v2 v3 v4 : Fin 8) : ℝ :=
  c.vertices v1 + c.vertices v2 + c.vertices v3 + c.vertices v4

-- Define the sum of all face sums
def total_face_sum (c : Cube) : ℝ :=
  face_sum c 0 1 2 3 +
  face_sum c 0 1 4 5 +
  face_sum c 0 3 4 7 +
  face_sum c 1 2 5 6 +
  face_sum c 2 3 6 7 +
  face_sum c 4 5 6 7

-- Define the sum of all vertex values
def vertex_sum (c : Cube) : ℝ :=
  c.vertices 0 + c.vertices 1 + c.vertices 2 + c.vertices 3 +
  c.vertices 4 + c.vertices 5 + c.vertices 6 + c.vertices 7

-- Theorem statement
theorem cube_sum_theorem (c : Cube) :
  total_face_sum c = 2019 → vertex_sum c = 673 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3575_357540


namespace NUMINAMATH_CALUDE_fruit_purchase_problem_l3575_357576

/-- Fruit purchase problem -/
theorem fruit_purchase_problem (x y : ℝ) :
  let apple_weight : ℝ := 2
  let orange_weight : ℝ := 5 * apple_weight
  let total_weight : ℝ := apple_weight + orange_weight
  let total_cost : ℝ := x * apple_weight + y * orange_weight
  (orange_weight = 10 ∧ total_cost = 2*x + 10*y) ∧ total_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_problem_l3575_357576


namespace NUMINAMATH_CALUDE_visitors_not_enjoy_not_understand_l3575_357551

-- Define the total number of visitors
def V : ℕ := 560

-- Define the number of visitors who enjoyed the painting
def E : ℕ := (3 * V) / 4

-- Define the number of visitors who understood the painting
def U : ℕ := E

-- Theorem to prove
theorem visitors_not_enjoy_not_understand : V - E = 140 := by
  sorry

end NUMINAMATH_CALUDE_visitors_not_enjoy_not_understand_l3575_357551


namespace NUMINAMATH_CALUDE_firefighter_net_sag_l3575_357507

/-- The net sag for a person jumping onto a firefighter rescue net -/
def net_sag (m₁ m₂ h₁ h₂ x₁ : ℝ) (x₂ : ℝ) : Prop :=
  m₁ > 0 ∧ m₂ > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ x₁ > 0 ∧ x₂ > 0 ∧
  28 * x₂^2 - x₂ - 29 = 0

theorem firefighter_net_sag (m₁ m₂ h₁ h₂ x₁ : ℝ) (hm₁ : m₁ = 78.75) (hm₂ : m₂ = 45)
    (hh₁ : h₁ = 15) (hh₂ : h₂ = 29) (hx₁ : x₁ = 1) :
  ∃ x₂, net_sag m₁ m₂ h₁ h₂ x₁ x₂ :=
by sorry

end NUMINAMATH_CALUDE_firefighter_net_sag_l3575_357507


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3575_357506

/-- Represents the discount percentage on bulk photocopy orders -/
def discount_percentage : ℝ := 25

/-- Represents the regular cost per photocopy in dollars -/
def regular_cost_per_copy : ℝ := 0.02

/-- Represents the number of copies in a bulk order -/
def bulk_order_size : ℕ := 160

/-- Represents the individual savings when placing a bulk order -/
def individual_savings : ℝ := 0.40

/-- Represents the total savings when two people place a bulk order together -/
def total_savings : ℝ := 2 * individual_savings

/-- Proves that the discount percentage is correct given the problem conditions -/
theorem discount_percentage_proof :
  discount_percentage = (total_savings / (regular_cost_per_copy * bulk_order_size)) * 100 :=
by sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3575_357506


namespace NUMINAMATH_CALUDE_haunted_house_entry_exit_l3575_357543

theorem haunted_house_entry_exit (total_windows : ℕ) (magical_barrier : ℕ) : 
  total_windows = 8 →
  magical_barrier = 1 →
  (total_windows - magical_barrier - 1) * (total_windows - 2) + 
  magical_barrier * (total_windows - 1) = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_haunted_house_entry_exit_l3575_357543


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3575_357582

theorem triangle_angle_calculation (A B C : ℝ) :
  A + B + C = 180 →
  B = 4 * A →
  C - B = 27 →
  A = 17 ∧ B = 68 ∧ C = 95 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3575_357582


namespace NUMINAMATH_CALUDE_store_discount_income_increase_l3575_357500

theorem store_discount_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (quantity_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1)
  (h2 : quantity_increase_rate = 0.12)
  : (1 + quantity_increase_rate) * (1 - discount_rate) - 1 = 0.008 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_income_increase_l3575_357500


namespace NUMINAMATH_CALUDE_no_adjacent_x_probability_l3575_357574

-- Define the number of X tiles and O tiles
def num_x : ℕ := 4
def num_o : ℕ := 3

-- Define the total number of tiles
def total_tiles : ℕ := num_x + num_o

-- Function to calculate the number of ways to arrange tiles
def arrange_tiles (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the number of valid arrangements (no adjacent X tiles)
def valid_arrangements : ℕ := 1

-- Theorem statement
theorem no_adjacent_x_probability :
  (valid_arrangements : ℚ) / (arrange_tiles total_tiles num_x) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_x_probability_l3575_357574


namespace NUMINAMATH_CALUDE_factorization_equality_l3575_357536

theorem factorization_equality (x y : ℝ) : x^2*y - 6*x*y + 9*y = y*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3575_357536


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_range_l3575_357532

theorem complex_in_second_quadrant_range (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 > 0) → 2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_range_l3575_357532


namespace NUMINAMATH_CALUDE_min_value_expression_l3575_357505

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 3 * b = 4) :
  1 / (a + 1) + 3 / (b + 1) ≥ 2 ∧
  (1 / (a + 1) + 3 / (b + 1) = 2 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3575_357505


namespace NUMINAMATH_CALUDE_two_color_similar_ngons_l3575_357597

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define similarity between two n-gons
def AreSimilarNGons (n : ℕ) (k : ℝ) (ngon1 ngon2 : Fin n → Point) : Prop :=
  ∃ (center : Point), ∀ (i : Fin n),
    let p1 := ngon1 i
    let p2 := ngon2 i
    (p2.x - center.x)^2 + (p2.y - center.y)^2 = k^2 * ((p1.x - center.x)^2 + (p1.y - center.y)^2)

theorem two_color_similar_ngons 
  (n : ℕ) 
  (h_n : n ≥ 3) 
  (k : ℝ) 
  (h_k : k > 0 ∧ k ≠ 1) 
  (coloring : Coloring) :
  ∃ (ngon1 ngon2 : Fin n → Point),
    AreSimilarNGons n k ngon1 ngon2 ∧
    (∃ (c : Color), (∀ (i : Fin n), coloring (ngon1 i) = c)) ∧
    (∃ (c : Color), (∀ (i : Fin n), coloring (ngon2 i) = c)) :=
by sorry

end NUMINAMATH_CALUDE_two_color_similar_ngons_l3575_357597


namespace NUMINAMATH_CALUDE_no_square_subdivision_l3575_357531

theorem no_square_subdivision : ¬ ∃ (s : ℝ) (n : ℕ), 
  s > 0 ∧ n > 0 ∧ 
  ∃ (a : ℝ), a > 0 ∧ 
  s * s = n * (1/2 * a * a * Real.sqrt 3) ∧
  s = a * Real.sqrt 3 ∨ s = 2 * a ∨ s = 3 * a :=
sorry

end NUMINAMATH_CALUDE_no_square_subdivision_l3575_357531


namespace NUMINAMATH_CALUDE_sweater_price_theorem_l3575_357538

def total_price_shirts : ℕ := 400
def num_shirts : ℕ := 25
def num_sweaters : ℕ := 75
def price_diff : ℕ := 4

theorem sweater_price_theorem :
  let avg_shirt_price := total_price_shirts / num_shirts
  let avg_sweater_price := avg_shirt_price + price_diff
  avg_sweater_price * num_sweaters = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_theorem_l3575_357538
