import Mathlib

namespace triangle_third_side_length_l3713_371322

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (ha : a = 8) 
  (hb : b = 15) 
  (hθ : θ = 30 * π / 180) :
  c = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by sorry

end triangle_third_side_length_l3713_371322


namespace x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y_l3713_371353

theorem x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y :
  ¬(∀ (x y : ℝ), x ≤ y → |x| ≤ |y|) ∧ ¬(∀ (x y : ℝ), |x| ≤ |y| → x ≤ y) := by
  sorry

end x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y_l3713_371353


namespace square_root_difference_l3713_371302

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def twos (n : ℕ) : ℕ := 2 * ones n

theorem square_root_difference (n : ℕ+) :
  (ones (2*n) - twos n).sqrt = ones (2*n - 1) * 3 :=
sorry

end square_root_difference_l3713_371302


namespace perpendicular_bisector_intersection_equidistant_l3713_371315

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to find the intersection point of perpendicular bisectors
def intersectionOfPerpendicularBisectors (t : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem perpendicular_bisector_intersection_equidistant (t : Triangle) :
  let P := intersectionOfPerpendicularBisectors t
  distance P t.A = distance P t.B ∧ distance P t.B = distance P t.C := by
  sorry

end perpendicular_bisector_intersection_equidistant_l3713_371315


namespace no_positive_integer_solutions_l3713_371372

theorem no_positive_integer_solutions (k n : ℕ+) (h : n > 2) :
  ¬∃ (x y : ℕ+), x^(n : ℕ) - y^(n : ℕ) = 2^(k : ℕ) := by
  sorry

end no_positive_integer_solutions_l3713_371372


namespace B_power_15_minus_3_power_14_l3713_371388

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 8; 0, -2] := by sorry

end B_power_15_minus_3_power_14_l3713_371388


namespace sum_of_reciprocals_of_roots_minus_one_l3713_371336

theorem sum_of_reciprocals_of_roots_minus_one (p q r : ℂ) : 
  (p^3 - p - 2 = 0) → (q^3 - q - 2 = 0) → (r^3 - r - 2 = 0) →
  (1 / (p - 1) + 1 / (q - 1) + 1 / (r - 1) = -2) := by sorry

end sum_of_reciprocals_of_roots_minus_one_l3713_371336


namespace product_of_sum_and_difference_l3713_371308

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 15 ∧ x - y = 11 → x * y = 26 := by
sorry

end product_of_sum_and_difference_l3713_371308


namespace min_chord_length_l3713_371378

def circle_center : ℝ × ℝ := (3, 2)
def circle_radius : ℝ := 3
def point : ℝ × ℝ := (1, 1)

theorem min_chord_length :
  let d := Real.sqrt ((circle_center.1 - point.1)^2 + (circle_center.2 - point.2)^2)
  2 * Real.sqrt (circle_radius^2 - d^2) = 4 := by sorry

end min_chord_length_l3713_371378


namespace roots_of_f12_l3713_371363

def quadratic_polynomial (i : ℕ) (b : ℕ → ℤ) (c : ℕ → ℤ) : ℝ → ℝ :=
  fun x => x^2 + (b i : ℝ) * x + (c i : ℝ)

theorem roots_of_f12 
  (b : ℕ → ℤ) 
  (c : ℕ → ℤ) 
  (h1 : ∀ i : ℕ, i ≥ 1 → b (i + 1) = 2 * b i)
  (h2 : ∀ i : ℕ, i ≥ 1 → c i = -32 * b i - 1024)
  (h3 : ∃ x y : ℝ, x = 32 ∧ y = -31 ∧ 
    (quadratic_polynomial 1 b c x = 0 ∧ quadratic_polynomial 1 b c y = 0)) :
  ∃ x y : ℝ, x = 2016 ∧ y = 32 ∧ 
    (quadratic_polynomial 12 b c x = 0 ∧ quadratic_polynomial 12 b c y = 0) :=
sorry

end roots_of_f12_l3713_371363


namespace smallest_solution_floor_equation_l3713_371320

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ x = 89/9 ∧
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 8 → x ≤ y) ∧
  ⌊x^2⌋ - x * ⌊x⌋ = 8 :=
sorry

end smallest_solution_floor_equation_l3713_371320


namespace age_doubling_time_l3713_371369

/-- Given the ages of Wesley and Breenah, calculate the number of years until their combined age doubles -/
theorem age_doubling_time (wesley_age breenah_age : ℕ) (h1 : wesley_age = 15) (h2 : breenah_age = 7) 
  (h3 : wesley_age + breenah_age = 22) : 
  (fun n : ℕ => wesley_age + breenah_age + 2 * n = 2 * (wesley_age + breenah_age)) 11 := by
  sorry

end age_doubling_time_l3713_371369


namespace total_fish_l3713_371364

/-- The number of fish Billy has -/
def billy : ℕ := 10

/-- The number of fish Tony has -/
def tony : ℕ := 3 * billy

/-- The number of fish Sarah has -/
def sarah : ℕ := tony + 5

/-- The number of fish Bobby has -/
def bobby : ℕ := 2 * sarah

/-- The total number of fish all 4 people have -/
def total : ℕ := billy + tony + sarah + bobby

theorem total_fish : total = 145 := by sorry

end total_fish_l3713_371364


namespace no_rain_probability_l3713_371392

theorem no_rain_probability (pMonday pTuesday pBoth : ℝ) 
  (hMonday : pMonday = 0.6)
  (hTuesday : pTuesday = 0.55)
  (hBoth : pBoth = 0.4) :
  1 - (pMonday + pTuesday - pBoth) = 0.25 := by
sorry

end no_rain_probability_l3713_371392


namespace abs_sum_eq_six_iff_in_interval_l3713_371360

theorem abs_sum_eq_six_iff_in_interval (x : ℝ) : 
  |x + 1| + |x - 5| = 6 ↔ x ∈ Set.Icc (-1) 5 := by
  sorry

end abs_sum_eq_six_iff_in_interval_l3713_371360


namespace min_value_of_function_l3713_371352

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  ∃ (y : ℝ), y = x + 4 / (x - 2) ∧ (∀ (z : ℝ), z = x + 4 / (x - 2) → y ≤ z) ∧ y = 6 :=
sorry

end min_value_of_function_l3713_371352


namespace complex_equation_solution_l3713_371387

theorem complex_equation_solution : ∃ z : ℂ, z * (1 + Complex.I) + Complex.I = 0 ∧ z = -1/2 - Complex.I/2 := by
  sorry

end complex_equation_solution_l3713_371387


namespace difference_largest_smallest_valid_numbers_l3713_371374

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 - (n / 10) % 10 = (n / 10) % 10 - n % 10)

def largest_valid_number : ℕ := 951

def smallest_valid_number : ℕ := 159

theorem difference_largest_smallest_valid_numbers :
  largest_valid_number - smallest_valid_number = 792 ∧
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  ∀ n : ℕ, is_valid_number n → 
    smallest_valid_number ≤ n ∧ n ≤ largest_valid_number := by
  sorry

end difference_largest_smallest_valid_numbers_l3713_371374


namespace triangle_side_length_l3713_371381

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- No specific conditions needed here

-- Define a point on a line segment
def PointOnSegment (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

-- Define perpendicularity
def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define equality of distances
def EqualDistances (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2

-- Main theorem
theorem triangle_side_length 
  (P Q R E G : ℝ × ℝ) 
  (triangle : Triangle P Q R) 
  (e_on_pq : PointOnSegment P Q E)
  (g_on_pr : PointOnSegment P R G)
  (pq_perp_pr : Perpendicular P Q P R)
  (pg_perp_pr : Perpendicular P G P R)
  (qe_eq_eg : EqualDistances Q E E G)
  (eg_eq_gr : EqualDistances E G G R)
  (gr_eq_3 : EqualDistances G R P (P.1 + 3, P.2)) :
  EqualDistances P R P (P.1 + 6, P.2) :=
sorry

end triangle_side_length_l3713_371381


namespace jacoby_work_hours_l3713_371356

/-- The problem of calculating Jacoby's work hours -/
theorem jacoby_work_hours :
  let trip_cost : ℕ := 5000
  let hourly_wage : ℕ := 20
  let cookies_sold : ℕ := 24
  let cookie_price : ℕ := 4
  let lottery_ticket_cost : ℕ := 10
  let lottery_winnings : ℕ := 500
  let sister_gift : ℕ := 500
  let remaining_needed : ℕ := 3214

  let cookie_earnings := cookies_sold * cookie_price
  let gifts := sister_gift * 2
  let other_income := cookie_earnings + lottery_winnings + gifts - lottery_ticket_cost
  let total_earned := trip_cost - remaining_needed
  let job_earnings := total_earned - other_income
  let hours_worked := job_earnings / hourly_wage

  hours_worked = 10 := by sorry

end jacoby_work_hours_l3713_371356


namespace right_triangle_area_l3713_371370

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 26) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 120 := by
  sorry

end right_triangle_area_l3713_371370


namespace original_ratio_first_term_l3713_371319

theorem original_ratio_first_term 
  (original_first : ℚ) 
  (original_second : ℚ) 
  (added_number : ℚ) 
  (new_ratio_first : ℚ) 
  (new_ratio_second : ℚ) :
  original_first / original_second = 4 / 15 →
  added_number = 29 →
  (original_first + added_number) / (original_second + added_number) = new_ratio_first / new_ratio_second →
  new_ratio_first / new_ratio_second = 3 / 4 →
  original_first = 4 :=
by sorry

end original_ratio_first_term_l3713_371319


namespace initial_birds_count_l3713_371389

theorem initial_birds_count (total : ℕ) (additional : ℕ) (initial : ℕ) : 
  total = 42 → additional = 13 → initial + additional = total → initial = 29 := by
  sorry

end initial_birds_count_l3713_371389


namespace largest_satisfying_number_l3713_371342

/-- A function that returns all possible two-digit numbers from three digits -/
def twoDigitNumbers (a b c : Nat) : List Nat :=
  [10*a+b, 10*a+c, 10*b+a, 10*b+c, 10*c+a, 10*c+b]

/-- The property that a three-digit number satisfies the given conditions -/
def satisfiesCondition (n : Nat) : Prop :=
  ∃ a b c : Nat,
    n = 100*a + 10*b + c ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (twoDigitNumbers a b c).sum = n

theorem largest_satisfying_number :
  satisfiesCondition 396 ∧
  ∀ m : Nat, satisfiesCondition m → m ≤ 396 :=
sorry

end largest_satisfying_number_l3713_371342


namespace pauls_initial_pens_l3713_371366

theorem pauls_initial_pens (initial_books : ℕ) (books_left : ℕ) (pens_left : ℕ) (books_sold : ℕ) :
  initial_books = 108 →
  books_left = 66 →
  pens_left = 59 →
  books_sold = 42 →
  initial_books - books_left = books_sold →
  ∃ (initial_pens : ℕ), initial_pens = 101 ∧ initial_pens - books_sold = pens_left :=
by sorry

end pauls_initial_pens_l3713_371366


namespace problem_solution_l3713_371317

theorem problem_solution : (((2304 + 88) - 2400)^2 : ℚ) / 121 = 64 / 121 := by
  sorry

end problem_solution_l3713_371317


namespace total_pencils_l3713_371344

/-- Given the number of pencils in different locations, prove the total number of pencils -/
theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (dan_added : ℕ)
  (h1 : drawer = 43)
  (h2 : desk_initial = 19)
  (h3 : dan_added = 16) :
  drawer + desk_initial + dan_added = 78 := by
  sorry

end total_pencils_l3713_371344


namespace calculate_expression_l3713_371347

theorem calculate_expression : 
  (-2)^2 + Real.sqrt 8 - abs (1 - Real.sqrt 2) + (2023 - Real.pi)^0 = 6 + Real.sqrt 2 := by
  sorry

end calculate_expression_l3713_371347


namespace hypotenuse_increase_bound_l3713_371313

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt ((x + 1)^2 + (y + 1)^2) - Real.sqrt (x^2 + y^2) ≤ Real.sqrt 2 := by
  sorry

end hypotenuse_increase_bound_l3713_371313


namespace four_digit_square_same_digits_l3713_371395

theorem four_digit_square_same_digits : ∃! N : ℕ,
  (1000 ≤ N) ∧ (N ≤ 9999) ∧
  (∃ k : ℕ, N = k^2) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ N = 1100*a + 11*b) ∧
  N = 7744 := by
sorry

end four_digit_square_same_digits_l3713_371395


namespace child_growth_l3713_371354

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) :
  current_height - previous_height = 3 := by
  sorry

end child_growth_l3713_371354


namespace quadratic_radical_simplification_l3713_371328

theorem quadratic_radical_simplification :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x + y) = Real.sqrt x + Real.sqrt y → x = 0 ∨ y = 0) ∧
  (Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2) ∧
  (Real.sqrt (8 + 4 * Real.sqrt 3) = Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end quadratic_radical_simplification_l3713_371328


namespace custom_mul_unique_identity_l3713_371305

/-- Custom multiplication operation -/
def custom_mul (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mul_unique_identity
  (a b c : ℝ)
  (h1 : custom_mul a b c 1 2 = 3)
  (h2 : custom_mul a b c 2 3 = 4)
  (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), custom_mul a b c x m = x) :
  ∃ (m : ℝ), m = 4 ∧ m ≠ 0 ∧ ∀ (x : ℝ), custom_mul a b c x m = x :=
by sorry

end custom_mul_unique_identity_l3713_371305


namespace constant_function_property_l3713_371386

theorem constant_function_property (f : ℝ → ℝ) (h : ∀ x, f (4 * x) = 4) :
  ∀ x, f (2 * x) = 4 := by
sorry

end constant_function_property_l3713_371386


namespace isosceles_triangle_unique_point_l3713_371341

-- Define the triangle and point
def Triangle (A B C P : ℝ × ℝ) : Prop :=
  ∃ (s t : ℝ),
    -- Triangle ABC is isosceles with AB = AC = s
    dist A B = s ∧ dist A C = s ∧
    -- BC = t
    dist B C = t ∧
    -- Point P is inside the triangle (simplified assumption)
    true ∧
    -- AP = 2
    dist A P = 2 ∧
    -- BP = √5
    dist B P = Real.sqrt 5 ∧
    -- CP = 3
    dist C P = 3

-- The theorem to prove
theorem isosceles_triangle_unique_point 
  (A B C P : ℝ × ℝ) 
  (h : Triangle A B C P) : 
  ∃ (s t : ℝ), s = 2 * Real.sqrt 3 ∧ t = Real.sqrt 5 :=
sorry

end isosceles_triangle_unique_point_l3713_371341


namespace pool_filling_proof_l3713_371316

/-- The amount of water in gallons that Tina's pail can hold -/
def tinas_pail : ℝ := 4

/-- The amount of water in gallons that Tommy's pail can hold -/
def tommys_pail : ℝ := tinas_pail + 2

/-- The amount of water in gallons that Timmy's pail can hold -/
def timmys_pail : ℝ := 2 * tommys_pail

/-- The number of trips each person makes -/
def num_trips : ℕ := 3

/-- The total amount of water in gallons filled in the pool after 3 trips each -/
def total_water : ℝ := num_trips * (tinas_pail + tommys_pail + timmys_pail)

theorem pool_filling_proof : total_water = 66 := by
  sorry

end pool_filling_proof_l3713_371316


namespace cost_sharing_equalization_l3713_371394

theorem cost_sharing_equalization (A B : ℝ) (h : A < B) : 
  let total_cost := A + B
  let equal_share := total_cost / 2
  let amount_to_pay := equal_share - A
  amount_to_pay = (B - A) / 2 := by
sorry

end cost_sharing_equalization_l3713_371394


namespace math_problems_l3713_371390

theorem math_problems :
  (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) ∧
  (∃ x y : ℝ, |x| > |y| ∧ x ≤ y) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 2 ∧ x < 3 → 3*x - a < 0) → a ≥ 9) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) ↔ m < 0) :=
by sorry

end math_problems_l3713_371390


namespace expression_value_l3713_371373

theorem expression_value (x : ℝ) (h : x^2 - 3*x = 12) : 3*x^2 - 9*x + 5 = 41 := by
  sorry

end expression_value_l3713_371373


namespace russian_players_pairing_probability_l3713_371334

/-- The probability of all Russian players pairing only with other Russian players in a random pairing -/
theorem russian_players_pairing_probability 
  (total_players : ℕ) 
  (russian_players : ℕ) 
  (h1 : total_players = 10) 
  (h2 : russian_players = 4) 
  (h3 : russian_players ≤ total_players) :
  (russian_players.choose 2 : ℚ) / total_players.choose 2 = 1 / 21 := by
  sorry

end russian_players_pairing_probability_l3713_371334


namespace austins_highest_wave_l3713_371368

/-- Represents the height of a surfer and related wave measurements. -/
structure SurferMeasurements where
  surfboard_length : ℝ
  shortest_wave : ℝ
  highest_wave : ℝ
  surfer_height : ℝ

/-- Calculates the height of the highest wave caught by a surfer given the measurements. -/
def highest_wave_height (m : SurferMeasurements) : Prop :=
  m.surfboard_length = 7 ∧
  m.shortest_wave = m.surfboard_length + 3 ∧
  m.shortest_wave = m.surfer_height + 4 ∧
  m.highest_wave = 4 * m.surfer_height + 2 ∧
  m.highest_wave = 26

/-- Theorem stating that the highest wave Austin caught was 26 feet tall. -/
theorem austins_highest_wave :
  ∃ m : SurferMeasurements, highest_wave_height m :=
sorry

end austins_highest_wave_l3713_371368


namespace calculate_expression_l3713_371357

theorem calculate_expression : (-2)^2 - (1/8 - 3/4 + 1/2) * (-24) = 1 := by
  sorry

end calculate_expression_l3713_371357


namespace standard_notation_expression_l3713_371324

/-- A predicate to check if an expression conforms to standard algebraic notation -/
def is_standard_notation : String → Prop := sorry

/-- The set of given expressions -/
def expressions : Set String :=
  {"18 * b", "1 1/4 x", "-b / a^2", "m ÷ 2n"}

/-- Theorem stating that "-b / a^2" conforms to standard algebraic notation -/
theorem standard_notation_expression :
  ∃ e ∈ expressions, is_standard_notation e ∧ e = "-b / a^2" := by sorry

end standard_notation_expression_l3713_371324


namespace billboard_shorter_side_l3713_371382

theorem billboard_shorter_side (length width : ℝ) : 
  length * width = 91 →
  2 * (length + width) = 40 →
  length > 0 →
  width > 0 →
  min length width = 7 := by
sorry

end billboard_shorter_side_l3713_371382


namespace probability_more_heads_l3713_371358

/-- 
Given two players A and B, where A flips a fair coin n+1 times and B flips a fair coin n times,
this theorem states that the probability of A having more heads than B is 1/2.
-/
theorem probability_more_heads (n : ℕ) : ℝ := by
  sorry

#check probability_more_heads

end probability_more_heads_l3713_371358


namespace rose_more_expensive_l3713_371393

/-- The price of a single rose -/
def rose_price : ℝ := sorry

/-- The price of a single carnation -/
def carnation_price : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is greater than 24 yuan -/
axiom condition1 : 6 * rose_price + 3 * carnation_price > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * rose_price + 5 * carnation_price < 22

/-- The price of 2 roses is higher than the price of 3 carnations -/
theorem rose_more_expensive : 2 * rose_price > 3 * carnation_price := by
  sorry

end rose_more_expensive_l3713_371393


namespace interest_rate_equivalence_l3713_371359

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_equivalence : ∃ (rate : ℝ),
  simple_interest 100 0.05 8 = simple_interest 200 rate 2 ∧ rate = 0.1 := by
  sorry

end interest_rate_equivalence_l3713_371359


namespace min_distance_to_origin_l3713_371332

theorem min_distance_to_origin (x y : ℝ) : 
  8 * x + 15 * y = 120 → x ≥ 0 → y ≥ 0 → 
  ∀ x' y' : ℝ, 8 * x' + 15 * y' = 120 → x' ≥ 0 → y' ≥ 0 → 
  Real.sqrt (x^2 + y^2) ≤ Real.sqrt (x'^2 + y'^2) → 
  Real.sqrt (x^2 + y^2) = 120 / 17 := by
sorry

end min_distance_to_origin_l3713_371332


namespace min_value_of_f_min_value_is_zero_l3713_371331

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 3

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem min_value_of_f :
  ∀ x : ℝ, domain x → ∀ y : ℝ, domain y → f y ≥ f 3 := by
  sorry

-- The minimum value is f(3) = 0
theorem min_value_is_zero : f 3 = 0 := by
  sorry

end min_value_of_f_min_value_is_zero_l3713_371331


namespace tetrahedron_volume_in_cube_l3713_371327

/-- The volume of the tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (s : ℝ) (h : s = 8) : 
  let cube_volume := s^3
  let small_tetrahedron_volume := (1/3) * (1/2 * s^2) * s
  let purple_tetrahedron_volume := cube_volume - 4 * small_tetrahedron_volume
  purple_tetrahedron_volume = 512 - (1024/3) :=
by sorry

end tetrahedron_volume_in_cube_l3713_371327


namespace right_triangle_area_l3713_371397

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 8 * Real.sqrt 3 →
  angle = 30 * π / 180 →
  let a := h / 2
  let b := a * Real.sqrt 3
  (1 / 2) * a * b = 24 * Real.sqrt 3 := by sorry

end right_triangle_area_l3713_371397


namespace smallest_x_sqrt_3x_eq_5x_l3713_371300

theorem smallest_x_sqrt_3x_eq_5x (x : ℝ) :
  x ≥ 0 ∧ Real.sqrt (3 * x) = 5 * x → x = 0 := by sorry

end smallest_x_sqrt_3x_eq_5x_l3713_371300


namespace final_position_total_consumption_l3713_371307

-- Define the list of mileage values
def mileage : List Int := [-6, -2, 8, -3, 6, -4, 6, 3]

-- Define the electricity consumption rate per kilometer
def consumption_rate : Float := 0.15

-- Theorem for the final position
theorem final_position (m : List Int := mileage) :
  m.sum = 8 := by sorry

-- Theorem for total electricity consumption
theorem total_consumption (m : List Int := mileage) (r : Float := consumption_rate) :
  (m.map Int.natAbs).sum.toFloat * r = 5.7 := by sorry

end final_position_total_consumption_l3713_371307


namespace bike_ride_distance_l3713_371325

/-- Calculates the total distance traveled given the conditions of the bike ride --/
theorem bike_ride_distance (total_time : ℝ) (speed_out speed_back : ℝ) : 
  total_time = 7 ∧ speed_out = 24 ∧ speed_back = 18 →
  2 * (total_time / (1 / speed_out + 1 / speed_back)) = 144 := by
  sorry


end bike_ride_distance_l3713_371325


namespace perfume_price_change_l3713_371375

-- Define the original price
def original_price : ℝ := 1200

-- Define the increase percentage
def increase_percent : ℝ := 10

-- Define the decrease percentage
def decrease_percent : ℝ := 15

-- Theorem statement
theorem perfume_price_change :
  let increased_price := original_price * (1 + increase_percent / 100)
  let final_price := increased_price * (1 - decrease_percent / 100)
  original_price - final_price = 78 := by
sorry

end perfume_price_change_l3713_371375


namespace cubic_as_difference_of_squares_l3713_371367

theorem cubic_as_difference_of_squares (a : ℕ) :
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := by
  sorry

end cubic_as_difference_of_squares_l3713_371367


namespace jake_balloons_l3713_371361

theorem jake_balloons (total : ℕ) (allan_extra : ℕ) (h1 : total = 56) (h2 : allan_extra = 8) :
  ∃ (jake : ℕ), jake + (jake + allan_extra) = total ∧ jake = 24 :=
by sorry

end jake_balloons_l3713_371361


namespace tangent_product_simplification_l3713_371337

theorem tangent_product_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
sorry

end tangent_product_simplification_l3713_371337


namespace temperature_difference_l3713_371365

def highest_temp : ℝ := 8
def lowest_temp : ℝ := -2

theorem temperature_difference : highest_temp - lowest_temp = 10 := by
  sorry

end temperature_difference_l3713_371365


namespace crosswalks_per_intersection_l3713_371345

/-- Given a road with intersections and crosswalks, prove the number of crosswalks per intersection. -/
theorem crosswalks_per_intersection
  (num_intersections : ℕ)
  (lines_per_crosswalk : ℕ)
  (total_lines : ℕ)
  (h1 : num_intersections = 5)
  (h2 : lines_per_crosswalk = 20)
  (h3 : total_lines = 400) :
  total_lines / lines_per_crosswalk / num_intersections = 4 :=
by sorry

end crosswalks_per_intersection_l3713_371345


namespace two_number_problem_l3713_371311

theorem two_number_problem :
  ∃ (x y : ℕ), x > y ∧ x - y = 4 ∧ x * y = 80 ∧ (Even x ∨ Even y) ∧ x + y = 20 := by
  sorry

end two_number_problem_l3713_371311


namespace factorial_fraction_equality_l3713_371306

theorem factorial_fraction_equality : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 2 / 7 := by
  sorry

end factorial_fraction_equality_l3713_371306


namespace fishing_result_l3713_371329

/-- The total number of fishes Will and Henry have after fishing -/
def total_fishes (will_catfish : ℕ) (will_eels : ℕ) (henry_trout_ratio : ℕ) : ℕ :=
  let will_total := will_catfish + will_eels
  let henry_total := will_catfish * henry_trout_ratio
  let henry_kept := henry_total / 2
  will_total + henry_kept

/-- Theorem stating the total number of fishes Will and Henry have -/
theorem fishing_result : total_fishes 16 10 3 = 50 := by
  sorry

#eval total_fishes 16 10 3

end fishing_result_l3713_371329


namespace gardening_project_total_cost_l3713_371376

/-- The cost of the gardening project -/
def gardening_project_cost (
  num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℕ)
  (gardener_hourly_rate : ℕ)
  (gardener_hours_per_day : ℕ)
  (gardener_days : ℕ)
  (soil_volume : ℕ)
  (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush +
  gardener_hourly_rate * gardener_hours_per_day * gardener_days +
  soil_volume * soil_cost_per_unit

/-- The theorem stating the total cost of the gardening project -/
theorem gardening_project_total_cost :
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end gardening_project_total_cost_l3713_371376


namespace slope_is_plus_minus_two_l3713_371326

/-- The slope of a line passing through (-1,0) that intersects the parabola y^2 = 4x
    such that the midpoint of the intersection points lies on x = 3 -/
def slope_of_intersecting_line : ℝ → Prop :=
  λ k : ℝ => ∃ (x₁ x₂ y₁ y₂ : ℝ),
    -- Line equation
    y₁ = k * (x₁ + 1) ∧
    y₂ = k * (x₂ + 1) ∧
    -- Parabola equation
    y₁^2 = 4 * x₁ ∧
    y₂^2 = 4 * x₂ ∧
    -- Midpoint condition
    (x₁ + x₂) / 2 = 3

theorem slope_is_plus_minus_two :
  ∀ k : ℝ, slope_of_intersecting_line k ↔ k = 2 ∨ k = -2 :=
sorry

end slope_is_plus_minus_two_l3713_371326


namespace max_distinct_pairs_l3713_371391

/-- Given a set of integers from 1 to 3000, we can choose at most 1199 pairs
    such that each pair sum is distinct and no greater than 3000. -/
theorem max_distinct_pairs : ∀ (k : ℕ) (a b : ℕ → ℕ),
  (∀ i, i < k → 1 ≤ a i ∧ a i < b i ∧ b i ≤ 3000) →
  (∀ i j, i < k → j < k → i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) →
  (∀ i j, i < k → j < k → i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i, i < k → a i + b i ≤ 3000) →
  k ≤ 1199 :=
by sorry

end max_distinct_pairs_l3713_371391


namespace triangle_inequality_l3713_371349

/-- Given a triangle with sides a, b, c and area S, prove that a^2 + b^2 + c^2 ≥ 4S√3,
    with equality if and only if the triangle is equilateral -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0)
  (h_S : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) := by
  sorry


end triangle_inequality_l3713_371349


namespace flat_transaction_l3713_371312

theorem flat_transaction (x y : ℝ) : 
  0.14 * x - 0.14 * y = 1.96 ↔ 
  ∃ (gain loss : ℝ), 
    gain = 0.14 * x ∧ 
    loss = 0.14 * y ∧ 
    gain - loss = 1.96 :=
sorry

end flat_transaction_l3713_371312


namespace complex_distance_problem_l3713_371339

theorem complex_distance_problem (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^3 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^6 - 1) = 5 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 3 ∨ α = -Complex.I * Real.sqrt 3 := by
  sorry

end complex_distance_problem_l3713_371339


namespace negation_equivalence_l3713_371321

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end negation_equivalence_l3713_371321


namespace gillian_spending_theorem_l3713_371377

/-- Calculates the total amount Gillian spent at the farmer's market after tax -/
def gillian_total_spending (sandi_initial: ℝ) (sandi_market_fraction: ℝ) (sandi_discount: ℝ) 
  (gillian_extra: ℝ) (gillian_tax: ℝ) : ℝ :=
  let sandi_market := sandi_initial * sandi_market_fraction
  let sandi_after_discount := sandi_market * (1 - sandi_discount)
  let gillian_before_tax := 3 * sandi_after_discount + gillian_extra
  gillian_before_tax * (1 + gillian_tax)

/-- Theorem stating that Gillian's total spending at the farmer's market after tax is $957 -/
theorem gillian_spending_theorem :
  gillian_total_spending 600 0.5 0.2 150 0.1 = 957 := by
  sorry

end gillian_spending_theorem_l3713_371377


namespace ellipse_focal_length_l3713_371318

/-- Given an ellipse equation (x²/(10-m)) + (y²/(m-2)) = 1 with focal length 4,
    prove that the possible values of m are 4 and 8. -/
theorem ellipse_focal_length (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m)) + (y^2 / (m - 2)) = 1) →
  (∃ a b c : ℝ, a^2 = 10 - m ∧ b^2 = m - 2 ∧ c = 4 ∧ a^2 - b^2 = c^2) →
  m = 4 ∨ m = 8 := by
sorry


end ellipse_focal_length_l3713_371318


namespace cones_paths_count_l3713_371330

/-- Represents a position in the diagram --/
structure Position :=
  (row : Fin 5) (col : Fin 5)

/-- Represents a letter in the diagram --/
inductive Letter
  | C | O | N | E | S

/-- The diagram structure --/
def diagram : Position → Option Letter := sorry

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Represents a valid path in the diagram --/
def ValidPath : List Position → Prop := sorry

/-- Checks if a path spells "CONES" --/
def spellsCONES (path : List Position) : Prop := sorry

/-- The main theorem to prove --/
theorem cones_paths_count :
  (∃! (paths : Finset (List Position)),
    (∀ path ∈ paths, ValidPath path ∧ spellsCONES path) ∧
    paths.card = 6) := by sorry

end cones_paths_count_l3713_371330


namespace train_meeting_point_l3713_371348

/-- Two trains moving towards each other on a bridge --/
theorem train_meeting_point 
  (bridge_length : ℝ) 
  (train_a_speed : ℝ) 
  (train_b_speed : ℝ) 
  (h1 : bridge_length = 9000) 
  (h2 : train_a_speed = 15) 
  (h3 : train_b_speed = train_a_speed) :
  ∃ (meeting_time meeting_point : ℝ),
    meeting_time = 300 ∧ 
    meeting_point = bridge_length / 2 ∧
    meeting_point = train_a_speed * meeting_time :=
by sorry

end train_meeting_point_l3713_371348


namespace min_sum_of_product_72_l3713_371398

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 72 ∧ a₀ + b₀ = -73 :=
by sorry

end min_sum_of_product_72_l3713_371398


namespace problem_statement_l3713_371351

theorem problem_statement :
  (∀ x : ℝ, |x| ≥ 0) ∧
  (1^2 + 1 + 1 ≠ 0) ∧
  ((∀ x : ℝ, |x| ≥ 0) ∧ (1^2 + 1 + 1 ≠ 0)) := by
  sorry

end problem_statement_l3713_371351


namespace complement_intersection_theorem_l3713_371355

def U : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1}
def B : Set Nat := {1, 2, 3}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 3} := by sorry

end complement_intersection_theorem_l3713_371355


namespace bus_journey_stoppage_time_l3713_371323

/-- Calculates the total stoppage time for a bus journey with three stops -/
def total_stoppage_time (stop1 stop2 stop3 : ℕ) : ℕ :=
  stop1 + stop2 + stop3

/-- Theorem stating that the total stoppage time for the given stop durations is 23 minutes -/
theorem bus_journey_stoppage_time :
  total_stoppage_time 5 8 10 = 23 :=
by sorry

end bus_journey_stoppage_time_l3713_371323


namespace teacher_age_l3713_371399

/-- Given a class of students and their teacher, proves the teacher's age based on average ages. -/
theorem teacher_age (num_students : ℕ) (student_avg_age teacher_age : ℝ) (total_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = total_avg_age →
  teacher_age = 36 := by
  sorry

end teacher_age_l3713_371399


namespace school_purchase_cost_l3713_371314

theorem school_purchase_cost : 
  let projector_count : ℕ := 8
  let computer_count : ℕ := 32
  let projector_cost : ℕ := 7500
  let computer_cost : ℕ := 3600
  (projector_count * projector_cost + computer_count * computer_cost : ℕ) = 175200 := by
  sorry

end school_purchase_cost_l3713_371314


namespace expression_evaluation_l3713_371383

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (3 * x^2 * y - x * y^2) - 2 * (-2 * x * y^2 + x^2 * y) = 2 :=
by sorry

end expression_evaluation_l3713_371383


namespace perpendicular_line_equation_l3713_371380

/-- A line passing through point (2,1) and perpendicular to x-2y+1=0 has equation 2x + y - 5 = 0 -/
theorem perpendicular_line_equation : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ p : ℝ × ℝ, p ∈ l ↔ 2 * p.1 + p.2 - 5 = 0) → -- l is defined by 2x + y - 5 = 0
    ((2, 1) ∈ l) →  -- l passes through (2,1)
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → 
      (p.1 - q.1) * (1 - 2) + (p.2 - q.2) * (1 - (-1/2)) = 0) → -- l is perpendicular to x-2y+1=0
    True := by sorry

end perpendicular_line_equation_l3713_371380


namespace intersects_implies_a_in_range_l3713_371396

/-- A function f(x) that always intersects the x-axis -/
def f (m a x : ℝ) : ℝ := m * (x^2 - 1) + x - a

/-- The property that f(x) always intersects the x-axis for all m -/
def always_intersects (a : ℝ) : Prop :=
  ∀ m : ℝ, ∃ x : ℝ, f m a x = 0

/-- Theorem: If f(x) always intersects the x-axis for all m, then a is in [-1, 1] -/
theorem intersects_implies_a_in_range (a : ℝ) :
  always_intersects a → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end intersects_implies_a_in_range_l3713_371396


namespace blue_face_prob_five_eighths_l3713_371362

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces : ℕ
  total_is_sum : total_faces = blue_faces + red_faces
  total_is_eight : total_faces = 8

/-- The probability of rolling a blue face on a colored octahedron -/
def blue_face_probability (o : ColoredOctahedron) : ℚ :=
  o.blue_faces / o.total_faces

/-- Theorem: The probability of rolling a blue face on an octahedron with 5 blue faces and 3 red faces is 5/8 -/
theorem blue_face_prob_five_eighths (o : ColoredOctahedron) 
    (h1 : o.blue_faces = 5) 
    (h2 : o.red_faces = 3) : 
    blue_face_probability o = 5 / 8 := by
  sorry

#check blue_face_prob_five_eighths

end blue_face_prob_five_eighths_l3713_371362


namespace docked_amount_is_five_l3713_371335

/-- Calculates the amount docked per late arrival given the hourly rate, weekly hours, 
    number of late arrivals, and actual pay. -/
def amount_docked_per_late_arrival (hourly_rate : ℚ) (weekly_hours : ℚ) 
  (late_arrivals : ℕ) (actual_pay : ℚ) : ℚ :=
  ((hourly_rate * weekly_hours) - actual_pay) / late_arrivals

/-- Proves that the amount docked per late arrival is $5 given the specific conditions. -/
theorem docked_amount_is_five :
  amount_docked_per_late_arrival 30 18 3 525 = 5 := by
  sorry

end docked_amount_is_five_l3713_371335


namespace abc_sum_theorem_l3713_371333

theorem abc_sum_theorem (a b c : ℚ) (h : a * b * c > 0) :
  (|a| / a + |b| / b + |c| / c : ℚ) = 3 ∨ (|a| / a + |b| / b + |c| / c : ℚ) = -1 :=
by sorry

end abc_sum_theorem_l3713_371333


namespace incenter_x_coordinate_is_one_l3713_371309

/-- The triangle formed by the x-axis, y-axis, and the line x + y = 2 -/
structure Triangle where
  A : ℝ × ℝ := (0, 2)  -- y-intercept
  B : ℝ × ℝ := (2, 0)  -- x-intercept
  O : ℝ × ℝ := (0, 0)  -- origin

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between a point and a line -/
def distancePointToLine (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := sorry

theorem incenter_x_coordinate_is_one (t : Triangle) :
  (incenter t).1 = 1 ∧
  distancePointToLine (incenter t) (fun x => 0) =
  distancePointToLine (incenter t) (fun x => x) ∧
  distancePointToLine (incenter t) (fun x => 0) =
  distancePointToLine (incenter t) (fun x => 2 - x) :=
sorry

end incenter_x_coordinate_is_one_l3713_371309


namespace child_ticket_cost_l3713_371303

theorem child_ticket_cost (adult_price : ℕ) (total_sales : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) :
  adult_price = 5 →
  total_sales = 178 →
  total_tickets = 42 →
  child_tickets = 16 →
  ∃ (child_price : ℕ), child_price = 3 ∧
    total_sales = adult_price * (total_tickets - child_tickets) + child_price * child_tickets :=
by
  sorry

end child_ticket_cost_l3713_371303


namespace circle_intersection_range_l3713_371343

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 25 - r^2 = 0

-- Define the condition for common points
def have_common_points (r : ℝ) : Prop :=
  ∃ x y, circle1 x y ∧ circle2 x y r

-- State the theorem
theorem circle_intersection_range :
  ∃ m n : ℝ, (∀ r, have_common_points r ↔ m ≤ r ∧ r ≤ n) ∧ n - m = 10 :=
sorry

end circle_intersection_range_l3713_371343


namespace company_salary_theorem_l3713_371379

/-- Proves that given a company with 15 managers earning an average of $90,000 and 75 associates,
    if the company's overall average salary is $40,000, then the average salary of associates is $30,000. -/
theorem company_salary_theorem (num_managers : ℕ) (num_associates : ℕ) 
    (avg_salary_managers : ℝ) (avg_salary_company : ℝ) : 
    num_managers = 15 →
    num_associates = 75 →
    avg_salary_managers = 90000 →
    avg_salary_company = 40000 →
    (num_managers * avg_salary_managers + num_associates * (30000 : ℝ)) / (num_managers + num_associates) = avg_salary_company :=
by sorry

end company_salary_theorem_l3713_371379


namespace cos_135_degrees_l3713_371384

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l3713_371384


namespace at_least_two_equal_l3713_371340

theorem at_least_two_equal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : a^2 + b + c = 1/a)
  (eq2 : b^2 + c + a = 1/b)
  (eq3 : c^2 + a + b = 1/c) :
  ¬(a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by sorry

end at_least_two_equal_l3713_371340


namespace x_1971_approximation_l3713_371346

/-- A sequence satisfying the given recurrence relation -/
def recurrence_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 3 * x n - x (n - 1) = n

theorem x_1971_approximation
  (x : ℕ → ℝ)
  (h_recurrence : recurrence_sequence x)
  (h_x1_bound : |x 1| < 1971) :
  |x 1971 - 985.250000| < 0.000001 := by
  sorry

end x_1971_approximation_l3713_371346


namespace solution_set_implies_sum_l3713_371338

/-- Given that the solution set of ax^2 - bx + 2 < 0 is {x | 1 < x < 2}, prove that a + b = -2 -/
theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + 2 < 0 ↔ 1 < x ∧ x < 2) → 
  a + b = -2 := by
  sorry

end solution_set_implies_sum_l3713_371338


namespace smallest_c_value_l3713_371310

-- Define the polynomial
def polynomial (c d x : ℤ) : ℤ := x^3 - c*x^2 + d*x - 2730

-- Define the property that the polynomial has three positive integer roots
def has_three_positive_integer_roots (c d : ℤ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ x, polynomial c d x = (x - r₁) * (x - r₂) * (x - r₃)

-- Theorem statement
theorem smallest_c_value (c d : ℤ) :
  has_three_positive_integer_roots c d → c ≥ 54 :=
by sorry

end smallest_c_value_l3713_371310


namespace hypotenuse_length_l3713_371385

/-- Given a right triangle with an acute angle α and a circle of radius R
    touching the hypotenuse and the extensions of the two legs,
    the length of the hypotenuse is R * (1 - tan(α/2)) / cos(α) -/
theorem hypotenuse_length (α R : Real) (h1 : 0 < α ∧ α < π/2) (h2 : R > 0) :
  ∃ x, x > 0 ∧ x = R * (1 - Real.tan (α/2)) / Real.cos α :=
by sorry

end hypotenuse_length_l3713_371385


namespace system_of_equations_substitution_l3713_371301

theorem system_of_equations_substitution :
  ∀ x y : ℝ,
  (2 * x - 5 * y = 4) →
  (3 * x - y = 1) →
  (2 * x - 5 * (3 * x - 1) = 4) :=
by
  sorry

end system_of_equations_substitution_l3713_371301


namespace soccer_team_theorem_l3713_371304

def soccer_team_problem (total_players starting_players first_half_subs : ℕ) : ℕ :=
  let second_half_subs := first_half_subs + (first_half_subs + 1) / 2
  let total_played := starting_players + first_half_subs + second_half_subs
  total_players - total_played

theorem soccer_team_theorem :
  soccer_team_problem 36 11 3 = 17 := by
  sorry

end soccer_team_theorem_l3713_371304


namespace awards_distribution_l3713_371350

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution :
  distribute_awards 6 4 = 1560 :=
sorry

end awards_distribution_l3713_371350


namespace X_inverse_of_A_l3713_371371

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -1, 0; -3, 5, 0; 0, 0, 2]

def X : Matrix (Fin 3) (Fin 3) ℚ := !![5/7, 1/7, 0; 3/7, 2/7, 0; 0, 0, 1/2]

theorem X_inverse_of_A : X * A = 1 := by sorry

end X_inverse_of_A_l3713_371371
