import Mathlib

namespace NUMINAMATH_CALUDE_airport_visit_total_l1403_140393

theorem airport_visit_total (first_graders : ℕ) (second_graders_difference : ℕ) : 
  first_graders = 358 →
  second_graders_difference = 64 →
  first_graders + (first_graders - second_graders_difference) = 652 :=
by sorry

end NUMINAMATH_CALUDE_airport_visit_total_l1403_140393


namespace NUMINAMATH_CALUDE_second_prime_range_l1403_140365

theorem second_prime_range (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  15 < p * q ∧ p * q ≤ 36 → 2 < p ∧ p < 6 → p * q = 33 → q = 11 := by
  sorry

end NUMINAMATH_CALUDE_second_prime_range_l1403_140365


namespace NUMINAMATH_CALUDE_sequence_term_proof_l1403_140371

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℚ := (2/3) * n^2 - (1/3) * n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℚ := (4/3) * n - 1

theorem sequence_term_proof (n : ℕ) (h : n > 0) : 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l1403_140371


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l1403_140317

/-- Theorem: In an election with 3 candidates receiving 2500, 5000, and 15000 votes respectively,
    the winning candidate received 75% of the total votes. -/
theorem winning_candidate_percentage (votes : Fin 3 → ℕ)
  (h1 : votes 0 = 2500)
  (h2 : votes 1 = 5000)
  (h3 : votes 2 = 15000) :
  (votes 2 : ℚ) / (votes 0 + votes 1 + votes 2) * 100 = 75 := by
  sorry


end NUMINAMATH_CALUDE_winning_candidate_percentage_l1403_140317


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1403_140300

theorem division_remainder_problem (R Q D : ℕ) : 
  D = 3 * Q →
  D = 3 * R + 3 →
  251 = D * Q + R →
  R = 8 := by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1403_140300


namespace NUMINAMATH_CALUDE_license_plate_increase_l1403_140364

def old_plates : ℕ := 26^3 * 10^3
def new_plates_a : ℕ := 26^2 * 10^4
def new_plates_b : ℕ := 26^4 * 10^2
def avg_new_plates : ℚ := (new_plates_a + new_plates_b) / 2

theorem license_plate_increase : 
  (avg_new_plates : ℚ) / old_plates = 468 / 10 := by sorry

end NUMINAMATH_CALUDE_license_plate_increase_l1403_140364


namespace NUMINAMATH_CALUDE_terry_lunch_options_l1403_140343

theorem terry_lunch_options :
  ∀ (lettuce_types tomato_types olive_types soup_types : ℕ),
    lettuce_types = 2 →
    tomato_types = 3 →
    olive_types = 4 →
    soup_types = 2 →
    (lettuce_types * tomato_types * olive_types * soup_types) = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_terry_lunch_options_l1403_140343


namespace NUMINAMATH_CALUDE_movie_remaining_time_l1403_140380

/-- Calculates the remaining time of a movie given the total duration and watched duration. -/
def remaining_movie_time (total_duration watch_duration : ℕ) : ℕ :=
  total_duration - watch_duration

/-- Theorem: Given a 3-hour movie and a viewing duration of 2 hours and 24 minutes, 
    the remaining time to watch is 36 minutes. -/
theorem movie_remaining_time : 
  let total_duration : ℕ := 3 * 60  -- 3 hours in minutes
  let watch_duration : ℕ := 2 * 60 + 24  -- 2 hours and 24 minutes
  remaining_movie_time total_duration watch_duration = 36 := by
sorry

end NUMINAMATH_CALUDE_movie_remaining_time_l1403_140380


namespace NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l1403_140376

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 12) :
  let side := d / Real.sqrt 2
  4 * side = 24 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l1403_140376


namespace NUMINAMATH_CALUDE_non_indian_percentage_approx_l1403_140322

/-- Represents the number of attendees in a category and the percentage of Indians in that category -/
structure AttendeeCategory where
  total : ℕ
  indianPercentage : ℚ

/-- Calculates the number of non-Indian attendees in a category -/
def nonIndianCount (category : AttendeeCategory) : ℚ :=
  category.total * (1 - category.indianPercentage)

/-- Data for the climate conference -/
def conferenceData : List AttendeeCategory := [
  ⟨1200, 25/100⟩,  -- Male participants
  ⟨800, 40/100⟩,   -- Male volunteers
  ⟨1000, 35/100⟩,  -- Female participants
  ⟨500, 15/100⟩,   -- Female volunteers
  ⟨1800, 10/100⟩,  -- Children
  ⟨500, 45/100⟩,   -- Male scientists
  ⟨250, 30/100⟩,   -- Female scientists
  ⟨350, 55/100⟩,   -- Male government officials
  ⟨150, 50/100⟩    -- Female government officials
]

/-- Total number of attendees -/
def totalAttendees : ℕ := 6550

/-- Theorem stating that the percentage of non-Indian attendees is approximately 72.61% -/
theorem non_indian_percentage_approx :
  abs ((List.sum (List.map nonIndianCount conferenceData) / totalAttendees) - 72.61/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_non_indian_percentage_approx_l1403_140322


namespace NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l1403_140384

theorem modular_inverse_35_mod_36 : ∃ x : ℤ, (35 * x) % 36 = 1 ∧ x % 36 = 35 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l1403_140384


namespace NUMINAMATH_CALUDE_f_min_value_l1403_140348

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 4 * y^2 - 12 * x - 8 * y

/-- The theorem stating the minimum value and where it occurs -/
theorem f_min_value :
  (∀ x y : ℝ, f x y ≥ -28) ∧
  f (8/3) (-1) = -28 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l1403_140348


namespace NUMINAMATH_CALUDE_tenth_line_correct_l1403_140311

def ninthLine : String := "311311222113111231131112322211231231131112"

def countConsecutive (s : String) : String :=
  sorry

theorem tenth_line_correct : 
  countConsecutive ninthLine = "13211321322111312211" := by
  sorry

end NUMINAMATH_CALUDE_tenth_line_correct_l1403_140311


namespace NUMINAMATH_CALUDE_rectangular_box_problem_l1403_140337

theorem rectangular_box_problem :
  ∃! (a b c : ℕ+),
    (a ≤ b ∧ b ≤ c) ∧
    (a * b * c = 2 * (2 * (a * b + b * c + c * a))) ∧
    (4 * a = c) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_problem_l1403_140337


namespace NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l1403_140367

theorem max_cube_sum_on_sphere (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  x^3 + y^3 + z^3 ≤ 27 ∧ ∃ x y z : ℝ, x^2 + y^2 + z^2 = 9 ∧ x^3 + y^3 + z^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l1403_140367


namespace NUMINAMATH_CALUDE_extremum_values_l1403_140341

theorem extremum_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/(2*y + 1) ≥ (3 + 2*Real.sqrt 2)/3) := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l1403_140341


namespace NUMINAMATH_CALUDE_unique_n_reaching_three_l1403_140321

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 3 else n / 2

def iterateG (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => g (iterateG n k)

theorem unique_n_reaching_three :
  ∃! n : ℕ, n ∈ Finset.range 100 ∧ ∃ k : ℕ, iterateG n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_reaching_three_l1403_140321


namespace NUMINAMATH_CALUDE_positive_A_value_l1403_140328

def hash (k : ℝ) (A B : ℝ) : ℝ := A^2 + k * B^2

theorem positive_A_value (k : ℝ) (A : ℝ) :
  k = 3 →
  hash k A 7 = 196 →
  A > 0 →
  A = 7 := by
sorry

end NUMINAMATH_CALUDE_positive_A_value_l1403_140328


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_not_sufficient_nor_necessary_l1403_140353

/-- A sequence is geometric if the ratio between consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement "If {a_n} is geometric, then {a_n + a_{n+1}} is geometric" is neither sufficient nor necessary. -/
theorem geometric_sequence_sum_not_sufficient_nor_necessary :
  (∃ a : ℕ → ℝ, IsGeometric a ∧ ¬IsGeometric (fun n ↦ a n + a (n + 1))) ∧
  (∃ a : ℕ → ℝ, ¬IsGeometric a ∧ IsGeometric (fun n ↦ a n + a (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_not_sufficient_nor_necessary_l1403_140353


namespace NUMINAMATH_CALUDE_books_on_shelf_l1403_140327

/-- The number of books on a shelf after adding more books is equal to the sum of the initial number of books and the number of books added. -/
theorem books_on_shelf (initial_books additional_books : ℕ) :
  initial_books + additional_books = initial_books + additional_books :=
by sorry

end NUMINAMATH_CALUDE_books_on_shelf_l1403_140327


namespace NUMINAMATH_CALUDE_prob_five_diamond_three_l1403_140302

-- Define a standard deck of cards
def standard_deck : Nat := 52

-- Define the number of 5s in a deck
def num_fives : Nat := 4

-- Define the number of diamonds in a deck
def num_diamonds : Nat := 13

-- Define the number of 3s in a deck
def num_threes : Nat := 4

-- Define our specific event
def event_probability : ℚ :=
  (num_fives : ℚ) / standard_deck *
  (num_diamonds : ℚ) / (standard_deck - 1) *
  (num_threes : ℚ) / (standard_deck - 2)

-- Theorem statement
theorem prob_five_diamond_three :
  event_probability = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_diamond_three_l1403_140302


namespace NUMINAMATH_CALUDE_inverse_prop_of_matrix_l1403_140313

/-- Given a 2x2 matrix B and a constant k, proves that if B^(-1) = k * B, 
    then the bottom-right element of B is -4 and k = 1/72 -/
theorem inverse_prop_of_matrix (B : Matrix (Fin 2) (Fin 2) ℝ) (k : ℝ) 
    (h1 : B 0 0 = 4) (h2 : B 0 1 = 7) (h3 : B 1 0 = 8) :
  B⁻¹ = k • B → (B 1 1 = -4 ∧ k = 1/72) := by
  sorry

end NUMINAMATH_CALUDE_inverse_prop_of_matrix_l1403_140313


namespace NUMINAMATH_CALUDE_alice_initial_cookies_count_l1403_140389

/-- The number of chocolate chip cookies Alice initially baked -/
def alices_initial_cookies : ℕ := 91

/-- The number of peanut butter cookies Bob initially baked -/
def bobs_initial_cookies : ℕ := 7

/-- The number of cookies thrown on the floor -/
def thrown_cookies : ℕ := 29

/-- The number of additional cookies Alice baked after the accident -/
def alices_additional_cookies : ℕ := 5

/-- The number of additional cookies Bob baked after the accident -/
def bobs_additional_cookies : ℕ := 36

/-- The total number of edible cookies at the end -/
def total_edible_cookies : ℕ := 93

theorem alice_initial_cookies_count :
  alices_initial_cookies = 91 :=
by
  sorry

#check alice_initial_cookies_count

end NUMINAMATH_CALUDE_alice_initial_cookies_count_l1403_140389


namespace NUMINAMATH_CALUDE_factorization_of_cyclic_expression_l1403_140346

theorem factorization_of_cyclic_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + a*c + b*c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cyclic_expression_l1403_140346


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_squared_l1403_140386

theorem reciprocal_of_negative_five_squared :
  ((-5 : ℝ)^2)⁻¹ = (1 / 25 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_squared_l1403_140386


namespace NUMINAMATH_CALUDE_private_pilot_course_cost_l1403_140329

/-- The cost of a private pilot course -/
theorem private_pilot_course_cost :
  ∀ (flight_cost ground_cost total_cost : ℕ),
    flight_cost = 950 →
    ground_cost = 325 →
    flight_cost = ground_cost + 625 →
    total_cost = flight_cost + ground_cost →
    total_cost = 1275 := by
  sorry

end NUMINAMATH_CALUDE_private_pilot_course_cost_l1403_140329


namespace NUMINAMATH_CALUDE_jacket_markup_percentage_l1403_140378

/-- Proves that the markup percentage is 40% given the conditions of the jacket sale problem -/
theorem jacket_markup_percentage (purchase_price : ℝ) (selling_price : ℝ) (markup_percentage : ℝ) 
  (sale_discount : ℝ) (gross_profit : ℝ) :
  purchase_price = 48 →
  selling_price = purchase_price + markup_percentage * selling_price →
  sale_discount = 0.2 →
  gross_profit = 16 →
  (1 - sale_discount) * selling_price - purchase_price = gross_profit →
  markup_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_jacket_markup_percentage_l1403_140378


namespace NUMINAMATH_CALUDE_watch_cost_is_20_l1403_140377

-- Define the given conditions
def evans_initial_money : ℕ := 1
def money_given_to_evan : ℕ := 12
def additional_money_needed : ℕ := 7

-- Define the cost of the watch
def watch_cost : ℕ := evans_initial_money + money_given_to_evan + additional_money_needed

-- Theorem to prove
theorem watch_cost_is_20 : watch_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_is_20_l1403_140377


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1403_140390

theorem trigonometric_equation_solution (x : ℝ) :
  (2 * Real.sin x ^ 3 + 2 * Real.sin x ^ 2 * Real.cos x - Real.sin x * Real.cos x ^ 2 - Real.cos x ^ 3 = 0) ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨
  (∃ k : ℤ, x = Real.arctan (Real.sqrt 2 / 2) + k * π) ∨
  (∃ k : ℤ, x = -Real.arctan (Real.sqrt 2 / 2) + k * π) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1403_140390


namespace NUMINAMATH_CALUDE_zoey_reading_schedule_l1403_140339

def days_to_read (n : ℕ) : ℕ := n

def total_days (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem zoey_reading_schedule (num_books : ℕ) (start_day : ℕ) 
  (h1 : num_books = 20)
  (h2 : start_day = 5) -- Friday is represented as 5 (0 is Sunday)
  : day_of_week start_day (total_days num_books) = start_day := by
  sorry

#check zoey_reading_schedule

end NUMINAMATH_CALUDE_zoey_reading_schedule_l1403_140339


namespace NUMINAMATH_CALUDE_physical_fitness_test_probability_l1403_140379

theorem physical_fitness_test_probability 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (selected_students : ℕ) :
  total_students = male_students + female_students →
  male_students = 3 →
  female_students = 2 →
  selected_students = 2 →
  (Nat.choose male_students 1 * Nat.choose female_students 1) / 
  Nat.choose total_students selected_students = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_physical_fitness_test_probability_l1403_140379


namespace NUMINAMATH_CALUDE_zachary_needs_money_l1403_140334

/-- The additional amount of money Zachary needs to buy football equipment -/
def additional_money_needed (football_price : ℝ) (shorts_price : ℝ) (shoes_price : ℝ) 
  (socks_price : ℝ) (water_bottle_price : ℝ) (eur_to_usd : ℝ) (gbp_to_usd : ℝ) 
  (jpy_to_usd : ℝ) (krw_to_usd : ℝ) (discount_rate : ℝ) (current_money : ℝ) : ℝ :=
  let total_cost := football_price * eur_to_usd + 2 * shorts_price * gbp_to_usd + 
    shoes_price + 4 * socks_price * jpy_to_usd + water_bottle_price * krw_to_usd
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost - current_money

/-- Theorem stating the additional amount Zachary needs -/
theorem zachary_needs_money : 
  additional_money_needed 3.756 2.498 11.856 135.29 7834 1.19 1.38 0.0088 0.00085 0.1 24.042 = 7.127214 := by
  sorry

end NUMINAMATH_CALUDE_zachary_needs_money_l1403_140334


namespace NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l1403_140309

theorem cal_anthony_transaction_ratio :
  ∀ (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ),
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    jade_transactions = 85 →
    jade_transactions = cal_transactions + 19 →
    cal_transactions * 3 = anthony_transactions * 2 :=
by sorry

end NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l1403_140309


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l1403_140351

theorem shirt_cost_calculation (C : ℝ) : 
  (C * (1 + 0.3) * 0.5 = 13) → C = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l1403_140351


namespace NUMINAMATH_CALUDE_inverse_evaluation_l1403_140344

def problem (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧
  f 4 = 7 ∧
  f 6 = 3 ∧
  f 3 = 6 ∧
  f_inv ∘ f = id ∧
  f ∘ f_inv = id

theorem inverse_evaluation (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h : problem f f_inv) : 
  f_inv (f_inv 6 + f_inv 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_evaluation_l1403_140344


namespace NUMINAMATH_CALUDE_chessboard_diagonal_squares_l1403_140366

/-- The number of squares a diagonal passes through on a chessboard -/
def diagonalSquares (width : Nat) (height : Nat) : Nat :=
  width + height + Nat.gcd width height - 2

/-- Theorem: The diagonal of a 1983 × 999 chessboard passes through 2979 squares -/
theorem chessboard_diagonal_squares :
  diagonalSquares 1983 999 = 2979 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_diagonal_squares_l1403_140366


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_radii_relation_l1403_140357

/-- Theorem about the relationship between radii of spheres in a tetrahedron -/
theorem tetrahedron_sphere_radii_relation 
  (r r_a r_b r_c r_d : ℝ) 
  (S_a S_b S_c S_d V : ℝ) 
  (h_r : r = 3 * V / (S_a + S_b + S_c + S_d))
  (h_r_a : 1 / r_a = (-S_a + S_b + S_c + S_d) / (3 * V))
  (h_r_b : 1 / r_b = (S_a - S_b + S_c + S_d) / (3 * V))
  (h_r_c : 1 / r_c = (S_a + S_b - S_c + S_d) / (3 * V))
  (h_r_d : 1 / r_d = (S_a + S_b + S_c - S_d) / (3 * V))
  (h_positive : r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r_d > 0) :
  1 / r_a + 1 / r_b + 1 / r_c + 1 / r_d = 2 / r :=
by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_radii_relation_l1403_140357


namespace NUMINAMATH_CALUDE_only_C_has_inverse_l1403_140374

-- Define the set of graph labels
inductive GraphLabel
  | A | B | C | D | E

-- Define a predicate for functions that have inverses
def has_inverse (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.C => True
  | _ => False

-- Theorem statement
theorem only_C_has_inverse :
  ∀ g : GraphLabel, has_inverse g ↔ g = GraphLabel.C :=
by sorry

end NUMINAMATH_CALUDE_only_C_has_inverse_l1403_140374


namespace NUMINAMATH_CALUDE_line_intersecting_parabola_l1403_140360

/-- The equation of a line that intersects a parabola at two points 8 units apart vertically -/
theorem line_intersecting_parabola (m b : ℝ) (h1 : b ≠ 0) :
  (∃ k : ℝ, abs ((k^2 + 4*k + 4) - (m*k + b)) = 8) →
  (9 = 2*m + b) →
  (m = 2 ∧ b = 5) :=
by sorry

end NUMINAMATH_CALUDE_line_intersecting_parabola_l1403_140360


namespace NUMINAMATH_CALUDE_extreme_perimeter_rectangles_l1403_140310

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a rectangle with width w and height h -/
structure Rectangle where
  w : ℝ
  h : ℝ
  h_pos_w : 0 < w
  h_pos_h : 0 < h

/-- Predicate to check if a rectangle touches the given ellipse -/
def touches (e : Ellipse) (r : Rectangle) : Prop :=
  ∃ (x y : ℝ), (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ∧
    (x = r.w / 2 ∨ x = -r.w / 2 ∨ y = r.h / 2 ∨ y = -r.h / 2)

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.w + r.h)

/-- Theorem stating the properties of rectangles with extreme perimeters touching an ellipse -/
theorem extreme_perimeter_rectangles (e : Ellipse) :
  ∃ (r_min r_max : Rectangle),
    touches e r_min ∧ touches e r_max ∧
    (∀ r : Rectangle, touches e r → perimeter r_min ≤ perimeter r) ∧
    (∀ r : Rectangle, touches e r → perimeter r ≤ perimeter r_max) ∧
    r_min.w = 2 * e.b ∧ r_min.h = 2 * Real.sqrt (e.a^2 - e.b^2) ∧
    r_max.w = r_max.h ∧ r_max.w = 2 * Real.sqrt ((e.a^2 + e.b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_extreme_perimeter_rectangles_l1403_140310


namespace NUMINAMATH_CALUDE_susan_apples_l1403_140314

/-- The number of apples each person has -/
structure Apples where
  phillip : ℝ
  ben : ℝ
  tom : ℝ
  susan : ℝ

/-- The conditions of the problem -/
def apple_conditions (a : Apples) : Prop :=
  a.phillip = 38.25 ∧
  a.ben = a.phillip + 8.5 ∧
  a.tom = (3/8) * a.ben ∧
  a.susan = (1/2) * a.tom + 7

/-- The theorem stating that under the given conditions, Susan has 15.765625 apples -/
theorem susan_apples (a : Apples) (h : apple_conditions a) : a.susan = 15.765625 := by
  sorry

end NUMINAMATH_CALUDE_susan_apples_l1403_140314


namespace NUMINAMATH_CALUDE_wuyang_cup_result_l1403_140312

-- Define the teams
inductive Team : Type
| A : Team
| B : Team
| C : Team
| D : Team

-- Define the positions
inductive Position : Type
| Champion : Position
| RunnerUp : Position
| Third : Position
| Last : Position

-- Define the result type
def Result := Team → Position

-- Define the predictor type
inductive Predictor : Type
| Jia : Predictor
| Yi : Predictor
| Bing : Predictor

-- Define the prediction type
def Prediction := Predictor → Team → Position

-- Define the correctness of a prediction
def is_correct (pred : Prediction) (result : Result) (p : Predictor) (t : Team) : Prop :=
  pred p t = result t

-- Define the condition that each predictor is half right and half wrong
def half_correct (pred : Prediction) (result : Result) (p : Predictor) : Prop :=
  (∃ t1 t2 : Team, t1 ≠ t2 ∧ is_correct pred result p t1 ∧ is_correct pred result p t2) ∧
  (∃ t3 t4 : Team, t3 ≠ t4 ∧ ¬is_correct pred result p t3 ∧ ¬is_correct pred result p t4)

-- Define the predictions
def predictions (pred : Prediction) : Prop :=
  pred Predictor.Jia Team.C = Position.RunnerUp ∧
  pred Predictor.Jia Team.D = Position.Third ∧
  pred Predictor.Yi Team.D = Position.Last ∧
  pred Predictor.Yi Team.A = Position.RunnerUp ∧
  pred Predictor.Bing Team.C = Position.Champion ∧
  pred Predictor.Bing Team.B = Position.RunnerUp

-- State the theorem
theorem wuyang_cup_result :
  ∀ (pred : Prediction) (result : Result),
    predictions pred →
    (∀ p : Predictor, half_correct pred result p) →
    result Team.C = Position.Champion ∧
    result Team.A = Position.RunnerUp ∧
    result Team.D = Position.Third ∧
    result Team.B = Position.Last :=
sorry

end NUMINAMATH_CALUDE_wuyang_cup_result_l1403_140312


namespace NUMINAMATH_CALUDE_circle_packing_theorem_l1403_140388

theorem circle_packing_theorem :
  ∃ (n : ℕ+), (n : ℝ) / 2 > 2008 ∧
  ∀ (i j : Fin (n^2)), i ≠ j →
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x - (i.val % n : ℕ) / n)^2 + (y - (i.val / n : ℕ) / n)^2 ≤ (1 / (2*n))^2 ∧
  (x - (j.val % n : ℕ) / n)^2 + (y - (j.val / n : ℕ) / n)^2 ≤ (1 / (2*n))^2 →
  (x - (i.val % n : ℕ) / n)^2 + (y - (i.val / n : ℕ) / n)^2 = (1 / (2*n))^2 ∨
  (x - (j.val % n : ℕ) / n)^2 + (y - (j.val / n : ℕ) / n)^2 = (1 / (2*n))^2 ∨
  ((x - (i.val % n : ℕ) / n) - (x - (j.val % n : ℕ) / n))^2 +
  ((y - (i.val / n : ℕ) / n) - (y - (j.val / n : ℕ) / n))^2 ≥ (1 / n)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_packing_theorem_l1403_140388


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1403_140397

noncomputable def line_l (x y : ℝ) : Prop := x + 2*y = 0

noncomputable def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 2

theorem intersection_chord_length :
  ∀ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    A ≠ B →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 1435 / 35 := by
  sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1403_140397


namespace NUMINAMATH_CALUDE_inequality_solution_l1403_140352

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 1) ↔ (x > -8 ∧ x < -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1403_140352


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1403_140330

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 3) :
  1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 1 ∧
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1403_140330


namespace NUMINAMATH_CALUDE_correct_regression_equation_l1403_140307

-- Define the sample means
def x_mean : ℝ := 2
def y_mean : ℝ := 1.5

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := -2 * x + 5.5

-- State the theorem
theorem correct_regression_equation :
  -- Condition: x and y are negatively correlated
  (∃ k : ℝ, k < 0 ∧ ∀ x y : ℝ, y = k * x + linear_regression x_mean - k * x_mean) →
  -- The linear regression equation passes through the point (x_mean, y_mean)
  linear_regression x_mean = y_mean := by
  sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l1403_140307


namespace NUMINAMATH_CALUDE_transaction_fraction_proof_l1403_140398

theorem transaction_fraction_proof (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ) :
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  jade_transactions = 83 →
  jade_transactions = cal_transactions + 17 →
  3 * cal_transactions = 2 * anthony_transactions :=
by
  sorry

#check transaction_fraction_proof

end NUMINAMATH_CALUDE_transaction_fraction_proof_l1403_140398


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1403_140387

theorem algebraic_expression_value 
  (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : |x - 2| = 3) : 
  (a + b - m * n) * x + (a + b) ^ 2022 + (-m * n) ^ 2023 = -6 ∨ 
  (a + b - m * n) * x + (a + b) ^ 2022 + (-m * n) ^ 2023 = 0 :=
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1403_140387


namespace NUMINAMATH_CALUDE_pencils_needed_is_90_l1403_140318

/-- Calculates the number of pencils needed to be purchased for a school supply distribution problem. -/
def pencils_to_purchase (box_a_pencils box_b_pencils : ℕ) (box_a_classrooms box_b_classrooms : ℕ) (additional_pencils_needed : ℕ) : ℕ :=
  let total_classrooms := box_a_classrooms + box_b_classrooms
  let box_a_per_classroom := box_a_pencils / box_a_classrooms
  let box_b_per_classroom := box_b_pencils / box_b_classrooms
  let total_per_classroom := box_a_per_classroom + box_b_per_classroom
  let shortage_per_classroom := (additional_pencils_needed + total_classrooms - 1) / total_classrooms
  shortage_per_classroom * total_classrooms

/-- Theorem stating that given the specific conditions of the problem, 90 pencils need to be purchased. -/
theorem pencils_needed_is_90 : 
  pencils_to_purchase 480 735 6 9 85 = 90 := by
  sorry

end NUMINAMATH_CALUDE_pencils_needed_is_90_l1403_140318


namespace NUMINAMATH_CALUDE_complex_number_proof_l1403_140359

def i : ℂ := Complex.I

def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_number_proof (z : ℂ) 
  (h1 : is_real (z + 2*i)) 
  (h2 : is_real (z / (2 - i))) : 
  z = 4 - 2*i ∧ Complex.abs (z / (1 + i)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_proof_l1403_140359


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1403_140333

/-- The length of the diagonal of a rectangle with length 40 and width 40√2 is 40√3 -/
theorem rectangle_diagonal : 
  ∀ (l w d : ℝ), 
  l = 40 → 
  w = 40 * Real.sqrt 2 → 
  d = Real.sqrt (l^2 + w^2) → 
  d = 40 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1403_140333


namespace NUMINAMATH_CALUDE_blue_faces_cube_l1403_140358

theorem blue_faces_cube (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_faces_cube_l1403_140358


namespace NUMINAMATH_CALUDE_valid_pairings_count_l1403_140369

def num_bowls : ℕ := 6
def num_glasses : ℕ := 6

theorem valid_pairings_count :
  (num_bowls * num_glasses : ℕ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairings_count_l1403_140369


namespace NUMINAMATH_CALUDE_elijah_masking_tape_order_l1403_140395

/-- The amount of masking tape needed for Elijah's living room -/
def masking_tape_needed (narrow_wall_width : ℕ) (wide_wall_width : ℕ) : ℕ :=
  2 * narrow_wall_width + 2 * wide_wall_width

/-- Theorem stating the amount of masking tape Elijah needs to order -/
theorem elijah_masking_tape_order :
  masking_tape_needed 4 6 = 20 := by
sorry

end NUMINAMATH_CALUDE_elijah_masking_tape_order_l1403_140395


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1403_140382

theorem quadratic_equation_solution (c : ℝ) : 
  ((-5 : ℝ)^2 + c * (-5) - 45 = 0) → c = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1403_140382


namespace NUMINAMATH_CALUDE_kylie_final_coins_l1403_140356

/-- Calculates the number of US coins Kylie has left after converting all coins and giving some away --/
def kylie_coins_left (initial_us : ℝ) (euro : ℝ) (canadian : ℝ) (given_away : ℝ) 
  (euro_to_us : ℝ) (canadian_to_us : ℝ) : ℝ :=
  initial_us + euro * euro_to_us + canadian * canadian_to_us - given_away

/-- Theorem stating that Kylie is left with 15.58 US coins --/
theorem kylie_final_coins : 
  kylie_coins_left 15 13 8 21 1.18 0.78 = 15.58 := by
  sorry

end NUMINAMATH_CALUDE_kylie_final_coins_l1403_140356


namespace NUMINAMATH_CALUDE_ratio_calculation_l1403_140305

theorem ratio_calculation (x y a b : ℚ) 
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / (3 * b - y) = 3) :
  a / b = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1403_140305


namespace NUMINAMATH_CALUDE_wheel_diameter_l1403_140362

/-- The diameter of a wheel given its revolutions and distance covered -/
theorem wheel_diameter (revolutions : ℝ) (distance : ℝ) (π : ℝ) :
  revolutions = 8.007279344858963 →
  distance = 1056 →
  π = 3.14159 →
  ∃ (diameter : ℝ), abs (diameter - 41.975) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_diameter_l1403_140362


namespace NUMINAMATH_CALUDE_union_equality_implies_a_equals_two_l1403_140326

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, 2+a}

theorem union_equality_implies_a_equals_two :
  ∀ a : ℝ, A a ∪ B a = A a → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_equals_two_l1403_140326


namespace NUMINAMATH_CALUDE_no_geometric_sequence_cosines_l1403_140304

theorem no_geometric_sequence_cosines :
  ¬ ∃ a : ℝ, 0 < a ∧ a < 2 * π ∧
    ∃ r : ℝ, (Real.cos (2 * a) = r * Real.cos a) ∧
             (Real.cos (3 * a) = r * Real.cos (2 * a)) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_cosines_l1403_140304


namespace NUMINAMATH_CALUDE_distance_downstream_20min_l1403_140324

/-- Calculates the distance traveled downstream by a boat -/
def distance_traveled (boat_speed wind_speed current_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + current_speed + 0.1 * wind_speed) * time

/-- Theorem: Distance traveled downstream in 20 minutes -/
theorem distance_downstream_20min (c w : ℝ) :
  distance_traveled 26 w c (1/3) = 26/3 + c/3 + 0.1*w/3 := by
  sorry

end NUMINAMATH_CALUDE_distance_downstream_20min_l1403_140324


namespace NUMINAMATH_CALUDE_rectangle_area_l1403_140301

theorem rectangle_area (a b : ℝ) (h1 : a + b = 8) (h2 : 2*a^2 + 2*b^2 = 68) :
  a * b = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1403_140301


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1403_140320

/-- The area of the shaded region formed by a sector of a circle and an equilateral triangle -/
theorem shaded_area_calculation (r : ℝ) (θ : ℝ) (a : ℝ) (h1 : r = 12) (h2 : θ = 112) (h3 : a = 12) :
  let sector_area := (θ / 360) * π * r^2
  let triangle_area := (Real.sqrt 3 / 4) * a^2
  abs ((sector_area - triangle_area) - 78.0211) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1403_140320


namespace NUMINAMATH_CALUDE_election_votes_l1403_140349

theorem election_votes (total_votes : ℕ) 
  (winning_percentage : ℚ) (vote_majority : ℕ) :
  winning_percentage = 70 / 100 →
  vote_majority = 192 →
  (winning_percentage * total_votes - (1 - winning_percentage) * total_votes : ℚ) = vote_majority →
  total_votes = 480 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l1403_140349


namespace NUMINAMATH_CALUDE_monday_sales_correct_l1403_140396

/-- Represents the inventory and sales data for Danivan Drugstore --/
structure DrugstoreData where
  initial_inventory : ℕ
  tuesday_sales : ℕ
  daily_sales_wed_to_sun : ℕ
  saturday_delivery : ℕ
  end_week_inventory : ℕ

/-- Calculates the number of bottles sold on Monday --/
def monday_sales (data : DrugstoreData) : ℕ :=
  data.initial_inventory - data.tuesday_sales - (5 * data.daily_sales_wed_to_sun) + data.saturday_delivery - data.end_week_inventory

/-- Theorem stating that the number of bottles sold on Monday is 2445 --/
theorem monday_sales_correct (data : DrugstoreData) 
  (h1 : data.initial_inventory = 4500)
  (h2 : data.tuesday_sales = 900)
  (h3 : data.daily_sales_wed_to_sun = 50)
  (h4 : data.saturday_delivery = 650)
  (h5 : data.end_week_inventory = 1555) :
  monday_sales data = 2445 := by
  sorry

#eval monday_sales {
  initial_inventory := 4500,
  tuesday_sales := 900,
  daily_sales_wed_to_sun := 50,
  saturday_delivery := 650,
  end_week_inventory := 1555
}

end NUMINAMATH_CALUDE_monday_sales_correct_l1403_140396


namespace NUMINAMATH_CALUDE_sanitizer_dilution_l1403_140350

/-- Proves that adding 6 ounces of water to 12 ounces of 60% alcohol hand sanitizer 
    results in a solution with 40% alcohol concentration. -/
theorem sanitizer_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (final_concentration : ℝ) (water_added : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.6 →
  final_concentration = 0.4 →
  water_added = 6 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_sanitizer_dilution_l1403_140350


namespace NUMINAMATH_CALUDE_ruby_starting_lineup_combinations_l1403_140399

def total_players : ℕ := 15
def all_stars : ℕ := 5
def starting_lineup : ℕ := 7

theorem ruby_starting_lineup_combinations :
  Nat.choose (total_players - all_stars) (starting_lineup - all_stars) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ruby_starting_lineup_combinations_l1403_140399


namespace NUMINAMATH_CALUDE_jellybean_mass_theorem_l1403_140315

/-- The price of jellybeans in cents per gram -/
def price_per_gram : ℚ := 750 / 250

/-- The mass of jellybeans in grams that can be bought for 180 cents -/
def mass_for_180_cents : ℚ := 180 / price_per_gram

theorem jellybean_mass_theorem :
  mass_for_180_cents = 60 := by sorry

end NUMINAMATH_CALUDE_jellybean_mass_theorem_l1403_140315


namespace NUMINAMATH_CALUDE_baker_extra_donuts_l1403_140375

theorem baker_extra_donuts (total_donuts : ℕ) (num_boxes : ℕ) 
  (h1 : total_donuts = 48) 
  (h2 : num_boxes = 7) : 
  total_donuts % num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_extra_donuts_l1403_140375


namespace NUMINAMATH_CALUDE_expression_evaluation_l1403_140319

theorem expression_evaluation : 
  let x : ℚ := 1/2
  (((x^2 - 2*x + 1) / (x^2 - 1) - 1 / (x + 1)) / ((2*x - 4) / (x^2 + x))) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1403_140319


namespace NUMINAMATH_CALUDE_leon_payment_l1403_140372

/-- The total amount Leon paid for toy organizers, gaming chairs, and delivery fee. -/
def total_paid (toy_organizer_sets : ℕ) (toy_organizer_price : ℚ) 
                (gaming_chairs : ℕ) (gaming_chair_price : ℚ) 
                (delivery_fee_percentage : ℚ) : ℚ :=
  let total_sales := toy_organizer_sets * toy_organizer_price + gaming_chairs * gaming_chair_price
  let delivery_fee := delivery_fee_percentage * total_sales
  total_sales + delivery_fee

/-- Theorem stating that Leon paid $420 in total -/
theorem leon_payment : 
  total_paid 3 78 2 83 (5/100) = 420 := by
  sorry

end NUMINAMATH_CALUDE_leon_payment_l1403_140372


namespace NUMINAMATH_CALUDE_square_circle_puzzle_l1403_140331

theorem square_circle_puzzle (x y : ℚ) 
  (eq1 : 5 * x + 2 * y = 39)
  (eq2 : 3 * x + 3 * y = 27) :
  x = 7 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_puzzle_l1403_140331


namespace NUMINAMATH_CALUDE_digit_property_l1403_140392

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def S (n : ℕ) : ℕ :=
  (digits n).sum

def P (n : ℕ) : ℕ :=
  (digits n).prod

theorem digit_property :
  ({ n : ℕ | n > 0 ∧ n = P n } = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  ({ n : ℕ | n > 0 ∧ n = S n + P n } = {19, 29, 39, 49, 59, 69, 79, 89, 99}) :=
by sorry

end NUMINAMATH_CALUDE_digit_property_l1403_140392


namespace NUMINAMATH_CALUDE_inequality_proof_l1403_140340

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1403_140340


namespace NUMINAMATH_CALUDE_tori_height_l1403_140336

/-- Tori's initial height in feet -/
def initial_height : ℝ := 4.4

/-- The amount Tori grew in feet -/
def growth : ℝ := 2.86

/-- Tori's current height in feet -/
def current_height : ℝ := initial_height + growth

theorem tori_height : current_height = 7.26 := by
  sorry

end NUMINAMATH_CALUDE_tori_height_l1403_140336


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l1403_140345

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) : 
  cube_side = 8 → cylinder_radius = 2 → 
  (cube_side ^ 3 : ℝ) - π * cylinder_radius ^ 2 * cube_side = 512 - 32 * π := by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l1403_140345


namespace NUMINAMATH_CALUDE_worker_daily_hours_l1403_140368

/-- Represents the number of work hours per day for a worker -/
def daily_hours (total_hours : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℚ :=
  total_hours / (days_per_week * weeks_per_month)

/-- Theorem stating that under given conditions, a worker's daily work hours are 10 -/
theorem worker_daily_hours :
  let total_hours : ℕ := 200
  let days_per_week : ℕ := 5
  let weeks_per_month : ℕ := 4
  daily_hours total_hours days_per_week weeks_per_month = 10 := by
  sorry

end NUMINAMATH_CALUDE_worker_daily_hours_l1403_140368


namespace NUMINAMATH_CALUDE_shaded_area_sum_l1403_140373

theorem shaded_area_sum (r₁ : ℝ) (r₂ : ℝ) : 
  r₁ > 0 → 
  r₂ > 0 → 
  r₁ = 8 → 
  r₂ = r₁ / 2 → 
  (π * r₁^2) / 2 + (π * r₂^2) / 2 = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l1403_140373


namespace NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l1403_140394

theorem tens_digit_of_N_power_20 (N : ℕ) (h1 : Even N) (h2 : ¬ (10 ∣ N)) :
  (N^20 % 100) / 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l1403_140394


namespace NUMINAMATH_CALUDE_final_balance_is_correct_l1403_140335

/-- Represents a currency with its exchange rate to USD -/
structure Currency where
  name : String
  exchange_rate : Float

/-- Represents a transaction with amount, currency, and discount -/
structure Transaction where
  amount : Float
  currency : Currency
  discount : Float

def initial_balance : Float := 126.00

def gbp : Currency := { name := "GBP", exchange_rate := 1.39 }
def eur : Currency := { name := "EUR", exchange_rate := 1.18 }
def jpy : Currency := { name := "JPY", exchange_rate := 0.0091 }
def usd : Currency := { name := "USD", exchange_rate := 1.0 }

def uk_transaction : Transaction := { amount := 50.0, currency := gbp, discount := 0.1 }
def france_transaction : Transaction := { amount := 70.0, currency := eur, discount := 0.15 }
def japan_transaction : Transaction := { amount := 10000.0, currency := jpy, discount := 0.05 }
def us_gas_transaction : Transaction := { amount := 25.0, currency := gbp, discount := 0.0 }
def return_transaction : Transaction := { amount := 45.0, currency := usd, discount := 0.0 }

def monthly_interest_rate : Float := 0.015

def calculate_final_balance (initial_balance : Float) 
  (transactions : List Transaction) 
  (return_transaction : Transaction)
  (monthly_interest_rate : Float) : Float :=
  sorry

theorem final_balance_is_correct :
  calculate_final_balance initial_balance 
    [uk_transaction, france_transaction, japan_transaction, us_gas_transaction]
    return_transaction
    monthly_interest_rate = 340.00 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_is_correct_l1403_140335


namespace NUMINAMATH_CALUDE_painted_cube_problem_l1403_140325

theorem painted_cube_problem (n : ℕ) (h1 : n > 2) :
  (2 * (n - 2)^2 = 2 * (n - 2) * n^2) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l1403_140325


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1403_140316

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1403_140316


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1403_140363

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1403_140363


namespace NUMINAMATH_CALUDE_distance_AB_l1403_140342

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, -Real.sqrt 3 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * Real.sin (θ - Real.pi / 6)

def intersection_OA (t : ℝ) : Prop :=
  ∃ θ : ℝ, line_l t = curve_C1 θ

def intersection_OB (t : ℝ) : Prop :=
  ∃ θ : ℝ, line_l t = (curve_C2 θ * Real.cos θ, curve_C2 θ * Real.sin θ)

theorem distance_AB :
  ∀ t₁ t₂ : ℝ, intersection_OA t₁ → intersection_OB t₂ →
    Real.sqrt ((t₂ - t₁)^2 + (-Real.sqrt 3 * t₂ + Real.sqrt 3 * t₁)^2) = 4 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_l1403_140342


namespace NUMINAMATH_CALUDE_star_op_result_l1403_140306

/-- The * operation for non-zero integers -/
def star_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem stating the result of the star operation given the conditions -/
theorem star_op_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_cond : a + b = 15) (prod_cond : a * b = 36) : 
  star_op a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_op_result_l1403_140306


namespace NUMINAMATH_CALUDE_notebook_boxes_l1403_140370

theorem notebook_boxes (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) :
  total_notebooks / notebooks_per_box = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_boxes_l1403_140370


namespace NUMINAMATH_CALUDE_math_books_in_same_box_probability_l1403_140308

/-- Represents a box that can hold textbooks -/
structure Box where
  capacity : Nat

/-- Represents the collection of textbooks -/
structure Textbooks where
  total : Nat
  math : Nat

/-- Represents the problem setup -/
structure TextbookProblem where
  boxes : List Box
  books : Textbooks

/-- The probability of all math textbooks being in the same box -/
def mathBooksInSameBoxProbability (problem : TextbookProblem) : Rat :=
  18/1173

/-- The main theorem stating the probability of all math textbooks being in the same box -/
theorem math_books_in_same_box_probability 
  (problem : TextbookProblem)
  (h1 : problem.boxes.length = 3)
  (h2 : problem.books.total = 15)
  (h3 : problem.books.math = 4)
  (h4 : problem.boxes.map Box.capacity = [4, 5, 6]) :
  mathBooksInSameBoxProbability problem = 18/1173 := by
  sorry

end NUMINAMATH_CALUDE_math_books_in_same_box_probability_l1403_140308


namespace NUMINAMATH_CALUDE_pond_length_l1403_140347

/-- Given a rectangular field and a square pond, prove the length of the pond. -/
theorem pond_length (field_length field_width pond_area_ratio : ℝ) 
  (h1 : field_length = 96)
  (h2 : field_width = 48)
  (h3 : field_length = 2 * field_width)
  (h4 : pond_area_ratio = 1 / 72) : 
  Real.sqrt (pond_area_ratio * field_length * field_width) = 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l1403_140347


namespace NUMINAMATH_CALUDE_total_pies_baked_l1403_140381

/-- The number of pies Eddie can bake in a day -/
def eddie_pies_per_day : ℕ := 3

/-- The number of pies Eddie's sister can bake in a day -/
def sister_pies_per_day : ℕ := 6

/-- The number of pies Eddie's mother can bake in a day -/
def mother_pies_per_day : ℕ := 8

/-- The number of days they will bake pies -/
def days : ℕ := 7

/-- Theorem stating the total number of pies baked in 7 days -/
theorem total_pies_baked : 
  (eddie_pies_per_day * days + sister_pies_per_day * days + mother_pies_per_day * days) = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_baked_l1403_140381


namespace NUMINAMATH_CALUDE_koi_fish_count_l1403_140303

theorem koi_fish_count : ∃ k : ℕ, (2 * k - 14 = 64) ∧ (k = 39) := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_count_l1403_140303


namespace NUMINAMATH_CALUDE_fraction_invariance_l1403_140354

theorem fraction_invariance (x y : ℝ) (hx : x ≠ 0) : (y + x) / x = (3*y + 3*x) / (3*x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1403_140354


namespace NUMINAMATH_CALUDE_contrapositive_odd_sum_l1403_140391

theorem contrapositive_odd_sum (x y : ℤ) :
  (¬(Odd (x + y)) → ¬(Odd x ∧ Odd y)) ↔
  (∀ x y : ℤ, (Odd x ∧ Odd y) → Odd (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_odd_sum_l1403_140391


namespace NUMINAMATH_CALUDE_blanket_collection_l1403_140323

/-- Proves the number of blankets collected on the last day of a three-day collection drive -/
theorem blanket_collection (team_size : ℕ) (first_day_per_person : ℕ) (total_blankets : ℕ) : 
  team_size = 15 → 
  first_day_per_person = 2 → 
  total_blankets = 142 → 
  total_blankets - (team_size * first_day_per_person + 3 * (team_size * first_day_per_person)) = 22 := by
  sorry

#check blanket_collection

end NUMINAMATH_CALUDE_blanket_collection_l1403_140323


namespace NUMINAMATH_CALUDE_independent_recruitment_probabilities_l1403_140383

/-- Represents a student in the independent recruitment process -/
inductive Student
| A
| B
| C

/-- The probability of passing the review for each student -/
def review_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.5
  | Student.B => 0.6
  | Student.C => 0.4

/-- The probability of passing the cultural test after passing the review for each student -/
def cultural_test_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.6
  | Student.B => 0.5
  | Student.C => 0.75

/-- The probability of obtaining qualification for independent recruitment for each student -/
def qualification_prob (s : Student) : ℝ :=
  review_prob s * cultural_test_prob s

/-- The number of students who obtain qualification for independent recruitment -/
def num_qualified : Fin 4 → ℝ
| 0 => (1 - qualification_prob Student.A) * (1 - qualification_prob Student.B) * (1 - qualification_prob Student.C)
| 1 => 3 * qualification_prob Student.A * (1 - qualification_prob Student.B) * (1 - qualification_prob Student.C)
| 2 => 3 * qualification_prob Student.A * qualification_prob Student.B * (1 - qualification_prob Student.C)
| 3 => qualification_prob Student.A * qualification_prob Student.B * qualification_prob Student.C

/-- The expected value of the number of students who obtain qualification -/
def expected_num_qualified : ℝ :=
  1 * num_qualified 1 + 2 * num_qualified 2 + 3 * num_qualified 3

theorem independent_recruitment_probabilities :
  (∀ s : Student, qualification_prob s = 0.3) ∧ expected_num_qualified = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_independent_recruitment_probabilities_l1403_140383


namespace NUMINAMATH_CALUDE_log_inequality_l1403_140361

theorem log_inequality (x : ℝ) : 
  0 < x → x < 4 → (Real.log x / Real.log 9 ≥ (Real.log (Real.sqrt (1 - x / 4)) / Real.log 3)^2 ↔ x = 2 ∨ (4/5 ≤ x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l1403_140361


namespace NUMINAMATH_CALUDE_anna_apples_total_l1403_140385

def apples_eaten (tuesday wednesday thursday : ℕ) : ℕ :=
  tuesday + wednesday + thursday

theorem anna_apples_total :
  ∀ (tuesday : ℕ),
    tuesday = 4 →
    ∀ (wednesday thursday : ℕ),
      wednesday = 2 * tuesday →
      thursday = tuesday / 2 →
      apples_eaten tuesday wednesday thursday = 14 := by
sorry

end NUMINAMATH_CALUDE_anna_apples_total_l1403_140385


namespace NUMINAMATH_CALUDE_sin_sum_simplification_l1403_140338

theorem sin_sum_simplification :
  Real.sin (119 * π / 180) * Real.sin (181 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_simplification_l1403_140338


namespace NUMINAMATH_CALUDE_max_b_no_lattice_points_l1403_140332

theorem max_b_no_lattice_points :
  let max_b : ℚ := 67 / 199
  ∀ m : ℚ, 1/3 < m → m < max_b →
    ∀ x : ℕ, 0 < x → x ≤ 200 →
      ∀ y : ℤ, y ≠ ⌊m * x + 3⌋ ∧
    ∀ b : ℚ, b > max_b →
      ∃ m : ℚ, 1/3 < m ∧ m < b ∧
        ∃ x : ℕ, 0 < x ∧ x ≤ 200 ∧
          ∃ y : ℤ, y = ⌊m * x + 3⌋ := by
  sorry

end NUMINAMATH_CALUDE_max_b_no_lattice_points_l1403_140332


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1403_140355

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + m + 3 = 0 ∧ y^2 + m*y + m + 3 = 0) ↔ 
  (m < -2 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1403_140355
