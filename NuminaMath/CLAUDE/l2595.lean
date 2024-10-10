import Mathlib

namespace three_numbers_sum_l2595_259514

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →
  y = 10 →
  (x + y + z) / 3 = x + 20 →
  (x + y + z) / 3 = z - 25 →
  x + y + z = 45 := by
  sorry

end three_numbers_sum_l2595_259514


namespace f_has_max_iff_a_ge_e_l2595_259523

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ a then Real.log x else a / x

-- Theorem statement
theorem f_has_max_iff_a_ge_e (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), x > 0 → f a x ≤ M) ↔ a ≥ Real.exp 1 := by
  sorry

end

end f_has_max_iff_a_ge_e_l2595_259523


namespace exponents_gp_iff_n_3_6_10_l2595_259501

/-- A function that returns the sequence of exponents in the prime factorization of n! --/
def exponents_of_factorial (n : ℕ) : List ℕ :=
  sorry

/-- Check if a list of natural numbers forms a geometric progression --/
def is_geometric_progression (l : List ℕ) : Prop :=
  sorry

/-- The main theorem stating that the exponents in the prime factorization of n!
    form a geometric progression if and only if n is 3, 6, or 10 --/
theorem exponents_gp_iff_n_3_6_10 (n : ℕ) :
  n ≥ 3 → (is_geometric_progression (exponents_of_factorial n) ↔ n = 3 ∨ n = 6 ∨ n = 10) :=
sorry

end exponents_gp_iff_n_3_6_10_l2595_259501


namespace erased_length_cm_l2595_259507

-- Define the original length in meters
def original_length_m : ℝ := 1

-- Define the final length in centimeters
def final_length_cm : ℝ := 76

-- Define the conversion factor from meters to centimeters
def m_to_cm : ℝ := 100

-- Theorem to prove
theorem erased_length_cm : 
  (original_length_m * m_to_cm - final_length_cm) = 24 := by
  sorry

end erased_length_cm_l2595_259507


namespace smallest_n_square_and_cube_l2595_259586

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧ 
  (∀ (n' : ℕ), n' > 0 → 
    (∃ (k : ℕ), 5 * n' = k^2) → 
    (∃ (m : ℕ), 3 * n' = m^3) → 
    n ≤ n') ∧
  n = 225 := by
sorry

end smallest_n_square_and_cube_l2595_259586


namespace anna_left_probability_l2595_259562

/-- The probability that the girl on the right is lying -/
def P_right_lying : ℚ := 1/4

/-- The probability that the girl on the left is lying -/
def P_left_lying : ℚ := 1/5

/-- The event that Anna is sitting on the left -/
def A : Prop := sorry

/-- The event that both girls claim to be Brigitte -/
def B : Prop := sorry

/-- The probability of event A given event B -/
def P_A_given_B : ℚ := sorry

theorem anna_left_probability : P_A_given_B = 3/7 := by sorry

end anna_left_probability_l2595_259562


namespace square_difference_of_even_integers_l2595_259582

theorem square_difference_of_even_integers (x y : ℕ) : 
  Even x → Even y → x > y → x + y = 68 → x - y = 20 → x^2 - y^2 = 1360 :=
by
  sorry

end square_difference_of_even_integers_l2595_259582


namespace jameson_medals_l2595_259519

theorem jameson_medals (total_medals : ℕ) (badminton_medals : ℕ) 
  (h1 : total_medals = 20)
  (h2 : badminton_medals = 5) :
  ∃ track_medals : ℕ, 
    track_medals + 2 * track_medals + badminton_medals = total_medals ∧ 
    track_medals = 5 := by
  sorry

end jameson_medals_l2595_259519


namespace minoxidil_concentration_l2595_259524

/-- Proves that the initial concentration of Minoxidil is 2% --/
theorem minoxidil_concentration 
  (initial_volume : ℝ) 
  (added_volume : ℝ) 
  (added_concentration : ℝ) 
  (final_volume : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 70)
  (h2 : added_volume = 35)
  (h3 : added_concentration = 0.05)
  (h4 : final_volume = 105)
  (h5 : final_concentration = 0.03)
  (h6 : final_volume = initial_volume + added_volume) :
  ∃ (initial_concentration : ℝ), 
    initial_concentration = 0.02 ∧ 
    initial_volume * initial_concentration + added_volume * added_concentration = 
    final_volume * final_concentration :=
by sorry

end minoxidil_concentration_l2595_259524


namespace board_traversal_paths_bound_l2595_259550

/-- A piece on an n × n board that can move one step at a time (up, down, left, or right) --/
structure Piece (n : ℕ) where
  x : Fin n
  y : Fin n

/-- The number of unique paths to traverse the entire n × n board --/
def t (n : ℕ) : ℕ := sorry

/-- The theorem to be proved --/
theorem board_traversal_paths_bound {n : ℕ} (h : n ≥ 100) :
  (1.25 : ℝ) < (t n : ℝ) ^ (1 / (n^2 : ℝ)) ∧ (t n : ℝ) ^ (1 / (n^2 : ℝ)) < Real.sqrt 3 := by
  sorry

end board_traversal_paths_bound_l2595_259550


namespace geometric_sequence_first_term_l2595_259512

/-- A geometric sequence with common ratio 2 and fourth term 16 has first term equal to 2 -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 4 = 16 →                    -- fourth term is 16
  a 1 = 2 :=                    -- prove that first term is 2
by
  sorry


end geometric_sequence_first_term_l2595_259512


namespace class_mood_distribution_l2595_259542

theorem class_mood_distribution (total_children : Nat) (happy_children : Nat) (sad_children : Nat) (anxious_children : Nat)
  (total_boys : Nat) (total_girls : Nat) (happy_boys : Nat) (sad_girls : Nat) (anxious_girls : Nat)
  (h1 : total_children = 80)
  (h2 : happy_children = 25)
  (h3 : sad_children = 15)
  (h4 : anxious_children = 20)
  (h5 : total_boys = 35)
  (h6 : total_girls = 45)
  (h7 : happy_boys = 10)
  (h8 : sad_girls = 6)
  (h9 : anxious_girls = 12)
  (h10 : total_children = total_boys + total_girls) :
  (total_boys - (happy_boys + (sad_children - sad_girls) + (anxious_children - anxious_girls)) = 8) ∧
  (happy_children - happy_boys = 15) :=
by sorry

end class_mood_distribution_l2595_259542


namespace M_divisible_by_49_l2595_259513

/-- M is the concatenated number formed by writing integers from 1 to 48 in order -/
def M : ℕ := sorry

/-- Theorem stating that M is divisible by 49 -/
theorem M_divisible_by_49 : 49 ∣ M := by sorry

end M_divisible_by_49_l2595_259513


namespace unique_colors_count_l2595_259522

/-- The total number of unique colored pencils owned by Serenity, Jordan, and Alex -/
def total_unique_colors (serenity_colors jordan_colors alex_colors
                         serenity_jordan_shared serenity_alex_shared jordan_alex_shared
                         all_shared : ℕ) : ℕ :=
  serenity_colors + jordan_colors + alex_colors
  - (serenity_jordan_shared + serenity_alex_shared + jordan_alex_shared - 2 * all_shared)
  - all_shared

/-- Theorem stating the total number of unique colored pencils -/
theorem unique_colors_count :
  total_unique_colors 24 36 30 8 5 10 3 = 73 := by
  sorry

end unique_colors_count_l2595_259522


namespace calculate_expression_l2595_259547

theorem calculate_expression : 18 - (-16) / (2^3) = 20 := by
  sorry

end calculate_expression_l2595_259547


namespace age_difference_l2595_259528

theorem age_difference (A B C : ℕ) (h1 : C = A - 17) : A + B - (B + C) = 17 := by
  sorry

end age_difference_l2595_259528


namespace sequence_relation_l2595_259508

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : (y n)^2 = 3 * (x n)^2 + 1 := by
  sorry

end sequence_relation_l2595_259508


namespace area_STUV_l2595_259576

/-- A semicircle with an inscribed square PQRS and another square STUV -/
structure SemicircleWithSquares where
  /-- The radius of the semicircle -/
  r : ℝ
  /-- The side length of the inscribed square PQRS -/
  s : ℝ
  /-- The side length of the square STUV -/
  x : ℝ
  /-- The radius is determined by the side length of PQRS -/
  h_radius : r = s * Real.sqrt 2 / 2
  /-- PQRS is inscribed in the semicircle -/
  h_inscribed : s^2 + s^2 = (2*r)^2
  /-- STUV has a vertex on the semicircle -/
  h_on_semicircle : 6^2 + x^2 = r^2

/-- The area of square STUV is 36 -/
theorem area_STUV (c : SemicircleWithSquares) : c.x^2 = 36 := by
  sorry

#check area_STUV

end area_STUV_l2595_259576


namespace oranges_given_eq_difference_l2595_259556

/-- The number of oranges Clarence gave to Joyce -/
def oranges_given : ℝ := sorry

/-- Clarence's initial number of oranges -/
def initial_oranges : ℝ := 5.0

/-- Clarence's remaining number of oranges -/
def remaining_oranges : ℝ := 2.0

/-- Theorem stating that the number of oranges given is equal to the difference between initial and remaining oranges -/
theorem oranges_given_eq_difference : 
  oranges_given = initial_oranges - remaining_oranges := by sorry

end oranges_given_eq_difference_l2595_259556


namespace cube_paint_theorem_l2595_259500

/-- Given a cube of side length n, prove that if exactly one-third of the total number of faces
    of the n^3 unit cubes (obtained by cutting the original cube) are red, then n = 3. -/
theorem cube_paint_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by
  sorry

end cube_paint_theorem_l2595_259500


namespace absolute_value_equation_solution_l2595_259598

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 6| = 3*x + 6) ↔ (x = 0) :=
by sorry

end absolute_value_equation_solution_l2595_259598


namespace f_inequality_l2595_259552

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, f (x + 1) = f (-(x + 1)))
variable (h2 : ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y)
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ < 0)
variable (h4 : x₂ > 0)
variable (h5 : x₁ + x₂ < -2)

-- State the theorem
theorem f_inequality : f (-x₁) > f (-x₂) := by sorry

end f_inequality_l2595_259552


namespace sine_cosine_inequality_l2595_259555

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end sine_cosine_inequality_l2595_259555


namespace rectangular_hall_dimensions_l2595_259516

theorem rectangular_hall_dimensions (length width area : ℝ) : 
  width = (1/2) * length →
  area = length * width →
  area = 200 →
  length - width = 10 := by
sorry

end rectangular_hall_dimensions_l2595_259516


namespace salon_buys_33_cans_l2595_259574

/-- Represents the number of cans of hairspray a salon buys daily. -/
def salon_hairspray_cans (customers : ℕ) (cans_per_customer : ℕ) (extra_cans : ℕ) : ℕ :=
  customers * cans_per_customer + extra_cans

/-- Theorem stating that the salon buys 33 cans of hairspray daily. -/
theorem salon_buys_33_cans :
  salon_hairspray_cans 14 2 5 = 33 := by
  sorry

#eval salon_hairspray_cans 14 2 5

end salon_buys_33_cans_l2595_259574


namespace max_product_sum_2004_l2595_259540

theorem max_product_sum_2004 :
  ∃ (a b : ℤ), a + b = 2004 ∧
  ∀ (x y : ℤ), x + y = 2004 → x * y ≤ a * b ∧
  a * b = 1004004 := by
sorry

end max_product_sum_2004_l2595_259540


namespace paul_cookie_price_l2595_259558

/-- Represents a cookie baker -/
structure Baker where
  name : String
  num_cookies : ℕ
  price_per_cookie : ℚ

/-- The total amount of dough used by all bakers -/
def total_dough : ℝ := 120

theorem paul_cookie_price 
  (art paul : Baker)
  (h1 : art.name = "Art")
  (h2 : paul.name = "Paul")
  (h3 : art.num_cookies = 10)
  (h4 : paul.num_cookies = 20)
  (h5 : art.price_per_cookie = 1/2)
  (h6 : (total_dough / art.num_cookies) = (total_dough / paul.num_cookies)) :
  paul.price_per_cookie = 1/4 := by
sorry

end paul_cookie_price_l2595_259558


namespace self_inverse_matrix_l2595_259592

theorem self_inverse_matrix (c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; c, d]
  A * A = 1 → c = 7.5 ∧ d = -4 := by
sorry

end self_inverse_matrix_l2595_259592


namespace tens_digit_of_11_pow_2045_l2595_259560

theorem tens_digit_of_11_pow_2045 : ∃ k : ℕ, 11^2045 ≡ 50 + k [ZMOD 100] :=
by
  sorry

end tens_digit_of_11_pow_2045_l2595_259560


namespace elvis_album_songs_l2595_259539

/-- Calculates the number of songs on Elvis' new album given the studio time constraints. -/
theorem elvis_album_songs (
  total_studio_time : ℕ
  ) (record_time : ℕ) (edit_time : ℕ) (write_time : ℕ) 
  (h1 : total_studio_time = 5 * 60)  -- 5 hours in minutes
  (h2 : record_time = 12)            -- 12 minutes to record each song
  (h3 : edit_time = 30)              -- 30 minutes to edit all songs
  (h4 : write_time = 15)             -- 15 minutes to write each song
  : ℕ := by
  
  -- The number of songs is equal to the available time for writing and recording
  -- divided by the time needed for writing and recording one song
  have num_songs : ℕ := (total_studio_time - edit_time) / (write_time + record_time)
  
  -- Prove that num_songs equals 10
  sorry

#eval (5 * 60 - 30) / (15 + 12)  -- Should evaluate to 10

end elvis_album_songs_l2595_259539


namespace elderly_people_not_well_defined_l2595_259544

-- Define a structure for a potential set
structure PotentialSet where
  elements : String
  is_well_defined : Bool

-- Define the criteria for a well-defined set
def is_well_defined_set (s : PotentialSet) : Prop :=
  s.is_well_defined = true

-- Define the set of elderly people
def elderly_people : PotentialSet :=
  { elements := "All elderly people", is_well_defined := false }

-- Theorem stating that the set of elderly people is not well-defined
theorem elderly_people_not_well_defined : ¬(is_well_defined_set elderly_people) := by
  sorry

#check elderly_people_not_well_defined

end elderly_people_not_well_defined_l2595_259544


namespace min_posts_for_specific_plot_l2595_259511

/-- Calculates the number of fence posts required for a given length -/
def posts_for_length (length : ℕ) : ℕ :=
  length / 10 + 1

/-- Represents a rectangular garden plot -/
structure GardenPlot where
  width : ℕ
  length : ℕ
  wall_length : ℕ

/-- Calculates the minimum number of fence posts required for a garden plot -/
def min_posts (plot : GardenPlot) : ℕ :=
  posts_for_length plot.length + 2 * (posts_for_length plot.width - 1)

/-- Theorem stating the minimum number of posts for the specific garden plot -/
theorem min_posts_for_specific_plot :
  ∃ (plot : GardenPlot), plot.width = 30 ∧ plot.length = 50 ∧ plot.wall_length = 80 ∧ min_posts plot = 12 := by
  sorry

end min_posts_for_specific_plot_l2595_259511


namespace simple_interest_rate_calculation_l2595_259553

theorem simple_interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * (1 + 7 * R / 100) = 7 / 6 * P → R = 100 / 49 := by
  sorry

end simple_interest_rate_calculation_l2595_259553


namespace total_letters_written_l2595_259584

/-- The number of letters Nathan can write in one hour -/
def nathan_speed : ℕ := 25

/-- Jacob's writing speed relative to Nathan's -/
def jacob_relative_speed : ℕ := 2

/-- The number of hours they write together -/
def total_hours : ℕ := 10

/-- Theorem stating the total number of letters Jacob and Nathan can write together -/
theorem total_letters_written : 
  (nathan_speed + jacob_relative_speed * nathan_speed) * total_hours = 750 := by
  sorry

end total_letters_written_l2595_259584


namespace complete_square_m_values_l2595_259566

/-- A polynomial of the form x^2 + mx + 4 can be factored using the complete square formula -/
def is_complete_square (m : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 4 = (x + a)^2

/-- If a polynomial x^2 + mx + 4 can be factored using the complete square formula,
    then m = 4 or m = -4 -/
theorem complete_square_m_values (m : ℝ) :
  is_complete_square m → m = 4 ∨ m = -4 :=
by sorry

end complete_square_m_values_l2595_259566


namespace inequality_solution_set_l2595_259506

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1) * x + 1 > 0) ↔ -1 < a ∧ a < 3 := by sorry

end inequality_solution_set_l2595_259506


namespace no_snow_probability_l2595_259570

def probability_of_snow : ℚ := 2/3

def days : ℕ := 5

theorem no_snow_probability :
  (1 - probability_of_snow) ^ days = 1/243 := by
  sorry

end no_snow_probability_l2595_259570


namespace arithmetic_expression_equality_l2595_259578

theorem arithmetic_expression_equality : 68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 := by
  sorry

end arithmetic_expression_equality_l2595_259578


namespace angle_less_iff_sin_less_l2595_259569

theorem angle_less_iff_sin_less (A B : Real) (hA : 0 < A) (hB : B < π) (hAB : A + B < π) :
  A < B ↔ Real.sin A < Real.sin B := by sorry

end angle_less_iff_sin_less_l2595_259569


namespace complex_z_value_l2595_259561

def is_negative_real (z : ℂ) : Prop := ∃ (r : ℝ), r < 0 ∧ z = r

def is_purely_imaginary (z : ℂ) : Prop := ∃ (r : ℝ), z = r * Complex.I

theorem complex_z_value (z : ℂ) 
  (h1 : is_negative_real ((z - 3*Complex.I) / (z + Complex.I)))
  (h2 : is_purely_imaginary ((z - 3) / (z + 1))) :
  z = Complex.I * Real.sqrt 3 := by
sorry

end complex_z_value_l2595_259561


namespace function_property_l2595_259559

theorem function_property (f : ℕ → ℝ) :
  f 1 = 3/2 ∧
  (∀ x y : ℕ, f (x + y) = (1 + y / (x + 1 : ℝ)) * f x + (1 + x / (y + 1 : ℝ)) * f y + x^2 * y + x * y + x * y^2) →
  ∀ x : ℕ, f x = (1/4 : ℝ) * x * (x + 1) * (2 * x + 1) :=
by sorry

end function_property_l2595_259559


namespace greatest_divisor_with_remainders_l2595_259593

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a > r1) (hb : b > r2) :
  let d := Nat.gcd (a - r1) (b - r2)
  d = Nat.gcd a b ∧ 
  a % d = r1 ∧ 
  b % d = r2 ∧ 
  ∀ m : ℕ, m > d → (a % m = r1 ∧ b % m = r2) → False :=
by sorry

end greatest_divisor_with_remainders_l2595_259593


namespace sonika_deposit_l2595_259526

/-- Calculates the final amount after simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem sonika_deposit :
  ∀ (P R : ℝ),
  simpleInterest P (R / 100) 3 = 10200 →
  simpleInterest P ((R + 2) / 100) 3 = 10680 →
  P = 8000 := by
sorry

end sonika_deposit_l2595_259526


namespace consecutive_numbers_with_divisible_digit_sums_l2595_259517

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exist two consecutive natural numbers whose sum of digits are both divisible by 7 -/
theorem consecutive_numbers_with_divisible_digit_sums :
  ∃ n : ℕ, 7 ∣ sumOfDigits n ∧ 7 ∣ sumOfDigits (n + 1) := by sorry

end consecutive_numbers_with_divisible_digit_sums_l2595_259517


namespace f_always_negative_iff_a_in_range_l2595_259571

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem f_always_negative_iff_a_in_range :
  (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 := by sorry

end f_always_negative_iff_a_in_range_l2595_259571


namespace seven_lines_regions_l2595_259557

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- Seven lines in a plane with no two parallel and no three concurrent -/
def seven_lines : ℕ := 7

theorem seven_lines_regions :
  regions seven_lines = 29 := by sorry

end seven_lines_regions_l2595_259557


namespace even_function_implies_a_zero_l2595_259564

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x + a|

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end even_function_implies_a_zero_l2595_259564


namespace least_prime_factor_of_8_pow_4_minus_8_pow_3_l2595_259579

theorem least_prime_factor_of_8_pow_4_minus_8_pow_3 :
  Nat.minFac (8^4 - 8^3) = 2 := by
  sorry

end least_prime_factor_of_8_pow_4_minus_8_pow_3_l2595_259579


namespace downstream_distance_l2595_259573

-- Define the given constants
def boat_speed : ℝ := 22
def stream_speed : ℝ := 5
def time_downstream : ℝ := 2

-- Define the theorem
theorem downstream_distance :
  let effective_speed := boat_speed + stream_speed
  effective_speed * time_downstream = 54 := by
  sorry

end downstream_distance_l2595_259573


namespace jims_age_fraction_l2595_259580

theorem jims_age_fraction (tom_age_5_years_ago : ℕ) (jim_age_in_2_years : ℕ) : 
  tom_age_5_years_ago = 32 →
  jim_age_in_2_years = 29 →
  ∃ f : ℚ, 
    (jim_age_in_2_years - 9 : ℚ) = f * (tom_age_5_years_ago + 2 : ℚ) + 5 ∧ 
    f = 1/2 := by
  sorry

end jims_age_fraction_l2595_259580


namespace only_zero_factorizable_l2595_259599

/-- The polynomial we're considering -/
def poly (m : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + m*y - 2*m

/-- A linear factor with integer coefficients -/
def linear_factor (a b c : ℤ) (x y : ℤ) : ℤ := a*x + b*y + c

/-- Predicate to check if the polynomial can be factored into two linear factors with integer coefficients -/
def can_be_factored (m : ℤ) : Prop :=
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℤ), ∀ (x y : ℤ),
    poly m x y = linear_factor a₁ b₁ c₁ x y * linear_factor a₂ b₂ c₂ x y

theorem only_zero_factorizable :
  ∀ m : ℤ, can_be_factored m ↔ m = 0 :=
sorry

end only_zero_factorizable_l2595_259599


namespace platform_length_l2595_259543

theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_platform = 27) 
  (h3 : time_pole = 18) : 
  ∃ (platform_length : ℝ), platform_length = 150 ∧ 
  (train_length + platform_length) / time_platform = train_length / time_pole :=
sorry

end platform_length_l2595_259543


namespace product_formula_l2595_259529

theorem product_formula (a b : ℕ) :
  (100 - a) * (100 + b) = ((b + (200 - a) - 100) * 100) - a * b := by
  sorry

end product_formula_l2595_259529


namespace angle_measure_in_acute_triangle_l2595_259515

theorem angle_measure_in_acute_triangle (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < Real.pi/2 →
  0 < B ∧ B < Real.pi/2 →
  0 < C ∧ C < Real.pi/2 →
  A + B + C = Real.pi →
  a = Real.sin B * (Real.sin C / Real.sin A) →
  b = Real.sin C * (Real.sin A / Real.sin B) →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  A = Real.pi/3 := by
sorry

end angle_measure_in_acute_triangle_l2595_259515


namespace optimal_pencil_purchase_l2595_259588

theorem optimal_pencil_purchase : ∀ (x y : ℕ),
  -- Cost constraint
  27 * x + 23 * y ≤ 940 →
  -- Difference constraint
  y ≤ x + 10 →
  -- Optimality
  (∀ (x' y' : ℕ), 27 * x' + 23 * y' ≤ 940 → y' ≤ x' + 10 → x' + y' ≤ x + y) →
  -- Minimizing red pencils
  (∀ (x' : ℕ), x' < x → ∃ (y' : ℕ), 27 * x' + 23 * y' ≤ 940 ∧ y' ≤ x' + 10 ∧ x' + y' < x + y) →
  x = 14 ∧ y = 24 :=
by sorry

end optimal_pencil_purchase_l2595_259588


namespace max_plumber_earnings_l2595_259581

def toilet_rate : ℕ := 50
def shower_rate : ℕ := 40
def sink_rate : ℕ := 30

def job1_earnings : ℕ := 3 * toilet_rate + 3 * sink_rate
def job2_earnings : ℕ := 2 * toilet_rate + 5 * sink_rate
def job3_earnings : ℕ := 1 * toilet_rate + 2 * shower_rate + 3 * sink_rate

theorem max_plumber_earnings :
  max job1_earnings (max job2_earnings job3_earnings) = 250 := by
  sorry

end max_plumber_earnings_l2595_259581


namespace water_in_bucket_l2595_259590

theorem water_in_bucket (initial_amount poured_out : ℚ) 
  (h1 : initial_amount = 15/8)
  (h2 : poured_out = 9/8) : 
  initial_amount - poured_out = 3/4 := by
  sorry

end water_in_bucket_l2595_259590


namespace braking_velocities_l2595_259525

/-- The displacement function representing the braking system -/
def s (t : ℝ) : ℝ := -3 * t^3 + t^2 + 20

/-- The velocity function (derivative of displacement) -/
def v (t : ℝ) : ℝ := -9 * t^2 + 2 * t

/-- Theorem stating the average and instantaneous velocities during braking -/
theorem braking_velocities :
  (∀ t ∈ Set.Icc 0 2, s t ≥ 0) →  -- Braking completes within 2 seconds
  ((s 1 - s 0) / 1 = -2) ∧        -- Average velocity in first second
  ((s 2 - s 1) / 1 = -18) ∧       -- Average velocity between 1 and 2 seconds
  (v 1 = -7)                      -- Instantaneous velocity at 1 second
:= by sorry

end braking_velocities_l2595_259525


namespace a_50_equals_6_5_l2595_259575

-- Define the sequence a_n
def a : ℕ → ℚ
| n => sorry

-- Theorem statement
theorem a_50_equals_6_5 : a 50 = 6/5 := by sorry

end a_50_equals_6_5_l2595_259575


namespace arrangement_theorem_l2595_259538

def arrangement_count (n : ℕ) (zeros : ℕ) : ℕ :=
  if n = 27 ∧ zeros = 13 then 14
  else if n = 26 ∧ zeros = 13 then 105
  else 0

theorem arrangement_theorem (n : ℕ) (zeros : ℕ) :
  (n = 27 ∨ n = 26) ∧ zeros = 13 →
  arrangement_count n zeros = 
    (if n = 27 then 14 else 105) :=
by sorry

end arrangement_theorem_l2595_259538


namespace f_properties_l2595_259534

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x^2 - 3
  else if x < 0 then -x^2 + 3
  else 0

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x^2 - 3) ∧  -- given condition for x > 0
  (∀ x < 0, f x = -x^2 + 3) ∧  -- prove this for x < 0
  (f 0 = 0) ∧  -- prove f(0) = 0
  ({x : ℝ | f x = 2 * x} = {-3, 0, 3}) :=  -- prove the solution set
by sorry

end f_properties_l2595_259534


namespace joy_reading_time_l2595_259505

/-- Given that Joy can read 8 pages in 20 minutes, prove that it takes her 5 hours to read 120 pages. -/
theorem joy_reading_time : 
  -- Define Joy's reading speed
  let pages_per_20_min : ℚ := 8
  let total_pages : ℚ := 120
  -- Calculate the time in hours
  let time_in_hours : ℚ := (total_pages / pages_per_20_min) * (20 / 60)
  -- Prove that the time is 5 hours
  ∀ (pages_per_20_min total_pages time_in_hours : ℚ), 
    pages_per_20_min = 8 → 
    total_pages = 120 → 
    time_in_hours = (total_pages / pages_per_20_min) * (20 / 60) → 
    time_in_hours = 5 := by
  sorry


end joy_reading_time_l2595_259505


namespace zero_real_necessary_not_sufficient_for_purely_imaginary_l2595_259549

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

theorem zero_real_necessary_not_sufficient_for_purely_imaginary :
  ∃ (a b : ℝ), (isPurelyImaginary (Complex.mk a b) → a = 0) ∧
                ¬(a = 0 → isPurelyImaginary (Complex.mk a b)) :=
by sorry

end zero_real_necessary_not_sufficient_for_purely_imaginary_l2595_259549


namespace mass_of_man_equals_240kg_l2595_259531

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating that the mass of the man is 240 kg under given conditions. -/
theorem mass_of_man_equals_240kg 
  (boat_length : ℝ) 
  (boat_breadth : ℝ) 
  (boat_sinking : ℝ) 
  (water_density : ℝ) 
  (h1 : boat_length = 8) 
  (h2 : boat_breadth = 3) 
  (h3 : boat_sinking = 0.01) 
  (h4 : water_density = 1000) :
  mass_of_man boat_length boat_breadth boat_sinking water_density = 240 := by
  sorry

end mass_of_man_equals_240kg_l2595_259531


namespace ferry_distance_ratio_l2595_259577

/-- Represents a ferry with speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- The problem setup -/
def ferryProblem : Prop :=
  ∃ (P Q : Ferry),
    P.speed = 8 ∧
    P.time = 2 ∧
    Q.speed = P.speed + 4 ∧
    Q.time = P.time + 2 ∧
    Q.speed * Q.time / (P.speed * P.time) = 3

/-- The theorem to prove -/
theorem ferry_distance_ratio :
  ferryProblem := by sorry

end ferry_distance_ratio_l2595_259577


namespace count_valid_s_l2595_259585

def is_valid_sequence (n p q r s : ℕ) : Prop :=
  p < q ∧ q < r ∧ r < s ∧ s ≤ n ∧ 100 < p ∧
  ((q = p + 1 ∧ r = q + 1) ∨ (r = q + 1 ∧ s = r + 1) ∨ (q = p + 1 ∧ r = q + 1 ∧ s = r + 1))

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_removed (p q r s : ℕ) : ℕ := p + q + r + s

def remaining_sum (n p q r s : ℕ) : ℕ := sum_first_n n - sum_removed p q r s

def average_is_correct (n p q r s : ℕ) : Prop :=
  (remaining_sum n p q r s : ℚ) / (n - 4 : ℚ) = 89.5625

theorem count_valid_s (n : ℕ) : 
  (∃ p q r s, is_valid_sequence n p q r s ∧ average_is_correct n p q r s) →
  (∃! (valid_s : Finset ℕ), 
    (∀ s, s ∈ valid_s ↔ ∃ p q r, is_valid_sequence n p q r s ∧ average_is_correct n p q r s) ∧
    valid_s.card = 22) :=
sorry

end count_valid_s_l2595_259585


namespace train_crossing_time_l2595_259567

/-- Calculates the time taken for two trains to cross each other. -/
theorem train_crossing_time (length1 length2 speed1 speed2 initial_distance : ℝ) 
  (h1 : length1 = 135.5)
  (h2 : length2 = 167.2)
  (h3 : speed1 = 55)
  (h4 : speed2 = 43)
  (h5 : initial_distance = 250) :
  ∃ (time : ℝ), (abs (time - 20.3) < 0.1) ∧ 
  (time = (length1 + length2 + initial_distance) / ((speed1 + speed2) * (5/18))) :=
sorry

end train_crossing_time_l2595_259567


namespace square_difference_equals_24_l2595_259548

theorem square_difference_equals_24 (x y : ℝ) (h1 : x + y = 4) (h2 : x - y = 6) :
  x^2 - y^2 = 24 := by
  sorry

end square_difference_equals_24_l2595_259548


namespace megan_total_markers_l2595_259568

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := 217

/-- The number of markers Robert gave to Megan -/
def received_markers : ℕ := 109

/-- The total number of markers Megan has -/
def total_markers : ℕ := initial_markers + received_markers

theorem megan_total_markers : total_markers = 326 := by
  sorry

end megan_total_markers_l2595_259568


namespace one_in_linked_triple_l2595_259509

def is_linked (m n : ℕ+) : Prop :=
  (m.val ∣ 3 * n.val + 1) ∧ (n.val ∣ 3 * m.val + 1)

theorem one_in_linked_triple (a b c : ℕ+) :
  a ≠ b → b ≠ c → a ≠ c →
  is_linked a b → is_linked b c →
  1 ∈ ({a.val, b.val, c.val} : Set ℕ) :=
sorry

end one_in_linked_triple_l2595_259509


namespace prob_one_red_bag_with_three_red_balls_l2595_259545

/-- A bag containing red and non-red balls -/
structure Bag where
  red : ℕ
  nonRed : ℕ

/-- The probability of drawing exactly one red ball in two consecutive draws with replacement -/
def probOneRedWithReplacement (b : Bag) : ℚ :=
  let totalBalls := b.red + b.nonRed
  let probRed := b.red / totalBalls
  let probNonRed := b.nonRed / totalBalls
  2 * (probRed * probNonRed)

/-- The probability of drawing exactly one red ball in two consecutive draws without replacement -/
def probOneRedWithoutReplacement (b : Bag) : ℚ :=
  let totalBalls := b.red + b.nonRed
  let probRedFirst := b.red / totalBalls
  let probNonRedSecond := b.nonRed / (totalBalls - 1)
  2 * (probRedFirst * probNonRedSecond)

theorem prob_one_red_bag_with_three_red_balls :
  let b : Bag := { red := 3, nonRed := 3 }
  probOneRedWithReplacement b = 1/2 ∧ probOneRedWithoutReplacement b = 3/5 := by
  sorry

end prob_one_red_bag_with_three_red_balls_l2595_259545


namespace train_speed_l2595_259589

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 160)
  (h2 : bridge_length = 215)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end train_speed_l2595_259589


namespace cube_cutting_l2595_259591

theorem cube_cutting (n s : ℕ) : 
  n > s → 
  n^3 - s^3 = 152 → 
  n = 6 ∧ s = 4 := by sorry

end cube_cutting_l2595_259591


namespace smallest_divisible_by_495_l2595_259597

/-- Represents a number in the sequence with n digits of 5 -/
def sequenceNumber (n : ℕ) : ℕ :=
  (10^n - 1) / 9 * 5

/-- The target number we want to prove is the smallest divisible by 495 -/
def targetNumber : ℕ := sequenceNumber 18

/-- Checks if a number is in the sequence -/
def isInSequence (k : ℕ) : Prop :=
  ∃ n : ℕ, sequenceNumber n = k

theorem smallest_divisible_by_495 :
  (targetNumber % 495 = 0) ∧
  (∀ k : ℕ, k < targetNumber → isInSequence k → k % 495 ≠ 0) :=
sorry

end smallest_divisible_by_495_l2595_259597


namespace smallest_n_cubic_minus_n_divisibility_l2595_259520

theorem smallest_n_cubic_minus_n_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 0 < m ∧ m < n → 
    ∀ (k : ℕ), 1 ≤ k ∧ k ≤ m + 2 → (m^3 - m) % k = 0) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n + 2 ∧ (n^3 - n) % k ≠ 0) ∧
  n = 5 :=
by sorry

end smallest_n_cubic_minus_n_divisibility_l2595_259520


namespace sum_of_three_digit_numbers_divisible_by_37_l2595_259527

/-- A function that generates all possible three-digit numbers from three digits -/
def generateThreeDigitNumbers (a b c : ℕ) : List ℕ :=
  [100*a + 10*b + c,
   100*a + 10*c + b,
   100*b + 10*a + c,
   100*b + 10*c + a,
   100*c + 10*a + b,
   100*c + 10*b + a]

/-- Theorem: The sum of all possible three-digit numbers formed from three distinct non-zero digits is divisible by 37 -/
theorem sum_of_three_digit_numbers_divisible_by_37 
  (a b c : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  37 ∣ (List.sum (generateThreeDigitNumbers a b c)) :=
sorry

end sum_of_three_digit_numbers_divisible_by_37_l2595_259527


namespace ethanol_in_fuel_mix_l2595_259518

/-- Calculates the total ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- Theorem: The total ethanol in the specified fuel mix is 30 gallons -/
theorem ethanol_in_fuel_mix :
  total_ethanol 200 49.99999999999999 0.12 0.16 = 30 := by
  sorry

end ethanol_in_fuel_mix_l2595_259518


namespace expected_groups_l2595_259504

/-- The expected number of alternating groups in a random sequence of zeros and ones -/
theorem expected_groups (k m : ℕ) : 
  let total := k + m
  let prob_diff := (2 * k * m) / (total * (total - 1))
  1 + (total - 1) * prob_diff = 1 + (2 * k * m) / total := by
  sorry

end expected_groups_l2595_259504


namespace watermelon_weight_is_reasonable_l2595_259565

/-- The approximate weight of a typical watermelon in grams -/
def watermelon_weight : ℕ := 4000

/-- Predicate to determine if a given weight is a reasonable approximation for a watermelon -/
def is_reasonable_watermelon_weight (weight : ℕ) : Prop :=
  3500 ≤ weight ∧ weight ≤ 4500

/-- Theorem stating that the defined watermelon weight is a reasonable approximation -/
theorem watermelon_weight_is_reasonable : 
  is_reasonable_watermelon_weight watermelon_weight := by
  sorry

end watermelon_weight_is_reasonable_l2595_259565


namespace count_common_divisors_l2595_259537

/-- The number of positive divisors that 9240 and 10080 have in common -/
def common_divisors_count : ℕ := 32

/-- The first given number -/
def n1 : ℕ := 9240

/-- The second given number -/
def n2 : ℕ := 10080

/-- Theorem stating that the number of positive divisors that n1 and n2 have in common is equal to common_divisors_count -/
theorem count_common_divisors : 
  (Finset.filter (λ d => d ∣ n1 ∧ d ∣ n2) (Finset.range (min n1 n2 + 1))).card = common_divisors_count := by
  sorry


end count_common_divisors_l2595_259537


namespace monday_sales_proof_l2595_259563

/-- Represents the daily pastry sales for a week -/
structure WeeklySales :=
  (monday : ℕ)
  (increase_per_day : ℕ)
  (days_per_week : ℕ)

/-- Calculates the total sales for the week -/
def total_sales (s : WeeklySales) : ℕ :=
  s.days_per_week * s.monday + (s.days_per_week * (s.days_per_week - 1) * s.increase_per_day) / 2

/-- Theorem: If daily sales increase by 1 for 7 days and average 5 per day, Monday's sales were 2 -/
theorem monday_sales_proof (s : WeeklySales) 
  (h1 : s.increase_per_day = 1)
  (h2 : s.days_per_week = 7)
  (h3 : total_sales s / s.days_per_week = 5) :
  s.monday = 2 := by
  sorry


end monday_sales_proof_l2595_259563


namespace james_restaurant_revenue_l2595_259503

theorem james_restaurant_revenue :
  -- Define the constants
  let beef_amount : ℝ := 20
  let pork_amount : ℝ := beef_amount / 2
  let meat_per_meal : ℝ := 1.5
  let price_per_meal : ℝ := 20

  -- Calculate total meat
  let total_meat : ℝ := beef_amount + pork_amount

  -- Calculate number of meals
  let number_of_meals : ℝ := total_meat / meat_per_meal

  -- Calculate total revenue
  let total_revenue : ℝ := number_of_meals * price_per_meal

  -- Prove that the total revenue is $400
  total_revenue = 400 := by sorry

end james_restaurant_revenue_l2595_259503


namespace largest_unorderable_number_l2595_259535

def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_unorderable_number : 
  (∀ m > 43, is_orderable m) ∧ ¬(is_orderable 43) := by
  sorry

end largest_unorderable_number_l2595_259535


namespace savings_ratio_l2595_259595

theorem savings_ratio (initial_savings : ℝ) (final_savings : ℝ) (months : ℕ) 
  (h1 : initial_savings = 10)
  (h2 : final_savings = 160)
  (h3 : months = 5) :
  ∃ (ratio : ℝ), ratio = 2 ∧ final_savings = initial_savings * ratio ^ (months - 1) :=
sorry

end savings_ratio_l2595_259595


namespace semicircle_radius_l2595_259554

theorem semicircle_radius (D E F : ℝ × ℝ) : 
  -- Triangle DEF has a right angle at D
  (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0 →
  -- Area of semicircle on DE = 12.5π
  (1/2) * Real.pi * ((E.1 - D.1)^2 + (E.2 - D.2)^2) / 4 = 12.5 * Real.pi →
  -- Arc length of semicircle on DF = 7π
  Real.pi * ((F.1 - D.1)^2 + (F.2 - D.2)^2).sqrt / 2 = 7 * Real.pi →
  -- The radius of the semicircle on EF is √74
  ((E.1 - F.1)^2 + (E.2 - F.2)^2).sqrt / 2 = Real.sqrt 74 := by
sorry

end semicircle_radius_l2595_259554


namespace cow_husk_consumption_l2595_259541

/-- Given that 55 cows eat 55 bags of husk in 55 days, prove that one cow will eat one bag of husk in 55 days. -/
theorem cow_husk_consumption (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 55) 
  (h2 : num_bags = 55) 
  (h3 : num_days = 55) : 
  num_days = 55 := by
  sorry

end cow_husk_consumption_l2595_259541


namespace range_of_2a_minus_b_l2595_259587

theorem range_of_2a_minus_b (a b : ℝ) (ha : 2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := by
  sorry

end range_of_2a_minus_b_l2595_259587


namespace complex_division_simplification_l2595_259572

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  let z : ℂ := 5 / (2 - i)
  z = 2 + i := by sorry

end complex_division_simplification_l2595_259572


namespace valid_selections_count_l2595_259502

/-- The number of male intern teachers --/
def male_teachers : ℕ := 5

/-- The number of female intern teachers --/
def female_teachers : ℕ := 4

/-- The total number of intern teachers --/
def total_teachers : ℕ := male_teachers + female_teachers

/-- The number of teachers to be selected --/
def selected_teachers : ℕ := 3

/-- The number of ways to select 3 teachers from the total pool --/
def total_selections : ℕ := Nat.descFactorial total_teachers selected_teachers

/-- The number of ways to select 3 male teachers --/
def all_male_selections : ℕ := Nat.descFactorial male_teachers selected_teachers

/-- The number of ways to select 3 female teachers --/
def all_female_selections : ℕ := Nat.descFactorial female_teachers selected_teachers

/-- The number of valid selection schemes --/
def valid_selections : ℕ := total_selections - (all_male_selections + all_female_selections)

theorem valid_selections_count : valid_selections = 420 := by
  sorry

end valid_selections_count_l2595_259502


namespace sum_difference_is_450_l2595_259521

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_down_to_10 (n : ℕ) : ℕ := (n / 10) * 10

def kate_sum (n : ℕ) : ℕ := 
  (List.range n).map round_down_to_10 |> List.sum

theorem sum_difference_is_450 (n : ℕ) (h : n = 100) : 
  (sum_first_n n) - (kate_sum n) = 450 := by
  sorry

end sum_difference_is_450_l2595_259521


namespace complex_sum_real_imag_l2595_259533

theorem complex_sum_real_imag (a : ℝ) : 
  let z : ℂ := a / (2 + Complex.I) + (2 + Complex.I) / 5
  (z.re + z.im = 1) → a = 3 := by
sorry

end complex_sum_real_imag_l2595_259533


namespace sum_y_equals_375_l2595_259546

variable (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ)

-- Define the sum of x values
def sum_x : ℝ := x₁ + x₂ + x₃ + x₄ + x₅

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.67 * x + 54.9

-- State the theorem
theorem sum_y_equals_375 
  (h_sum_x : sum_x = 150) : 
  y₁ + y₂ + y₃ + y₄ + y₅ = 375 := by
  sorry

end sum_y_equals_375_l2595_259546


namespace crosswalk_stripe_distance_l2595_259536

/-- Given a street with parallel curbs and a crosswalk, prove the distance between stripes. -/
theorem crosswalk_stripe_distance
  (curb_distance : ℝ)
  (curb_length : ℝ)
  (stripe_length : ℝ)
  (h_curb_distance : curb_distance = 60)
  (h_curb_length : curb_length = 20)
  (h_stripe_length : stripe_length = 75) :
  curb_length * curb_distance / stripe_length = 16 := by
  sorry

end crosswalk_stripe_distance_l2595_259536


namespace train_passing_platform_l2595_259583

/-- The time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  platform_length = 390 →
  train_speed_kmh = 45 →
  (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end train_passing_platform_l2595_259583


namespace condition_necessary_not_sufficient_l2595_259594

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - 2*x ≤ 0 → -1 ≤ x ∧ x ≤ 3) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x > 0) := by
  sorry

end condition_necessary_not_sufficient_l2595_259594


namespace cake_box_width_l2595_259596

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

theorem cake_box_width :
  let carton := BoxDimensions.mk 25 42 60
  let cakeBox := BoxDimensions.mk 8 W 5
  let maxBoxes := 210
  boxVolume carton = maxBoxes * boxVolume cakeBox →
  W = 7.5 := by
sorry

end cake_box_width_l2595_259596


namespace complex_fourth_quadrant_range_l2595_259510

theorem complex_fourth_quadrant_range (a : ℝ) : 
  let z₁ : ℂ := 3 - a * Complex.I
  let z₂ : ℂ := 1 + 2 * Complex.I
  (0 < (z₁ / z₂).re ∧ (z₁ / z₂).im < 0) → (-6 < a ∧ a < 3/2) :=
by sorry

end complex_fourth_quadrant_range_l2595_259510


namespace partial_fraction_decomposition_l2595_259532

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ), 
    P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 ∧
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 5*x + 6) / ((x - 1)*(x - 4)*(x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end partial_fraction_decomposition_l2595_259532


namespace odd_function_property_l2595_259551

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 9

-- Theorem statement
theorem odd_function_property (h1 : ∀ x, f (-x) = -f x) 
                              (h2 : g (-2) = 3) : 
  f 2 = 6 := by
  sorry

end odd_function_property_l2595_259551


namespace max_value_of_expression_l2595_259530

theorem max_value_of_expression (x y : ℝ) (h : x^2 + y^2 ≠ 0) :
  (3*x^2 + 16*x*y + 15*y^2) / (x^2 + y^2) ≤ 19 := by
  sorry

end max_value_of_expression_l2595_259530
