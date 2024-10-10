import Mathlib

namespace car_speed_in_kmph_l4047_404795

/-- Proves that a car covering 375 meters in 15 seconds has a speed of 90 kmph -/
theorem car_speed_in_kmph : 
  let distance : ℝ := 375 -- distance in meters
  let time : ℝ := 15 -- time in seconds
  let conversion_factor : ℝ := 3.6 -- conversion factor from m/s to kmph
  (distance / time) * conversion_factor = 90 := by
  sorry

end car_speed_in_kmph_l4047_404795


namespace reading_time_calculation_gwendolyn_reading_time_l4047_404798

/-- Calculates the time needed to read a book given reading speed and book properties -/
theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : ℕ :=
  let sentences_per_page := paragraphs_per_page * sentences_per_paragraph
  let total_sentences := sentences_per_page * total_pages
  total_sentences / reading_speed

/-- Proves that Gwendolyn will take 225 hours to read the book -/
theorem gwendolyn_reading_time : 
  reading_time_calculation 200 30 15 100 = 225 := by
  sorry

end reading_time_calculation_gwendolyn_reading_time_l4047_404798


namespace correct_number_probability_l4047_404714

def first_three_digits : ℕ := 3

def last_four_digits : List ℕ := [0, 1, 1, 7]

def permutations_of_last_four : ℕ := 12

theorem correct_number_probability :
  (1 : ℚ) / (first_three_digits * permutations_of_last_four) = 1 / 36 := by
  sorry

end correct_number_probability_l4047_404714


namespace min_sum_of_squares_l4047_404755

theorem min_sum_of_squares (x y : ℝ) (h : x + y = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (a b : ℝ), a + b = 2 → x^2 + y^2 ≤ a^2 + b^2 := by
sorry

end min_sum_of_squares_l4047_404755


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l4047_404790

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The condition that a = 3 is sufficient for the lines to be parallel -/
theorem sufficient_condition (a : ℝ) :
  a = 3 → are_parallel 2 a 1 (a - 1) 3 (-2) :=
by sorry

/-- The condition that a = 3 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ 3 ∧ are_parallel 2 a 1 (a - 1) 3 (-2) :=
by sorry

/-- The main theorem stating that a = 3 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = 3 → are_parallel 2 a 1 (a - 1) 3 (-2)) ∧
  (∃ a : ℝ, a ≠ 3 ∧ are_parallel 2 a 1 (a - 1) 3 (-2)) :=
by sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l4047_404790


namespace expression_equals_x_plus_one_l4047_404791

theorem expression_equals_x_plus_one (x : ℝ) (h : x ≠ -1) :
  (x^2 + 2*x + 1) / (x + 1) = x + 1 := by
  sorry

end expression_equals_x_plus_one_l4047_404791


namespace factorial_99_trailing_zeros_l4047_404735

/-- The number of trailing zeros in n factorial -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 99! has 22 trailing zeros -/
theorem factorial_99_trailing_zeros :
  trailingZeros 99 = 22 := by
  sorry

end factorial_99_trailing_zeros_l4047_404735


namespace cylinder_height_relationship_l4047_404779

/-- Theorem: Relationship between heights of two cylinders with equal volumes and different radii -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by
  sorry

end cylinder_height_relationship_l4047_404779


namespace consecutive_non_prime_powers_l4047_404778

/-- For any positive integer n, there exists a positive integer m such that
    for all k in the range 0 ≤ k < n, m + k is not an integer power of a prime number. -/
theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ m : ℕ+, ∀ k : ℕ, k < n → ¬∃ (p : ℕ) (e : ℕ), Prime p ∧ (m + k : ℕ) = p ^ e :=
sorry

end consecutive_non_prime_powers_l4047_404778


namespace unique_six_digit_number_l4047_404772

def is_valid_number (n : ℕ) : Prop :=
  (100000 ≤ n) ∧ (n < 1000000) ∧ (n / 100000 = 1) ∧
  ((n % 100000) * 10 + 1 = 3 * n)

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
sorry

end unique_six_digit_number_l4047_404772


namespace sarahs_bowling_score_l4047_404786

theorem sarahs_bowling_score (s g : ℕ) : 
  s = g + 50 ∧ (s + g) / 2 = 105 → s = 130 := by
  sorry

end sarahs_bowling_score_l4047_404786


namespace geometric_sequence_properties_l4047_404770

theorem geometric_sequence_properties (a₁ q : ℝ) (h_q : -1 < q ∧ q < 0) :
  let a : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  (∀ n : ℕ, a n * a (n + 1) < 0) ∧
  (∀ n : ℕ, |a n| > |a (n + 1)|) :=
by sorry

end geometric_sequence_properties_l4047_404770


namespace f_increasing_on_interval_l4047_404769

-- Define the function f(x) = (x - 1)^2 - 2
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y := by
  sorry

-- Note: Set.Ici 1 represents the interval [1, +∞)

end f_increasing_on_interval_l4047_404769


namespace bridge_length_calculation_l4047_404715

/-- Calculates the length of a bridge given train and crossing parameters -/
theorem bridge_length_calculation
  (train_length : ℝ)
  (initial_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (wind_resistance_factor : ℝ)
  (h_train_length : train_length = 750)
  (h_initial_speed : initial_speed_kmh = 120)
  (h_crossing_time : crossing_time = 45)
  (h_wind_resistance : wind_resistance_factor = 0.9)
  : ∃ (bridge_length : ℝ), bridge_length = 600 :=
by
  sorry

#check bridge_length_calculation

end bridge_length_calculation_l4047_404715


namespace impossibleToDetectAllGenuine_l4047_404723

/-- Represents a diamond --/
inductive Diamond
| genuine
| fake

/-- Represents the expert's response --/
inductive ExpertResponse
| zero
| one
| two

/-- A strategy for the expert to choose which pair to reveal --/
def ExpertStrategy := (Diamond × Diamond × Diamond) → (Diamond × Diamond)

/-- The expert's response based on the chosen pair --/
def expertRespond (pair : Diamond × Diamond) : ExpertResponse :=
  match pair with
  | (Diamond.genuine, Diamond.genuine) => ExpertResponse.two
  | (Diamond.fake, Diamond.fake) => ExpertResponse.zero
  | _ => ExpertResponse.one

/-- Represents the state of our knowledge about the diamonds --/
def Knowledge := Fin 100 → Option Diamond

theorem impossibleToDetectAllGenuine :
  ∃ (strategy : ExpertStrategy),
    ∀ (initialState : Knowledge),
    ∃ (finalState : Knowledge),
      (∃ i j, i ≠ j ∧ finalState i = none ∧ finalState j = none) ∧
      (∀ k, finalState k ≠ some Diamond.genuine → initialState k ≠ some Diamond.genuine) :=
by sorry

end impossibleToDetectAllGenuine_l4047_404723


namespace correct_factorization_l4047_404719

theorem correct_factorization (a : ℝ) : 2*a^2 - 4*a + 2 = 2*(a-1)^2 := by
  sorry

end correct_factorization_l4047_404719


namespace books_per_shelf_l4047_404704

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 504) (h2 : num_shelves = 9) :
  total_books / num_shelves = 56 := by
  sorry

end books_per_shelf_l4047_404704


namespace unfoldable_cylinder_volume_l4047_404732

/-- A cylinder with a lateral surface that unfolds into a rectangle -/
structure UnfoldableCylinder where
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The volume of an unfoldable cylinder -/
def cylinder_volume (c : UnfoldableCylinder) : Set ℝ :=
  { v | ∃ (r h : ℝ), 
    ((2 * Real.pi * r = c.rectangle_length ∧ h = c.rectangle_width) ∨
     (2 * Real.pi * r = c.rectangle_width ∧ h = c.rectangle_length)) ∧
    v = Real.pi * r^2 * h }

/-- Theorem: The volume of a cylinder with lateral surface unfolding to a 4π by 1 rectangle is either 4π or 1 -/
theorem unfoldable_cylinder_volume :
  let c := UnfoldableCylinder.mk (4 * Real.pi) 1
  cylinder_volume c = {4 * Real.pi, 1} := by
  sorry

end unfoldable_cylinder_volume_l4047_404732


namespace sum_of_digits_18_to_21_l4047_404768

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def sum_of_digits_range (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => sum_of_digits (a + i))

theorem sum_of_digits_18_to_21 :
  sum_of_digits_range 18 21 = 24 :=
by
  sorry

-- The following definition is provided as a condition from the problem
axiom sum_of_digits_0_to_99 : sum_of_digits_range 0 99 = 900

end sum_of_digits_18_to_21_l4047_404768


namespace special_function_property_l4047_404748

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 ≠ 0) :
  (f 3 - f 1) / f 2 = 2 := by
  sorry

end special_function_property_l4047_404748


namespace prob_no_adjacent_same_five_people_l4047_404767

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of faces on the standard die -/
def d : ℕ := 6

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ :=
  (d - 1)^(n - 1) * (d - 2) / d^n

theorem prob_no_adjacent_same_five_people (h : n = 5) :
  prob_no_adjacent_same = 625 / 1944 := by
  sorry

end prob_no_adjacent_same_five_people_l4047_404767


namespace circle_points_m_value_l4047_404718

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points lie on the same circle -/
def onSameCircle (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (D E F : ℝ),
    p1.x^2 + p1.y^2 + D*p1.x + E*p1.y + F = 0 ∧
    p2.x^2 + p2.y^2 + D*p2.x + E*p2.y + F = 0 ∧
    p3.x^2 + p3.y^2 + D*p3.x + E*p3.y + F = 0 ∧
    p4.x^2 + p4.y^2 + D*p4.x + E*p4.y + F = 0

/-- Theorem: If (2,1), (4,2), (3,4), and (1,m) lie on the same circle, then m = 2 or m = 3 -/
theorem circle_points_m_value :
  ∀ (m : ℝ),
    onSameCircle
      (Point.mk 2 1)
      (Point.mk 4 2)
      (Point.mk 3 4)
      (Point.mk 1 m) →
    m = 2 ∨ m = 3 := by
  sorry

end circle_points_m_value_l4047_404718


namespace henri_reads_670_words_l4047_404707

def total_time : ℝ := 8
def movie_durations : List ℝ := [3.5, 1.5, 1.25, 0.75]
def reading_speeds : List (ℝ × ℝ) := [(30, 12), (20, 8)]

def calculate_words_read (total_time : ℝ) (movie_durations : List ℝ) (reading_speeds : List (ℝ × ℝ)) : ℕ :=
  sorry

theorem henri_reads_670_words :
  calculate_words_read total_time movie_durations reading_speeds = 670 := by
  sorry

end henri_reads_670_words_l4047_404707


namespace odd_power_of_seven_plus_one_divisible_by_eight_l4047_404760

theorem odd_power_of_seven_plus_one_divisible_by_eight (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, (7^n : ℤ) + 1 = 8 * k := by
  sorry

end odd_power_of_seven_plus_one_divisible_by_eight_l4047_404760


namespace parabola_shift_down_2_l4047_404764

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Shifts a parabola vertically -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

theorem parabola_shift_down_2 :
  let original := Parabola.mk 2 4
  let shifted := shift_parabola original (-2)
  shifted = Parabola.mk 2 2 := by
  sorry

end parabola_shift_down_2_l4047_404764


namespace painted_cube_probability_l4047_404753

/-- Represents a rectangular prism with painted faces -/
structure PaintedPrism where
  length : ℕ
  width : ℕ
  height : ℕ
  painted_face1 : ℕ × ℕ
  painted_face2 : ℕ × ℕ

/-- Calculates the total number of unit cubes in the prism -/
def total_cubes (p : PaintedPrism) : ℕ :=
  p.length * p.width * p.height

/-- Calculates the number of cubes with exactly one painted face -/
def cubes_with_one_painted_face (p : PaintedPrism) : ℕ :=
  (p.painted_face1.1 - 2) * (p.painted_face1.2 - 2) +
  (p.painted_face2.1 - 2) * (p.painted_face2.2 - 2) + 2

/-- Calculates the number of cubes with no painted faces -/
def cubes_with_no_painted_faces (p : PaintedPrism) : ℕ :=
  total_cubes p - (p.painted_face1.1 * p.painted_face1.2 +
                   p.painted_face2.1 * p.painted_face2.2 -
                   (p.painted_face1.1 + p.painted_face2.1))

/-- The main theorem to be proved -/
theorem painted_cube_probability (p : PaintedPrism)
  (h1 : p.length = 4)
  (h2 : p.width = 3)
  (h3 : p.height = 3)
  (h4 : p.painted_face1 = (4, 3))
  (h5 : p.painted_face2 = (3, 3)) :
  (cubes_with_one_painted_face p * cubes_with_no_painted_faces p : ℚ) /
  (total_cubes p * (total_cubes p - 1) / 2) = 221 / 630 := by
  sorry

end painted_cube_probability_l4047_404753


namespace no_real_roots_l4047_404785

theorem no_real_roots : ∀ x : ℝ, x^2 + 3*x + 5 ≠ 0 := by
  sorry

end no_real_roots_l4047_404785


namespace apple_basket_problem_l4047_404752

theorem apple_basket_problem (x : ℕ) : 
  (x / 2 - 2) - ((x / 2 - 2) / 2 - 3) = 24 → x = 88 := by
  sorry

end apple_basket_problem_l4047_404752


namespace function_inequality_l4047_404725

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end function_inequality_l4047_404725


namespace rosys_age_l4047_404771

theorem rosys_age (rosy_age : ℕ) : 
  (rosy_age + 12 + 4 = 2 * (rosy_age + 4)) → rosy_age = 8 := by
  sorry

end rosys_age_l4047_404771


namespace well_diameter_l4047_404784

/-- The diameter of a circular well given its depth and volume -/
theorem well_diameter (depth : ℝ) (volume : ℝ) (h1 : depth = 14) (h2 : volume = 43.982297150257104) :
  let radius := Real.sqrt (volume / (Real.pi * depth))
  2 * radius = 2 := by sorry

end well_diameter_l4047_404784


namespace correct_system_of_equations_l4047_404749

/-- Represents the number of students in a grade -/
def total_students : ℕ := 246

/-- Theorem: The system of equations {x + y = 246, y = 2x + 2} correctly represents
    the scenario where the total number of students is 246, and the number of boys (y)
    is 2 more than twice the number of girls (x). -/
theorem correct_system_of_equations (x y : ℕ) :
  x + y = total_students ∧ y = 2 * x + 2 →
  x + y = total_students ∧ y = 2 * x + 2 :=
by sorry

end correct_system_of_equations_l4047_404749


namespace alex_coin_distribution_l4047_404751

/-- The minimum number of additional coins needed for distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_coins_needed := (num_friends * (num_friends + 1)) / 2
  if total_coins_needed > initial_coins then
    total_coins_needed - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed for Alex's distribution. -/
theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 80) :
  min_additional_coins num_friends initial_coins = 40 := by
  sorry

end alex_coin_distribution_l4047_404751


namespace tray_height_l4047_404743

theorem tray_height (side_length : ℝ) (corner_distance : ℝ) (cut_angle : ℝ) 
  (h1 : side_length = 150)
  (h2 : corner_distance = 5)
  (h3 : cut_angle = 45) : 
  let tray_height := corner_distance * Real.sqrt 2 * Real.sin (cut_angle * π / 180)
  tray_height = 5 := by sorry

end tray_height_l4047_404743


namespace candle_burning_l4047_404750

/-- Candle burning problem -/
theorem candle_burning (h₀ : ℕ) (burn_rate : ℕ → ℝ) (T : ℝ) : 
  (h₀ = 150) →
  (∀ k, burn_rate k = 15 * k) →
  (T = (15 : ℝ) * (h₀ * (h₀ + 1) / 2)) →
  (∃ m : ℕ, 
    (7.5 * m * (m + 1) ≤ T / 2) ∧ 
    (T / 2 < 7.5 * (m + 1) * (m + 2)) ∧
    (h₀ - m = 45)) :=
by sorry

end candle_burning_l4047_404750


namespace figure_area_theorem_l4047_404716

theorem figure_area_theorem (y : ℝ) :
  (3 * y)^2 + (7 * y)^2 + (1/2 * 3 * y * 7 * y) = 1200 → y = 10 := by
  sorry

end figure_area_theorem_l4047_404716


namespace hyperbola_eccentricity_range_l4047_404745

/-- Given a hyperbola with semi-major axis a and semi-minor axis b, 
    and a point P on its right branch satisfying |PF₁| = 4|PF₂|, 
    prove that the eccentricity e is in the range (1, 5/3] -/
theorem hyperbola_eccentricity_range (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) 
  (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)  -- P is on the hyperbola
  (h₄ : P.1 > 0)  -- P is on the right branch
  (h₅ : ‖P - F₁‖ = 4 * ‖P - F₂‖)  -- |PF₁| = 4|PF₂|
  (h₆ : F₁.1 < 0 ∧ F₂.1 > 0)  -- F₁ is left focus, F₂ is right focus
  (h₇ : ‖F₁ - F₂‖ = 2 * (a^2 + b^2).sqrt)  -- distance between foci
  : 1 < (a^2 + b^2).sqrt / a ∧ (a^2 + b^2).sqrt / a ≤ 5/3 :=
by sorry

end hyperbola_eccentricity_range_l4047_404745


namespace programmer_is_odd_one_out_l4047_404713

/-- Represents a profession --/
inductive Profession
  | Dentist
  | ElementarySchoolTeacher
  | Programmer

/-- Represents whether a profession has special pension benefits --/
def has_special_pension_benefits (p : Profession) : Prop :=
  match p with
  | Profession.Dentist => true
  | Profession.ElementarySchoolTeacher => true
  | Profession.Programmer => false

/-- Theorem stating that the programmer is the odd one out --/
theorem programmer_is_odd_one_out :
  ∃! p : Profession, ¬(has_special_pension_benefits p) :=
sorry


end programmer_is_odd_one_out_l4047_404713


namespace chess_pool_theorem_l4047_404717

theorem chess_pool_theorem (U : Type) 
  (A : Set U) -- Set of people who play chess
  (B : Set U) -- Set of people who are not interested in mathematics
  (C : Set U) -- Set of people who bathe in the pool every day
  (h1 : (A ∩ B).Nonempty) -- Condition 1
  (h2 : (C ∩ B ∩ A) = ∅) -- Condition 2
  : ¬(A ⊆ C) := by
  sorry

end chess_pool_theorem_l4047_404717


namespace complex_magnitude_equation_l4047_404776

theorem complex_magnitude_equation (z : ℂ) : 
  (z + Complex.I) * (1 - Complex.I) = 1 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_equation_l4047_404776


namespace triangle_properties_l4047_404742

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_CM : ℝ → ℝ → ℝ
  altitude_BH : ℝ → ℝ → ℝ

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  B := sorry
  C := sorry
  median_CM := λ x y ↦ 2*x - y - 5
  altitude_BH := λ x y ↦ x - 2*y - 5

theorem triangle_properties (t : Triangle) (h : t = given_triangle) : 
  t.C = (4, 3) ∧ 
  (λ x y ↦ 6*x - 5*y - 9) = (λ x y ↦ 0) :=
sorry

end triangle_properties_l4047_404742


namespace combined_average_age_l4047_404710

theorem combined_average_age (people_c people_d : ℕ) (avg_age_c avg_age_d : ℚ) :
  people_c = 8 →
  people_d = 6 →
  avg_age_c = 30 →
  avg_age_d = 35 →
  (people_c * avg_age_c + people_d * avg_age_d) / (people_c + people_d) = 32 := by
  sorry

end combined_average_age_l4047_404710


namespace point_symmetry_y_axis_l4047_404758

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetric_y_axis (A B : Point) : Prop :=
  A.x = -B.x ∧ A.y = B.y

theorem point_symmetry_y_axis : 
  let A : Point := ⟨-1, 8⟩
  ∀ B : Point, symmetric_y_axis A B → B = ⟨1, 8⟩ := by
  sorry

end point_symmetry_y_axis_l4047_404758


namespace triangle_ratios_l4047_404726

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = 8 ∧ d t.A t.C = 6 ∧ d t.B t.C = 4

-- Define angle bisector
def isAngleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.B D / d t.C D = d t.A t.B / d t.A t.C

-- Define the intersection point P
def intersectionPoint (t : Triangle) (D E : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual calculation of the intersection point

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry  -- The actual calculation of the circumcenter

-- Main theorem
theorem triangle_ratios (t : Triangle) (D E : ℝ × ℝ) :
  isValidTriangle t →
  isAngleBisector t D →
  isAngleBisector t E →
  let P := intersectionPoint t D E
  let O := circumcenter t
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.B P / d P E = 2 ∧
  d O D / d D t.A = 1/3 :=
by
  sorry


end triangle_ratios_l4047_404726


namespace unique_prime_arith_seq_l4047_404733

/-- An arithmetic sequence of three prime numbers with common difference 80. -/
structure PrimeArithSeq where
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  prime_p₁ : Nat.Prime p₁
  prime_p₂ : Nat.Prime p₂
  prime_p₃ : Nat.Prime p₃
  diff_p₂_p₁ : p₂ = p₁ + 80
  diff_p₃_p₂ : p₃ = p₂ + 80

/-- There exists exactly one arithmetic sequence of three prime numbers with common difference 80. -/
theorem unique_prime_arith_seq : ∃! seq : PrimeArithSeq, True :=
  sorry

end unique_prime_arith_seq_l4047_404733


namespace equation_solution_l4047_404765

theorem equation_solution : ∃! x : ℝ, (2 / 3) * x - 2 = 4 ∧ x = 9 := by sorry

end equation_solution_l4047_404765


namespace line_perpendicular_to_parallel_planes_l4047_404788

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel α β → perpendicular m β :=
sorry

end line_perpendicular_to_parallel_planes_l4047_404788


namespace least_integer_with_divisibility_pattern_l4047_404756

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_pattern :
  ∃ (n : ℕ) (a : ℕ),
    n > 0 ∧
    a ≥ 1 ∧ a < 30 ∧
    consecutive_pair a (a + 1) ∧
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 30 ∧ i ≠ a ∧ i ≠ (a + 1) → is_divisible n i) ∧
    ¬(is_divisible n a) ∧
    ¬(is_divisible n (a + 1)) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ (b : ℕ),
        b ≥ 1 ∧ b < 30 ∧
        consecutive_pair b (b + 1) ∧
        (∀ i : ℕ, 1 ≤ i ∧ i ≤ 30 ∧ i ≠ b ∧ i ≠ (b + 1) → is_divisible m i) ∧
        ¬(is_divisible m b) ∧
        ¬(is_divisible m (b + 1)))) ∧
    n = 12252240 :=
by sorry

end least_integer_with_divisibility_pattern_l4047_404756


namespace candy_sharing_l4047_404734

theorem candy_sharing (bags : ℕ) (candies_per_bag : ℕ) (people : ℕ) :
  bags = 25 →
  candies_per_bag = 16 →
  people = 2 →
  (bags * candies_per_bag) / people = 200 := by
  sorry

end candy_sharing_l4047_404734


namespace curve_is_line_segment_l4047_404703

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the range of t
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 5}

-- Define the curve as a set of points
def curve : Set (ℝ × ℝ) := {(x t, y t) | t ∈ t_range}

-- Theorem statement
theorem curve_is_line_segment : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ (p : ℝ × ℝ), p ∈ curve → a * p.1 + b * p.2 + c = 0) ∧
  (∃ (p q : ℝ × ℝ), p ∈ curve ∧ q ∈ curve ∧ p ≠ q ∧
    ∀ (r : ℝ × ℝ), r ∈ curve → ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ r = (1 - t) • p + t • q) :=
sorry

end curve_is_line_segment_l4047_404703


namespace imaginary_part_of_z_l4047_404702

theorem imaginary_part_of_z (z : ℂ) (h : z = Complex.I * (2 - z)) : z.im = 1 := by
  sorry

end imaginary_part_of_z_l4047_404702


namespace bobs_family_adults_l4047_404700

theorem bobs_family_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) 
  (h1 : total_apples = 1200)
  (h2 : num_children = 45)
  (h3 : apples_per_child = 15)
  (h4 : apples_per_adult = 5) :
  (total_apples - num_children * apples_per_child) / apples_per_adult = 105 :=
by
  sorry

end bobs_family_adults_l4047_404700


namespace inequality_iff_solution_set_l4047_404773

def inequality (x : ℝ) : Prop :=
  (2 / (x + 2)) + (8 / (x + 6)) ≥ 2

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ -6 < x ∧ x ≤ 1 :=
by sorry

end inequality_iff_solution_set_l4047_404773


namespace equation_solution_l4047_404737

theorem equation_solution : 
  ∃ x : ℝ, x ≠ 2 ∧ (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 := by
  sorry

end equation_solution_l4047_404737


namespace least_number_divisibility_l4047_404799

theorem least_number_divisibility (n : ℕ) : n = 215988 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 48 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 64 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 72 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 108 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 125 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    (n + 12) = 48 * k₁ ∧
    (n + 12) = 64 * k₂ ∧
    (n + 12) = 72 * k₃ ∧
    (n + 12) = 108 * k₄ ∧
    (n + 12) = 125 * k₅) :=
by sorry

end least_number_divisibility_l4047_404799


namespace abc_log_sum_l4047_404720

theorem abc_log_sum (A B C : ℕ+) (h_coprime : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) :
  A + B + C = 5 := by
  sorry

end abc_log_sum_l4047_404720


namespace problem_solution_l4047_404736

theorem problem_solution (a b c d e : ℝ) 
  (eq1 : a - b - c + d = 18)
  (eq2 : a + b - c - d = 6)
  (eq3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 := by
sorry

end problem_solution_l4047_404736


namespace validMSetIs0And8_l4047_404766

/-- The function f(x) = x^2 + mx - 2m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2*m - 1

/-- Predicate to check if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Predicate to check if all roots of f are integers -/
def hasOnlyIntegerRoots (m : ℝ) : Prop :=
  ∀ x : ℝ, f m x = 0 → isInteger x

/-- The set of m values for which f has only integer roots -/
def validMSet : Set ℝ := {m | hasOnlyIntegerRoots m}

/-- Theorem stating that the set of valid m values is {0, -8} -/
theorem validMSetIs0And8 : validMSet = {0, -8} := by sorry

end validMSetIs0And8_l4047_404766


namespace tom_chocolate_boxes_l4047_404730

/-- The number of pieces Tom gave away -/
def pieces_given_away : ℕ := 8

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 3

/-- The number of pieces Tom still has -/
def pieces_remaining : ℕ := 18

/-- The number of boxes Tom bought initially -/
def boxes_bought : ℕ := 8

theorem tom_chocolate_boxes :
  boxes_bought * pieces_per_box = pieces_given_away + pieces_remaining :=
by sorry

end tom_chocolate_boxes_l4047_404730


namespace sheet_width_l4047_404705

/-- Given a rectangular sheet of paper with length 10 inches, a 1.5-inch margin all around,
    and a picture covering 38.5 square inches, prove that the width of the sheet is 8.5 inches. -/
theorem sheet_width (W : ℝ) (margin : ℝ) (picture_area : ℝ) : 
  margin = 1.5 →
  picture_area = 38.5 →
  (W - 2 * margin) * (10 - 2 * margin) = picture_area →
  W = 8.5 := by
sorry

end sheet_width_l4047_404705


namespace quadratic_functions_coincidence_l4047_404739

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two quadratic functions can coincide through parallel translation -/
def can_coincide (f g : QuadraticFunction) : Prop :=
  f.a = g.a ∧ f.a ≠ 0

/-- The three given quadratic functions -/
def A : QuadraticFunction := ⟨1, 0, -1⟩
def B : QuadraticFunction := ⟨-1, 0, 1⟩
def C : QuadraticFunction := ⟨1, 2, -1⟩

theorem quadratic_functions_coincidence :
  can_coincide A C ∧ ¬can_coincide A B ∧ ¬can_coincide B C := by sorry

end quadratic_functions_coincidence_l4047_404739


namespace parallel_line_x_coordinate_l4047_404708

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines when two points form a line segment parallel to the y-axis. -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

/-- The problem statement -/
theorem parallel_line_x_coordinate 
  (M N : Point)
  (h_parallel : parallelToYAxis M N)
  (h_M : M = ⟨3, -5⟩)
  (h_N : N = ⟨N.x, 2⟩) :
  N.x = 3 := by
  sorry

#check parallel_line_x_coordinate

end parallel_line_x_coordinate_l4047_404708


namespace activity_ratio_theorem_l4047_404780

/-- Represents the ratio of time spent on two activities -/
structure TimeRatio where
  activity1 : ℝ
  activity2 : ℝ

/-- Calculates the score based on time spent on an activity -/
def calculateScore (pointsPerHour : ℝ) (hours : ℝ) : ℝ :=
  pointsPerHour * hours

/-- Theorem stating the relationship between activities and score -/
theorem activity_ratio_theorem (timeActivity1 : ℝ) (pointsPerHour : ℝ) (finalScore : ℝ) :
  timeActivity1 = 9 →
  pointsPerHour = 15 →
  finalScore = 45 →
  ∃ (ratio : TimeRatio),
    ratio.activity1 = timeActivity1 ∧
    ratio.activity2 = finalScore / pointsPerHour ∧
    ratio.activity2 / ratio.activity1 = 1 / 3 := by
  sorry

end activity_ratio_theorem_l4047_404780


namespace cone_vertex_angle_l4047_404797

theorem cone_vertex_angle (r l : ℝ) (h : r > 0) (h2 : l > 0) : 
  (π * r * l) / (π * r^2) = 2 → 2 * Real.arcsin (r / l) = π / 3 :=
by sorry

end cone_vertex_angle_l4047_404797


namespace nonagon_triangle_probability_l4047_404709

/-- The number of vertices in a regular nonagon -/
def nonagon_vertices : ℕ := 9

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of ways to choose 3 vertices from 9 vertices -/
def total_triangles : ℕ := Nat.choose nonagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the nonagon -/
def favorable_triangles : ℕ := 54

/-- The probability of forming a triangle with at least one side being a side of the nonagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem nonagon_triangle_probability : probability = 9 / 14 := by sorry

end nonagon_triangle_probability_l4047_404709


namespace english_failure_percentage_l4047_404796

/-- The percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 25

/-- The percentage of students who failed in both Hindi and English -/
def failed_both : ℝ := 27

/-- The percentage of students who passed in both subjects -/
def passed_both : ℝ := 54

/-- The percentage of students who failed in English -/
def failed_english : ℝ := 100 - passed_both - failed_hindi + failed_both

theorem english_failure_percentage :
  failed_english = 48 :=
sorry

end english_failure_percentage_l4047_404796


namespace arithmetic_sequence_increasing_iff_a1_lt_a3_l4047_404763

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: For an arithmetic sequence, a_1 < a_3 iff the sequence is monotonically increasing -/
theorem arithmetic_sequence_increasing_iff_a1_lt_a3 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 < a 3 ↔ MonotonicallyIncreasing a) :=
sorry

end arithmetic_sequence_increasing_iff_a1_lt_a3_l4047_404763


namespace triangle_foldable_to_2020_layers_l4047_404775

/-- A triangle in a plane --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A folding method that transforms a triangle into a uniformly thick object --/
structure FoldingMethod where
  apply : Triangle → ℕ

/-- The theorem stating that any triangle can be folded into 2020 layers --/
theorem triangle_foldable_to_2020_layers :
  ∀ (t : Triangle), ∃ (f : FoldingMethod), f.apply t = 2020 :=
sorry

end triangle_foldable_to_2020_layers_l4047_404775


namespace transistor_count_2005_l4047_404747

/-- Calculates the number of transistors in a CPU after applying Moore's law and an additional growth law over a specified time period. -/
def transistor_count (initial_count : ℕ) (years : ℕ) : ℕ :=
  let doubling_cycles := years / 2
  let tripling_cycles := years / 6
  initial_count * 2^doubling_cycles + initial_count * 3^tripling_cycles

/-- Theorem stating that the number of transistors in a CPU in 2005 is 68,500,000,
    given an initial count of 500,000 in 1990 and the application of Moore's law
    and an additional growth law. -/
theorem transistor_count_2005 :
  transistor_count 500000 15 = 68500000 := by
  sorry

end transistor_count_2005_l4047_404747


namespace equation_representation_l4047_404759

/-- Given that a number is 5 more than three times a and equals 9, 
    prove that the equation is 3a + 5 = 9 -/
theorem equation_representation (a : ℝ) : 
  (3 * a + 5 = 9) ↔ (∃ x, x = 3 * a + 5 ∧ x = 9) := by sorry

end equation_representation_l4047_404759


namespace election_winner_percentage_l4047_404762

theorem election_winner_percentage (total_votes winner_majority : ℕ) 
  (h_total : total_votes = 500) 
  (h_majority : winner_majority = 200) : 
  (((total_votes + winner_majority) / 2) / total_votes : ℚ) = 7/10 := by
  sorry

end election_winner_percentage_l4047_404762


namespace function_equality_implies_m_zero_l4047_404794

/-- Given two functions f and g, prove that m = 0 when 3f(3) = 2g(3) -/
theorem function_equality_implies_m_zero (m : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 3*x + 2*m
  let g := fun (x : ℝ) => 2*x^2 - 6*x + 5*m
  3 * f 3 = 2 * g 3 → m = 0 := by
sorry

end function_equality_implies_m_zero_l4047_404794


namespace expected_sixes_two_dice_l4047_404754

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling a 6 on a single die -/
def prob_six : ℚ := 1 / num_sides

/-- The expected number of 6's when rolling two dice -/
def expected_sixes : ℚ := 1 / 4

/-- Theorem stating that the expected number of 6's when rolling two eight-sided dice is 1/4 -/
theorem expected_sixes_two_dice : 
  expected_sixes = 2 * prob_six := by sorry

end expected_sixes_two_dice_l4047_404754


namespace cd_player_only_percentage_l4047_404728

theorem cd_player_only_percentage
  (power_windows : ℝ)
  (anti_lock_brakes : ℝ)
  (cd_player : ℝ)
  (gps_system : ℝ)
  (pw_abs : ℝ)
  (abs_cd : ℝ)
  (pw_cd : ℝ)
  (gps_abs : ℝ)
  (gps_cd : ℝ)
  (pw_gps : ℝ)
  (h1 : power_windows = 60)
  (h2 : anti_lock_brakes = 40)
  (h3 : cd_player = 75)
  (h4 : gps_system = 50)
  (h5 : pw_abs = 10)
  (h6 : abs_cd = 15)
  (h7 : pw_cd = 20)
  (h8 : gps_abs = 12)
  (h9 : gps_cd = 18)
  (h10 : pw_gps = 25)
  (h11 : ∀ x, x ≤ 100) -- Assuming percentages are ≤ 100%
  : cd_player - (abs_cd + pw_cd + gps_cd) = 22 :=
by sorry


end cd_player_only_percentage_l4047_404728


namespace min_difference_f_g_l4047_404706

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_difference_f_g :
  ∃ (min_val : ℝ), min_val = 1/2 + 1/2 * Real.log 2 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
sorry

end min_difference_f_g_l4047_404706


namespace ashley_state_quarters_amount_l4047_404724

/-- The amount Ashley receives for her state quarters -/
def ashley_amount (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) : ℚ :=
  (num_quarters : ℚ) * face_value * (percentage : ℚ) / 100

/-- Theorem stating the amount Ashley receives for her state quarters -/
theorem ashley_state_quarters_amount :
  ashley_amount 6 0.25 1500 = 22.50 := by
  sorry

end ashley_state_quarters_amount_l4047_404724


namespace hyperbola_equation_l4047_404727

/-- A hyperbola with center at the origin, a focus at (√2, 0), and the distance
    from this focus to an asymptote being 1 has the equation x^2 - y^2 = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 = 1) ↔
  (0, 0) ∈ C ∧ 
  F = (Real.sqrt 2, 0) ∧ 
  F ∈ C ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C → a * y = b * x ∨ a * y = -b * x) ∧
    (abs (b * Real.sqrt 2) / Real.sqrt (a^2 + b^2) = 1)) := by
  sorry

end hyperbola_equation_l4047_404727


namespace correction_is_subtract_30x_l4047_404793

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "half-dollar" => 50
  | "dollar" => 100
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculates the correction needed for mistaken coin counts -/
def correction_amount (x : ℕ) : ℤ :=
  (coin_value "dollar" - coin_value "half-dollar") * x -
  (coin_value "quarter" - coin_value "nickel") * x

theorem correction_is_subtract_30x (x : ℕ) :
  correction_amount x = -30 * x :=
sorry

end correction_is_subtract_30x_l4047_404793


namespace odd_prime_sum_of_squares_l4047_404744

theorem odd_prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) :
  (∃ (a b : ℕ+), a.val^2 + b.val^2 = p) ↔ p % 4 = 1 := by
  sorry

end odd_prime_sum_of_squares_l4047_404744


namespace polygon_sides_l4047_404740

theorem polygon_sides (n : ℕ) (h : n ≥ 3) : 
  (n - 2) * 180 + 360 = 1260 → n = 7 := by
sorry

end polygon_sides_l4047_404740


namespace no_rational_points_on_sqrt3_circle_l4047_404711

theorem no_rational_points_on_sqrt3_circle : 
  ¬∃ (x y : ℚ), x^2 + y^2 = 3 := by
sorry

end no_rational_points_on_sqrt3_circle_l4047_404711


namespace BC_completion_time_l4047_404712

/-- The time it takes for a group of workers to complete a job -/
def completion_time (work_rate : ℚ) : ℚ := 1 / work_rate

/-- The work rate of a single worker A -/
def work_rate_A : ℚ := 1 / 10

/-- The combined work rate of workers A and B -/
def work_rate_AB : ℚ := 1 / 5

/-- The combined work rate of workers A, B, and C -/
def work_rate_ABC : ℚ := 1 / 3

/-- The combined work rate of workers B and C -/
def work_rate_BC : ℚ := work_rate_ABC - work_rate_A

theorem BC_completion_time :
  completion_time work_rate_BC = 30 / 7 := by
  sorry

end BC_completion_time_l4047_404712


namespace slope_condition_l4047_404781

/-- Given two points A(-3, 10) and B(5, y) in a coordinate plane, 
    if the slope of the line through A and B is -4/3, then y = -2/3. -/
theorem slope_condition (y : ℚ) : 
  let A : ℚ × ℚ := (-3, 10)
  let B : ℚ × ℚ := (5, y)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -4/3 → y = -2/3 := by
sorry

end slope_condition_l4047_404781


namespace incorrect_comparison_l4047_404721

theorem incorrect_comparison : ¬((-5.2 : ℚ) > -5.1) := by
  sorry

end incorrect_comparison_l4047_404721


namespace expand_product_l4047_404774

theorem expand_product (x : ℝ) : (7*x + 5) * (5*x^2 - 2*x + 4) = 35*x^3 + 11*x^2 + 18*x + 20 := by
  sorry

end expand_product_l4047_404774


namespace complement_union_A_B_l4047_404783

def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}
def U : Set Int := Set.univ

theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : Int | ∃ k : Int, x = 3 * k} :=
by sorry

end complement_union_A_B_l4047_404783


namespace simple_interest_rate_l4047_404761

/-- Given a principal amount and a simple interest rate, if the sum of money
becomes 7/6 of itself in 2 years, then the rate is 1/12 -/
theorem simple_interest_rate (P : ℝ) (R : ℝ) (P_pos : P > 0) :
  P * (1 + 2 * R) = (7 / 6) * P → R = 1 / 12 := by
  sorry

end simple_interest_rate_l4047_404761


namespace absolute_value_inequality_l4047_404738

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem absolute_value_inequality
  (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_f_1 : f 1 = 0)
  (h_functional_equation : ∀ x y, x > 0 → y > 0 → f x + f y = f (x * y)) :
  ∀ x y, 0 < x → x < y → y < 1 → |f x| > |f y| :=
sorry

end absolute_value_inequality_l4047_404738


namespace greg_read_more_than_brad_l4047_404787

/-- Calculates the difference in pages read between Greg and Brad --/
def pages_difference : ℕ :=
  let greg_week1 := 7 * 18
  let greg_week2_3 := 14 * 22
  let greg_total := greg_week1 + greg_week2_3
  let brad_days1_5 := 5 * 26
  let brad_days6_17 := 12 * 20
  let brad_total := brad_days1_5 + brad_days6_17
  greg_total - brad_total

/-- The total number of pages both Greg and Brad need to read --/
def total_pages : ℕ := 800

/-- Theorem stating the difference in pages read between Greg and Brad --/
theorem greg_read_more_than_brad : pages_difference = 64 ∧ greg_total + brad_total = total_pages :=
  sorry

end greg_read_more_than_brad_l4047_404787


namespace max_value_sqrt7_plus_2xy_l4047_404777

theorem max_value_sqrt7_plus_2xy (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) 
  (h3 : x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32) : 
  ∃ (M : ℝ), M = 16 ∧ ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
  x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32 → Real.sqrt 7*(x + 2*y) + 2*x*y ≤ M :=
by
  sorry

end max_value_sqrt7_plus_2xy_l4047_404777


namespace marge_final_plant_count_l4047_404722

/-- Calculates the number of plants Marge ended up with in her garden -/
def marges_garden (total_seeds : ℕ) (non_growing_seeds : ℕ) (weed_kept : ℕ) : ℕ :=
  let growing_plants := total_seeds - non_growing_seeds
  let eaten_plants := growing_plants / 3
  let uneaten_plants := growing_plants - eaten_plants
  let strangled_plants := uneaten_plants / 3
  let surviving_plants := uneaten_plants - strangled_plants
  surviving_plants + weed_kept

/-- Theorem stating that Marge ended up with 9 plants in her garden -/
theorem marge_final_plant_count :
  marges_garden 23 5 1 = 9 := by
  sorry

end marge_final_plant_count_l4047_404722


namespace fraction_sum_equality_l4047_404731

theorem fraction_sum_equality : (3 : ℚ) / 30 + 9 / 300 + 27 / 3000 = 0.139 := by
  sorry

end fraction_sum_equality_l4047_404731


namespace percentage_problem_l4047_404792

theorem percentage_problem (P : ℝ) : (P / 100) * 150 - 40 = 50 → P = 60 := by
  sorry

end percentage_problem_l4047_404792


namespace inequality_solution_set_l4047_404782

theorem inequality_solution_set (x : ℝ) : 
  (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x ≤ 2 := by sorry

end inequality_solution_set_l4047_404782


namespace probability_of_three_in_three_eighths_l4047_404757

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

theorem probability_of_three_in_three_eighths :
  let digits := decimal_representation 3 8
  (digits.count 3) / (digits.length : ℚ) = 1/3 := by
  sorry

end probability_of_three_in_three_eighths_l4047_404757


namespace quadratic_function_conditions_l4047_404746

/-- A quadratic function passing through (1, -4) with vertex at (-1, 0) -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 1

/-- Theorem stating that f(x) satisfies the required conditions -/
theorem quadratic_function_conditions :
  (f 1 = -4) ∧ (∀ x : ℝ, f x ≥ f (-1)) ∧ (f (-1) = 0) := by sorry

end quadratic_function_conditions_l4047_404746


namespace smallest_abs_value_rational_l4047_404741

theorem smallest_abs_value_rational : ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end smallest_abs_value_rational_l4047_404741


namespace profit_conditions_l4047_404789

/-- Represents the profit function given the price increase -/
def profit_function (x : ℝ) : ℝ := (50 - 40 + x) * (500 - 10 * x)

/-- Represents the selling price given the price increase -/
def selling_price (x : ℝ) : ℝ := x + 50

/-- Represents the number of units sold given the price increase -/
def units_sold (x : ℝ) : ℝ := 500 - 10 * x

/-- Theorem stating the conditions for achieving a profit of 8000 yuan -/
theorem profit_conditions :
  (∃ x : ℝ, profit_function x = 8000 ∧
    ((selling_price x = 60 ∧ units_sold x = 400) ∨
     (selling_price x = 80 ∧ units_sold x = 200))) :=
by sorry

end profit_conditions_l4047_404789


namespace paco_cookies_left_l4047_404729

/-- The number of cookies Paco has left -/
def cookies_left (initial : ℕ) (given_away : ℕ) (eaten : ℕ) : ℕ :=
  initial - given_away - eaten

/-- Theorem stating that Paco has 12 cookies left -/
theorem paco_cookies_left :
  cookies_left 36 14 10 = 12 := by
  sorry

end paco_cookies_left_l4047_404729


namespace subset_condition_l4047_404701

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) / (x - 2) < 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 > 0}

-- Define the complement of A in ℝ
def A_complement : Set ℝ := {x | ¬ (x ∈ A)}

-- State the theorem
theorem subset_condition (a : ℝ) :
  0 < a ∧ a ≤ 1/2 → B a ⊆ A_complement :=
by sorry

end subset_condition_l4047_404701
