import Mathlib

namespace foreign_stamps_count_l1408_140827

/-- A collection of stamps with various properties -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreignAndOld : ℕ
  neitherForeignNorOld : ℕ

/-- The number of foreign stamps in the collection -/
def foreignStamps (sc : StampCollection) : ℕ :=
  sc.total - sc.old + sc.foreignAndOld - sc.neitherForeignNorOld

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (sc : StampCollection)
    (h1 : sc.total = 200)
    (h2 : sc.old = 50)
    (h3 : sc.foreignAndOld = 20)
    (h4 : sc.neitherForeignNorOld = 90) :
    foreignStamps sc = 80 := by
  sorry

end foreign_stamps_count_l1408_140827


namespace two_digit_product_4536_l1408_140841

theorem two_digit_product_4536 (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100) 
  (h3 : a * b = 4536) 
  (h4 : a ≤ b) : 
  a = 21 := by
sorry

end two_digit_product_4536_l1408_140841


namespace max_value_when_m_2_range_of_sum_when_parallel_tangents_l1408_140858

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1/m) * Real.log x + 1/x - x

theorem max_value_when_m_2 :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 2 x ≥ f 2 y ∧ f 2 x = 5/2 * Real.log 2 - 3/2 := by sorry

theorem range_of_sum_when_parallel_tangents :
  ∀ (m : ℝ), m ≥ 3 →
    ∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
      (deriv (f m) x₁ = deriv (f m) x₂) →
        x₁ + x₂ > 6/5 ∧ ∀ (ε : ℝ), ε > 0 →
          ∃ (y₁ y₂ : ℝ), y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧
            (deriv (f m) y₁ = deriv (f m) y₂) ∧
            y₁ + y₂ < 6/5 + ε := by sorry

end max_value_when_m_2_range_of_sum_when_parallel_tangents_l1408_140858


namespace racing_game_cost_l1408_140857

/-- The cost of the racing game given the total spent and the cost of the basketball game -/
theorem racing_game_cost (total_spent basketball_cost : ℚ) 
  (h1 : total_spent = 9.43)
  (h2 : basketball_cost = 5.20) : 
  total_spent - basketball_cost = 4.23 := by
  sorry

end racing_game_cost_l1408_140857


namespace tank_insulation_cost_l1408_140855

def tank_length : ℝ := 4
def tank_width : ℝ := 5
def tank_height : ℝ := 3
def insulation_cost_per_sqft : ℝ := 20

def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

def total_cost (sa : ℝ) (cost_per_sqft : ℝ) : ℝ := sa * cost_per_sqft

theorem tank_insulation_cost :
  total_cost (surface_area tank_length tank_width tank_height) insulation_cost_per_sqft = 1880 := by
  sorry

end tank_insulation_cost_l1408_140855


namespace complex_multiplication_l1408_140871

theorem complex_multiplication : (Complex.I : ℂ) * (1 - Complex.I) = 1 + Complex.I := by sorry

end complex_multiplication_l1408_140871


namespace sin_330_degrees_l1408_140891

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l1408_140891


namespace combined_weight_of_acids_l1408_140837

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.01

/-- The atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- The atomic mass of sulfur in g/mol -/
def sulfur_mass : ℝ := 32.07

/-- The molar mass of C6H8O7 in g/mol -/
def citric_acid_mass : ℝ := 6 * carbon_mass + 8 * hydrogen_mass + 7 * oxygen_mass

/-- The molar mass of H2SO4 in g/mol -/
def sulfuric_acid_mass : ℝ := 2 * hydrogen_mass + sulfur_mass + 4 * oxygen_mass

/-- The number of moles of C6H8O7 -/
def citric_acid_moles : ℝ := 8

/-- The number of moles of H2SO4 -/
def sulfuric_acid_moles : ℝ := 4

/-- The combined weight of C6H8O7 and H2SO4 in grams -/
def combined_weight : ℝ := citric_acid_moles * citric_acid_mass + sulfuric_acid_moles * sulfuric_acid_mass

theorem combined_weight_of_acids : combined_weight = 1929.48 := by
  sorry

end combined_weight_of_acids_l1408_140837


namespace correct_factorization_l1408_140810

theorem correct_factorization (x y : ℝ) : x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end correct_factorization_l1408_140810


namespace expression_simplification_l1408_140805

theorem expression_simplification (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end expression_simplification_l1408_140805


namespace marks_ratio_l1408_140884

def total_marks : ℕ := 170
def science_marks : ℕ := 17

def english_math_ratio : ℚ := 1 / 4

theorem marks_ratio : 
  ∃ (english_marks math_marks : ℕ),
    english_marks + math_marks + science_marks = total_marks ∧
    english_marks / math_marks = english_math_ratio ∧
    english_marks / science_marks = 31 / 17 :=
by sorry

end marks_ratio_l1408_140884


namespace range_of_m_l1408_140866

/-- An odd function f: ℝ → ℝ with domain [-2,2] -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 2)

/-- f is monotonically decreasing on [0,2] -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → f y < f x

/-- The main theorem -/
theorem range_of_m (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : MonoDecreasing f) :
  {m : ℝ | f (1 + m) + f m < 0} = Set.Ioo (-1/2) 1 := by
  sorry

end range_of_m_l1408_140866


namespace joggers_meeting_times_l1408_140870

theorem joggers_meeting_times (road_length : ℝ) (speed_a : ℝ) (speed_b : ℝ) (duration : ℝ) :
  road_length = 400 ∧
  speed_a = 3 ∧
  speed_b = 2.5 ∧
  duration = 20 * 60 →
  ∃ n : ℕ, n = 8 ∧ 
    (road_length + (n - 1) * 2 * road_length) / (speed_a + speed_b) = duration :=
by sorry

end joggers_meeting_times_l1408_140870


namespace die_roll_frequency_l1408_140849

/-- The frequency of an event in an experiment -/
def frequency (occurrences : ℕ) (totalTrials : ℕ) : ℚ :=
  occurrences / totalTrials

/-- The number of times the die was rolled -/
def totalRolls : ℕ := 100

/-- The number of times "even numbers facing up" occurred -/
def evenOccurrences : ℕ := 47

/-- The expected frequency of "even numbers facing up" -/
def expectedFrequency : ℚ := 47 / 100

theorem die_roll_frequency :
  frequency evenOccurrences totalRolls = expectedFrequency := by
  sorry

end die_roll_frequency_l1408_140849


namespace martha_lasagna_meat_amount_l1408_140869

-- Define the constants
def cheese_amount : Real := 1.5
def cheese_price_per_kg : Real := 6
def meat_price_per_kg : Real := 8
def total_cost : Real := 13

-- Define the theorem
theorem martha_lasagna_meat_amount :
  let cheese_cost := cheese_amount * cheese_price_per_kg
  let meat_cost := total_cost - cheese_cost
  let meat_amount_kg := meat_cost / meat_price_per_kg
  let meat_amount_g := meat_amount_kg * 1000
  meat_amount_g = 500 := by
  sorry

end martha_lasagna_meat_amount_l1408_140869


namespace negation_of_universal_square_geq_one_l1408_140881

theorem negation_of_universal_square_geq_one :
  (¬ ∀ x : ℝ, x^2 ≥ 1) ↔ (∃ x : ℝ, x^2 < 1) := by
  sorry

end negation_of_universal_square_geq_one_l1408_140881


namespace segments_form_triangle_l1408_140836

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (4, 5, 7) can form a triangle -/
theorem segments_form_triangle : can_form_triangle 4 5 7 := by
  sorry

end segments_form_triangle_l1408_140836


namespace min_value_and_inequality_l1408_140845

theorem min_value_and_inequality (a b x₁ x₂ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hab : a + b = 1) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ / a + x₂ / b + 2 / (x₁ * x₂) ≥ 6) ∧
  (a * x₁ + b * x₂) * (a * x₂ + b * x₁) ≥ x₁ * x₂ := by
  sorry

end min_value_and_inequality_l1408_140845


namespace triangle_circumradius_l1408_140813

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 7.5 := by
  sorry

end triangle_circumradius_l1408_140813


namespace all_equal_l1408_140867

-- Define the sequence type
def Sequence := Fin 2020 → ℕ

-- Define the divisibility condition for six consecutive numbers
def DivisibleSix (a : Sequence) : Prop :=
  ∀ n : Fin 2015, a n ∣ a (n + 5)

-- Define the divisibility condition for nine consecutive numbers
def DivisibleNine (a : Sequence) : Prop :=
  ∀ n : Fin 2012, a (n + 8) ∣ a n

-- State the theorem
theorem all_equal (a : Sequence) (h1 : DivisibleSix a) (h2 : DivisibleNine a) :
  ∀ i j : Fin 2020, a i = a j :=
sorry

end all_equal_l1408_140867


namespace simplest_fraction_sum_l1408_140806

theorem simplest_fraction_sum (p q : ℕ+) : 
  (p : ℚ) / q = 83125 / 100000 ∧ 
  ∀ (a b : ℕ+), (a : ℚ) / b = p / q → a ≤ p ∧ b ≤ q →
  p + q = 293 := by
sorry

end simplest_fraction_sum_l1408_140806


namespace ratio_xyz_l1408_140823

theorem ratio_xyz (x y z : ℝ) (h1 : 0.1 * x = 0.2 * y) (h2 : 0.3 * y = 0.4 * z) :
  ∃ (k : ℝ), k > 0 ∧ x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
sorry

end ratio_xyz_l1408_140823


namespace inequality_proof_l1408_140880

theorem inequality_proof (a₁ a₂ a₃ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃) : 
  let b₁ := (1 + a₁ * a₂ / (a₁ - a₂)) * (1 + a₁ * a₃ / (a₁ - a₃))
  let b₂ := (1 + a₂ * a₁ / (a₂ - a₁)) * (1 + a₂ * a₃ / (a₂ - a₃))
  let b₃ := (1 + a₃ * a₁ / (a₃ - a₁)) * (1 + a₃ * a₂ / (a₃ - a₂))
  1 + |a₁ * b₁ + a₂ * b₂ + a₃ * b₃| ≤ (1 + |a₁|) * (1 + |a₂|) * (1 + |a₃|) :=
by
  sorry

end inequality_proof_l1408_140880


namespace cubic_equation_with_geometric_roots_l1408_140895

/-- Given a cubic equation x^3 - 14x^2 + ax - 27 = 0 with three distinct real roots in geometric progression, prove that a = 42 -/
theorem cubic_equation_with_geometric_roots (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧  -- distinct roots
    (∃ r : ℝ, r ≠ 0 ∧ x₂ = x₁ * r ∧ x₃ = x₂ * r) ∧  -- geometric progression
    (x₁^3 - 14*x₁^2 + a*x₁ - 27 = 0) ∧
    (x₂^3 - 14*x₂^2 + a*x₂ - 27 = 0) ∧
    (x₃^3 - 14*x₃^2 + a*x₃ - 27 = 0)) →
  a = 42 := by
sorry

end cubic_equation_with_geometric_roots_l1408_140895


namespace school_trip_distances_l1408_140826

/-- Represents the problem of finding the distances in the school trip scenario. -/
theorem school_trip_distances 
  (total_distance : ℝ) 
  (walking_speed : ℝ) 
  (bus_speed : ℝ) 
  (rest_time : ℝ) : 
  total_distance = 21 ∧ 
  walking_speed = 4 ∧ 
  bus_speed = 60 ∧ 
  rest_time = 1/6 →
  ∃ (distance_to_A : ℝ) (distance_walked : ℝ),
    distance_to_A = 19 ∧
    distance_walked = 2 ∧
    distance_to_A + distance_walked = total_distance ∧
    distance_to_A / bus_speed + total_distance / bus_speed = 
      rest_time + distance_walked / walking_speed :=
by sorry

end school_trip_distances_l1408_140826


namespace yolanda_rate_l1408_140877

def total_distance : ℝ := 31
def bob_distance : ℝ := 20
def bob_rate : ℝ := 2

theorem yolanda_rate (total_distance : ℝ) (bob_distance : ℝ) (bob_rate : ℝ) :
  total_distance = 31 →
  bob_distance = 20 →
  bob_rate = 2 →
  ∃ yolanda_rate : ℝ,
    yolanda_rate = (total_distance - bob_distance) / (bob_distance / bob_rate) ∧
    yolanda_rate = 1.1 :=
by sorry

end yolanda_rate_l1408_140877


namespace evaluate_expression_l1408_140889

theorem evaluate_expression (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π + 2)) = Real.sin 2 ^ 2 - Real.cos 2 ^ 2 := by
  sorry

end evaluate_expression_l1408_140889


namespace expression_evaluation_l1408_140834

theorem expression_evaluation : (16 : ℝ) * 0.5 - (4.5 - 0.125 * 8) = 9/2 := by
  sorry

end expression_evaluation_l1408_140834


namespace ceiling_square_fraction_plus_eighth_l1408_140892

theorem ceiling_square_fraction_plus_eighth : ⌈(-7/4)^2 + 1/8⌉ = 4 := by
  sorry

end ceiling_square_fraction_plus_eighth_l1408_140892


namespace ball_probability_l1408_140879

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) 
  (h2 : red = 6) 
  (h3 : purple = 9) : 
  (total - (red + purple)) / total = 3 / 4 := by
  sorry

end ball_probability_l1408_140879


namespace area_of_triangle_pqr_l1408_140875

-- Define the square pyramid
structure SquarePyramid where
  base_side : ℝ
  altitude : ℝ

-- Define points P, Q, R
structure PyramidPoints where
  p_ratio : ℝ
  q_ratio : ℝ
  r_ratio : ℝ

-- Define the theorem
theorem area_of_triangle_pqr 
  (pyramid : SquarePyramid) 
  (points : PyramidPoints) 
  (h1 : pyramid.base_side = 4) 
  (h2 : pyramid.altitude = 8) 
  (h3 : points.p_ratio = 1/4) 
  (h4 : points.q_ratio = 1/4) 
  (h5 : points.r_ratio = 3/4) : 
  ∃ (area : ℝ), area = 2 * Real.sqrt 5 := by
  sorry

end area_of_triangle_pqr_l1408_140875


namespace total_pokemon_cards_l1408_140807

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 6

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 100

/-- Theorem: The total number of Pokemon cards for 6 people, each having 100 cards, is equal to 600 -/
theorem total_pokemon_cards : num_people * cards_per_person = 600 := by
  sorry

end total_pokemon_cards_l1408_140807


namespace distance_eq_speed_times_time_l1408_140819

/-- The distance between Martin's house and Lawrence's house -/
def distance : ℝ := 12

/-- The time Martin spent walking -/
def time : ℝ := 6

/-- Martin's walking speed -/
def speed : ℝ := 2

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_eq_speed_times_time : distance = speed * time := by
  sorry

end distance_eq_speed_times_time_l1408_140819


namespace box_office_scientific_notation_l1408_140886

/-- Converts a number in billions to scientific notation -/
def billionsToScientificNotation (x : ℝ) : ℝ × ℤ :=
  let mantissa := x * 10^(9 % 3)
  let exponent := 9 - (9 % 3)
  (mantissa, exponent)

/-- The box office revenue in billions of yuan -/
def boxOfficeRevenue : ℝ := 53.96

theorem box_office_scientific_notation :
  billionsToScientificNotation boxOfficeRevenue = (5.396, 9) := by
  sorry

end box_office_scientific_notation_l1408_140886


namespace function_characterization_l1408_140882

def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b c : ℤ, a + b + c = 0 →
    f a ^ 2 + f b ^ 2 + f c ^ 2 = 2 * f a * f b + 2 * f b * f c + 2 * f c * f a

def IsZeroFunction (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = 0

def IsQuadraticFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ, f x = k * x ^ 2

def IsEvenOddFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ, 
    (Even x → f x = 0) ∧ 
    (Odd x → f x = k)

def IsModFourFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ,
    (x % 4 = 0 → f x = 0) ∧
    (x % 4 = 1 → f x = k) ∧
    (x % 4 = 2 → f x = 4 * k)

theorem function_characterization (f : ℤ → ℤ) : 
  SatisfiesEquation f → 
    IsZeroFunction f ∨ 
    IsQuadraticFunction f ∨ 
    IsEvenOddFunction f ∨ 
    IsModFourFunction f := by
  sorry

end function_characterization_l1408_140882


namespace marius_darius_score_difference_l1408_140802

/-- The difference in scores between Marius and Darius in a table football game -/
theorem marius_darius_score_difference :
  ∀ (marius_score darius_score matt_score : ℕ),
    darius_score = 10 →
    matt_score = darius_score + 5 →
    marius_score + darius_score + matt_score = 38 →
    marius_score - darius_score = 3 := by
  sorry

end marius_darius_score_difference_l1408_140802


namespace only_solution_is_one_l1408_140844

theorem only_solution_is_one : 
  ∀ n : ℕ, (2 * n - 1 : ℚ) / (n^5 : ℚ) = 3 - 2 / (n : ℚ) ↔ n = 1 := by
  sorry

end only_solution_is_one_l1408_140844


namespace divisibility_by_112_l1408_140820

theorem divisibility_by_112 (m : ℕ) (h1 : m > 0) (h2 : m % 2 = 1) (h3 : m % 3 ≠ 0) :
  112 ∣ ⌊4^m - (2 + Real.sqrt 2)^m⌋ := by
  sorry

end divisibility_by_112_l1408_140820


namespace absolute_value_equation_solution_l1408_140853

theorem absolute_value_equation_solution :
  ∃ x : ℚ, (|x - 1| = |x - 2|) ∧ (x = 3/2) :=
by sorry

end absolute_value_equation_solution_l1408_140853


namespace negation_equivalence_l1408_140876

theorem negation_equivalence : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 1 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 1 ≤ 0) :=
by sorry

end negation_equivalence_l1408_140876


namespace vector_c_value_l1408_140865

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_c_value :
  ∀ c : ℝ × ℝ,
  (4 • a) + (3 • b - 2 • a) + c = (0, 0) →
  c = (4, -6) :=
by sorry

end vector_c_value_l1408_140865


namespace loan_principal_calculation_l1408_140851

theorem loan_principal_calculation (principal : ℝ) : 
  principal * 0.05 * 5 = principal - 2250 → principal = 3000 := by
  sorry

end loan_principal_calculation_l1408_140851


namespace jacque_suitcase_weight_l1408_140801

/-- The weight of Jacque's suitcase when he arrived in France -/
def initial_weight : ℝ := 5

/-- The weight of one bottle of perfume in ounces -/
def perfume_weight : ℝ := 1.2

/-- The number of bottles of perfume Jacque bought -/
def perfume_count : ℕ := 5

/-- The weight of chocolate in pounds -/
def chocolate_weight : ℝ := 4

/-- The weight of one bar of soap in ounces -/
def soap_weight : ℝ := 5

/-- The number of bars of soap Jacque bought -/
def soap_count : ℕ := 2

/-- The weight of one jar of jam in ounces -/
def jam_weight : ℝ := 8

/-- The number of jars of jam Jacque bought -/
def jam_count : ℕ := 2

/-- The number of ounces in a pound -/
def ounces_per_pound : ℝ := 16

/-- The total weight of Jacque's suitcase on the return flight in pounds -/
def return_weight : ℝ := 11

theorem jacque_suitcase_weight :
  initial_weight + 
  (perfume_weight * perfume_count + soap_weight * soap_count + jam_weight * jam_count) / ounces_per_pound + 
  chocolate_weight = return_weight := by
  sorry

end jacque_suitcase_weight_l1408_140801


namespace simplify_fraction_product_l1408_140847

theorem simplify_fraction_product : 
  (36 : ℚ) / 51 * 35 / 24 * 68 / 49 = 20 / 7 := by sorry

end simplify_fraction_product_l1408_140847


namespace f_inequality_implies_a_range_l1408_140878

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + 4*x else Real.log (x + 1)

-- State the theorem
theorem f_inequality_implies_a_range :
  (∀ x, |f x| ≥ a * x) → a ∈ Set.Icc (-4) 0 :=
by sorry

end f_inequality_implies_a_range_l1408_140878


namespace job_completion_time_l1408_140824

/-- Proves that the initial estimated time to finish the job is 8 days given the problem conditions. -/
theorem job_completion_time : 
  ∀ (initial_workers : ℕ) 
    (additional_workers : ℕ) 
    (days_before_joining : ℕ) 
    (days_after_joining : ℕ),
  initial_workers = 6 →
  additional_workers = 4 →
  days_before_joining = 3 →
  days_after_joining = 3 →
  ∃ (initial_estimate : ℕ),
    initial_estimate * initial_workers = 
      (initial_workers * days_before_joining + 
       (initial_workers + additional_workers) * days_after_joining) ∧
    initial_estimate = 8 := by
  sorry

#check job_completion_time

end job_completion_time_l1408_140824


namespace nine_by_nine_min_unoccupied_l1408_140856

/-- Represents a chessboard with grasshoppers -/
structure Chessboard :=
  (size : Nat)
  (initial_grasshoppers : Nat)
  (diagonal_jump : Bool)

/-- Calculates the minimum number of unoccupied squares after jumps -/
def min_unoccupied_squares (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating the minimum number of unoccupied squares for a 9x9 board -/
theorem nine_by_nine_min_unoccupied (board : Chessboard) : 
  board.size = 9 ∧ 
  board.initial_grasshoppers = 9 * 9 ∧ 
  board.diagonal_jump = true →
  min_unoccupied_squares board = 9 :=
sorry

end nine_by_nine_min_unoccupied_l1408_140856


namespace consecutive_integers_sqrt_33_l1408_140839

theorem consecutive_integers_sqrt_33 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 33) → (Real.sqrt 33 < b) → (a + b = 11) := by
  sorry

end consecutive_integers_sqrt_33_l1408_140839


namespace intermediate_circle_radius_l1408_140852

theorem intermediate_circle_radius 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 5)
  (h₂ : r₃ = 13)
  (h₃ : π * r₁^2 = π * r₃^2 - π * r₂^2) :
  r₂ = 12 := by
sorry

end intermediate_circle_radius_l1408_140852


namespace movies_watched_l1408_140898

theorem movies_watched (total_movies : ℕ) (movies_left : ℕ) (h1 : total_movies = 8) (h2 : movies_left = 4) :
  total_movies - movies_left = 4 := by
  sorry

end movies_watched_l1408_140898


namespace partition_seven_students_l1408_140814

/-- The number of ways to partition 7 students into groups of 2 or 3 -/
def partition_ways : ℕ := 105

/-- The number of students -/
def num_students : ℕ := 7

/-- The possible group sizes -/
def group_sizes : List ℕ := [2, 3]

/-- Theorem stating that the number of ways to partition 7 students into groups of 2 or 3 is 105 -/
theorem partition_seven_students :
  (∀ g ∈ group_sizes, g ≤ num_students) →
  (∃ f : List ℕ, (∀ x ∈ f, x ∈ group_sizes) ∧ f.sum = num_students) →
  partition_ways = 105 := by
  sorry

end partition_seven_students_l1408_140814


namespace sqrt_equation_solution_l1408_140894

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y - 5) = 9 → y = 86 := by
  sorry

end sqrt_equation_solution_l1408_140894


namespace cape_may_august_sharks_l1408_140842

/-- The number of sharks in Daytona Beach in July -/
def daytona_july : ℕ := 23

/-- The number of sharks in Cape May in July -/
def cape_may_july : ℕ := 2 * daytona_july

/-- The number of sharks in Daytona Beach in August -/
def daytona_august : ℕ := daytona_july

/-- The number of sharks in Cape May in August -/
def cape_may_august : ℕ := 5 + 3 * daytona_august

theorem cape_may_august_sharks : cape_may_august = 74 := by
  sorry

end cape_may_august_sharks_l1408_140842


namespace ram_krish_efficiency_ratio_l1408_140896

/-- Ram's efficiency -/
def ram_efficiency : ℝ := 1

/-- Krish's efficiency -/
def krish_efficiency : ℝ := 2

/-- Time taken by Ram alone to complete the task -/
def ram_alone_time : ℝ := 30

/-- Time taken by Ram and Krish together to complete the task -/
def combined_time : ℝ := 10

/-- The amount of work to be done -/
def work : ℝ := ram_efficiency * ram_alone_time

theorem ram_krish_efficiency_ratio :
  ram_efficiency / krish_efficiency = 1 / 2 ∧
  work = ram_efficiency * ram_alone_time ∧
  work = (ram_efficiency + krish_efficiency) * combined_time :=
by sorry

end ram_krish_efficiency_ratio_l1408_140896


namespace initial_mixture_volume_l1408_140860

/-- Given a mixture of milk and water with an initial ratio of 2:1, 
    prove that if 60 litres of water is added to change the ratio to 1:2, 
    the initial volume of the mixture was 60 litres. -/
theorem initial_mixture_volume 
  (initial_milk : ℝ) 
  (initial_water : ℝ) 
  (h1 : initial_milk = 2 * initial_water) 
  (h2 : initial_milk = (initial_water + 60) / 2) : 
  initial_milk + initial_water = 60 := by
  sorry

#check initial_mixture_volume

end initial_mixture_volume_l1408_140860


namespace remainder_of_n_squared_plus_2n_plus_4_l1408_140864

theorem remainder_of_n_squared_plus_2n_plus_4 (n : ℤ) (k : ℤ) 
  (h : n = 75 * k - 1) : 
  (n^2 + 2*n + 4) % 75 = 3 := by
sorry

end remainder_of_n_squared_plus_2n_plus_4_l1408_140864


namespace all_positive_integers_in_A_l1408_140872

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define the properties of set A
def HasPropertyA (A : Set ℕ) : Prop :=
  A ⊆ PositiveIntegers ∧
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (∀ m : ℕ, m ∈ A → ∀ d : ℕ, d > 0 ∧ m % d = 0 → d ∈ A) ∧
  (∀ b c : ℕ, b ∈ A → c ∈ A → 1 < b → b < c → (1 + b * c) ∈ A)

-- Theorem statement
theorem all_positive_integers_in_A (A : Set ℕ) (h : HasPropertyA A) :
  A = PositiveIntegers := by
  sorry

end all_positive_integers_in_A_l1408_140872


namespace cubic_equation_real_root_l1408_140822

theorem cubic_equation_real_root (a b : ℝ) : ∃ x : ℝ, x^3 + a*x + b = 0 := by
  sorry

end cubic_equation_real_root_l1408_140822


namespace wrong_value_correction_l1408_140833

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 20)
  (h2 : initial_mean = 150)
  (h3 : correct_mean = 151.25)
  (h4 : correct_value = 160) :
  ∃ x : ℝ, n * initial_mean - x + correct_value = n * correct_mean ∧ x = 135 := by
  sorry

end wrong_value_correction_l1408_140833


namespace usable_field_area_l1408_140897

/-- Calculates the area of a usable rectangular field with an L-shaped obstacle -/
theorem usable_field_area
  (breadth : ℕ)
  (h1 : breadth + 30 = 150)  -- Length is 30 meters more than breadth
  (h2 : 2 * (breadth + (breadth + 30)) = 540)  -- Perimeter is 540 meters
  : (breadth - 5) * (breadth + 30 - 10) = 16100 :=
by sorry

end usable_field_area_l1408_140897


namespace whole_number_between_bounds_l1408_140885

theorem whole_number_between_bounds (M : ℤ) :
  (9.5 < (M : ℚ) / 5 ∧ (M : ℚ) / 5 < 10.5) ↔ (M = 49 ∨ M = 50 ∨ M = 51) := by
  sorry

end whole_number_between_bounds_l1408_140885


namespace unhappy_redheads_ratio_l1408_140888

theorem unhappy_redheads_ratio 
  (x y z : ℕ) -- x: happy subjects, y: redheads, z: total subjects
  (h1 : (40 : ℚ) / 100 * x = (60 : ℚ) / 100 * y) -- Condition 1
  (h2 : z = x + (40 : ℚ) / 100 * y) -- Condition 2
  : (y - ((40 : ℚ) / 100 * y).floor) / z = 4 / 19 := by
  sorry


end unhappy_redheads_ratio_l1408_140888


namespace ratio_is_sixteen_thirteenths_l1408_140825

/-- An arithmetic sequence with a non-zero common difference where a₉, a₃, and a₁ form a geometric sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : a 3 ^ 2 = a 1 * a 9

/-- The ratio of (a₂ + a₄ + a₁₀) to (a₁ + a₃ + a₉) is 16/13 -/
theorem ratio_is_sixteen_thirteenths (seq : ArithmeticSequence) :
  (seq.a 2 + seq.a 4 + seq.a 10) / (seq.a 1 + seq.a 3 + seq.a 9) = 16 / 13 := by
  sorry

end ratio_is_sixteen_thirteenths_l1408_140825


namespace lcm_and_prime_factorization_l1408_140804

theorem lcm_and_prime_factorization :
  let a := 48
  let b := 180
  let c := 250
  let lcm_result := Nat.lcm (Nat.lcm a b) c
  lcm_result = 18000 ∧ 
  18000 = 2^4 * 3^2 * 5^3 := by
sorry

end lcm_and_prime_factorization_l1408_140804


namespace min_honey_purchase_l1408_140803

def is_valid_purchase (o h : ℕ) : Prop :=
  o ≥ 7 + h / 2 ∧ 
  o ≤ 3 * h ∧ 
  2 * o + 3 * h ≤ 36

theorem min_honey_purchase : 
  (∃ (o h : ℕ), is_valid_purchase o h) ∧ 
  (∀ (o h : ℕ), is_valid_purchase o h → h ≥ 4) ∧
  (∃ (o : ℕ), is_valid_purchase o 4) :=
sorry

end min_honey_purchase_l1408_140803


namespace sum_max_l1408_140862

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_odd : a 1 + a 3 + a 5 = 156
  sum_even : a 2 + a 4 + a 6 = 147

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq.a |>.sum

/-- The theorem stating that the sum reaches its maximum at n = 20 -/
theorem sum_max (seq : ArithmeticSequence) :
  ∀ k : ℕ, sum_n seq 20 ≥ sum_n seq k :=
sorry

end sum_max_l1408_140862


namespace find_other_number_l1408_140830

theorem find_other_number (A B : ℕ+) (hA : A = 24) (hHCF : Nat.gcd A B = 13) (hLCM : Nat.lcm A B = 312) : B = 169 := by
  sorry

end find_other_number_l1408_140830


namespace largest_exponent_inequality_l1408_140811

theorem largest_exponent_inequality (n : ℕ) : 64^8 > 4^n ↔ n ≤ 23 := by
  sorry

end largest_exponent_inequality_l1408_140811


namespace segment_length_is_70_l1408_140818

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  position : ℝ
  h1 : A ≤ position
  h2 : position ≤ B

/-- The line segment AB -/
def lineSegment (A B : ℝ) := {x : ℝ | A ≤ x ∧ x ≤ B}

theorem segment_length_is_70 
  (A B : ℝ) 
  (P Q : PointOnSegment A B) 
  (h_order : P.position < Q.position) 
  (h_same_side : (P.position - A) / (B - A) < 1/2 ∧ (Q.position - A) / (B - A) < 1/2) 
  (h_P_ratio : (P.position - A) / (B - P.position) = 2/3) 
  (h_Q_ratio : (Q.position - A) / (B - Q.position) = 3/4) 
  (h_PQ_length : Q.position - P.position = 2) :
  B - A = 70 := by
  sorry

#check segment_length_is_70

end segment_length_is_70_l1408_140818


namespace john_remaining_money_l1408_140829

theorem john_remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : 
  initial_amount = 1600 →
  initial_amount - spent_amount = spent_amount - 600 →
  initial_amount - spent_amount = 500 :=
by sorry

end john_remaining_money_l1408_140829


namespace train_length_l1408_140835

theorem train_length (t : ℝ) 
  (h1 : (t + 100) / 15 = (t + 250) / 20) : t = 350 := by
  sorry

#check train_length

end train_length_l1408_140835


namespace cone_volume_l1408_140883

/-- The volume of a cone whose lateral surface unfolds to a semicircle with radius 2 -/
theorem cone_volume (r : Real) (h : Real) : 
  r = 1 → h = Real.sqrt 3 → (1/3 : Real) * π * r^2 * h = (Real.sqrt 3 / 3) * π := by
  sorry

end cone_volume_l1408_140883


namespace expand_polynomial_l1408_140874

theorem expand_polynomial (x : ℝ) : (x + 3) * (2*x^2 - x + 4) = 2*x^3 + 5*x^2 + x + 12 := by
  sorry

end expand_polynomial_l1408_140874


namespace cone_lateral_surface_area_l1408_140846

/-- The lateral surface area of a cone with base radius 2 and height 1 is 2√5π -/
theorem cone_lateral_surface_area :
  let r : ℝ := 2  -- base radius
  let h : ℝ := 1  -- height
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  r * l * Real.pi = 2 * Real.sqrt 5 * Real.pi := by sorry

end cone_lateral_surface_area_l1408_140846


namespace matrix_equation_result_l1408_140817

/-- Given two 2x2 matrices A and B, where A is fixed and B has variable entries,
    if AB = BA and 4y ≠ z, then (x - w) / (z - 4y) = 3/8 -/
theorem matrix_equation_result (x y z w : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  4 * y ≠ z →
  A * B = B * A →
  (x - w) / (z - 4 * y) = 3 / 8 := by
sorry

end matrix_equation_result_l1408_140817


namespace smallest_visible_sum_l1408_140863

/-- Represents a standard die with opposite faces summing to 7 -/
structure StandardDie :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- Represents the 4x4x4 cube constructed from standard dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → StandardDie

/-- Calculates the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the smallest possible sum of visible faces is 144 -/
theorem smallest_visible_sum (cube : LargeCube) : 
  ∃ (min_cube : LargeCube), visibleSum min_cube = 144 ∧ ∀ (c : LargeCube), visibleSum c ≥ 144 :=
sorry

end smallest_visible_sum_l1408_140863


namespace money_distribution_l1408_140868

/-- Given that A, B, and C have a total of 500 Rs between them,
    B and C together have 320 Rs, and C has 20 Rs,
    prove that A and C together have 200 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 →
  B + C = 320 →
  C = 20 →
  A + C = 200 := by
  sorry

end money_distribution_l1408_140868


namespace compound_interest_years_l1408_140808

/-- Compound interest calculation --/
theorem compound_interest_years (P : ℝ) (r : ℝ) (CI : ℝ) (n : ℕ) : 
  P > 0 → r > 0 → CI > 0 → n > 0 →
  let A := P + CI
  let t := Real.log (A / P) / Real.log (1 + r / n)
  P = 1200 → r = 0.20 → n = 1 → CI = 873.60 →
  ⌈t⌉ = 3 := by sorry

#check compound_interest_years

end compound_interest_years_l1408_140808


namespace square_field_area_l1408_140812

theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 3.5 →
  total_cost = 2331 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    (4 * side_length - num_gates * gate_width) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end square_field_area_l1408_140812


namespace reciprocal_sum_fractions_l1408_140832

theorem reciprocal_sum_fractions : (((3 : ℚ) / 4 + (1 : ℚ) / 6)⁻¹) = (12 : ℚ) / 11 := by sorry

end reciprocal_sum_fractions_l1408_140832


namespace labourer_savings_is_30_l1408_140893

/-- Calculates the savings of a labourer after 10 months, given specific spending patterns -/
def labourerSavings (monthlyIncome : ℕ) (expenseFirst6Months : ℕ) (expenseLast4Months : ℕ) : ℤ :=
  let totalIncome : ℕ := monthlyIncome * 10
  let totalExpense : ℕ := expenseFirst6Months * 6 + expenseLast4Months * 4
  (totalIncome : ℤ) - (totalExpense : ℤ)

/-- Theorem stating that the labourer's savings after 10 months is 30 -/
theorem labourer_savings_is_30 :
  labourerSavings 75 80 60 = 30 := by
  sorry

#eval labourerSavings 75 80 60

end labourer_savings_is_30_l1408_140893


namespace hannah_savings_l1408_140854

theorem hannah_savings (first_week : ℝ) : first_week = 4 := by
  have total_goal : ℝ := 80
  have fifth_week : ℝ := 20
  have savings_sum : first_week + 2 * first_week + 4 * first_week + 8 * first_week + fifth_week = total_goal := by sorry
  sorry

end hannah_savings_l1408_140854


namespace inequality_solution_set_l1408_140843

theorem inequality_solution_set (a : ℝ) :
  let f := fun x : ℝ => (a^2 - 4) * x^2 + 4 * x - 1
  (∀ x, f x > 0 ↔ 
    (a = 2 ∨ a = -2 → x > 1/4) ∧
    (a > 2 → x > 1/(a+2) ∨ x < 1/(2-a)) ∧
    (a < -2 → x < 1/(a+2) ∨ x > 1/(2-a)) ∧
    (-2 < a ∧ a < 2 → 1/(a+2) < x ∧ x < 1/(2-a))) :=
by sorry

end inequality_solution_set_l1408_140843


namespace right_triangle_existence_condition_l1408_140859

/-- A right triangle with hypotenuse c and median s_a to one of the legs. -/
structure RightTriangle (c s_a : ℝ) :=
  (hypotenuse_positive : c > 0)
  (median_positive : s_a > 0)
  (right_angle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2)
  (median_property : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ s_a^2 = (a/2)^2 + (c/2)^2)

/-- The existence condition for a right triangle with given hypotenuse and median. -/
theorem right_triangle_existence_condition (c s_a : ℝ) :
  (∃ (t : RightTriangle c s_a), True) ↔ (c/2 < s_a ∧ s_a < c) :=
sorry

end right_triangle_existence_condition_l1408_140859


namespace power_function_decreasing_first_quadrant_l1408_140828

/-- A power function with negative exponent is decreasing in the first quadrant -/
theorem power_function_decreasing_first_quadrant (n : ℝ) (h : n < 0) :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂^n < x₁^n :=
by sorry


end power_function_decreasing_first_quadrant_l1408_140828


namespace conditional_probability_rain_given_east_wind_l1408_140890

theorem conditional_probability_rain_given_east_wind
  (p_east_wind : ℚ)
  (p_rain : ℚ)
  (p_both : ℚ)
  (h1 : p_east_wind = 3 / 10)
  (h2 : p_rain = 11 / 30)
  (h3 : p_both = 8 / 30) :
  p_both / p_east_wind = 8 / 9 :=
sorry

end conditional_probability_rain_given_east_wind_l1408_140890


namespace inverse_g_87_l1408_140899

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 6

-- Theorem statement
theorem inverse_g_87 : g⁻¹ 87 = 3 := by
  sorry

end inverse_g_87_l1408_140899


namespace hyperbola_standard_equation_l1408_140850

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The asymptotic equations of the hyperbola are y = ± (slope * x) -/
  slope : ℝ
  /-- The focal length of the hyperbola -/
  focal_length : ℝ

/-- Checks if the given equation is a valid standard form for the hyperbola -/
def is_standard_equation (h : Hyperbola) (eq : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, eq x y = 0 ↔ x^2 / 4 - y^2 = 1) ∨
  (∀ x y, eq x y = 0 ↔ y^2 - x^2 / 4 = 1)

/-- The main theorem stating the standard equation of the hyperbola -/
theorem hyperbola_standard_equation (h : Hyperbola) 
  (h_slope : h.slope = 1/2) 
  (h_focal : h.focal_length = 2 * Real.sqrt 5) :
  ∃ eq : ℝ → ℝ → ℝ, is_standard_equation h eq :=
sorry

end hyperbola_standard_equation_l1408_140850


namespace set_relationship_l1408_140873

def M : Set ℝ := {x : ℝ | ∃ m : ℤ, x = m + 1/6}
def S : Set ℝ := {x : ℝ | ∃ s : ℤ, x = 1/2 * s - 1/3}
def P : Set ℝ := {x : ℝ | ∃ p : ℤ, x = 1/2 * p + 1/6}

theorem set_relationship : M ⊆ S ∧ S = P := by sorry

end set_relationship_l1408_140873


namespace consecutive_six_product_not_776965920_l1408_140848

theorem consecutive_six_product_not_776965920 (n : ℕ) : 
  n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920 := by
  sorry

end consecutive_six_product_not_776965920_l1408_140848


namespace correct_result_l1408_140861

theorem correct_result (mistaken_result : ℕ) 
  (ones_digit_mistake : ℕ) (tens_digit_mistake : ℕ) : 
  mistaken_result = 387 ∧ 
  ones_digit_mistake = 8 - 3 ∧ 
  tens_digit_mistake = 90 - 50 → 
  mistaken_result - ones_digit_mistake + tens_digit_mistake = 422 :=
by sorry

end correct_result_l1408_140861


namespace building_units_count_l1408_140838

/-- Represents the number of units in a building -/
structure Building where
  oneBedroom : ℕ
  twoBedroom : ℕ

/-- The total cost of all units in the building -/
def totalCost (b : Building) : ℕ := 360 * b.oneBedroom + 450 * b.twoBedroom

/-- The total number of units in the building -/
def totalUnits (b : Building) : ℕ := b.oneBedroom + b.twoBedroom

theorem building_units_count :
  ∃ (b : Building),
    totalCost b = 4950 ∧
    b.twoBedroom = 7 ∧
    totalUnits b = 12 := by
  sorry

end building_units_count_l1408_140838


namespace factor_expression_l1408_140800

theorem factor_expression (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) = (x + y) * (y + z) * (z + x) :=
by sorry

end factor_expression_l1408_140800


namespace geometric_sequence_a3_l1408_140887

/-- Given a geometric sequence {a_n} with common ratio q > 1,
    if a_5 - a_1 = 15 and a_4 - a_2 = 6, then a_3 = 4 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 1 →  -- common ratio greater than 1
  a 5 - a 1 = 15 →  -- condition on a_5 and a_1
  a 4 - a 2 = 6 →  -- condition on a_4 and a_2
  a 3 = 4 := by
sorry

end geometric_sequence_a3_l1408_140887


namespace y_divisibility_l1408_140809

def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_divisibility :
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 9 * k) ∧
  (∃ k : ℕ, y = 27 * k) ∧
  (∃ k : ℕ, y = 81 * k) :=
by sorry

end y_divisibility_l1408_140809


namespace angle_c_value_l1408_140821

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem angle_c_value (t : Triangle) 
  (h : t.a^2 + t.b^2 = t.c^2 + Real.sqrt 3 * t.a * t.b) : 
  t.C = π/6 := by
  sorry

end angle_c_value_l1408_140821


namespace elderly_selected_in_scenario_l1408_140815

/-- Represents the number of elderly people selected in a stratified sampling -/
def elderly_selected (total_population : ℕ) (elderly_population : ℕ) (sample_size : ℕ) : ℚ :=
  (sample_size : ℚ) * (elderly_population : ℚ) / (total_population : ℚ)

/-- Theorem stating the number of elderly people selected in the given scenario -/
theorem elderly_selected_in_scenario : 
  elderly_selected 100 60 20 = 12 := by
  sorry

end elderly_selected_in_scenario_l1408_140815


namespace two_digit_numbers_with_difference_56_l1408_140840

-- Define the property for two numbers to have the same last two digits in their squares
def SameLastTwoDigitsSquared (a b : ℕ) : Prop :=
  a ^ 2 % 100 = b ^ 2 % 100

-- Main theorem
theorem two_digit_numbers_with_difference_56 :
  ∀ x y : ℕ,
    10 ≤ x ∧ x < 100 →  -- x is a two-digit number
    10 ≤ y ∧ y < 100 →  -- y is a two-digit number
    x - y = 56 →        -- their difference is 56
    SameLastTwoDigitsSquared x y →  -- last two digits of their squares are the same
    (x = 78 ∧ y = 22) :=
by sorry

end two_digit_numbers_with_difference_56_l1408_140840


namespace restaurant_glasses_count_l1408_140831

theorem restaurant_glasses_count :
  ∀ (x y : ℕ),
  -- x is the number of small boxes (12 glasses each)
  -- y is the number of large boxes (16 glasses each)
  y = x + 16 →  -- There are 16 more large boxes
  (12 * x + 16 * y) / (x + y) = 15 →  -- Average number of glasses per box is 15
  12 * x + 16 * y = 480  -- Total number of glasses
  := by sorry

end restaurant_glasses_count_l1408_140831


namespace completing_square_min_value_compare_expressions_l1408_140816

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Theorem 1: Completing the square
theorem completing_square : ∀ x : ℝ, f x = (x - 2)^2 + 2 := by sorry

-- Theorem 2: Minimum value and corresponding x
theorem min_value : 
  (∃ x_min : ℝ, ∀ x : ℝ, f x ≥ f x_min) ∧
  (∃ x_min : ℝ, f x_min = 2) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f x = 2 → x = x_min) ∧
  (∃ x_min : ℝ, x_min = 2) := by sorry

-- Theorem 3: Comparison of two expressions
theorem compare_expressions : ∀ x : ℝ, x^2 - 1 > 2*x - 3 := by sorry

end completing_square_min_value_compare_expressions_l1408_140816
