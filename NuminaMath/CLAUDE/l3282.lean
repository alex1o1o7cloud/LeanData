import Mathlib

namespace subtraction_digit_sum_l3282_328260

theorem subtraction_digit_sum : ∃ (a b : ℕ), 
  (a < 10) ∧ (b < 10) ∧ 
  (a * 10 + 9) - (1800 + b * 10 + 8) = 1 ∧
  a + b = 14 := by
sorry

end subtraction_digit_sum_l3282_328260


namespace train_speed_l3282_328251

/-- Proves that the speed of a train is 72 km/hr, given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 9) :
  (length / time) * 3.6 = 72 := by
  sorry

end train_speed_l3282_328251


namespace truck_capacity_l3282_328262

theorem truck_capacity (large small : ℝ) 
  (h1 : 2 * large + 3 * small = 15.5)
  (h2 : 5 * large + 6 * small = 35) :
  3 * large + 2 * small = 17 := by
sorry

end truck_capacity_l3282_328262


namespace odd_function_extension_l3282_328236

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (1 + 3 * x)) :
  ∀ x < 0, f x = x * (1 - 3 * x) := by
sorry

end odd_function_extension_l3282_328236


namespace megan_markers_proof_l3282_328240

def final_markers (initial : ℕ) (robert_factor : ℕ) (elizabeth_taken : ℕ) : ℕ :=
  initial + robert_factor * initial - elizabeth_taken

theorem megan_markers_proof :
  final_markers 2475 3 1650 = 8250 := by
  sorry

end megan_markers_proof_l3282_328240


namespace cricket_score_theorem_l3282_328231

/-- Represents the score of a cricket match -/
def CricketScore := ℕ

/-- Calculates the total runs from boundaries -/
def boundaryRuns (boundaries : ℕ) : ℕ := 4 * boundaries

/-- Calculates the total runs from sixes -/
def sixRuns (sixes : ℕ) : ℕ := 6 * sixes

/-- Theorem: Given the conditions of the cricket match, prove the total score is 142 runs -/
theorem cricket_score_theorem (boundaries sixes : ℕ) 
  (h1 : boundaries = 12)
  (h2 : sixes = 2)
  (h3 : (57.74647887323944 : ℚ) / 100 * 142 = 142 - boundaryRuns boundaries - sixRuns sixes) :
  142 = boundaryRuns boundaries + sixRuns sixes + 
    ((57.74647887323944 : ℚ) / 100 * 142).floor :=
by sorry

end cricket_score_theorem_l3282_328231


namespace max_value_quadratic_l3282_328206

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  x^2 + 2*x*y + 3*y^2 ≤ 24 + 12*Real.sqrt 3 := by
  sorry

end max_value_quadratic_l3282_328206


namespace trapezoid_angle_sequence_l3282_328214

theorem trapezoid_angle_sequence (a d : ℝ) : 
  (a > 0) →
  (d > 0) →
  (a + 2*d = 105) →
  (4*a + 6*d = 360) →
  (a + d = 85) :=
by sorry

end trapezoid_angle_sequence_l3282_328214


namespace parabola_hyperbola_equations_l3282_328219

-- Define the parabola and hyperbola
def parabola (x y : ℝ) : Prop := y^2 = -4*x
def hyperbola (x y : ℝ) : Prop := x^2/(1/4) - y^2/(3/4) = 1

-- Define the conditions
def parabola_vertex_origin (p : ℝ → ℝ → Prop) : Prop := p 0 0
def parabola_axis_perpendicular_x (p : ℝ → ℝ → Prop) : Prop := ∀ y, p 1 y
def parabola_hyperbola_intersection (p h : ℝ → ℝ → Prop) : Prop := p (-3/2) (Real.sqrt 6) ∧ h (-3/2) (Real.sqrt 6)
def hyperbola_focus (h : ℝ → ℝ → Prop) : Prop := h 1 0

-- Main theorem
theorem parabola_hyperbola_equations :
  ∀ p h : ℝ → ℝ → Prop,
  parabola_vertex_origin p →
  parabola_axis_perpendicular_x p →
  parabola_hyperbola_intersection p h →
  hyperbola_focus h →
  (∀ x y, p x y ↔ parabola x y) ∧
  (∀ x y, h x y ↔ hyperbola x y) :=
sorry

end parabola_hyperbola_equations_l3282_328219


namespace music_books_cost_l3282_328282

/-- Calculates the amount spent on music books including tax --/
def amount_spent_on_music_books (total_budget : ℝ) (math_book_price : ℝ) (math_book_count : ℕ) 
  (math_book_discount : ℝ) (science_book_price : ℝ) (art_book_price : ℝ) (art_book_tax : ℝ) 
  (music_book_tax : ℝ) : ℝ :=
  let math_books_cost := math_book_count * math_book_price * (1 - math_book_discount)
  let science_books_cost := (math_book_count + 6) * science_book_price
  let art_books_cost := 2 * math_book_count * art_book_price * (1 + art_book_tax)
  let remaining_budget := total_budget - (math_books_cost + science_books_cost + art_books_cost)
  remaining_budget

/-- Theorem stating that the amount spent on music books including tax is $160 --/
theorem music_books_cost (total_budget : ℝ) (math_book_price : ℝ) (math_book_count : ℕ) 
  (math_book_discount : ℝ) (science_book_price : ℝ) (art_book_price : ℝ) (art_book_tax : ℝ) 
  (music_book_tax : ℝ) :
  total_budget = 500 ∧ 
  math_book_price = 20 ∧ 
  math_book_count = 4 ∧ 
  math_book_discount = 0.1 ∧ 
  science_book_price = 10 ∧ 
  art_book_price = 20 ∧ 
  art_book_tax = 0.05 ∧ 
  music_book_tax = 0.07 → 
  amount_spent_on_music_books total_budget math_book_price math_book_count math_book_discount 
    science_book_price art_book_price art_book_tax music_book_tax = 160 := by
  sorry


end music_books_cost_l3282_328282


namespace max_modulus_complex_l3282_328246

theorem max_modulus_complex (z : ℂ) : 
  ∀ z, Complex.abs (z + z⁻¹) = 1 → Complex.abs z ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end max_modulus_complex_l3282_328246


namespace parallelogram_area_calculation_l3282_328216

/-- The area of a parallelogram generated by two vectors -/
def parallelogram_area (a b : ℝ × ℝ) : ℝ := sorry

theorem parallelogram_area_calculation 
  (a b : ℝ × ℝ) 
  (h1 : parallelogram_area a b = 20)
  (u : ℝ × ℝ := (1/2 : ℝ) • a + (5/2 : ℝ) • b)
  (v : ℝ × ℝ := 3 • a - 2 • b) :
  parallelogram_area u v = 130 := by sorry

end parallelogram_area_calculation_l3282_328216


namespace no_natural_solutions_l3282_328283

theorem no_natural_solutions : ¬∃ (x y : ℕ), x^4 - 2*y^2 = 1 := by
  sorry

end no_natural_solutions_l3282_328283


namespace alex_phone_bill_l3282_328278

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (included_texts : ℕ) (text_cost : ℚ) 
                         (included_hours : ℕ) (minute_cost : ℚ)
                         (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let extra_texts := max (texts_sent - included_texts) 0
  let extra_minutes := max ((hours_talked - included_hours) * 60) 0
  base_cost + (extra_texts : ℚ) * text_cost + (extra_minutes : ℚ) * minute_cost

theorem alex_phone_bill :
  calculate_phone_bill 25 20 0.1 20 (15 / 100) 150 25 = 83 := by
  sorry

end alex_phone_bill_l3282_328278


namespace not_right_triangle_l3282_328273

theorem not_right_triangle : ∀ a b c : ℕ,
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ 
  (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ 
  (a = 7 ∧ b = 8 ∧ c = 13) →
  (a^2 + b^2 ≠ c^2) ↔ (a = 7 ∧ b = 8 ∧ c = 13) :=
by sorry

end not_right_triangle_l3282_328273


namespace simplify_and_evaluate_l3282_328212

theorem simplify_and_evaluate (x y : ℝ) (h : x / y = 3) :
  (1 + y^2 / (x^2 - y^2)) * ((x - y) / x) = 3 / 4 := by
  sorry

end simplify_and_evaluate_l3282_328212


namespace congruence_conditions_and_smallest_n_l3282_328250

theorem congruence_conditions_and_smallest_n :
  ∀ (r s : ℕ+),
  (2^(r : ℕ) - 16^(s : ℕ)) % 7 = 5 →
  (r : ℕ) % 3 = 1 ∧ (s : ℕ) % 3 = 2 ∧
  (∀ (r' s' : ℕ+),
    (2^(r' : ℕ) - 16^(s' : ℕ)) % 7 = 5 →
    2^(r : ℕ) - 16^(s : ℕ) ≤ 2^(r' : ℕ) - 16^(s' : ℕ)) ∧
  2^(r : ℕ) - 16^(s : ℕ) = 768 :=
by sorry

end congruence_conditions_and_smallest_n_l3282_328250


namespace even_decreasing_function_a_equals_two_l3282_328229

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if f(x) > f(y) for all 0 < x < y -/
def IsDecreasingOnPositives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

/-- The main theorem -/
theorem even_decreasing_function_a_equals_two (a : ℤ) :
  IsEven (fun x => x^(a^2 - 4*a)) →
  IsDecreasingOnPositives (fun x => x^(a^2 - 4*a)) →
  a = 2 := by
  sorry

end even_decreasing_function_a_equals_two_l3282_328229


namespace mans_swimming_speed_l3282_328270

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem mans_swimming_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 51) 
  (h2 : upstream_distance = 18) (h3 : downstream_time = 3) (h4 : upstream_time = 3) :
  ∃ (v_m : ℝ), v_m = 11.5 ∧ 
    (downstream_distance / downstream_time + upstream_distance / upstream_time) / 2 = v_m :=
by sorry

end mans_swimming_speed_l3282_328270


namespace spongebob_fry_price_l3282_328220

/-- Calculates the price of each large fry given the number of burgers sold, 
    price per burger, number of large fries sold, and total earnings. -/
def price_of_large_fry (num_burgers : ℕ) (price_per_burger : ℚ) 
                       (num_fries : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - num_burgers * price_per_burger) / num_fries

/-- Theorem stating that the price of each large fry is $1.50 
    given Spongebob's sales information. -/
theorem spongebob_fry_price : 
  price_of_large_fry 30 2 12 78 = (3/2) := by
  sorry

end spongebob_fry_price_l3282_328220


namespace linear_coefficient_is_correct_l3282_328223

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

-- Define the coefficient of the linear term
def linear_coefficient : ℝ := -2

-- Theorem statement
theorem linear_coefficient_is_correct :
  ∃ (a c : ℝ), ∀ x, quadratic_equation x ↔ x^2 + linear_coefficient * x + c = 0 :=
sorry

end linear_coefficient_is_correct_l3282_328223


namespace singing_competition_result_l3282_328290

def singing_competition (total_contestants : ℕ) 
                        (female_solo_percent : ℚ) 
                        (male_solo_percent : ℚ) 
                        (group_percent : ℚ) 
                        (male_young_percent : ℚ) 
                        (female_young_percent : ℚ) : Prop :=
  let female_solo := ⌊(female_solo_percent * total_contestants : ℚ)⌋
  let male_solo := ⌊(male_solo_percent * total_contestants : ℚ)⌋
  let male_young := ⌊(male_young_percent * male_solo : ℚ)⌋
  let female_young := ⌊(female_young_percent * female_solo : ℚ)⌋
  total_contestants = 18 ∧
  female_solo_percent = 35/100 ∧
  male_solo_percent = 25/100 ∧
  group_percent = 40/100 ∧
  male_young_percent = 30/100 ∧
  female_young_percent = 20/100 ∧
  male_young = 1 ∧
  female_young = 1

theorem singing_competition_result : 
  singing_competition 18 (35/100) (25/100) (40/100) (30/100) (20/100) := by
  sorry

end singing_competition_result_l3282_328290


namespace expression_simplification_l3282_328208

theorem expression_simplification (a : ℝ) (h : a = 2 * Real.sqrt 3 + 3) :
  (1 - 1 / (a - 2)) / ((a^2 - 6*a + 9) / (2*a - 4)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l3282_328208


namespace inequality_system_solution_l3282_328211

theorem inequality_system_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | x > -2*a ∧ x > 3*a} = {x : ℝ | x > -2*a} := by
  sorry

end inequality_system_solution_l3282_328211


namespace courtyard_length_l3282_328264

theorem courtyard_length (stone_length : ℝ) (stone_width : ℝ) (courtyard_width : ℝ) (total_stones : ℕ) :
  stone_length = 2.5 →
  stone_width = 2 →
  courtyard_width = 16.5 →
  total_stones = 198 →
  ∃ courtyard_length : ℝ, courtyard_length = 60 ∧ 
    courtyard_length * courtyard_width = (stone_length * stone_width) * total_stones :=
by
  sorry

end courtyard_length_l3282_328264


namespace symmetry_and_inverse_l3282_328257

/-- A function that is symmetric about the line y = x + 1 -/
def SymmetricAboutXPlus1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (y - 1) = x + 1

/-- Definition of g in terms of f and b -/
def g (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x ↦ f (x + b)

/-- A function that is identical to its inverse -/
def IdenticalToInverse (h : ℝ → ℝ) : Prop :=
  ∀ x, h (h x) = x

theorem symmetry_and_inverse (f : ℝ → ℝ) (b : ℝ) 
  (h_sym : SymmetricAboutXPlus1 f) :
  IdenticalToInverse (g f b) ↔ b = -1 := by
  sorry

end symmetry_and_inverse_l3282_328257


namespace existence_condition_range_l3282_328275

theorem existence_condition_range (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, -x₀^2 + 3*x₀ + a > 0) ↔ a > -2 :=
sorry

end existence_condition_range_l3282_328275


namespace roots_in_specific_intervals_roots_in_unit_interval_l3282_328238

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 2*m + 1

-- Part I
theorem roots_in_specific_intervals (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-1 : ℝ) 0 ∧ 
             x₂ ∈ Set.Ioo 1 2 ∧ 
             f m x₁ = 0 ∧ 
             f m x₂ = 0) →
  m ∈ Set.Ioo (-5/6 : ℝ) (-1/2) :=
sorry

-- Part II
theorem roots_in_unit_interval (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo 0 1 ∧ 
             x₂ ∈ Set.Ioo 0 1 ∧ 
             f m x₁ = 0 ∧ 
             f m x₂ = 0) →
  m ∈ Set.Ico (-1/2 : ℝ) (1 - Real.sqrt 2) :=
sorry

end roots_in_specific_intervals_roots_in_unit_interval_l3282_328238


namespace prove_necklace_sum_l3282_328200

def necklace_sum (H J x S : ℕ) : Prop :=
  (H = J + 5) ∧ 
  (x = J / 2) ∧ 
  (S = 2 * H) ∧ 
  (H = 25) →
  H + J + x + S = 105

theorem prove_necklace_sum : 
  ∀ (H J x S : ℕ), necklace_sum H J x S :=
by
  sorry

end prove_necklace_sum_l3282_328200


namespace f_equals_g_l3282_328263

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end f_equals_g_l3282_328263


namespace tan_half_difference_l3282_328280

theorem tan_half_difference (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5) 
  (h2 : Real.sin a + Real.sin b = 2/5) : 
  Real.tan ((a - b)/2) = 2/3 := by
  sorry

end tan_half_difference_l3282_328280


namespace proportional_sampling_l3282_328254

theorem proportional_sampling :
  let total_population : ℕ := 162
  let elderly_population : ℕ := 27
  let middle_aged_population : ℕ := 54
  let young_population : ℕ := 81
  let sample_size : ℕ := 36

  let elderly_sample : ℕ := 6
  let middle_aged_sample : ℕ := 12
  let young_sample : ℕ := 18

  elderly_population + middle_aged_population + young_population = total_population →
  elderly_sample + middle_aged_sample + young_sample = sample_size →
  (elderly_sample : ℚ) / sample_size = (elderly_population : ℚ) / total_population ∧
  (middle_aged_sample : ℚ) / sample_size = (middle_aged_population : ℚ) / total_population ∧
  (young_sample : ℚ) / sample_size = (young_population : ℚ) / total_population :=
by
  sorry

end proportional_sampling_l3282_328254


namespace modulus_of_z_values_of_a_and_b_l3282_328286

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)

-- Theorem for the modulus of z
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

-- Theorem for the values of a and b
theorem values_of_a_and_b :
  ∀ (a b : ℝ), z^2 + a*z + b = 1 + i → a = -3 ∧ b = 4 := by sorry

end modulus_of_z_values_of_a_and_b_l3282_328286


namespace count_valid_a_l3282_328213

-- Define the system of inequalities
def system_inequalities (a : ℤ) (x : ℤ) : Prop :=
  6 * x - 5 ≥ a ∧ (x : ℚ) / 4 - (x - 1 : ℚ) / 6 < 1 / 2

-- Define the equation
def equation (a : ℤ) (y : ℚ) : Prop :=
  4 * y - 3 * (a : ℚ) = 2 * (y - 3)

-- Main theorem
theorem count_valid_a : 
  (∃ (s : Finset ℤ), s.card = 5 ∧ 
    (∀ a : ℤ, a ∈ s ↔ 
      (∃! (sol : Finset ℤ), sol.card = 2 ∧ 
        (∀ x : ℤ, x ∈ sol ↔ system_inequalities a x)) ∧
      (∃ y : ℚ, y > 0 ∧ equation a y))) := by sorry

end count_valid_a_l3282_328213


namespace gathering_attendees_l3282_328255

theorem gathering_attendees (n : ℕ) : 
  (n * (n - 1) / 2 = 55) → n = 11 := by
  sorry

end gathering_attendees_l3282_328255


namespace sod_area_calculation_l3282_328239

-- Define the dimensions
def yard_width : ℕ := 200
def yard_depth : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def front_flowerbed_depth : ℕ := 4
def front_flowerbed_length : ℕ := 25
def third_flowerbed_width : ℕ := 10
def third_flowerbed_length : ℕ := 12
def fourth_flowerbed_width : ℕ := 7
def fourth_flowerbed_length : ℕ := 8

-- Define the theorem
theorem sod_area_calculation : 
  let total_area := yard_width * yard_depth
  let sidewalk_area := sidewalk_width * sidewalk_length
  let front_flowerbeds_area := 2 * (front_flowerbed_depth * front_flowerbed_length)
  let third_flowerbed_area := third_flowerbed_width * third_flowerbed_length
  let fourth_flowerbed_area := fourth_flowerbed_width * fourth_flowerbed_length
  let non_sod_area := sidewalk_area + front_flowerbeds_area + third_flowerbed_area + fourth_flowerbed_area
  total_area - non_sod_area = 9474 := by
sorry

end sod_area_calculation_l3282_328239


namespace parabola_equation_and_min_ratio_l3282_328261

/-- Represents a parabola with focus F and parameter p -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  h : p > 0

/-- A point on the parabola -/
def PointOnParabola (para : Parabola) (P : ℝ × ℝ) : Prop :=
  P.2^2 = 2 * para.p * P.1

theorem parabola_equation_and_min_ratio 
  (para : Parabola) 
  (P : ℝ × ℝ) 
  (h_P_on_parabola : PointOnParabola para P)
  (h_P_ordinate : P.2 = 4)
  (h_PF_distance : Real.sqrt ((P.1 - para.F.1)^2 + (P.2 - para.F.2)^2) = 4) :
  -- 1. The equation of the parabola is y^2 = 8x
  (∀ (x y : ℝ), PointOnParabola para (x, y) ↔ y^2 = 8*x) ∧
  -- 2. Minimum value of |MF| / |AB| is 1/2
  (∀ (A B : ℝ × ℝ) (h_A_on_parabola : PointOnParabola para A) 
                    (h_B_on_parabola : PointOnParabola para B)
                    (h_A_ne_B : A ≠ B)
                    (h_A_ne_P : A ≠ P)
                    (h_B_ne_P : B ≠ P),
   ∃ (M : ℝ × ℝ),
     -- Angle bisector of ∠APB is perpendicular to x-axis
     (∃ (k : ℝ), (A.2 - P.2) / (A.1 - P.1) = k ∧ (B.2 - P.2) / (B.1 - P.1) = -1/k) →
     -- M is on x-axis and perpendicular bisector of AB
     (M.2 = 0 ∧ (M.1 - (A.1 + B.1)/2) * ((B.2 - A.2)/(B.1 - A.1)) + (M.2 - (A.2 + B.2)/2) = 0) →
     -- Minimum value of |MF| / |AB| is 1/2
     (Real.sqrt ((M.1 - para.F.1)^2 + (M.2 - para.F.2)^2) / 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ 1/2)) := by
  sorry

end parabola_equation_and_min_ratio_l3282_328261


namespace equal_fractions_imply_equal_values_l3282_328299

theorem equal_fractions_imply_equal_values (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : (a + b) / c = (c + b) / a) 
  (h2 : (c + b) / a = (a + c) / b) : 
  a = b ∧ b = c := by
  sorry

end equal_fractions_imply_equal_values_l3282_328299


namespace trig_identity_l3282_328201

theorem trig_identity (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : 2 * (Real.sin α)^2 - Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = 0) :
  Real.sin (α + π / 4) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = Real.sqrt 26 / 8 := by
  sorry

end trig_identity_l3282_328201


namespace coloring_books_sale_result_l3282_328284

/-- The number of coloring books gotten rid of in a store sale -/
def books_gotten_rid_of (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  initial_stock - (shelves * books_per_shelf)

/-- Theorem stating that the number of coloring books gotten rid of is 39 -/
theorem coloring_books_sale_result : 
  books_gotten_rid_of 120 9 9 = 39 := by
  sorry

end coloring_books_sale_result_l3282_328284


namespace fence_cost_square_plot_l3282_328267

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 25) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 1160 := by
  sorry

end fence_cost_square_plot_l3282_328267


namespace quadratic_equation_roots_l3282_328285

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 16) / 4 ∨ x = (-8 - Real.sqrt 16) / 4) →
  c = 6 := by
  sorry

end quadratic_equation_roots_l3282_328285


namespace train_crossing_signal_pole_l3282_328259

/-- Given a train and a platform with the following properties:
  * The train is 300 meters long
  * The platform is 250 meters long
  * The train crosses the platform in 33 seconds
  This theorem proves that the time taken for the train to cross a signal pole is 18 seconds. -/
theorem train_crossing_signal_pole
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 250)
  (h3 : platform_crossing_time = 33)
  : ℝ :=
let total_distance := train_length + platform_length
let train_speed := total_distance / platform_crossing_time
let signal_pole_crossing_time := train_length / train_speed
18

/-- The proof of the theorem -/
lemma train_crossing_signal_pole_proof
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 250)
  (h3 : platform_crossing_time = 33)
  : train_crossing_signal_pole train_length platform_length platform_crossing_time h1 h2 h3 = 18 := by
  sorry

end train_crossing_signal_pole_l3282_328259


namespace suitable_squares_are_1_4_9_49_l3282_328269

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A number is suitable if it's the smallest among all natural numbers with the same digit sum -/
def is_suitable (n : ℕ) : Prop :=
  ∀ m : ℕ, digit_sum m = digit_sum n → n ≤ m

/-- The set of all suitable square numbers -/
def suitable_squares : Set ℕ :=
  {n : ℕ | is_suitable n ∧ ∃ k : ℕ, n = k^2}

/-- Theorem: The set of suitable square numbers is exactly {1, 4, 9, 49} -/
theorem suitable_squares_are_1_4_9_49 : suitable_squares = {1, 4, 9, 49} := by sorry

end suitable_squares_are_1_4_9_49_l3282_328269


namespace sum_of_integers_l3282_328274

theorem sum_of_integers (s l : ℤ) : 
  s = 10 → 2 * l = 5 * s - 10 → s + l = 30 := by
  sorry

end sum_of_integers_l3282_328274


namespace both_runners_in_photo_probability_l3282_328265

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photography setup -/
structure Photo where
  trackFraction : ℚ
  timeRange : Set ℕ

/-- Calculates the probability of both runners being in the photo -/
def probabilityBothInPhoto (r1 r2 : Runner) (p : Photo) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem both_runners_in_photo_probability
  (rachel : Runner)
  (robert : Runner)
  (photo : Photo)
  (h1 : rachel.name = "Rachel" ∧ rachel.lapTime = 75 ∧ rachel.direction = true)
  (h2 : robert.name = "Robert" ∧ robert.lapTime = 95 ∧ robert.direction = false)
  (h3 : photo.trackFraction = 1/5)
  (h4 : photo.timeRange = {t | 900 ≤ t ∧ t < 960}) :
  probabilityBothInPhoto rachel robert photo = 1/4 :=
sorry

end both_runners_in_photo_probability_l3282_328265


namespace abs_y_plus_sqrt_y_plus_two_squared_l3282_328294

theorem abs_y_plus_sqrt_y_plus_two_squared (y : ℝ) (h : y > 1) :
  |y + Real.sqrt ((y + 2)^2)| = 2*y + 2 := by
  sorry

end abs_y_plus_sqrt_y_plus_two_squared_l3282_328294


namespace initial_column_size_l3282_328207

theorem initial_column_size (total_people : ℕ) (initial_columns : ℕ) (people_per_column : ℕ) : 
  total_people = initial_columns * people_per_column →
  total_people = 40 * 12 →
  initial_columns = 16 →
  people_per_column = 30 := by
sorry

end initial_column_size_l3282_328207


namespace min_m_value_l3282_328202

open Real

-- Define the inequality function
def f (x m : ℝ) : Prop := x + m * log x + exp (-x) ≥ x^m

-- State the theorem
theorem min_m_value :
  (∀ x > 1, ∀ m : ℝ, f x m) →
  (∃ m₀ : ℝ, m₀ = -exp 1 ∧ (∀ m : ℝ, (∀ x > 1, f x m) → m ≥ m₀)) :=
sorry

end min_m_value_l3282_328202


namespace paths_in_4x4_grid_l3282_328227

/-- Number of paths in a grid -/
def num_paths (m n : ℕ) : ℕ :=
  if m = 0 ∨ n = 0 then 1
  else num_paths (m - 1) n + num_paths m (n - 1)

/-- Theorem: The number of paths from (0,0) to (3,3) in a 4x4 grid is 23 -/
theorem paths_in_4x4_grid : num_paths 3 3 = 23 := by
  sorry

end paths_in_4x4_grid_l3282_328227


namespace integral_inverse_cube_l3282_328289

theorem integral_inverse_cube (x : ℝ) (h : x ≠ 0) :
  ∫ (t : ℝ) in Set.Ioo 0 x, 1 / (t^3) = -1 / (2 * x^2) + 1 / 2 := by sorry

end integral_inverse_cube_l3282_328289


namespace ellipse_properties_l3282_328252

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the left focus
def LeftFocus : ℝ × ℝ := (-1, 0)

-- Define the line l
def Line (x y : ℝ) : Prop :=
  y = x + 1

-- Theorem statement
theorem ellipse_properties :
  -- Given conditions
  let C := Ellipse
  let e : ℝ := 1/2
  let max_distance : ℝ := 3
  let l := Line

  -- Prove
  (∀ x y, C x y → x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ A B : ℝ × ℝ,
    C A.1 A.2 ∧ C B.1 B.2 ∧
    l A.1 A.2 ∧ l B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 24/7) :=
by
  sorry

end ellipse_properties_l3282_328252


namespace prism_lateral_edges_parallel_equal_l3282_328247

structure Prism where
  -- A prism is a polyhedron
  is_polyhedron : Bool
  -- A prism has two congruent and parallel bases
  has_congruent_parallel_bases : Bool
  -- The lateral faces of a prism are parallelograms
  lateral_faces_are_parallelograms : Bool

/-- The lateral edges of a prism are parallel and equal in length -/
theorem prism_lateral_edges_parallel_equal (p : Prism) :
  p.is_polyhedron ∧ p.has_congruent_parallel_bases ∧ p.lateral_faces_are_parallelograms →
  (lateral_edges_parallel : Bool) ∧ (lateral_edges_equal_length : Bool) :=
by sorry

end prism_lateral_edges_parallel_equal_l3282_328247


namespace conjunction_implies_left_prop_l3282_328245

theorem conjunction_implies_left_prop (p q : Prop) : (p ∧ q) → p := by
  sorry

end conjunction_implies_left_prop_l3282_328245


namespace return_trip_duration_l3282_328293

def time_to_park : ℕ := 20 + 10

def return_trip_factor : ℕ := 3

theorem return_trip_duration : 
  return_trip_factor * time_to_park = 90 :=
by sorry

end return_trip_duration_l3282_328293


namespace oil_drop_probability_l3282_328234

/-- The probability of an oil drop falling into a square hole in a circular coin -/
theorem oil_drop_probability (coin_diameter : Real) (hole_side : Real) : 
  coin_diameter = 2 → hole_side = 0.5 → 
  (hole_side^2) / (π * (coin_diameter/2)^2) = 1 / (4 * π) := by
sorry

end oil_drop_probability_l3282_328234


namespace problem_solution_l3282_328222

theorem problem_solution (x : ℝ) : (0.75 * x = (1/3) * x + 110) → x = 264 := by
  sorry

end problem_solution_l3282_328222


namespace hiram_age_is_40_l3282_328215

/-- Hiram's age in years -/
def hiram_age : ℕ := sorry

/-- Allyson's age in years -/
def allyson_age : ℕ := 28

/-- Theorem stating Hiram's age based on the given conditions -/
theorem hiram_age_is_40 :
  (hiram_age + 12 = 2 * allyson_age - 4) → hiram_age = 40 := by
  sorry

end hiram_age_is_40_l3282_328215


namespace initial_worth_is_30_l3282_328217

/-- Represents the value of a single gold coin -/
def coin_value : ℕ := 6

/-- Represents the number of coins Roman sold -/
def sold_coins : ℕ := 3

/-- Represents the number of coins Roman has left after the sale -/
def remaining_coins : ℕ := 2

/-- Represents the amount of money Roman has after the sale -/
def money_after_sale : ℕ := 12

/-- Theorem stating that the initial total worth of Roman's gold coins was $30 -/
theorem initial_worth_is_30 :
  (sold_coins + remaining_coins) * coin_value = 30 :=
sorry

end initial_worth_is_30_l3282_328217


namespace count_solutions_3x_plus_5y_equals_501_l3282_328218

theorem count_solutions_3x_plus_5y_equals_501 :
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 5 * p.2 = 501 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 168) (Finset.range 101))).card = 34 := by
  sorry

end count_solutions_3x_plus_5y_equals_501_l3282_328218


namespace enclosing_polygons_sides_l3282_328288

/-- The number of sides of the central polygon -/
def m : ℕ := 12

/-- The number of polygons enclosing the central polygon -/
def enclosing_polygons : ℕ := 12

/-- The number of sides of each enclosing polygon -/
def n : ℕ := 12

/-- The interior angle of a regular polygon with k sides -/
def interior_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
def exterior_angle (k : ℕ) : ℚ := 360 / k

/-- Theorem: In a configuration where a regular polygon with m sides is exactly
    enclosed by 'enclosing_polygons' number of regular polygons each with n sides,
    the value of n must be equal to the number of sides of the central polygon. -/
theorem enclosing_polygons_sides (h1 : m = 12) (h2 : enclosing_polygons = 12) :
  n = m := by sorry

end enclosing_polygons_sides_l3282_328288


namespace pyramid_stack_balls_l3282_328243

/-- Represents a pyramid-shaped stack of balls -/
structure PyramidStack where
  top_layer : Nat
  layer_diff : Nat
  bottom_layer : Nat

/-- Calculates the number of layers in the pyramid stack -/
def num_layers (p : PyramidStack) : Nat :=
  (p.bottom_layer - p.top_layer) / p.layer_diff + 1

/-- Calculates the total number of balls in the pyramid stack -/
def total_balls (p : PyramidStack) : Nat :=
  let n := num_layers p
  n * (p.top_layer + p.bottom_layer) / 2

/-- Theorem: The total number of balls in the given pyramid stack is 247 -/
theorem pyramid_stack_balls :
  let p := PyramidStack.mk 1 3 37
  total_balls p = 247 := by
  sorry

end pyramid_stack_balls_l3282_328243


namespace shekar_average_marks_l3282_328233

theorem shekar_average_marks :
  let math_score := 76
  let science_score := 65
  let social_studies_score := 82
  let english_score := 67
  let biology_score := 95
  let total_score := math_score + science_score + social_studies_score + english_score + biology_score
  let num_subjects := 5
  (total_score / num_subjects : ℚ) = 77 := by
  sorry

end shekar_average_marks_l3282_328233


namespace triangle_inequality_l3282_328224

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  -- Add triangle inequality conditions
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.area ∧
  (t.a^2 + t.b^2 + t.c^2 = 4 * Real.sqrt 3 * t.area ↔ isEquilateral t) :=
sorry

end triangle_inequality_l3282_328224


namespace alien_legs_count_l3282_328266

/-- Represents the number of limbs for an alien or martian -/
structure Limbs where
  arms : ℕ
  legs : ℕ

/-- Defines the properties of alien limbs -/
def alien_limbs (l : ℕ) : Limbs :=
  { arms := 3, legs := l }

/-- Defines the properties of martian limbs based on alien legs -/
def martian_limbs (l : ℕ) : Limbs :=
  { arms := 2 * 3, legs := l / 2 }

/-- Theorem stating that aliens have 8 legs -/
theorem alien_legs_count : 
  ∃ l : ℕ, 
    (alien_limbs l).legs = 8 ∧ 
    5 * ((alien_limbs l).arms + (alien_limbs l).legs) = 
    5 * ((martian_limbs l).arms + (martian_limbs l).legs) + 5 :=
by
  sorry


end alien_legs_count_l3282_328266


namespace binomial_sum_l3282_328279

theorem binomial_sum : (Nat.choose 10 3) + (Nat.choose 10 4) = 330 := by
  sorry

end binomial_sum_l3282_328279


namespace ancient_tower_height_l3282_328271

/-- Proves that the height of an ancient tower is 14.4 meters given the conditions of Xiao Liang's height and shadow length, and the tower's shadow length. -/
theorem ancient_tower_height 
  (xiao_height : ℝ) 
  (xiao_shadow : ℝ) 
  (tower_shadow : ℝ) 
  (h1 : xiao_height = 1.6)
  (h2 : xiao_shadow = 2)
  (h3 : tower_shadow = 18) :
  (xiao_height / xiao_shadow) * tower_shadow = 14.4 :=
by sorry

end ancient_tower_height_l3282_328271


namespace smallest_x_and_corresponding_yzw_l3282_328272

theorem smallest_x_and_corresponding_yzw :
  ∀ (x y z w : ℝ),
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0 →
  y = x - 2003 →
  z = 2*y - 2003 →
  w = 3*z - 2003 →
  (x ≥ 10015/3 ∧ 
   (x = 10015/3 → y = 4006/3 ∧ z = 2003/3 ∧ w = 0)) :=
by sorry

end smallest_x_and_corresponding_yzw_l3282_328272


namespace binary_conversion_theorem_l3282_328291

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_conversion_theorem :
  let binary : List Bool := [true, false, true, true, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let base7 : List ℕ := decimal_to_base7 decimal
  decimal = 45 ∧ base7 = [6, 3] := by sorry

end binary_conversion_theorem_l3282_328291


namespace work_efficiency_increase_l3282_328209

theorem work_efficiency_increase (original_months : ℕ) (actual_months : ℕ) (x : ℚ) : 
  original_months = 20 →
  actual_months = 18 →
  (1 : ℚ) / actual_months = (1 : ℚ) / original_months * (1 + x / 100) :=
by sorry

end work_efficiency_increase_l3282_328209


namespace quadratic_coefficient_l3282_328292

theorem quadratic_coefficient (a b c : ℝ) (h1 : a ≠ 0) : 
  let f := fun x => a * x^2 + b * x + c
  let Δ := b^2 - 4*a*c
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ (x - y)^2 = 1) →
  Δ = 1/4 →
  a = -1/2 := by
sorry

end quadratic_coefficient_l3282_328292


namespace triangle_area_theorem_l3282_328297

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end triangle_area_theorem_l3282_328297


namespace intersection_condition_m_values_l3282_328296

theorem intersection_condition_m_values (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - x - 6 = 0}
  let B : Set ℝ := {x | x * m - 1 = 0}
  (A ∩ B = B) ↔ (m = 0 ∨ m = -1/2 ∨ m = 1/3) := by
  sorry

end intersection_condition_m_values_l3282_328296


namespace soccer_camp_afternoon_attendance_l3282_328268

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : total_kids / 2 = total_kids / 2) -- Half of kids go to soccer camp
  (h3 : (total_kids / 2) / 4 = (total_kids / 2) / 4) -- 1/4 of soccer camp kids go in the morning
  : total_kids / 2 - (total_kids / 2) / 4 = 750 := by
  sorry

end soccer_camp_afternoon_attendance_l3282_328268


namespace rabbit_count_l3282_328235

theorem rabbit_count (white_rabbits black_rabbits female_rabbits : ℕ) : 
  white_rabbits = 12 → black_rabbits = 9 → female_rabbits = 8 → 
  white_rabbits + black_rabbits - female_rabbits = 13 := by
sorry

end rabbit_count_l3282_328235


namespace board_numbers_count_l3282_328277

theorem board_numbers_count (M : ℝ) : ∃! k : ℕ,
  k > 0 ∧
  (∃ S : ℝ, M = S / k) ∧
  (S + 15) / (k + 1) = M + 2 ∧
  (S + 16) / (k + 2) = M + 1 :=
by
  sorry

end board_numbers_count_l3282_328277


namespace binomial_coefficient_equation_l3282_328221

theorem binomial_coefficient_equation (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 8 (x - 2) + Nat.choose 8 (x - 1) + Nat.choose 9 (2 * x - 3)) → 
  (x = 3 ∨ x = 4) := by
  sorry

end binomial_coefficient_equation_l3282_328221


namespace winter_olympics_volunteers_l3282_328225

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  (k - 1).choose (n - 1) * k.factorial

/-- The problem statement -/
theorem winter_olympics_volunteers : distribute 5 4 = 240 := by
  sorry

end winter_olympics_volunteers_l3282_328225


namespace fred_grew_38_cantelopes_l3282_328276

/-- The number of cantelopes Tim grew -/
def tims_cantelopes : ℕ := 44

/-- The total number of cantelopes Fred and Tim grew together -/
def total_cantelopes : ℕ := 82

/-- The number of cantelopes Fred grew -/
def freds_cantelopes : ℕ := total_cantelopes - tims_cantelopes

/-- Theorem stating that Fred grew 38 cantelopes -/
theorem fred_grew_38_cantelopes : freds_cantelopes = 38 := by
  sorry

end fred_grew_38_cantelopes_l3282_328276


namespace problem_grid_square_count_l3282_328226

/-- Represents the grid structure in the problem -/
structure GridStructure where
  width : Nat
  height : Nat
  largeSquares : Nat
  topRowExtraSquares : Nat
  bottomRowExtraSquares : Nat

/-- Counts the number of squares of a given size in the grid -/
def countSquares (g : GridStructure) (size : Nat) : Nat :=
  match size with
  | 1 => g.largeSquares * 6 + g.topRowExtraSquares + g.bottomRowExtraSquares
  | 2 => g.largeSquares * 4 + 4
  | 3 => g.largeSquares * 2 + 1
  | _ => 0

/-- The total number of squares in the grid -/
def totalSquares (g : GridStructure) : Nat :=
  (countSquares g 1) + (countSquares g 2) + (countSquares g 3)

/-- The specific grid structure from the problem -/
def problemGrid : GridStructure := {
  width := 5
  height := 5
  largeSquares := 2
  topRowExtraSquares := 5
  bottomRowExtraSquares := 4
}

theorem problem_grid_square_count :
  totalSquares problemGrid = 38 := by
  sorry

end problem_grid_square_count_l3282_328226


namespace smallest_positive_angle_l3282_328237

def angle_equation (x : ℝ) : Prop :=
  12 * (Real.sin x)^3 * (Real.cos x)^2 - 12 * (Real.sin x)^2 * (Real.cos x)^3 = 3/2

theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < π/2 ∧ angle_equation x ∧
  ∀ (y : ℝ), y > 0 ∧ y < x → ¬(angle_equation y) ∧
  x = 7.5 * π / 180 :=
sorry

end smallest_positive_angle_l3282_328237


namespace equation_holds_iff_sum_ten_l3282_328203

theorem equation_holds_iff_sum_ten (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 := by
sorry

end equation_holds_iff_sum_ten_l3282_328203


namespace coach_rental_equation_l3282_328248

/-- Represents the equation for renting coaches to transport a group of people -/
theorem coach_rental_equation (total_people : ℕ) (school_bus_capacity : ℕ) (coach_capacity : ℕ) (x : ℕ) :
  total_people = 328 →
  school_bus_capacity = 64 →
  coach_capacity = 44 →
  44 * x + 64 = 328 :=
by sorry

end coach_rental_equation_l3282_328248


namespace triangle_arithmetic_angle_sequence_l3282_328298

theorem triangle_arithmetic_angle_sequence (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  2 * B = A + C →    -- Angles form an arithmetic sequence
  (max A (max B C) + min A (min B C) = 120) := by
sorry

end triangle_arithmetic_angle_sequence_l3282_328298


namespace trip_expenses_l3282_328204

def david_initial : ℝ := 1800
def emma_initial : ℝ := 2400
def john_initial : ℝ := 1200

def david_spend_percent : ℝ := 0.60
def emma_spend_percent : ℝ := 0.75
def john_spend_percent : ℝ := 0.50

def david_remaining : ℝ := david_initial * (1 - david_spend_percent)
def emma_spent : ℝ := emma_initial * emma_spend_percent
def emma_remaining : ℝ := emma_spent - 800
def john_remaining : ℝ := john_initial * (1 - john_spend_percent)

theorem trip_expenses :
  david_remaining = 720 ∧
  emma_remaining = 1400 ∧
  john_remaining = 600 ∧
  emma_remaining = emma_spent - 800 :=
by sorry

end trip_expenses_l3282_328204


namespace natalia_novels_l3282_328258

/-- The number of novels Natalia has in her library -/
def number_of_novels : ℕ := sorry

/-- The number of comics in Natalia's library -/
def comics : ℕ := 271

/-- The number of documentaries in Natalia's library -/
def documentaries : ℕ := 419

/-- The number of albums in Natalia's library -/
def albums : ℕ := 209

/-- The capacity of each crate -/
def crate_capacity : ℕ := 9

/-- The total number of crates used -/
def total_crates : ℕ := 116

theorem natalia_novels :
  number_of_novels = 145 ∧
  comics + documentaries + albums + number_of_novels = crate_capacity * total_crates :=
by sorry

end natalia_novels_l3282_328258


namespace area_of_bounded_region_l3282_328281

/-- The equation of the boundary curve -/
def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * |x - y| + 4 * |x + y|

/-- The region bounded by the curve -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | boundary_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area : ℝ := sorry

theorem area_of_bounded_region :
  area = 64 + 32 * Real.pi :=
by sorry

end area_of_bounded_region_l3282_328281


namespace f_max_min_on_interval_l3282_328253

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 * (x - 1)

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-1) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-1) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1) 2, f x = min) ∧
    max = 4 ∧ min = -2 := by
  sorry


end f_max_min_on_interval_l3282_328253


namespace sum_difference_arithmetic_sequences_l3282_328287

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_difference (seq1 seq2 : List ℕ) : ℕ :=
  (seq1.zip seq2).map (λ (a, b) => a - b) |>.sum

theorem sum_difference_arithmetic_sequences : 
  let seq1 := arithmetic_sequence 2101 1 123
  let seq2 := arithmetic_sequence 401 1 123
  sum_difference seq1 seq2 = 209100 := by
sorry

#eval sum_difference (arithmetic_sequence 2101 1 123) (arithmetic_sequence 401 1 123)

end sum_difference_arithmetic_sequences_l3282_328287


namespace farm_ratio_l3282_328230

def cows : ℕ := 21
def horses : ℕ := 6

theorem farm_ratio : (cows / horses : ℚ) = 7 / 2 := by
  sorry

end farm_ratio_l3282_328230


namespace trigonometric_identity_l3282_328241

theorem trigonometric_identity (x y : ℝ) :
  3 * Real.cos (x + y) * Real.sin x + Real.sin (x + y) * Real.cos x =
  3 * Real.cos x * Real.cos y * Real.sin x - 3 * Real.sin x * Real.sin y * Real.sin x +
  Real.sin x * Real.cos y * Real.cos x + Real.cos x * Real.sin y * Real.cos x :=
by sorry

end trigonometric_identity_l3282_328241


namespace unique_division_solution_l3282_328249

theorem unique_division_solution :
  ∀ (dividend divisor quotient : ℕ),
    divisor ≥ 100 ∧ divisor < 1000 →
    quotient ≥ 10000 ∧ quotient < 100000 →
    (quotient / 1000) % 10 = 7 →
    dividend = divisor * quotient →
    (dividend, divisor, quotient) = (12128316, 124, 97809) := by
  sorry

end unique_division_solution_l3282_328249


namespace quirky_triangle_characterization_l3282_328244

/-- A triangle is quirky if there exist integers r₁, r₂, r₃, not all zero, 
    such that r₁θ₁ + r₂θ₂ + r₃θ₃ = 0, where θ₁, θ₂, θ₃ are the measures of the triangle's angles. -/
def IsQuirky (θ₁ θ₂ θ₃ : ℝ) : Prop :=
  ∃ r₁ r₂ r₃ : ℤ, (r₁ ≠ 0 ∨ r₂ ≠ 0 ∨ r₃ ≠ 0) ∧ r₁ * θ₁ + r₂ * θ₂ + r₃ * θ₃ = 0

/-- The angles of a triangle with side lengths n-1, n, n+1 -/
def TriangleAngles (n : ℕ) : (ℝ × ℝ × ℝ) :=
  sorry

theorem quirky_triangle_characterization (n : ℕ) (h : n ≥ 3) :
  let (θ₁, θ₂, θ₃) := TriangleAngles n
  IsQuirky θ₁ θ₂ θ₃ ↔ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7 :=
sorry

end quirky_triangle_characterization_l3282_328244


namespace sector_max_area_l3282_328295

/-- 
Given a sector with circumference c, this theorem states that:
1. The maximum area of the sector is c^2/16
2. The maximum area occurs when the arc length is c/2
-/
theorem sector_max_area (c : ℝ) (h : c > 0) :
  ∃ (max_area arc_length : ℝ),
    max_area = c^2 / 16 ∧
    arc_length = c / 2 ∧
    ∀ (area : ℝ), area ≤ max_area :=
by sorry

end sector_max_area_l3282_328295


namespace smaller_number_problem_l3282_328242

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 45) : 
  min x y = 3 := by
  sorry

end smaller_number_problem_l3282_328242


namespace junior_prom_dancer_ratio_l3282_328256

theorem junior_prom_dancer_ratio :
  let total_kids : ℕ := 140
  let slow_dancers : ℕ := 25
  let non_slow_dancers : ℕ := 10
  let total_dancers : ℕ := slow_dancers + non_slow_dancers
  (total_dancers : ℚ) / total_kids = 1 / 4 := by
  sorry

end junior_prom_dancer_ratio_l3282_328256


namespace recreation_percentage_is_twenty_percent_l3282_328232

/-- Calculates the percentage of earnings allocated for recreation and relaxation -/
def recreation_percentage (earnings_per_customer : ℚ) (fixed_expenses : ℚ) 
  (num_customers : ℕ) (savings : ℚ) : ℚ :=
  let total_earnings := earnings_per_customer * num_customers
  let total_expenses := fixed_expenses + savings
  let recreation_money := total_earnings - total_expenses
  (recreation_money / total_earnings) * 100

/-- Proves that the percentage of earnings allocated for recreation and relaxation is 20% -/
theorem recreation_percentage_is_twenty_percent :
  recreation_percentage 18 280 80 872 = 20 := by
  sorry

end recreation_percentage_is_twenty_percent_l3282_328232


namespace angle_A_is_30_degrees_max_area_is_3_l3282_328205

namespace TriangleProof

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Theorem 1: Angle A is 30° when a = 5/3
theorem angle_A_is_30_degrees (t : Triangle) 
  (h : triangle_conditions t) (ha : t.a = 5/3) : 
  t.A = Real.pi / 6 := by sorry

-- Theorem 2: Maximum area is 3
theorem max_area_is_3 (t : Triangle) 
  (h : triangle_conditions t) : 
  (∃ (max_area : ℝ), ∀ (s : Triangle), 
    triangle_conditions s → 
    (1/2 * s.a * s.c * Real.sin s.B) ≤ max_area ∧ 
    max_area = 3) := by sorry

end TriangleProof

end angle_A_is_30_degrees_max_area_is_3_l3282_328205


namespace min_value_expression_l3282_328210

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z) ≥ (5.5 : ℝ) :=
sorry

end min_value_expression_l3282_328210


namespace scientific_notation_of_86000_l3282_328228

theorem scientific_notation_of_86000 (average_price : ℝ) : 
  average_price = 86000 → average_price = 8.6 * (10 : ℝ)^4 := by
  sorry

end scientific_notation_of_86000_l3282_328228
