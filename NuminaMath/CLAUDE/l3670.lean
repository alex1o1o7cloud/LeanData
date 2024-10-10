import Mathlib

namespace point_coordinates_l3670_367025

def Point := ℝ × ℝ

def x_coordinate (p : Point) : ℝ := p.1

def distance_to_x_axis (p : Point) : ℝ := |p.2|

theorem point_coordinates (P : Point) 
  (h1 : x_coordinate P = -3)
  (h2 : distance_to_x_axis P = 5) :
  P = (-3, 5) ∨ P = (-3, -5) := by
  sorry

end point_coordinates_l3670_367025


namespace prob_ten_then_spade_value_l3670_367018

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing a 10 as the first card -/
def first_card_is_ten (d : Deck) : Prop :=
  ∃ c ∈ d.cards, c < 4

/-- Represents the event of drawing a spade as the second card -/
def second_card_is_spade (d : Deck) : Prop :=
  ∃ c ∈ d.cards, 39 ≤ c ∧ c < 52

/-- The probability of drawing a 10 as the first card -/
def prob_first_ten (d : Deck) : ℚ :=
  4 / 52

/-- The probability of drawing a spade as the second card -/
def prob_second_spade (d : Deck) : ℚ :=
  13 / 51

/-- The probability of drawing a 10 as the first card and a spade as the second card -/
def prob_ten_then_spade (d : Deck) : ℚ :=
  prob_first_ten d * prob_second_spade d

theorem prob_ten_then_spade_value (d : Deck) :
  prob_ten_then_spade d = 12 / 663 :=
sorry

end prob_ten_then_spade_value_l3670_367018


namespace triangle_abc_properties_l3670_367000

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.sqrt 3 * a * Real.sin C - c * (2 + Real.cos A) = 0 →
  a = Real.sqrt 13 →
  Real.sin C = 3 * Real.sin B →
  a ≥ b ∧ a ≥ c →
  (A = 2 * π / 3 ∧ b = 1) :=
by sorry

end triangle_abc_properties_l3670_367000


namespace simplify_expression_l3670_367076

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 6) - (x + 4)*(3*x - 2) = 4*x - 16 := by
  sorry

end simplify_expression_l3670_367076


namespace disease_probabilities_l3670_367075

/-- Represents a disease with its incidence rate and probability of showing symptom S -/
structure Disease where
  incidenceRate : ℝ
  probSymptomS : ℝ

/-- Given three diseases and their properties, proves statements about probabilities -/
theorem disease_probabilities (d₁ d₂ d₃ : Disease)
  (h₁ : d₁.incidenceRate = 0.02 ∧ d₁.probSymptomS = 0.4)
  (h₂ : d₂.incidenceRate = 0.05 ∧ d₂.probSymptomS = 0.18)
  (h₃ : d₃.incidenceRate = 0.005 ∧ d₃.probSymptomS = 0.6)
  (h_no_other : ∀ d, d ≠ d₁ ∧ d ≠ d₂ ∧ d ≠ d₃ → d.probSymptomS = 0) :
  let p_s := d₁.incidenceRate * d₁.probSymptomS +
             d₂.incidenceRate * d₂.probSymptomS +
             d₃.incidenceRate * d₃.probSymptomS
  p_s = 0.02 ∧
  (d₁.incidenceRate * d₁.probSymptomS) / p_s = 0.4 ∧
  (d₂.incidenceRate * d₂.probSymptomS) / p_s = 0.45 :=
by sorry


end disease_probabilities_l3670_367075


namespace contrapositive_square_sum_zero_l3670_367040

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ ((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0) := by sorry

end contrapositive_square_sum_zero_l3670_367040


namespace function_properties_l3670_367044

-- Define the function f(x) = ax^3 + bx^2
def f (x : ℝ) : ℝ := -6 * x^3 + 9 * x^2

-- State the theorem
theorem function_properties :
  (f 1 = 3) ∧ 
  (deriv f 1 = 0) ∧ 
  (∀ x : ℝ, f x ≥ 0) ∧
  (∃ x : ℝ, f x = 0) := by
  sorry

end function_properties_l3670_367044


namespace salt_percentage_in_water_l3670_367087

def salt_mass : ℝ := 10
def water_mass : ℝ := 40

theorem salt_percentage_in_water :
  (salt_mass / water_mass) * 100 = 25 := by
  sorry

end salt_percentage_in_water_l3670_367087


namespace cube_volume_l3670_367014

/-- The volume of a cube with total edge length 48 cm is 64 cm³. -/
theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 48) : 
  (edge_sum / 12)^3 = 64 := by
  sorry

end cube_volume_l3670_367014


namespace faye_country_albums_l3670_367047

/-- The number of country albums Faye bought -/
def country_albums : ℕ := sorry

/-- The number of pop albums Faye bought -/
def pop_albums : ℕ := 3

/-- The number of songs per album -/
def songs_per_album : ℕ := 6

/-- The total number of songs Faye bought -/
def total_songs : ℕ := 30

theorem faye_country_albums : 
  country_albums = 2 := by sorry

end faye_country_albums_l3670_367047


namespace total_flour_in_bowl_l3670_367078

-- Define the initial amount of flour in the bowl
def initial_flour : ℚ := 2 + 3/4

-- Define the amount of flour added
def added_flour : ℚ := 45/100

-- Theorem to prove
theorem total_flour_in_bowl :
  initial_flour + added_flour = 16/5 := by
  sorry

end total_flour_in_bowl_l3670_367078


namespace order_of_abc_l3670_367017

theorem order_of_abc (a b c : ℝ) 
  (ha : a = 0.1 * Real.exp 0.1)
  (hb : b = 1/9)
  (hc : c = -Real.log 0.9) : 
  c < a ∧ a < b := by
sorry

end order_of_abc_l3670_367017


namespace intersection_of_M_and_N_l3670_367084

def M : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2)}
def N : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (1, -2) + x • (2, 3)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-13, -23)} := by
  sorry

end intersection_of_M_and_N_l3670_367084


namespace no_distinct_positive_roots_l3670_367001

theorem no_distinct_positive_roots :
  ∀ (b c : ℤ), 0 ≤ b ∧ b ≤ 5 ∧ -10 ≤ c ∧ c ≤ 10 →
  ¬∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 + b * x₁ + c = 0 ∧ x₂^2 + b * x₂ + c = 0 :=
by sorry

end no_distinct_positive_roots_l3670_367001


namespace chord_length_l3670_367042

theorem chord_length (r : ℝ) (h : r = 10) : 
  let chord_length := 2 * (r^2 - (r/2)^2).sqrt
  chord_length = 10 * Real.sqrt 3 := by
  sorry

end chord_length_l3670_367042


namespace paths_A_to_C_via_B_l3670_367026

/-- The number of paths from A to B -/
def paths_A_to_B : ℕ := Nat.choose 6 2

/-- The number of paths from B to C -/
def paths_B_to_C : ℕ := Nat.choose 6 3

/-- The total number of steps from A to C -/
def total_steps : ℕ := 12

/-- The number of steps from A to B -/
def steps_A_to_B : ℕ := 6

/-- The number of steps from B to C -/
def steps_B_to_C : ℕ := 6

theorem paths_A_to_C_via_B : 
  paths_A_to_B * paths_B_to_C = 300 ∧ 
  steps_A_to_B + steps_B_to_C = total_steps :=
by sorry

end paths_A_to_C_via_B_l3670_367026


namespace age_difference_l3670_367007

theorem age_difference (older_age younger_age : ℕ) : 
  older_age + younger_age = 74 → older_age = 38 → older_age - younger_age = 2 :=
by
  sorry

end age_difference_l3670_367007


namespace chocolates_difference_l3670_367002

theorem chocolates_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 7)
  (h2 : nickel_chocolates = 3) :
  robert_chocolates - nickel_chocolates = 4 := by
sorry

end chocolates_difference_l3670_367002


namespace cubic_sum_geq_product_sum_l3670_367033

theorem cubic_sum_geq_product_sum {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c ≥ 1) :
  a^3 + b^3 + c^3 ≥ a*b + b*c + c*a ∧ 
  (a^3 + b^3 + c^3 = a*b + b*c + c*a ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end cubic_sum_geq_product_sum_l3670_367033


namespace min_value_problem_l3670_367030

-- Define the function f
def f (x a b c : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f x a b c ≥ 4) 
  (hex : ∃ x, f x a b c = 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 4 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end min_value_problem_l3670_367030


namespace infinitely_many_cube_sums_l3670_367066

theorem infinitely_many_cube_sums (n : ℕ) : 
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ 
  ∀ (k : ℕ), ∃ (m : ℕ+), (n^6 + 3 * (f k)) = m^3 := by
  sorry

end infinitely_many_cube_sums_l3670_367066


namespace magnitude_of_complex_power_l3670_367099

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) :
  z = 3/5 + 4/5 * I → n = 6 → Complex.abs (z^n) = 1 := by sorry

end magnitude_of_complex_power_l3670_367099


namespace erica_safari_animals_l3670_367090

/-- The number of animals Erica saw on Saturday -/
def saturday_animals : ℕ := 3 + 2

/-- The number of animals Erica saw on Sunday -/
def sunday_animals : ℕ := 2 + 5

/-- The number of animals Erica saw on Monday -/
def monday_animals : ℕ := 5 + 3

/-- The total number of animals Erica saw during her safari -/
def total_animals : ℕ := saturday_animals + sunday_animals + monday_animals

theorem erica_safari_animals : total_animals = 20 := by
  sorry

end erica_safari_animals_l3670_367090


namespace lawn_mowing_difference_l3670_367079

theorem lawn_mowing_difference (spring_mowings summer_mowings : ℕ) 
  (h1 : spring_mowings = 8) 
  (h2 : summer_mowings = 5) : 
  spring_mowings - summer_mowings = 3 := by
  sorry

end lawn_mowing_difference_l3670_367079


namespace aluminum_cans_collection_l3670_367091

theorem aluminum_cans_collection : 
  let sarah_yesterday : ℕ := 50
  let lara_yesterday : ℕ := sarah_yesterday + 30
  let sarah_today : ℕ := 40
  let lara_today : ℕ := 70
  let total_yesterday : ℕ := sarah_yesterday + lara_yesterday
  let total_today : ℕ := sarah_today + lara_today
  total_yesterday - total_today = 20 := by
sorry

end aluminum_cans_collection_l3670_367091


namespace rain_probability_l3670_367043

theorem rain_probability (p : ℚ) (n : ℕ) (hp : p = 4/5) (hn : n = 5) :
  1 - (1 - p)^n = 3124/3125 := by
  sorry

end rain_probability_l3670_367043


namespace range_of_abc_squared_l3670_367003

theorem range_of_abc_squared (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end range_of_abc_squared_l3670_367003


namespace quartic_at_one_equals_three_l3670_367054

/-- Horner's method for evaluating a quartic polynomial at x = 1 -/
def horner_quartic (a₄ a₃ a₂ a₁ a₀ : ℤ) : ℤ :=
  ((((1 * a₄ + a₃) * 1 + a₂) * 1 + a₁) * 1 + a₀)

/-- The given quartic polynomial evaluated at x = 1 equals 3 -/
theorem quartic_at_one_equals_three :
  horner_quartic 1 (-7) (-9) 11 7 = 3 := by
  sorry

end quartic_at_one_equals_three_l3670_367054


namespace circular_track_length_l3670_367096

/-- The length of the circular track in meters -/
def track_length : ℝ := 1250

/-- The initial speed of the Amur tiger in km/h -/
def amur_speed : ℝ := sorry

/-- The speed of the Bengal tiger in km/h -/
def bengal_speed : ℝ := sorry

/-- The number of additional laps run by the Amur tiger in the first 2 hours -/
def additional_laps_2h : ℝ := 6

/-- The speed increase of the Amur tiger after 2 hours in km/h -/
def speed_increase : ℝ := 10

/-- The total number of additional laps run by the Amur tiger in 3 hours -/
def total_additional_laps : ℝ := 17

theorem circular_track_length : 
  amur_speed - bengal_speed = 3 ∧
  (amur_speed - bengal_speed) * 2 = additional_laps_2h ∧
  (amur_speed + speed_increase - bengal_speed) * 1 + additional_laps_2h = total_additional_laps →
  track_length = 1250 := by sorry

end circular_track_length_l3670_367096


namespace negation_of_difference_l3670_367032

theorem negation_of_difference (a b : ℝ) : -(a - b) = -a + b := by sorry

end negation_of_difference_l3670_367032


namespace lucy_grocery_problem_l3670_367021

/-- Lucy's grocery shopping problem -/
theorem lucy_grocery_problem (total_packs cookies_packs noodles_packs : ℕ) :
  total_packs = 28 →
  cookies_packs = 12 →
  total_packs = cookies_packs + noodles_packs →
  noodles_packs = 16 := by
  sorry

end lucy_grocery_problem_l3670_367021


namespace marble_division_l3670_367048

theorem marble_division (x : ℚ) : 
  (4 * x + 2) + (2 * x - 1) + (3 * x + 3) = 100 → 
  2 * x - 1 = 61 / 3 := by
  sorry

end marble_division_l3670_367048


namespace quadratic_function_properties_l3670_367098

/-- Represents a quadratic function of the form y = x^2 + bx - c -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ

/-- Represents the roots of a quadratic function -/
structure Roots where
  m : ℝ
  h : m ≠ 0

theorem quadratic_function_properties (f : QuadraticFunction) (r : Roots) :
  (∀ x, f.b * x + x^2 - f.c = 0 ↔ x = r.m ∨ x = -2 * r.m) →
  f.c = 2 * f.b^2 ∧
  (f.b / 2 = -1 → f.b = 2 ∧ f.c = 8) := by
  sorry

#check quadratic_function_properties

end quadratic_function_properties_l3670_367098


namespace find_a_l3670_367020

def f (x : ℝ) := 3 * (x - 1) + 2

theorem find_a : ∃ a : ℝ, f a = 5 ∧ a = 2 := by sorry

end find_a_l3670_367020


namespace sheep_ratio_l3670_367080

theorem sheep_ratio (total : ℕ) (beth_sheep : ℕ) (h1 : total = 608) (h2 : beth_sheep = 76) :
  (total - beth_sheep) / beth_sheep = 133 / 19 := by
  sorry

end sheep_ratio_l3670_367080


namespace expression_evaluation_l3670_367094

theorem expression_evaluation : 5 + 7 * (2 + 1/4) = 20.75 := by
  sorry

end expression_evaluation_l3670_367094


namespace cone_volume_l3670_367050

/-- Given a cone with slant height 2 and base area π, its volume is (√3 * π) / 3 -/
theorem cone_volume (s : ℝ) (a : ℝ) (v : ℝ) 
  (h_slant : s = 2) 
  (h_area : a = π) 
  (h_volume : v = (Real.sqrt 3 * π) / 3) : 
  v = (Real.sqrt 3 * π) / 3 := by
sorry

end cone_volume_l3670_367050


namespace square_perimeter_l3670_367023

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (5 / 2 * s = 40) → (4 * s = 64) := by
  sorry

end square_perimeter_l3670_367023


namespace bees_after_five_days_l3670_367009

/-- The number of bees in the hive after n days -/
def bees_after_days (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * bees_after_days (n - 1)

/-- The theorem stating that after 5 days, there will be 1024 bees in the hive -/
theorem bees_after_five_days : bees_after_days 5 = 1024 := by
  sorry

end bees_after_five_days_l3670_367009


namespace point_not_on_line_l3670_367061

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def are_collinear (A B C : Point3D) : Prop :=
  ∃ k : ℝ, (C.x - A.x, C.y - A.y, C.z - A.z) = k • (B.x - A.x, B.y - A.y, B.z - A.z)

theorem point_not_on_line : 
  let A : Point3D := ⟨-1, 1, 2⟩
  let B : Point3D := ⟨3, 6, -1⟩
  let C : Point3D := ⟨7, 9, 0⟩
  ¬(are_collinear A B C) := by
  sorry

end point_not_on_line_l3670_367061


namespace point_on_line_l3670_367031

/-- Given a line passing through points (0, 2) and (-10, 0),
    prove that the point (25, 7) lies on this line. -/
theorem point_on_line : ∀ (x y : ℝ),
  (x = 25 ∧ y = 7) →
  (y - 2) * 10 = (x - 0) * 2 := by
  sorry

end point_on_line_l3670_367031


namespace solve_equation_l3670_367015

theorem solve_equation : 
  let y : ℚ := 45 / (8 - 3/7)
  y = 315 / 53 := by sorry

end solve_equation_l3670_367015


namespace fred_limes_picked_l3670_367062

theorem fred_limes_picked (total_limes : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) 
  (h1 : total_limes = 103)
  (h2 : alyssa_limes = 32)
  (h3 : nancy_limes = 35) :
  total_limes - (alyssa_limes + nancy_limes) = 36 := by
sorry

end fred_limes_picked_l3670_367062


namespace x0_value_l3670_367086

noncomputable def f (x : ℝ) : ℝ := x * (2014 + Real.log x)

theorem x0_value (x₀ : ℝ) (h : (deriv f) x₀ = 2015) : x₀ = 1 := by
  sorry

end x0_value_l3670_367086


namespace arccos_lt_arcsin_iff_l3670_367011

theorem arccos_lt_arcsin_iff (x : ℝ) : Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 := by
  sorry

end arccos_lt_arcsin_iff_l3670_367011


namespace angle_sum_proof_l3670_367046

theorem angle_sum_proof (x y : Real) (h1 : 0 < x ∧ x < π/2) (h2 : 0 < y ∧ y < π/2)
  (h3 : 4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 1)
  (h4 : 4 * Real.sin (2*x) + 3 * Real.sin (2*y) = 0) :
  x + 2*y = π/6*5 := by
sorry

end angle_sum_proof_l3670_367046


namespace equilateral_triangle_area_l3670_367069

theorem equilateral_triangle_area (perimeter : ℝ) (area : ℝ) : 
  perimeter = 30 → area = 25 * Real.sqrt 3 → 
  ∃ (side : ℝ), side > 0 ∧ 3 * side = perimeter ∧ area = (Real.sqrt 3 / 4) * side^2 :=
by sorry

end equilateral_triangle_area_l3670_367069


namespace fraction_numerator_l3670_367097

theorem fraction_numerator (y : ℝ) (n : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / 10 + (n / y) * y = 0.5 * y) : n = 3 := by
  sorry

end fraction_numerator_l3670_367097


namespace count_divisible_numbers_l3670_367013

theorem count_divisible_numbers : 
  let upper_bound := 242400
  let divisor := 303
  (Finset.filter 
    (fun k => (k^2 + 2*k) % divisor = 0) 
    (Finset.range (upper_bound + 1))).card = 3200 :=
by sorry

end count_divisible_numbers_l3670_367013


namespace smallest_a_value_l3670_367082

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (45 * x)) :
  a ≥ 45 ∧ ∃ a₀ : ℝ, a₀ ≥ 0 ∧ (∀ x : ℤ, Real.sin (a₀ * x + b) = Real.sin (45 * x)) ∧ a₀ = 45 :=
sorry

end smallest_a_value_l3670_367082


namespace inequality_system_solution_l3670_367028

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - 1 ≥ a ∧ 2*x - b < 3) ↔ (3 ≤ x ∧ x < 5)) → 
  a + b = 9 := by
sorry

end inequality_system_solution_l3670_367028


namespace vector_perpendicular_implies_x_value_l3670_367036

/-- Given vectors a and b in R^2, if a is perpendicular to 2a + b, then the x-coordinate of b is 10 -/
theorem vector_perpendicular_implies_x_value (a b : ℝ × ℝ) :
  a = (-2, 3) →
  b.2 = -2 →
  (a.1 * (2 * a.1 + b.1) + a.2 * (2 * a.2 + b.2) = 0) →
  b.1 = 10 := by
  sorry

end vector_perpendicular_implies_x_value_l3670_367036


namespace cube_diff_multiple_implies_sum_squares_multiple_of_sum_l3670_367005

theorem cube_diff_multiple_implies_sum_squares_multiple_of_sum (a b c : ℕ) : 
  a < 2017 → b < 2017 → c < 2017 →
  a ≠ b → b ≠ c → a ≠ c →
  (∃ k₁ k₂ k₃ : ℤ, a^3 - b^3 = k₁ * 2017 ∧ b^3 - c^3 = k₂ * 2017 ∧ c^3 - a^3 = k₃ * 2017) →
  ∃ m : ℤ, a^2 + b^2 + c^2 = m * (a + b + c) :=
by sorry

end cube_diff_multiple_implies_sum_squares_multiple_of_sum_l3670_367005


namespace prob_higher_2012_l3670_367064

-- Define the probability of guessing correctly
def p : ℝ := 0.25

-- Define the complementary probability
def q : ℝ := 1 - p

-- Define the binomial probability function
def binomProb (n : ℕ) (k : ℕ) : ℝ :=
  (n.choose k) * (p ^ k) * (q ^ (n - k))

-- Define the probability of passing in 2011
def prob2011 : ℝ :=
  1 - (binomProb 20 0 + binomProb 20 1 + binomProb 20 2)

-- Define the probability of passing in 2012
def prob2012 : ℝ :=
  1 - (binomProb 40 0 + binomProb 40 1 + binomProb 40 2 + binomProb 40 3 + binomProb 40 4 + binomProb 40 5)

-- Theorem statement
theorem prob_higher_2012 : prob2012 > prob2011 := by
  sorry

end prob_higher_2012_l3670_367064


namespace remainder_theorem_l3670_367073

theorem remainder_theorem (x y z a b c d e : ℕ) : 
  0 < a ∧ a < 211 ∧ 
  0 < b ∧ b < 211 ∧ 
  0 < c ∧ c < 211 ∧ 
  0 < d ∧ d < 251 ∧ 
  0 < e ∧ e < 251 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  x % 211 = a ∧ 
  y % 211 = b ∧ 
  z % 211 = c ∧
  x % 251 = c ∧
  y % 251 = d ∧
  z % 251 = e →
  ∃! R, 0 ≤ R ∧ R < 211 * 251 ∧ (2 * x - y + 3 * z + 47) % (211 * 251) = R :=
by sorry

end remainder_theorem_l3670_367073


namespace asteroid_fragments_proof_l3670_367016

theorem asteroid_fragments_proof :
  ∃ (X n : ℕ), 
    X > 0 ∧ 
    n > 0 ∧ 
    X / 5 + 26 + n * (X / 7) = X ∧ 
    X = 70 := by
  sorry

end asteroid_fragments_proof_l3670_367016


namespace grade_percentage_calculation_l3670_367051

theorem grade_percentage_calculation (total_students : ℕ) 
  (a_both : ℕ) (b_both : ℕ) (c_both : ℕ) (d_c : ℕ) :
  total_students = 40 →
  a_both = 4 →
  b_both = 6 →
  c_both = 3 →
  d_c = 2 →
  (((a_both + b_both + c_both + d_c : ℚ) / total_students) * 100 : ℚ) = 37.5 := by
sorry

end grade_percentage_calculation_l3670_367051


namespace max_sin_cos_product_l3670_367052

theorem max_sin_cos_product (x : Real) : 
  (∃ (y : Real), y = Real.sin x * Real.cos x ∧ ∀ (z : Real), z = Real.sin x * Real.cos x → z ≤ y) → 
  (∃ (max : Real), max = (1 : Real) / 2 ∧ ∀ (y : Real), y = Real.sin x * Real.cos x → y ≤ max) :=
sorry

end max_sin_cos_product_l3670_367052


namespace cookies_leftover_l3670_367006

theorem cookies_leftover (naomi oliver penelope : ℕ) 
  (h_naomi : naomi = 53)
  (h_oliver : oliver = 67)
  (h_penelope : penelope = 29)
  (package_size : ℕ) 
  (h_package : package_size = 15) : 
  (naomi + oliver + penelope) % package_size = 14 := by
  sorry

end cookies_leftover_l3670_367006


namespace two_students_same_school_probability_l3670_367083

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of schools --/
def num_schools : ℕ := 4

/-- The total number of possible outcomes --/
def total_outcomes : ℕ := num_schools ^ num_students

/-- The number of outcomes where exactly two students choose the same school --/
def favorable_outcomes : ℕ := num_students.choose 2 * num_schools * (num_schools - 1)

/-- The probability of exactly two students choosing the same school --/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem two_students_same_school_probability :
  probability = 9 / 16 := by sorry

end two_students_same_school_probability_l3670_367083


namespace polynomial_value_relation_l3670_367045

theorem polynomial_value_relation (y : ℝ) : 
  4 * y^2 - 2 * y + 5 = 7 → 2 * y^2 - y + 1 = 2 := by
  sorry

end polynomial_value_relation_l3670_367045


namespace proportion_equality_l3670_367058

theorem proportion_equality (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 := by
  sorry

end proportion_equality_l3670_367058


namespace parabola_locus_l3670_367089

-- Define the parabola and its properties
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the locus L
def locus (p : ℕ) (x y : ℝ) : Prop :=
  is_prime p ∧ p ≠ 2 ∧ y ≠ 0 ∧ 4 * y^2 = p * (x - p)

-- Theorem statement
theorem parabola_locus (p : ℕ) :
  is_prime p →
  p ≠ 2 →
  (∃ (x y : ℤ), locus p (x : ℝ) (y : ℝ)) ∧
  (∀ (x y : ℤ), locus p (x : ℝ) (y : ℝ) → ¬ ∃ (m : ℤ), (x : ℝ)^2 + (y : ℝ)^2 = (m : ℝ)^2) ∧
  (∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ locus p (x : ℝ) (y : ℝ)) :=
sorry

end parabola_locus_l3670_367089


namespace joe_fruit_probability_l3670_367071

def num_meals : ℕ := 4
def num_fruit_types : ℕ := 3

def prob_same_fruit_all_meals : ℚ := (1 / num_fruit_types) ^ num_meals

theorem joe_fruit_probability :
  1 - (num_fruit_types * prob_same_fruit_all_meals) = 26 / 27 := by
  sorry

end joe_fruit_probability_l3670_367071


namespace circle_properties_correct_l3670_367068

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a circle equation, compute its center and radius -/
def computeCircleProperties (eq : CircleEquation) : CircleProperties :=
  sorry

theorem circle_properties_correct (eq : CircleEquation) 
  (h : eq = CircleEquation.mk 4 (-8) 4 (-16) 20) : 
  computeCircleProperties eq = CircleProperties.mk (1, 2) 0 := by
  sorry

end circle_properties_correct_l3670_367068


namespace T_properties_l3670_367034

theorem T_properties (n : ℕ) : 
  let T := (10 * (10^n - 1)) / 81 - n / 9
  ∃ (k : ℕ), T = k ∧ T % 11 = ((n + 1) / 2) % 11 := by
  sorry

end T_properties_l3670_367034


namespace tom_balloons_remaining_l3670_367056

/-- Given that Tom has 30 violet balloons initially and gives away 16 balloons,
    prove that he has 14 violet balloons remaining. -/
theorem tom_balloons_remaining (initial : ℕ) (given_away : ℕ) (h1 : initial = 30) (h2 : given_away = 16) :
  initial - given_away = 14 := by
  sorry

end tom_balloons_remaining_l3670_367056


namespace jake_has_eight_peaches_l3670_367077

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference in peaches between Steven and Jake -/
def difference : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - difference

theorem jake_has_eight_peaches : jake_peaches = 8 := by
  sorry

end jake_has_eight_peaches_l3670_367077


namespace tank_full_time_l3670_367029

/-- Represents a water tank with pipes for filling and draining. -/
structure WaterTank where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the time required to fill the tank given the pipe rates and capacity. -/
def time_to_fill (tank : WaterTank) : ℕ :=
  let net_fill_per_cycle := tank.pipeA_rate + tank.pipeB_rate - tank.pipeC_rate
  let cycles := tank.capacity / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the given tank will be full after 48 minutes. -/
theorem tank_full_time (tank : WaterTank) 
  (h1 : tank.capacity = 800)
  (h2 : tank.pipeA_rate = 40)
  (h3 : tank.pipeB_rate = 30)
  (h4 : tank.pipeC_rate = 20) :
  time_to_fill tank = 48 := by
  sorry

#eval time_to_fill { capacity := 800, pipeA_rate := 40, pipeB_rate := 30, pipeC_rate := 20 }

end tank_full_time_l3670_367029


namespace greatest_b_for_non_range_l3670_367035

theorem greatest_b_for_non_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 15 ≠ -9) ↔ b ≤ 9 :=
sorry

end greatest_b_for_non_range_l3670_367035


namespace smallest_winning_number_l3670_367041

def ian_action (x : ℕ) : ℕ := 3 * x

def marcella_action (x : ℕ) : ℕ := x + 150

def game_sequence (m : ℕ) : ℕ := 
  ian_action (marcella_action (ian_action (marcella_action (ian_action m))))

theorem smallest_winning_number : 
  ∀ m : ℕ, 0 ≤ m ∧ m ≤ 1999 →
    (m < 112 → 
      game_sequence m ≤ 5000 ∧ 
      marcella_action (game_sequence m) ≤ 5000 ∧ 
      ian_action (marcella_action (game_sequence m)) > 5000) →
    (game_sequence 112 ≤ 5000 ∧ 
     marcella_action (game_sequence 112) ≤ 5000 ∧ 
     ian_action (marcella_action (game_sequence 112)) > 5000) :=
by sorry

#check smallest_winning_number

end smallest_winning_number_l3670_367041


namespace simplify_radical_product_l3670_367057

theorem simplify_radical_product (y : ℝ) (h : y > 0) :
  Real.sqrt (50 * y) * Real.sqrt (18 * y) * Real.sqrt (32 * y) = 30 * y * Real.sqrt (2 * y) := by
  sorry

end simplify_radical_product_l3670_367057


namespace geometric_sequence_ratio_l3670_367074

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 2 - a 3 / 2 = a 3 / 2 - a 1) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
sorry

end geometric_sequence_ratio_l3670_367074


namespace lemons_for_lemonade_l3670_367008

/-- Given that 30 lemons make 40 gallons of lemonade, 
    prove that 7.5 lemons are needed for 10 gallons. -/
theorem lemons_for_lemonade :
  let lemons_for_40 : ℚ := 30
  let gallons_40 : ℚ := 40
  let target_gallons : ℚ := 10
  (lemons_for_40 / gallons_40) * target_gallons = 7.5 := by sorry

end lemons_for_lemonade_l3670_367008


namespace current_rate_l3670_367081

/-- Proves that given a man who can row 3.3 km/hr in still water, and it takes him twice as long
    to row upstream as to row downstream, the rate of the current is 1.1 km/hr. -/
theorem current_rate (still_water_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) :
  still_water_speed = 3.3 ∧ upstream_time = 2 * downstream_time →
  ∃ current_rate : ℝ,
    current_rate = 1.1 ∧
    (still_water_speed + current_rate) * downstream_time =
    (still_water_speed - current_rate) * upstream_time :=
by sorry

end current_rate_l3670_367081


namespace degree_to_radian_conversion_l3670_367039

theorem degree_to_radian_conversion (θ_deg : ℝ) (θ_rad : ℝ) :
  θ_deg = 150 ∧ θ_rad = θ_deg * (π / 180) → θ_rad = 5 * π / 6 := by
  sorry

end degree_to_radian_conversion_l3670_367039


namespace initial_mixture_volume_l3670_367049

/-- Proves that the initial volume of a mixture is 45 litres, given the initial and final ratios of milk to water --/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 4 / 1)
  (h2 : initial_milk / (initial_water + 18) = 4 / 3) :
  initial_milk + initial_water = 45 :=
by sorry

end initial_mixture_volume_l3670_367049


namespace sqrt_sum_equality_l3670_367067

theorem sqrt_sum_equality : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end sqrt_sum_equality_l3670_367067


namespace sticker_distribution_l3670_367004

theorem sticker_distribution (d : ℕ) (h : d > 0) :
  let total_stickers : ℕ := 72
  let friends : ℕ := d
  let stickers_per_friend : ℚ := total_stickers / friends
  stickers_per_friend = 72 / d :=
by sorry

end sticker_distribution_l3670_367004


namespace brown_mushrooms_count_l3670_367095

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := sorry

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The total number of white-spotted mushrooms -/
def total_white_spotted : ℕ := 17

theorem brown_mushrooms_count :
  brown_mushrooms = 6 :=
by
  have h1 : (blue_mushrooms / 2 : ℕ) + (2 * red_mushrooms / 3 : ℕ) + brown_mushrooms = total_white_spotted :=
    sorry
  sorry

end brown_mushrooms_count_l3670_367095


namespace nonagon_diagonals_l3670_367085

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

/-- Theorem: The number of diagonals in a nonagon is 27 -/
theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end nonagon_diagonals_l3670_367085


namespace ivanov_net_worth_is_2300000_l3670_367059

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℤ :=
  let apartment_value : ℤ := 3000000
  let car_value : ℤ := 900000
  let bank_deposit : ℤ := 300000
  let securities_value : ℤ := 200000
  let liquid_cash : ℤ := 100000
  let mortgage_balance : ℤ := 1500000
  let car_loan_balance : ℤ := 500000
  let relatives_debt : ℤ := 200000
  let total_assets : ℤ := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities : ℤ := mortgage_balance + car_loan_balance + relatives_debt
  total_assets - total_liabilities

theorem ivanov_net_worth_is_2300000 : ivanov_net_worth = 2300000 := by
  sorry

end ivanov_net_worth_is_2300000_l3670_367059


namespace floor_plus_square_equals_72_l3670_367072

theorem floor_plus_square_equals_72 : 
  ∃! (x : ℝ), x > 0 ∧ ⌊x⌋ + x^2 = 72 :=
by sorry

end floor_plus_square_equals_72_l3670_367072


namespace value_range_cos_x_tan_x_l3670_367053

-- Define the function f(x) = cos x tan x
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.tan x

-- Theorem statement
theorem value_range_cos_x_tan_x :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ Real.pi / 2 + Real.pi * ↑k ∧ f x = y) ↔ -1 < y ∧ y < 1 :=
by sorry

end value_range_cos_x_tan_x_l3670_367053


namespace quadratic_factorization_l3670_367038

theorem quadratic_factorization (x : ℝ) : x^2 - 3*x - 4 = (x + 1)*(x - 4) := by
  sorry

end quadratic_factorization_l3670_367038


namespace child_ticket_price_is_correct_l3670_367063

/-- Calculates the price of a child's ticket given the group composition, adult ticket price, senior discount, and total bill. -/
def childTicketPrice (totalPeople adultCount seniorCount childCount : ℕ) 
                     (adultPrice : ℚ) (seniorDiscount : ℚ) (totalBill : ℚ) : ℚ :=
  let seniorPrice := adultPrice * (1 - seniorDiscount)
  let adultTotal := adultPrice * adultCount
  let seniorTotal := seniorPrice * seniorCount
  let childTotal := totalBill - adultTotal - seniorTotal
  childTotal / childCount

/-- Theorem stating that the child ticket price is $5.63 given the problem conditions. -/
theorem child_ticket_price_is_correct :
  childTicketPrice 50 25 15 10 15 0.25 600 = 5.63 := by
  sorry

end child_ticket_price_is_correct_l3670_367063


namespace combined_average_marks_l3670_367060

theorem combined_average_marks (class1_students class2_students class3_students : ℕ)
  (class1_avg class2_avg class3_avg : ℚ)
  (h1 : class1_students = 35)
  (h2 : class2_students = 45)
  (h3 : class3_students = 25)
  (h4 : class1_avg = 40)
  (h5 : class2_avg = 60)
  (h6 : class3_avg = 75) :
  (class1_students * class1_avg + class2_students * class2_avg + class3_students * class3_avg) /
  (class1_students + class2_students + class3_students) = 5975 / 105 :=
by sorry

end combined_average_marks_l3670_367060


namespace cube_root_and_square_root_problem_l3670_367027

theorem cube_root_and_square_root_problem :
  ∀ (a b : ℝ),
  (5 * a + 2) ^ (1/3 : ℝ) = 3 →
  (3 * a + b - 1) ^ (1/2 : ℝ) = 4 →
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3) ^ (1/2 : ℝ) = 4 :=
by sorry

end cube_root_and_square_root_problem_l3670_367027


namespace correct_operation_l3670_367012

theorem correct_operation (a b : ℝ) : 5 * a * b - 6 * a * b = -a * b := by
  sorry

end correct_operation_l3670_367012


namespace sqrt_inequality_solution_set_l3670_367010

theorem sqrt_inequality_solution_set (x : ℝ) : 
  (Real.sqrt (x + 3) < 2) ↔ (x ∈ Set.Icc (-3) 1) :=
sorry

end sqrt_inequality_solution_set_l3670_367010


namespace complex_exponential_sum_l3670_367092

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (1/2 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (1/2 : ℂ) * Complex.I := by
sorry

end complex_exponential_sum_l3670_367092


namespace total_students_at_concert_l3670_367065

/-- The number of buses going to the concert -/
def num_buses : ℕ := 18

/-- The number of students each bus took -/
def students_per_bus : ℕ := 65

/-- Theorem stating the total number of students who went to the concert -/
theorem total_students_at_concert : num_buses * students_per_bus = 1170 := by
  sorry

end total_students_at_concert_l3670_367065


namespace sum_of_rectangle_areas_l3670_367022

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℕ) : ℕ := width * height

/-- The sum of the areas of four rectangles -/
def totalArea (w1 h1 w2 h2 w3 h3 w4 h4 : ℕ) : ℕ :=
  rectangleArea w1 h1 + rectangleArea w2 h2 + rectangleArea w3 h3 + rectangleArea w4 h4

/-- Theorem stating that the sum of the areas of four specific rectangles is 56 -/
theorem sum_of_rectangle_areas :
  totalArea 7 6 3 2 3 1 5 1 = 56 := by
  sorry

#eval totalArea 7 6 3 2 3 1 5 1

end sum_of_rectangle_areas_l3670_367022


namespace investment_interest_rate_l3670_367024

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (second_part : ℝ) 
  (second_rate : ℝ) 
  (total_interest : ℝ) :
  total_investment = 3000 →
  first_part = 300 →
  second_part = total_investment - first_part →
  second_rate = 5 →
  total_interest = 144 →
  total_interest = (first_part * (3 : ℝ) / 100) + (second_part * second_rate / 100) :=
by
  sorry

end investment_interest_rate_l3670_367024


namespace unique_solution_condition_l3670_367037

theorem unique_solution_condition (h : ℝ) (h_neq_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 := by
  sorry

end unique_solution_condition_l3670_367037


namespace camerons_list_count_camerons_list_count_is_871_l3670_367070

theorem camerons_list_count : ℕ → Prop :=
  fun count =>
    let smallest_square := 900
    let smallest_cube := 27000
    (∀ k : ℕ, k < smallest_square → ¬∃ m : ℕ, k = 30 * m * m) ∧
    (∀ k : ℕ, k < smallest_cube → ¬∃ m : ℕ, k = 30 * m * m * m) ∧
    count = (smallest_cube / 30 - smallest_square / 30 + 1)

theorem camerons_list_count_is_871 : camerons_list_count 871 := by
  sorry

end camerons_list_count_camerons_list_count_is_871_l3670_367070


namespace probability_in_painted_cube_l3670_367055

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (three_face_cubes : ℕ)
  (no_face_cubes : ℕ)

/-- The probability of selecting one cube with three painted faces and one with no painted faces -/
def probability_three_and_none (cube : PaintedCube) : ℚ :=
  (cube.three_face_cubes * cube.no_face_cubes : ℚ) / (cube.total_cubes * (cube.total_cubes - 1) / 2)

/-- The theorem to be proved -/
theorem probability_in_painted_cube :
  ∃ (cube : PaintedCube),
    cube.size = 5 ∧
    cube.total_cubes = 125 ∧
    cube.three_face_cubes = 1 ∧
    cube.no_face_cubes = 76 ∧
    probability_three_and_none cube = 2 / 205 :=
sorry

end probability_in_painted_cube_l3670_367055


namespace three_lines_vertically_opposite_angles_l3670_367093

/-- The number of pairs of vertically opposite angles formed by three intersecting lines in a plane -/
def vertically_opposite_angles_count (n : ℕ) : ℕ :=
  if n = 3 then 6 else 0

/-- Theorem stating that three intersecting lines in a plane form 6 pairs of vertically opposite angles -/
theorem three_lines_vertically_opposite_angles :
  vertically_opposite_angles_count 3 = 6 := by
  sorry

end three_lines_vertically_opposite_angles_l3670_367093


namespace number_of_students_l3670_367088

-- Define the number of 8th-grade students
variable (x : ℕ)

-- Define the conditions
axiom retail_threshold : x < 250
axiom wholesale_threshold : x + 60 ≥ 250
axiom retail_cost : 240 / x * 240 = 240
axiom wholesale_cost : 260 / (x + 60) * (x + 60) = 260
axiom equal_cost : 260 / (x + 60) * 288 = 240 / x * 240

-- Theorem to prove
theorem number_of_students : x = 200 := by
  sorry

end number_of_students_l3670_367088


namespace max_value_f_l3670_367019

/-- The function f defined on ℝ -/
def f (a b x : ℝ) : ℝ := 2 * a * x + b

/-- The theorem stating the conditions and the result -/
theorem max_value_f (a b : ℝ) : 
  b > 0 ∧ 
  (∀ x ∈ Set.Icc (-1/2) (1/2), |f a b x| ≤ 2) ∧
  (∀ a' b' : ℝ, b' > 0 ∧ (∀ x ∈ Set.Icc (-1/2) (1/2), |f a' b' x| ≤ 2) → a * b ≥ a' * b') →
  f a b 2017 = 4035 := by
sorry

end max_value_f_l3670_367019
