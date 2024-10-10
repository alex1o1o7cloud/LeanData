import Mathlib

namespace x_plus_one_is_linear_l2613_261387

/-- A linear equation is an equation with variables of only the first power -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

/-- The function representing x + 1 = 0 -/
def f (x : ℝ) : ℝ := x + 1

theorem x_plus_one_is_linear : is_linear_equation f := by
  sorry

end x_plus_one_is_linear_l2613_261387


namespace quadratic_equation_solution_square_roots_solution_l2613_261324

theorem quadratic_equation_solution (x : ℝ) :
  25 * x^2 - 36 = 0 → x = 6/5 ∨ x = -6/5 := by sorry

theorem square_roots_solution (x a : ℝ) :
  a > 0 ∧ (x + 2)^2 = a ∧ (3*x - 10)^2 = a → x = 2 ∧ a = 16 := by sorry

end quadratic_equation_solution_square_roots_solution_l2613_261324


namespace kaleb_books_l2613_261328

theorem kaleb_books (initial_books : ℕ) : 
  initial_books - 17 + 7 = 24 → initial_books = 34 := by
  sorry

end kaleb_books_l2613_261328


namespace expression_value_l2613_261329

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (h : z ≠ 0) :
  (x^2 - x*y) / (y^2 + 2*z^2) = 10/11 := by
  sorry

end expression_value_l2613_261329


namespace dogs_not_liking_any_food_l2613_261361

theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : ℕ) 
  (watermelon_salmon watermelon_chicken salmon_chicken : ℕ) (all_three : ℕ)
  (h_total : total = 100)
  (h_watermelon : watermelon = 20)
  (h_salmon : salmon = 70)
  (h_chicken : chicken = 10)
  (h_watermelon_salmon : watermelon_salmon = 10)
  (h_salmon_chicken : salmon_chicken = 5)
  (h_watermelon_chicken : watermelon_chicken = 3)
  (h_all_three : all_three = 2) :
  total - (watermelon + salmon + chicken - watermelon_salmon - watermelon_chicken - salmon_chicken + all_three) = 28 := by
  sorry

end dogs_not_liking_any_food_l2613_261361


namespace rhombus_perimeter_l2613_261374

/-- The perimeter of a rhombus with diagonals 24 and 10 is 52 -/
theorem rhombus_perimeter (d1 d2 : ℝ) : 
  d1 = 24 → d2 = 10 → 4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l2613_261374


namespace max_graduates_of_interest_l2613_261369

theorem max_graduates_of_interest (n : ℕ) (u : ℕ) (g : ℕ) :
  n = 100 →  -- number of graduates
  u = 5 →    -- number of universities
  g = 50 →   -- number of graduates each university reached
  (∀ i : ℕ, i ≤ u → g = n / 2) →  -- each university reached half of the graduates
  (∃ x : ℕ, x ≥ 3 ∧ x ≤ u ∧ ∃ y : ℕ, y ≤ n ∧ ∀ i : ℕ, i ≤ x → y ≤ g) →  -- at least 3 universities couldn't reach some graduates
  (∃ m : ℕ, m ≤ 83 ∧ 
    (∀ k : ℕ, k > m → 
      ¬(∃ f : ℕ → ℕ, (∀ i : ℕ, i ≤ k → f i ≤ u) ∧ 
        (∀ i : ℕ, i ≤ k → (∃ j₁ j₂ j₃ : ℕ, j₁ < j₂ ∧ j₂ < j₃ ∧ j₃ ≤ u ∧ 
          f i = j₁ ∧ f i = j₂ ∧ f i = j₃))))) :=
by sorry

end max_graduates_of_interest_l2613_261369


namespace additional_area_codes_l2613_261395

/-- The number of available signs for area codes -/
def num_signs : ℕ := 124

/-- The number of 2-letter area codes -/
def two_letter_codes : ℕ := num_signs * (num_signs - 1)

/-- The number of 3-letter area codes -/
def three_letter_codes : ℕ := num_signs * (num_signs - 1) * (num_signs - 2)

/-- The additional number of area codes created with the 3-letter system compared to the 2-letter system -/
theorem additional_area_codes :
  three_letter_codes - two_letter_codes = 1845396 :=
by sorry

end additional_area_codes_l2613_261395


namespace triangle_area_bound_l2613_261351

/-- For any triangle with area S and semiperimeter p, S ≤ p^2 / (3√3) -/
theorem triangle_area_bound (S p : ℝ) (h_S : S > 0) (h_p : p > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧
    p = (a + b + c) / 2) :
  S ≤ p^2 / (3 * Real.sqrt 3) := by
sorry

end triangle_area_bound_l2613_261351


namespace composition_equality_l2613_261319

variables (m n p q : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + n * x

def g (x : ℝ) : ℝ := p * x + q

theorem composition_equality :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n :=
sorry

end composition_equality_l2613_261319


namespace min_gumballs_for_five_correct_l2613_261308

/-- Represents the number of gumballs of each color -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- The minimum number of gumballs needed to guarantee 5 of the same color -/
def minGumballsForFive (m : GumballMachine) : ℕ := 17

/-- Theorem stating that for the given gumball machine, 
    17 is the minimum number of gumballs needed to guarantee 5 of the same color -/
theorem min_gumballs_for_five_correct (m : GumballMachine) 
  (h_red : m.red = 12) 
  (h_white : m.white = 10) 
  (h_blue : m.blue = 9) 
  (h_green : m.green = 8) : 
  minGumballsForFive m = 17 := by
  sorry


end min_gumballs_for_five_correct_l2613_261308


namespace math_competition_score_l2613_261365

theorem math_competition_score 
  (a₁ a₂ a₃ a₄ a₅ : ℕ) 
  (h_distinct : a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅)
  (h_first_two : a₁ + a₂ = 10)
  (h_last_two : a₄ + a₅ = 18) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 35 :=
by sorry

end math_competition_score_l2613_261365


namespace new_crust_flour_amount_l2613_261345

/-- The amount of flour per new pie crust when changing the recipe -/
def flour_per_new_crust (original_crusts : ℕ) (original_flour_per_crust : ℚ) 
  (new_crusts : ℕ) : ℚ :=
  (original_crusts : ℚ) * original_flour_per_crust / (new_crusts : ℚ)

/-- Theorem stating that the amount of flour per new pie crust is 1/5 cup -/
theorem new_crust_flour_amount : 
  flour_per_new_crust 40 (1/8) 25 = 1/5 := by
  sorry

end new_crust_flour_amount_l2613_261345


namespace arithmetic_sequence_property_l2613_261398

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
by sorry

end arithmetic_sequence_property_l2613_261398


namespace election_votes_count_l2613_261337

theorem election_votes_count :
  ∀ (total_votes : ℕ) (harold_percentage : ℚ) (jacquie_percentage : ℚ),
    harold_percentage = 60 / 100 →
    jacquie_percentage = 1 - harold_percentage →
    (harold_percentage * total_votes : ℚ) - (jacquie_percentage * total_votes : ℚ) = 24 →
    total_votes = 120 :=
by
  sorry

end election_votes_count_l2613_261337


namespace power_5_2023_mod_17_l2613_261309

theorem power_5_2023_mod_17 : (5 : ℤ) ^ 2023 % 17 = 2 := by sorry

end power_5_2023_mod_17_l2613_261309


namespace club_membership_after_four_years_l2613_261354

/-- Represents the number of members in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 3 * club_members n - 16

/-- The club membership problem -/
theorem club_membership_after_four_years :
  club_members 4 = 980 := by
  sorry

end club_membership_after_four_years_l2613_261354


namespace bus_passengers_l2613_261356

/-- 
Given a bus that starts with 64 students and loses one-third of its 
passengers at each stop, prove that after four stops, 1024/81 students remain.
-/
theorem bus_passengers (initial_students : ℕ) (stops : ℕ) : 
  initial_students = 64 → 
  stops = 4 → 
  (initial_students : ℚ) * (2/3)^stops = 1024/81 := by sorry

end bus_passengers_l2613_261356


namespace product_mod_600_l2613_261300

theorem product_mod_600 : (1497 * 2003) % 600 = 291 := by
  sorry

end product_mod_600_l2613_261300


namespace island_age_conversion_l2613_261353

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The age of the island in base 7 and base 10 --/
theorem island_age_conversion :
  base7ToBase10 3 4 6 = 181 := by
  sorry

end island_age_conversion_l2613_261353


namespace quadratic_range_condition_l2613_261384

/-- A quadratic function f(x) = mx^2 - 2x + m has a value range of [0, +∞) if and only if m = 1 -/
theorem quadratic_range_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y ≥ 0 ∧ y = m * x^2 - 2 * x + m) ∧ 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, y = m * x^2 - 2 * x + m) ↔ 
  m = 1 :=
sorry

end quadratic_range_condition_l2613_261384


namespace remainder_sum_l2613_261366

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 75)
  (hd : d % 120 = 117) : 
  (c + d) % 40 = 32 := by
  sorry

end remainder_sum_l2613_261366


namespace man_speed_man_speed_is_6_l2613_261341

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5/18)
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * (18/5)
  man_speed_kmph

/-- The speed of the man is 6 kmph --/
theorem man_speed_is_6 :
  man_speed 220 60 12 = 6 := by
  sorry

end man_speed_man_speed_is_6_l2613_261341


namespace f_properties_l2613_261313

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Theorem stating the properties of f(x)
theorem f_properties :
  (f 0 = 1) ∧ 
  (f' 1 = 1) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (f 0 = 1) ∧
  (∀ x : ℝ, f x ≥ 23/27) ∧
  (f (2/3) = 23/27) :=
sorry

end f_properties_l2613_261313


namespace symmetry_about_x_axis_l2613_261338

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem symmetry_about_x_axis :
  let P : Point2D := { x := -1, y := 5 }
  symmetricAboutXAxis P = { x := -1, y := -5 } := by
  sorry

end symmetry_about_x_axis_l2613_261338


namespace sqrt_sum_inequality_l2613_261347

theorem sqrt_sum_inequality (a b c d : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_a : a ≤ 1)
  (h_ab : a + b ≤ 5)
  (h_abc : a + b + c ≤ 14)
  (h_abcd : a + b + c + d ≤ 30) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d ≤ 10 := by
  sorry

end sqrt_sum_inequality_l2613_261347


namespace cube_section_not_pentagon_cube_section_can_be_hexagon_l2613_261349

/-- A cube in 3D space --/
structure Cube where
  side : ℝ
  center : ℝ × ℝ × ℝ

/-- A plane in 3D space --/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a plane and a cube --/
def PlaneSection (c : Cube) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a regular polygon --/
def IsRegularPolygon (s : Set (ℝ × ℝ × ℝ)) (n : ℕ) : Prop :=
  sorry

theorem cube_section_not_pentagon (c : Cube) :
  ¬ ∃ p : Plane, IsRegularPolygon (PlaneSection c p) 5 :=
sorry

theorem cube_section_can_be_hexagon :
  ∃ c : Cube, ∃ p : Plane, IsRegularPolygon (PlaneSection c p) 6 :=
sorry

end cube_section_not_pentagon_cube_section_can_be_hexagon_l2613_261349


namespace vector_subtraction_l2613_261360

theorem vector_subtraction (c d : Fin 3 → ℝ) 
  (hc : c = ![5, -3, 2])
  (hd : d = ![-2, 1, 5]) :
  c - 4 • d = ![13, -7, -18] := by
sorry

end vector_subtraction_l2613_261360


namespace marbles_given_to_sam_l2613_261302

def initial_marbles : ℕ := 8
def remaining_marbles : ℕ := 4

theorem marbles_given_to_sam :
  initial_marbles - remaining_marbles = 4 :=
by sorry

end marbles_given_to_sam_l2613_261302


namespace polygon_area_l2613_261380

/-- An isosceles triangle with one angle of 100° and area 2 cm² --/
structure IsoscelesTriangle where
  angle : ℝ
  area : ℝ
  is_isosceles : angle = 100
  has_area : area = 2

/-- A polygon composed of isosceles triangles --/
structure Polygon where
  triangle : IsoscelesTriangle
  full_count : ℕ
  half_count : ℕ
  full_is_12 : full_count = 12
  half_is_4 : half_count = 4

/-- The area of the polygon is 28 cm² --/
theorem polygon_area (p : Polygon) : p.full_count * p.triangle.area + p.half_count * (p.triangle.area / 2) = 28 := by
  sorry

end polygon_area_l2613_261380


namespace geometric_sequence_ninth_term_l2613_261393

/-- Given a geometric sequence where the 3rd term is 5 and the 6th term is 40,
    the 9th term is 320. -/
theorem geometric_sequence_ninth_term : ∀ (a : ℕ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 4 / a 3)) →  -- Geometric sequence condition
  a 3 = 5 →                                   -- 3rd term is 5
  a 6 = 40 →                                  -- 6th term is 40
  a 9 = 320 :=                                -- 9th term is 320
by
  sorry


end geometric_sequence_ninth_term_l2613_261393


namespace ronald_egg_sharing_l2613_261304

def total_eggs : ℕ := 16
def eggs_per_friend : ℕ := 2

theorem ronald_egg_sharing :
  total_eggs / eggs_per_friend = 8 := by sorry

end ronald_egg_sharing_l2613_261304


namespace church_attendance_l2613_261343

theorem church_attendance (male_adults female_adults total_people : ℕ) 
  (h1 : male_adults = 60)
  (h2 : female_adults = 60)
  (h3 : total_people = 200) :
  total_people - (male_adults + female_adults) = 80 := by
sorry

end church_attendance_l2613_261343


namespace vydmans_formula_l2613_261318

theorem vydmans_formula (h b x r : ℝ) (h_pos : h > 0) (b_pos : b > 0) (x_pos : x > 0) :
  r = Real.sqrt ((b / 2)^2 + ((h^2 + (b / 2 - x)^2 - b^2 / 4) / (2 * h))^2) →
  r^2 = (b / 2)^2 + ((h^2 + (b / 2 - x)^2 - b^2 / 4) / (2 * h))^2 :=
by sorry

end vydmans_formula_l2613_261318


namespace sum_of_solutions_l2613_261394

-- Define the equation
def equation (x : ℝ) : Prop := (4 * x + 6) * (3 * x - 7) = 0

-- State the theorem
theorem sum_of_solutions : 
  ∃ (s : ℝ), (∀ (x : ℝ), equation x → x = s ∨ x = (5/6 - s)) ∧ s + (5/6 - s) = 5/6 :=
sorry

end sum_of_solutions_l2613_261394


namespace salt_production_theorem_l2613_261322

/-- Calculates the average daily salt production for a year given the initial production and monthly increase. -/
def averageDailyProduction (initialProduction : ℕ) (monthlyIncrease : ℕ) : ℚ :=
  let totalProduction := initialProduction + (initialProduction + monthlyIncrease + initialProduction + monthlyIncrease * 11) * 11 / 2
  totalProduction / 365

/-- Theorem stating that the average daily production is approximately 83.84 tonnes. -/
theorem salt_production_theorem (initialProduction monthlyIncrease : ℕ) 
  (h1 : initialProduction = 2000)
  (h2 : monthlyIncrease = 100) :
  ∃ ε > 0, |averageDailyProduction initialProduction monthlyIncrease - 83.84| < ε :=
sorry

end salt_production_theorem_l2613_261322


namespace root_property_l2613_261385

theorem root_property (a : ℝ) : 3 * a^2 - 4 * a + 1 = 0 → 6 * a^2 - 8 * a + 5 = 3 := by
  sorry

end root_property_l2613_261385


namespace cost_difference_between_cars_l2613_261359

/-- Represents a car with its associated costs and characteristics -/
structure Car where
  initialCost : ℕ
  fuelConsumption : ℕ
  annualInsurance : ℕ
  annualMaintenance : ℕ
  resaleValue : ℕ

/-- Calculates the total cost of owning a car for 5 years -/
def totalCost (c : Car) (annualDistance : ℕ) (fuelPrice : ℕ) (years : ℕ) : ℕ :=
  c.initialCost +
  (c.fuelConsumption * annualDistance / 100 * fuelPrice * years) +
  (c.annualInsurance * years) +
  (c.annualMaintenance * years) -
  c.resaleValue

/-- Theorem stating the difference in total cost between two cars -/
theorem cost_difference_between_cars :
  let carA : Car := {
    initialCost := 900000,
    fuelConsumption := 9,
    annualInsurance := 35000,
    annualMaintenance := 25000,
    resaleValue := 500000
  }
  let carB : Car := {
    initialCost := 600000,
    fuelConsumption := 10,
    annualInsurance := 32000,
    annualMaintenance := 20000,
    resaleValue := 350000
  }
  let annualDistance := 15000
  let fuelPrice := 40
  let years := 5

  totalCost carA annualDistance fuelPrice years -
  totalCost carB annualDistance fuelPrice years = 160000 := by
  sorry

end cost_difference_between_cars_l2613_261359


namespace log_not_always_decreasing_l2613_261331

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_not_always_decreasing :
  ∃ (a : ℝ), a > 1 ∧ ∀ (x y : ℝ), x > y → x > 0 → y > 0 → log a x > log a y :=
sorry

end log_not_always_decreasing_l2613_261331


namespace divisible_by_ten_l2613_261371

theorem divisible_by_ten (S : Finset ℤ) : 
  (Finset.card S = 5) →
  (∀ (a b c : ℤ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (10 ∣ a * b * c)) →
  (∃ x ∈ S, 10 ∣ x) :=
by sorry

end divisible_by_ten_l2613_261371


namespace x_minus_y_values_l2613_261368

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 4) (h3 : x < y) :
  x - y = -7 ∨ x - y = -1 := by
  sorry

end x_minus_y_values_l2613_261368


namespace test_score_ranges_l2613_261348

/-- Given three ranges of test scores, prove that R1 is 30 -/
theorem test_score_ranges (R1 R2 R3 : ℕ) : 
  R2 = 26 → 
  R3 = 32 → 
  (min R1 (min R2 R3) = 30) → 
  R1 = 30 := by
sorry

end test_score_ranges_l2613_261348


namespace wire_cutting_problem_l2613_261377

theorem wire_cutting_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 10 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 35 := by
sorry

end wire_cutting_problem_l2613_261377


namespace hyperbola_axes_length_l2613_261358

theorem hyperbola_axes_length (x y : ℝ) :
  x^2 - 8*y^2 = 32 →
  ∃ (real_axis imaginary_axis : ℝ),
    real_axis = 8 * Real.sqrt 2 ∧
    imaginary_axis = 4 :=
by sorry

end hyperbola_axes_length_l2613_261358


namespace surface_area_theorem_l2613_261399

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the solid RUVWX -/
structure SlicedSolid where
  prism : RightPrism
  V : Point3D
  W : Point3D
  X : Point3D

/-- Calculate the surface area of the sliced solid RUVWX -/
def surface_area_RUVWX (solid : SlicedSolid) : ℝ :=
  sorry

/-- The main theorem stating the surface area of RUVWX -/
theorem surface_area_theorem (solid : SlicedSolid) 
  (h1 : solid.prism.height = 20)
  (h2 : solid.prism.base_side_length = 10)
  (h3 : solid.V = Point3D.mk 5 0 10)
  (h4 : solid.W = Point3D.mk 5 (5 * Real.sqrt 3) 10)
  (h5 : solid.X = Point3D.mk 0 0 10) :
  surface_area_RUVWX solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
sorry

end surface_area_theorem_l2613_261399


namespace line_segment_length_l2613_261306

/-- Given points A, B, C, D, and E on a line in that order, prove that CD = 3 cm -/
theorem line_segment_length (A B C D E : ℝ) : 
  (B - A = 2) → 
  (C - A = 5) → 
  (D - B = 6) → 
  (∃ x, E - D = x) → 
  (E - B = 8) → 
  (E - A < 12) → 
  (D - C = 3) := by
  sorry

end line_segment_length_l2613_261306


namespace threeDigitNumberFormula_l2613_261317

/-- Given a natural number m, this function represents a three-digit number
    where the hundreds digit is 3m, the tens digit is m, and the units digit is m-1 -/
def threeDigitNumber (m : ℕ) : ℕ := 300 * m + 10 * m + (m - 1)

/-- Theorem stating that the three-digit number can be expressed as 311m - 1 -/
theorem threeDigitNumberFormula (m : ℕ) : 
  threeDigitNumber m = 311 * m - 1 := by
  sorry

end threeDigitNumberFormula_l2613_261317


namespace quadratic_radical_equality_l2613_261321

/-- If the simplest quadratic radical 2√(4m-1) is of the same type as √(2+3m), then m = 3. -/
theorem quadratic_radical_equality (m : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ k * (4 * m - 1) = 2 + 3 * m) → m = 3 := by
  sorry

end quadratic_radical_equality_l2613_261321


namespace train_speed_excluding_stoppages_l2613_261339

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop time. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stops = 30) 
  (h2 : stop_time = 24) : 
  speed_with_stops * (60 - stop_time) / 60 = 18 := by
  sorry

end train_speed_excluding_stoppages_l2613_261339


namespace arithmetic_sequence_and_binomial_expansion_l2613_261311

theorem arithmetic_sequence_and_binomial_expansion :
  let a : ℕ → ℤ := λ n => 3*n - 5
  let binomial_sum : ℕ → ℤ := λ k => Nat.choose 5 k + Nat.choose 6 k + Nat.choose 7 k
  a 20 = binomial_sum 4 := by sorry

end arithmetic_sequence_and_binomial_expansion_l2613_261311


namespace beaver_problem_l2613_261373

theorem beaver_problem (initial_beavers final_beavers : ℕ) : 
  final_beavers = initial_beavers + 1 → 
  final_beavers = 3 → 
  initial_beavers = 2 := by
sorry

end beaver_problem_l2613_261373


namespace team_order_l2613_261336

/-- Represents the points of a team in a sports league. -/
structure TeamPoints where
  points : ℕ

/-- Represents the points of all teams in the sports league. -/
structure LeaguePoints where
  A : TeamPoints
  B : TeamPoints
  C : TeamPoints
  D : TeamPoints

/-- Defines the conditions given in the problem. -/
def satisfiesConditions (lp : LeaguePoints) : Prop :=
  (lp.A.points + lp.C.points = lp.B.points + lp.D.points) ∧
  (lp.B.points + lp.A.points + 5 ≤ lp.D.points + lp.C.points) ∧
  (lp.B.points + lp.C.points ≥ lp.A.points + lp.D.points + 3)

/-- Defines the correct order of teams based on their points. -/
def correctOrder (lp : LeaguePoints) : Prop :=
  lp.C.points > lp.D.points ∧ lp.D.points > lp.B.points ∧ lp.B.points > lp.A.points

/-- Theorem stating that if the conditions are satisfied, the correct order of teams is C, D, B, A. -/
theorem team_order (lp : LeaguePoints) :
  satisfiesConditions lp → correctOrder lp := by
  sorry


end team_order_l2613_261336


namespace factorize_expression_1_l2613_261334

theorem factorize_expression_1 (x : ℝ) :
  (x^2 - 1 + x) * (x^2 - 1 + 3*x) + x^2 = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 := by
sorry

end factorize_expression_1_l2613_261334


namespace haircuts_to_goal_l2613_261312

/-- Given a person who has gotten 8 haircuts and is 80% towards their goal,
    prove that the number of additional haircuts needed to reach 100% of the goal is 2. -/
theorem haircuts_to_goal (current_haircuts : ℕ) (current_percentage : ℚ) : 
  current_haircuts = 8 → current_percentage = 80/100 → 
  (100/100 - current_percentage) / (current_percentage / current_haircuts) = 2 := by
sorry

end haircuts_to_goal_l2613_261312


namespace consecutive_even_numbers_l2613_261397

/-- Given three consecutive even integers and a condition, prove the value of k. -/
theorem consecutive_even_numbers (N₁ N₂ N₃ k : ℤ) : 
  N₃ = 19 →
  N₂ = N₁ + 2 →
  N₃ = N₂ + 2 →
  3 * N₁ = k * N₃ + 7 →
  k = 2 := by
sorry


end consecutive_even_numbers_l2613_261397


namespace a_33_mod_42_l2613_261355

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a_33 divided by 42 has a remainder of 20 -/
theorem a_33_mod_42 : a 33 % 42 = 20 := by sorry

end a_33_mod_42_l2613_261355


namespace tangent_line_at_2_2_increasing_intervals_decreasing_interval_l2613_261316

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2_2 :
  ∃ (a b c : ℝ), a * 2 + b * 2 + c = 0 ∧
  ∀ (x y : ℝ), y = f x → (y - f 2) = f_derivative 2 * (x - 2) →
  a * x + b * y + c = 0 :=
sorry

-- Theorem for increasing intervals
theorem increasing_intervals :
  ∀ x, (x < -1 ∨ x > 1) → f_derivative x > 0 :=
sorry

-- Theorem for decreasing interval
theorem decreasing_interval :
  ∀ x, -1 < x ∧ x < 1 → f_derivative x < 0 :=
sorry

end tangent_line_at_2_2_increasing_intervals_decreasing_interval_l2613_261316


namespace total_students_is_1480_l2613_261357

/-- Represents a campus in the school district -/
structure Campus where
  grades : ℕ  -- number of grades
  students_per_grade : ℕ  -- number of students per grade
  extra_students : ℕ  -- number of extra students in special programs

/-- Calculates the total number of students in a campus -/
def campus_total (c : Campus) : ℕ :=
  c.grades * c.students_per_grade + c.extra_students

/-- The school district with its three campuses -/
structure SchoolDistrict where
  campus_a : Campus
  campus_b : Campus
  campus_c : Campus

/-- Represents the specific school district described in the problem -/
def our_district : SchoolDistrict :=
  { campus_a := { grades := 5, students_per_grade := 100, extra_students := 30 }
  , campus_b := { grades := 5, students_per_grade := 120, extra_students := 0 }
  , campus_c := { grades := 2, students_per_grade := 150, extra_students := 50 }
  }

/-- Theorem stating that the total number of students in our school district is 1480 -/
theorem total_students_is_1480 : 
  campus_total our_district.campus_a + 
  campus_total our_district.campus_b + 
  campus_total our_district.campus_c = 1480 := by
  sorry

end total_students_is_1480_l2613_261357


namespace unique_solution_condition_l2613_261323

theorem unique_solution_condition (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 + k ∧ y = 3*x) ↔ k = 9/4 := by
  sorry

end unique_solution_condition_l2613_261323


namespace binomial_13_choose_10_l2613_261342

theorem binomial_13_choose_10 : Nat.choose 13 10 = 286 := by
  sorry

end binomial_13_choose_10_l2613_261342


namespace initial_bacteria_population_l2613_261352

/-- The number of seconds in 5 minutes -/
def totalTime : ℕ := 300

/-- The doubling time of the bacteria population in seconds -/
def doublingTime : ℕ := 30

/-- The number of bacteria after 5 minutes -/
def finalPopulation : ℕ := 1310720

/-- The number of doublings that occur in 5 minutes -/
def numberOfDoublings : ℕ := totalTime / doublingTime

theorem initial_bacteria_population :
  ∃ (initialPopulation : ℕ),
    initialPopulation * (2 ^ numberOfDoublings) = finalPopulation ∧
    initialPopulation = 1280 :=
by sorry

end initial_bacteria_population_l2613_261352


namespace expanded_expression_equals_804095_l2613_261310

theorem expanded_expression_equals_804095 :
  8 * 10^5 + 4 * 10^3 + 9 * 10 + 5 = 804095 := by
  sorry

end expanded_expression_equals_804095_l2613_261310


namespace inequality_reversal_l2613_261375

theorem inequality_reversal (a b : ℝ) (h : a > b) : ∃ m : ℝ, ¬(m * a < m * b) := by
  sorry

end inequality_reversal_l2613_261375


namespace tangent_line_at_ln2_max_k_for_f_greater_g_l2613_261340

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / (Real.exp x - 1)

def g (k : ℕ) (x : ℝ) : ℝ := k / (x + 1)

def tangent_line (x : ℝ) : ℝ := -2 * x + 2 * Real.log 2 + 2

theorem tangent_line_at_ln2 (x : ℝ) (h : x > 0) :
  tangent_line x = -2 * x + 2 * Real.log 2 + 2 :=
sorry

theorem max_k_for_f_greater_g :
  ∃ (k : ℕ), k = 3 ∧ 
  (∀ (x : ℝ), x > 0 → f x > g k x) ∧
  (∀ (k' : ℕ), k' > k → ∃ (x : ℝ), x > 0 ∧ f x ≤ g k' x) :=
sorry

end tangent_line_at_ln2_max_k_for_f_greater_g_l2613_261340


namespace program_outputs_divisors_l2613_261344

/-- The set of numbers output by the program for a given input n -/
def program_output (n : ℕ) : Set ℕ :=
  {i : ℕ | i ≤ n ∧ n % i = 0}

/-- The set of all divisors of n -/
def divisors (n : ℕ) : Set ℕ :=
  {i : ℕ | i ∣ n}

/-- Theorem stating that the program output is equal to the set of all divisors -/
theorem program_outputs_divisors (n : ℕ) : program_output n = divisors n := by
  sorry

end program_outputs_divisors_l2613_261344


namespace tony_additional_degrees_l2613_261320

/-- Represents the number of years Tony spent in school for various degrees -/
structure TonySchoolYears where
  science : ℕ
  physics : ℕ
  additional : ℕ
  total : ℕ

/-- Calculates the number of additional degrees Tony got -/
def additional_degrees (years : TonySchoolYears) : ℕ :=
  (years.total - years.science - years.physics) / years.science

/-- Theorem stating that Tony got 2 additional degrees -/
theorem tony_additional_degrees :
  ∀ (years : TonySchoolYears),
    years.science = 4 →
    years.physics = 2 →
    years.total = 14 →
    additional_degrees years = 2 := by
  sorry

#check tony_additional_degrees

end tony_additional_degrees_l2613_261320


namespace frustum_volume_l2613_261386

/-- The volume of a frustum with given ratio of radii and height, and slant height -/
theorem frustum_volume (r R h s : ℝ) (h1 : R = 4*r) (h2 : h = 4*r) (h3 : s = 10) 
  (h4 : s^2 = h^2 + (R - r)^2) : 
  (1/3 : ℝ) * Real.pi * h * (r^2 + R^2 + r*R) = 224 * Real.pi := by
  sorry

#check frustum_volume

end frustum_volume_l2613_261386


namespace valid_m_range_l2613_261381

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ (1 = -k ∧ 1 = -k ∧ m = k * |m|)

def q (m : ℝ) : Prop := (2*m + 1 > 0 ∧ m - 3 < 0) ∨ (2*m + 1 < 0 ∧ m - 3 > 0)

-- State the theorem
theorem valid_m_range :
  ∀ m : ℝ, (¬(p m) ∧ (p m ∨ q m)) → (0 < m ∧ m < 3) :=
by sorry

end valid_m_range_l2613_261381


namespace chord_length_circle_line_l2613_261335

/-- The chord length cut by a circle on a line --/
theorem chord_length_circle_line (x y : ℝ) : 
  let circle := fun x y => x^2 + y^2 - 8*x - 2*y + 1 = 0
  let line := fun x => Real.sqrt 3 * x + 1
  let center := (4, 1)
  let radius := 4
  let distance_center_to_line := 2 * Real.sqrt 3
  true → -- placeholder for the circle and line equations
  2 * Real.sqrt (radius^2 - distance_center_to_line^2) = 4 :=
by sorry

end chord_length_circle_line_l2613_261335


namespace function_max_implies_a_range_l2613_261382

/-- Given a function f(x) = (ax^2)/2 - (1+2a)x + 2ln(x) where a > 0,
    if f(x) has a maximum value in the interval (1/2, 1),
    then 1 < a < 2. -/
theorem function_max_implies_a_range (a : ℝ) (f : ℝ → ℝ) :
  a > 0 →
  (∀ x, f x = (a * x^2) / 2 - (1 + 2*a) * x + 2 * Real.log x) →
  (∃ x₀ ∈ Set.Ioo (1/2) 1, ∀ x ∈ Set.Ioo (1/2) 1, f x ≤ f x₀) →
  1 < a ∧ a < 2 :=
by sorry

end function_max_implies_a_range_l2613_261382


namespace sum_of_abc_l2613_261314

theorem sum_of_abc (a b c : ℝ) 
  (h1 : a^2*b + a^2*c + b^2*a + b^2*c + c^2*a + c^2*b + 3*a*b*c = 30)
  (h2 : a^2 + b^2 + c^2 = 13) : 
  a + b + c = 5 := by sorry

end sum_of_abc_l2613_261314


namespace power_function_through_point_l2613_261330

theorem power_function_through_point (k m : ℝ) : 
  k * (2 : ℝ)^m = 1/4 → m * k = -2 := by
  sorry

end power_function_through_point_l2613_261330


namespace bus_train_speed_ratio_l2613_261367

/-- Proves that the fraction of bus speed to train speed is 3/4 -/
theorem bus_train_speed_ratio :
  let train_car_speed_ratio : ℚ := 16 / 15
  let bus_distance : ℕ := 480
  let bus_time : ℕ := 8
  let car_distance : ℕ := 450
  let car_time : ℕ := 6
  let bus_speed : ℚ := bus_distance / bus_time
  let car_speed : ℚ := car_distance / car_time
  let train_speed : ℚ := car_speed * train_car_speed_ratio
  bus_speed / train_speed = 3 / 4 := by
  sorry

end bus_train_speed_ratio_l2613_261367


namespace bottle_problem_l2613_261391

/-- Represents a bottle in the case -/
inductive Bottle
  | FirstPrize
  | SecondPrize
  | NoPrize

/-- Represents the case of bottles -/
def Case : Finset Bottle := sorry

/-- The number of bottles in the case -/
def caseSize : ℕ := 6

/-- The number of bottles with prizes -/
def prizeBottles : ℕ := 2

/-- The number of bottles without prizes -/
def noPrizeBottles : ℕ := 4

/-- Person A's selection of bottles -/
def Selection : Finset Bottle := sorry

/-- The number of bottles selected -/
def selectionSize : ℕ := 2

/-- Event A: A did not win a prize -/
def EventA : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ ∀ b ∈ s, b = Bottle.NoPrize}

/-- Event B: A won the first prize -/
def EventB : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ Bottle.FirstPrize ∈ s}

/-- Event C: A won a prize -/
def EventC : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ (Bottle.FirstPrize ∈ s ∨ Bottle.SecondPrize ∈ s)}

/-- The probability measure on the sample space -/
noncomputable def P : Set (Finset Bottle) → ℝ := sorry

theorem bottle_problem :
  (EventA ∩ EventC = ∅) ∧ (P (EventB ∪ EventC) = P EventC) := by sorry

end bottle_problem_l2613_261391


namespace not_necessarily_regular_l2613_261392

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk ::

/-- Predicate to check if all edges of a polyhedron are equal -/
def all_edges_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if all dihedral angles of a polyhedron are equal -/
def all_dihedral_angles_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if all polyhedral angles of a polyhedron are equal -/
def all_polyhedral_angles_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if a polyhedron is regular -/
def is_regular (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating that a convex polyhedron with equal edges and either equal dihedral angles
    or equal polyhedral angles is not necessarily regular -/
theorem not_necessarily_regular :
  ∃ p : ConvexPolyhedron,
    (all_edges_equal p ∧ all_dihedral_angles_equal p ∧ ¬is_regular p) ∨
    (all_edges_equal p ∧ all_polyhedral_angles_equal p ∧ ¬is_regular p) :=
  sorry

end not_necessarily_regular_l2613_261392


namespace quadratic_root_difference_l2613_261383

/-- Given a quadratic equation x^2 + px - q = 0 where p and q are positive real numbers,
    if the difference between its roots is 2, then p = √(4 - 4q) -/
theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let r₁ := (-p + Real.sqrt (p^2 + 4*q)) / 2
  let r₂ := (-p - Real.sqrt (p^2 + 4*q)) / 2
  (r₁ - r₂ = 2) → p = Real.sqrt (4 - 4*q) :=
by sorry

end quadratic_root_difference_l2613_261383


namespace equal_group_formation_l2613_261325

-- Define the total number of people
def total_people : ℕ := 20

-- Define the number of boys
def num_boys : ℕ := 10

-- Define the number of girls
def num_girls : ℕ := 10

-- Define the size of the group to be formed
def group_size : ℕ := 10

-- Theorem statement
theorem equal_group_formation :
  Nat.choose total_people group_size = 184756 :=
by
  sorry

end equal_group_formation_l2613_261325


namespace decagon_diagonal_intersection_probability_l2613_261372

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of diagonals in a regular decagon -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose three diagonals that intersect at a single point -/
def num_intersecting_diagonals (d : RegularDecagon) : ℕ := 840

/-- The total number of ways to choose three diagonals -/
def total_diagonal_choices (d : RegularDecagon) : ℕ := 6545

/-- The probability that three randomly chosen diagonals in a regular decagon
    intersect at a single point inside the decagon -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  (num_intersecting_diagonals d : ℚ) / (total_diagonal_choices d : ℚ) = 840 / 6545 := by
  sorry

end decagon_diagonal_intersection_probability_l2613_261372


namespace parallelogram_area_calculation_l2613_261379

-- Define the parallelogram properties
def base : ℝ := 20
def total_length : ℝ := 26
def slant_height : ℝ := 7

-- Define the area function for a parallelogram
def parallelogram_area (b h : ℝ) : ℝ := b * h

-- Theorem statement
theorem parallelogram_area_calculation :
  ∃ (height : ℝ), 
    height^2 + (total_length - base)^2 = slant_height^2 ∧
    parallelogram_area base height = 20 * Real.sqrt 13 := by
  sorry

end parallelogram_area_calculation_l2613_261379


namespace symmetric_abs_sum_l2613_261332

/-- A function f is symmetric about a point c if f(c+x) = f(c-x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_abs_sum (a : ℝ) :
  SymmetricAbout (fun x ↦ |x + 1| + |x - a|) 1 → a = 3 := by
  sorry


end symmetric_abs_sum_l2613_261332


namespace max_value_theorem_l2613_261388

theorem max_value_theorem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 4) :
  (6 * a + 3 * b + 10 * c) ≤ Real.sqrt 41 ∧
  ∃ a₀ b₀ c₀ : ℝ, 9 * a₀^2 + 4 * b₀^2 + 25 * c₀^2 = 4 ∧ 6 * a₀ + 3 * b₀ + 10 * c₀ = Real.sqrt 41 :=
by sorry

end max_value_theorem_l2613_261388


namespace reflect_H_twice_l2613_261327

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the parallelogram EFGH
def E : Point2D := ⟨3, 6⟩
def F : Point2D := ⟨5, 10⟩
def G : Point2D := ⟨7, 6⟩
def H : Point2D := ⟨5, 2⟩

-- Define reflection across x-axis
def reflectX (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

-- Define reflection across y = x + 2
def reflectYXPlus2 (p : Point2D) : Point2D :=
  ⟨p.y - 2, p.x + 2⟩

-- Theorem statement
theorem reflect_H_twice (h : Point2D) :
  h = H →
  reflectYXPlus2 (reflectX h) = ⟨-4, 7⟩ :=
by sorry

end reflect_H_twice_l2613_261327


namespace wrench_handle_length_l2613_261378

/-- Represents the inverse relationship between force and handle length -/
def inverse_relation (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem wrench_handle_length
  (force₁ : ℝ) (length₁ : ℝ) (force₂ : ℝ) (length₂ : ℝ)
  (h_inverse : inverse_relation force₁ length₁ ∧ inverse_relation force₂ length₂)
  (h_force₁ : force₁ = 300)
  (h_length₁ : length₁ = 12)
  (h_force₂ : force₂ = 400) :
  length₂ = 9 := by
  sorry

end wrench_handle_length_l2613_261378


namespace existence_implies_a_bound_l2613_261363

/-- Given a > 0, prove that if there exists x₀ ∈ (0, 1/2] such that f(x₀) > g(x₀), then a > -3 + √17 -/
theorem existence_implies_a_bound (a : ℝ) (h₁ : a > 0) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioc 0 (1/2) ∧ 
    a^2 * x₀^3 - 3*a * x₀^2 + 2 > -3*a * x₀ + 3) → 
  a > -3 + Real.sqrt 17 := by
sorry

/-- Definition of f(x) -/
def f (a x : ℝ) : ℝ := a^2 * x^3 - 3*a * x^2 + 2

/-- Definition of g(x) -/
def g (a x : ℝ) : ℝ := -3*a * x + 3

end existence_implies_a_bound_l2613_261363


namespace milk_fraction_in_cup1_l2613_261370

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure TwoCups where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : TwoCups :=
  { cup1 := { coffee := 3, milk := 0 },
    cup2 := { coffee := 0, milk := 7 } }

def transfer_coffee (state : TwoCups) : TwoCups :=
  { cup1 := { coffee := state.cup1.coffee * 2/3, milk := state.cup1.milk },
    cup2 := { coffee := state.cup2.coffee + state.cup1.coffee * 1/3, milk := state.cup2.milk } }

def transfer_mixture (state : TwoCups) : TwoCups :=
  let total_cup2 := state.cup2.coffee + state.cup2.milk
  let transfer_amount := total_cup2 * 1/4
  let coffee_ratio := state.cup2.coffee / total_cup2
  let milk_ratio := state.cup2.milk / total_cup2
  { cup1 := { coffee := state.cup1.coffee + transfer_amount * coffee_ratio,
              milk := state.cup1.milk + transfer_amount * milk_ratio },
    cup2 := { coffee := state.cup2.coffee - transfer_amount * coffee_ratio,
              milk := state.cup2.milk - transfer_amount * milk_ratio } }

def final_state : TwoCups :=
  transfer_mixture (transfer_coffee initial_state)

theorem milk_fraction_in_cup1 :
  let total_liquid := final_state.cup1.coffee + final_state.cup1.milk
  final_state.cup1.milk / total_liquid = 7/16 := by sorry

end milk_fraction_in_cup1_l2613_261370


namespace pedestrian_cyclist_speed_problem_l2613_261362

/-- The problem setup and solution for the pedestrian and cyclist speed problem -/
theorem pedestrian_cyclist_speed_problem :
  let distance : ℝ := 40 -- km
  let pedestrian_start_time : ℝ := 0 -- 4:00 AM
  let first_cyclist_start_time : ℝ := 3 + 1/3 -- 7:20 AM
  let second_cyclist_start_time : ℝ := 4.5 -- 8:30 AM
  let meetup_distance : ℝ := distance / 2
  let second_meetup_time_diff : ℝ := 1 -- hour

  ∃ (pedestrian_speed cyclist_speed : ℝ),
    pedestrian_speed > 0 ∧
    cyclist_speed > 0 ∧
    pedestrian_speed < cyclist_speed ∧
    -- First cyclist catches up with pedestrian at midpoint
    meetup_distance = pedestrian_speed * (first_cyclist_start_time + 
      (meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed)) ∧
    -- Second cyclist meets pedestrian one hour after first meetup
    distance = pedestrian_speed * (second_cyclist_start_time + 
      (meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed) + 
      second_meetup_time_diff) + 
      cyclist_speed * ((distance - meetup_distance) / cyclist_speed - 
      ((meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed) + 
      second_meetup_time_diff)) ∧
    pedestrian_speed = 5 ∧
    cyclist_speed = 30
  := by sorry

end pedestrian_cyclist_speed_problem_l2613_261362


namespace intersection_A_complement_B_l2613_261390

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (1 : ℝ) / |x - 1| < 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 > 0}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

end intersection_A_complement_B_l2613_261390


namespace a_4_equals_4_l2613_261364

def sequence_term (n : ℕ) : ℤ := (-1)^n * n

theorem a_4_equals_4 : sequence_term 4 = 4 := by sorry

end a_4_equals_4_l2613_261364


namespace parabola_point_to_directrix_distance_l2613_261303

theorem parabola_point_to_directrix_distance 
  (C : ℝ → ℝ → Prop) 
  (p : ℝ) 
  (A : ℝ × ℝ) :
  (∀ x y, C x y ↔ y^2 = 2*p*x) →  -- Definition of parabola C
  C A.1 A.2 →  -- A lies on C
  A = (1, Real.sqrt 5) →  -- Coordinates of A
  (A.1 + p/2) = 9/4 :=  -- Distance formula to directrix
by sorry

end parabola_point_to_directrix_distance_l2613_261303


namespace cos_sin_negative_225_deg_l2613_261376

theorem cos_sin_negative_225_deg : Real.cos (-225 * π / 180) + Real.sin (-225 * π / 180) = 0 := by
  sorry

end cos_sin_negative_225_deg_l2613_261376


namespace library_code_probability_l2613_261389

/-- The number of digits in the code -/
def code_length : ℕ := 6

/-- The total number of possible digits -/
def total_digits : ℕ := 10

/-- The probability of selecting a code with all different digits and not starting with 0 -/
def probability : ℚ := 1496880 / 1000000

/-- Theorem stating the probability of selecting a code with all different digits 
    and not starting with 0 is 0.13608 -/
theorem library_code_probability : 
  probability = 1496880 / 1000000 ∧ 
  (1496880 : ℚ) / 1000000 = 0.13608 := by sorry

end library_code_probability_l2613_261389


namespace upper_bound_of_expression_l2613_261346

theorem upper_bound_of_expression (n : ℤ) (U : ℤ) : 
  (∃ (S : Finset ℤ), 
    (∀ m ∈ S, (4 * m + 7 > 1 ∧ 4 * m + 7 < U)) ∧ 
    S.card = 15 ∧
    (∀ m : ℤ, 4 * m + 7 > 1 ∧ 4 * m + 7 < U → m ∈ S)) →
  U ≥ 64 :=
sorry

end upper_bound_of_expression_l2613_261346


namespace brooke_homework_time_l2613_261307

/-- Calculates the total time Brooke spends on homework, including breaks -/
def total_homework_time (math_problems : ℕ) (social_studies_problems : ℕ) (science_problems : ℕ)
  (math_time_per_problem : ℚ) (social_studies_time_per_problem : ℚ) (science_time_per_problem : ℚ)
  (math_break : ℕ) (social_studies_break : ℕ) (science_break : ℕ) : ℚ :=
  let math_time := math_problems * math_time_per_problem
  let social_studies_time := social_studies_problems * social_studies_time_per_problem / 60
  let science_time := science_problems * science_time_per_problem
  let total_problem_time := math_time + social_studies_time + science_time
  let total_break_time := math_break + social_studies_break + science_break
  total_problem_time + total_break_time

theorem brooke_homework_time :
  total_homework_time 15 6 10 2 (1/2) (3/2) 5 10 15 = 78 := by
  sorry

end brooke_homework_time_l2613_261307


namespace max_tuesday_money_l2613_261305

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℝ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount (t : ℝ) : ℝ := 5 * t

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount (t : ℝ) : ℝ := wednesday_amount t + 9

theorem max_tuesday_money :
  ∃ t : ℝ, t = tuesday_amount ∧
    thursday_amount t = t + 41 :=
by sorry

end max_tuesday_money_l2613_261305


namespace books_on_cart_l2613_261301

def top_section : ℕ := 12 + 8 + 4

def bottom_section_non_mystery : ℕ := 5 + 6

def bottom_section : ℕ := 2 * bottom_section_non_mystery

def total_books : ℕ := top_section + bottom_section

theorem books_on_cart : total_books = 46 := by
  sorry

end books_on_cart_l2613_261301


namespace complex_sum_of_parts_l2613_261326

theorem complex_sum_of_parts (a b : ℝ) (h : (Complex.mk a b) = Complex.mk 1 (-1)) : a + b = 0 := by
  sorry

end complex_sum_of_parts_l2613_261326


namespace function_satisfies_equation_l2613_261333

/-- Proves that the given function satisfies the differential equation -/
theorem function_satisfies_equation (x a : ℝ) :
  let y := a + (7 * x) / (a * x + 1)
  let y' := 7 / ((a * x + 1) ^ 2)
  y - x * y' = a * (1 + x^2 * y') := by
  sorry

end function_satisfies_equation_l2613_261333


namespace jungkook_has_smallest_number_l2613_261396

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem jungkook_has_smallest_number :
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
sorry

end jungkook_has_smallest_number_l2613_261396


namespace sue_shoe_probability_l2613_261350

/-- Represents the distribution of shoes by color --/
structure ShoeDistribution where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the probability of selecting two shoes of the same color
    with one left and one right, given a shoe distribution --/
def samePairProbability (d : ShoeDistribution) : Rat :=
  let totalShoes := 2 * (d.black + d.brown + d.gray + d.red)
  let blackProb := (d.black : Rat) * (d.black - 1) / (totalShoes * (totalShoes - 1))
  let brownProb := (d.brown : Rat) * (d.brown - 1) / (totalShoes * (totalShoes - 1))
  let grayProb := (d.gray : Rat) * (d.gray - 1) / (totalShoes * (totalShoes - 1))
  let redProb := (d.red : Rat) * (d.red - 1) / (totalShoes * (totalShoes - 1))
  blackProb + brownProb + grayProb + redProb

theorem sue_shoe_probability :
  let sueShoes := ShoeDistribution.mk 7 4 2 1
  samePairProbability sueShoes = 20 / 63 := by
  sorry

end sue_shoe_probability_l2613_261350


namespace prob_two_red_or_blue_is_one_fifth_l2613_261315

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing two marbles sequentially without replacement
    where both marbles are either red or blue -/
def prob_two_red_or_blue (counts : MarbleCounts) : ℚ :=
  let total := counts.red + counts.blue + counts.green
  let red_or_blue := counts.red + counts.blue
  (red_or_blue / total) * ((red_or_blue - 1) / (total - 1))

/-- Theorem stating that the probability of drawing two red or blue marbles
    from a bag with 4 red, 3 blue, and 8 green marbles is 1/5 -/
theorem prob_two_red_or_blue_is_one_fifth :
  prob_two_red_or_blue ⟨4, 3, 8⟩ = 1 / 5 := by
  sorry

end prob_two_red_or_blue_is_one_fifth_l2613_261315
