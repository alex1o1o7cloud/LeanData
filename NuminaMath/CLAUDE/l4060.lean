import Mathlib

namespace half_of_recipe_l4060_406064

theorem half_of_recipe (original_recipe : ℚ) (half_recipe : ℚ) : 
  original_recipe = 4.5 → half_recipe = original_recipe / 2 → half_recipe = 2.25 := by
  sorry

end half_of_recipe_l4060_406064


namespace salary_decrease_percentage_l4060_406090

theorem salary_decrease_percentage (x : ℝ) : 
  (100 - x) / 100 * 130 / 100 = 65 / 100 → x = 50 := by
  sorry

end salary_decrease_percentage_l4060_406090


namespace iphone_savings_l4060_406092

def iphone_cost : ℝ := 600
def discount_rate : ℝ := 0.05
def num_phones : ℕ := 3

def individual_cost : ℝ := iphone_cost * num_phones
def discounted_cost : ℝ := individual_cost * (1 - discount_rate)
def savings : ℝ := individual_cost - discounted_cost

theorem iphone_savings : savings = 90 := by
  sorry

end iphone_savings_l4060_406092


namespace license_plate_combinations_l4060_406096

def num_consonants : ℕ := 21
def num_vowels : ℕ := 6
def num_digits : ℕ := 10
def num_special_chars : ℕ := 3

theorem license_plate_combinations : 
  num_consonants * num_vowels * num_consonants * num_digits * num_special_chars = 79380 :=
by sorry

end license_plate_combinations_l4060_406096


namespace sqrt_sum_reciprocal_l4060_406052

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_sum_reciprocal_l4060_406052


namespace survey_ratings_l4060_406059

theorem survey_ratings (total : ℕ) (excellent_percent : ℚ) (satisfactory_remaining_percent : ℚ) (needs_improvement : ℕ) :
  total = 120 →
  excellent_percent = 15 / 100 →
  satisfactory_remaining_percent = 80 / 100 →
  needs_improvement = 6 →
  ∃ (very_satisfactory_percent : ℚ),
    very_satisfactory_percent = 16 / 100 ∧
    excellent_percent + very_satisfactory_percent + 
    (satisfactory_remaining_percent * (1 - excellent_percent - needs_improvement / total)) +
    (needs_improvement / total) = 1 :=
by sorry

end survey_ratings_l4060_406059


namespace taco_truck_beef_amount_l4060_406094

/-- The taco truck problem -/
theorem taco_truck_beef_amount :
  ∀ (beef_amount : ℝ),
    (beef_amount > 0) →
    (0.25 * (beef_amount / 0.25) * (2 - 1.5) = 200) →
    beef_amount = 100 :=
by
  sorry

end taco_truck_beef_amount_l4060_406094


namespace ping_pong_theorem_l4060_406012

/-- Represents a ping-pong match result between two players -/
inductive MatchResult
| Win
| Lose

/-- Represents a ping-pong team -/
def Team := Fin 1000

/-- Represents the result of all matches between two teams -/
def MatchResults := Team → Team → MatchResult

theorem ping_pong_theorem (results : MatchResults) : 
  ∃ (winning_team : Bool) (subset : Finset Team),
    subset.card ≤ 10 ∧ 
    ∀ (player : Team), 
      ∃ (winner : Team), winner ∈ subset ∧ 
        (if winning_team then 
          results winner player = MatchResult.Win
        else
          results player winner = MatchResult.Lose) :=
sorry

end ping_pong_theorem_l4060_406012


namespace problem_solution_l4060_406079

theorem problem_solution (x : ℝ) : 3 * x = (26 - x) + 10 → x = 9 := by
  sorry

end problem_solution_l4060_406079


namespace inscribed_cube_volume_l4060_406048

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (sphere_diameter : ℝ) 
  (inner_cube_edge : ℝ) (inner_cube_volume : ℝ) :
  outer_cube_edge = 12 →
  sphere_diameter = outer_cube_edge →
  sphere_diameter = inner_cube_edge * Real.sqrt 3 →
  inner_cube_volume = inner_cube_edge ^ 3 →
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l4060_406048


namespace calculation_proof_l4060_406069

theorem calculation_proof (initial_amount : ℝ) (first_percentage : ℝ) 
  (discount_percentage : ℝ) (target_percentage : ℝ) (tax_rate : ℝ) : 
  initial_amount = 4000 ∧ 
  first_percentage = 0.15 ∧ 
  discount_percentage = 0.25 ∧ 
  target_percentage = 0.07 ∧ 
  tax_rate = 0.10 → 
  (1 + tax_rate) * (target_percentage * (1 - discount_percentage) * (first_percentage * initial_amount)) = 34.65 := by
sorry

#eval (1 + 0.10) * (0.07 * (1 - 0.25) * (0.15 * 4000))

end calculation_proof_l4060_406069


namespace power_difference_equality_l4060_406016

theorem power_difference_equality : (3^2)^3 - (2^3)^2 = 665 := by
  sorry

end power_difference_equality_l4060_406016


namespace line_parallel_to_polar_axis_l4060_406055

/-- Given a point P in polar coordinates (r, θ), prove that the equation r * sin(θ) = 1
    represents a line that passes through P and is parallel to the polar axis. -/
theorem line_parallel_to_polar_axis 
  (r : ℝ) (θ : ℝ) (h1 : r = 2) (h2 : θ = π / 6) :
  r * Real.sin θ = 1 := by sorry

end line_parallel_to_polar_axis_l4060_406055


namespace integer_sum_l4060_406086

theorem integer_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end integer_sum_l4060_406086


namespace march_first_is_sunday_l4060_406083

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific March with given properties -/
structure SpecificMarch where
  daysInMonth : Nat
  wednesdayCount : Nat
  saturdayCount : Nat
  firstDay : DayOfWeek

/-- Theorem stating that March 1st is a Sunday given the conditions -/
theorem march_first_is_sunday (march : SpecificMarch) 
  (h1 : march.daysInMonth = 31)
  (h2 : march.wednesdayCount = 4)
  (h3 : march.saturdayCount = 4) :
  march.firstDay = DayOfWeek.Sunday := by
  sorry

#check march_first_is_sunday

end march_first_is_sunday_l4060_406083


namespace equation_solution_l4060_406062

theorem equation_solution : ∃ x : ℝ, (((32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2)) ^ 2 = (1024 : ℝ) ^ (2 * x - 1)) ∧ x = 1/8 := by
  sorry

end equation_solution_l4060_406062


namespace steps_per_level_l4060_406099

theorem steps_per_level (total_blocks : ℕ) (blocks_per_step : ℕ) (levels : ℕ) : 
  total_blocks = 96 → blocks_per_step = 3 → levels = 4 → 
  (total_blocks / blocks_per_step) / levels = 8 := by
  sorry

end steps_per_level_l4060_406099


namespace quadratic_inequality_roots_l4060_406051

theorem quadratic_inequality_roots (k : ℝ) : 
  (∀ x, x^2 + k*x + 24 > 0 ↔ x < -6 ∨ x > 4) → k = 2 := by
  sorry

end quadratic_inequality_roots_l4060_406051


namespace fixed_point_exponential_function_l4060_406057

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 3 + a^(x + 2)
  f (-2) = 4 := by sorry

end fixed_point_exponential_function_l4060_406057


namespace apples_per_person_l4060_406095

/-- Given that Harold gave apples to 3.0 people and the total number of apples given was 45,
    prove that each person received 15 apples. -/
theorem apples_per_person (total_apples : ℕ) (num_people : ℝ) 
    (h1 : total_apples = 45) 
    (h2 : num_people = 3.0) : 
  (total_apples : ℝ) / num_people = 15 := by
  sorry

end apples_per_person_l4060_406095


namespace equation_solution_l4060_406026

theorem equation_solution (x y : ℚ) : 
  (0.009 / x = 0.01 / y) → (x + y = 50) → (x = 450 / 19 ∧ y = 500 / 19) := by
  sorry

end equation_solution_l4060_406026


namespace largest_integer_with_remainder_l4060_406019

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 80 ∧ n % 8 = 5 ∧ ∀ m, m < 80 ∧ m % 8 = 5 → m ≤ n → n = 77 := by
  sorry

end largest_integer_with_remainder_l4060_406019


namespace sum_of_roots_quadratic_l4060_406039

theorem sum_of_roots_quadratic (α β : ℝ) : 
  (∀ x : ℝ, x^2 + x - 2 = 0 ↔ x = α ∨ x = β) →
  α + β = -1 := by
sorry

end sum_of_roots_quadratic_l4060_406039


namespace stirling_second_kind_recurrence_stirling_second_kind_5_3_l4060_406077

def S (n k : ℕ) : ℕ := sorry

theorem stirling_second_kind_recurrence (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  S (n + 1) k = S n (k - 1) + k * S n k := by sorry

theorem stirling_second_kind_5_3 :
  S 5 3 = 25 := by sorry

end stirling_second_kind_recurrence_stirling_second_kind_5_3_l4060_406077


namespace apple_banana_ratio_l4060_406098

/-- Proves that the ratio of apples to bananas is 2:1 given the total number of fruits,
    number of bananas, and number of oranges in a bowl of fruit. -/
theorem apple_banana_ratio (total : ℕ) (bananas : ℕ) (oranges : ℕ)
    (h_total : total = 12)
    (h_bananas : bananas = 2)
    (h_oranges : oranges = 6) :
    (total - bananas - oranges) / bananas = 2 := by
  sorry

end apple_banana_ratio_l4060_406098


namespace right_triangle_perimeter_l4060_406044

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  (1/2) * a * b = (1/2) * c →  -- Area condition
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
by sorry

end right_triangle_perimeter_l4060_406044


namespace unique_solution_l4060_406032

/-- The price of Lunasa's violin -/
def violin_price : ℝ := sorry

/-- The price of Merlin's trumpet -/
def trumpet_price : ℝ := sorry

/-- The price of Lyrica's piano -/
def piano_price : ℝ := sorry

/-- Condition (a): If violin price is raised by 50% and trumpet price is decreased by 50%,
    violin is $50 more expensive than trumpet -/
axiom condition_a : 1.5 * violin_price = 0.5 * trumpet_price + 50

/-- Condition (b): If trumpet price is raised by 50% and piano price is decreased by 50%,
    trumpet is $50 more expensive than piano -/
axiom condition_b : 1.5 * trumpet_price = 0.5 * piano_price + 50

/-- The percentage m by which violin price is raised and piano price is decreased -/
def m : ℤ := sorry

/-- The price difference n between the adjusted violin and piano prices -/
def n : ℤ := sorry

/-- The relationship between adjusted violin and piano prices -/
axiom price_relationship : (100 + m) * violin_price / 100 = n + (100 - m) * piano_price / 100

theorem unique_solution : m = 80 ∧ n = 80 := by sorry

end unique_solution_l4060_406032


namespace race_win_probability_l4060_406009

theorem race_win_probability (pA pB pC pD pE : ℚ) 
  (hA : pA = 1/8) (hB : pB = 1/12) (hC : pC = 1/15) (hD : pD = 1/18) (hE : pE = 1/20)
  (h_mutually_exclusive : ∀ (x y : Fin 5), x ≠ y → pA + pB + pC + pD + pE ≤ 1) :
  pA + pB + pC + pD + pE = 137/360 := by
sorry

end race_win_probability_l4060_406009


namespace residue_of_negative_1001_mod_37_l4060_406056

theorem residue_of_negative_1001_mod_37 :
  -1001 ≡ 35 [ZMOD 37] := by sorry

end residue_of_negative_1001_mod_37_l4060_406056


namespace rectangle_longer_side_l4060_406085

theorem rectangle_longer_side (r : ℝ) (h1 : r = 6) (h2 : r > 0) : ∃ l w : ℝ,
  l > w ∧ w = 2 * r ∧ l * w = 2 * (π * r^2) ∧ l = 6 * π :=
sorry

end rectangle_longer_side_l4060_406085


namespace half_cutting_line_exists_l4060_406088

/-- Triangle ABC with vertices A(0, 10), B(4, 0), and C(10, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- The area of a triangle given its vertices -/
def triangle_area (t : Triangle) : ℝ := sorry

/-- Check if a line cuts a triangle in half -/
def cuts_in_half (l : Line) (t : Triangle) : Prop := sorry

/-- The theorem stating the existence of a line that cuts the triangle in half
    and the sum of its slope and y-intercept -/
theorem half_cutting_line_exists (t : Triangle) 
  (h1 : t.A = (0, 10))
  (h2 : t.B = (4, 0))
  (h3 : t.C = (10, 0)) :
  ∃ l : Line, cuts_in_half l t ∧ l.slope + l.y_intercept = 5.625 := by
  sorry

end half_cutting_line_exists_l4060_406088


namespace gdp_scientific_notation_equality_l4060_406068

-- Define the GDP value in yuan
def gdp : ℝ := 121 * 10^12

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.21 * 10^14

-- Theorem stating that the GDP is equal to its scientific notation representation
theorem gdp_scientific_notation_equality : gdp = scientific_notation := by
  sorry

end gdp_scientific_notation_equality_l4060_406068


namespace remainder_of_2583156_div_4_l4060_406030

theorem remainder_of_2583156_div_4 : 2583156 % 4 = 0 := by
  sorry

end remainder_of_2583156_div_4_l4060_406030


namespace our_circle_center_and_radius_l4060_406084

/-- A circle in the 2D plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle --/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle --/
def Circle.radius (c : Circle) : ℝ := sorry

/-- Our specific circle --/
def our_circle : Circle :=
  { equation := fun x y => x^2 + y^2 - 4*x - 6*y - 3 = 0 }

theorem our_circle_center_and_radius :
  Circle.center our_circle = (2, 3) ∧ Circle.radius our_circle = 4 := by sorry

end our_circle_center_and_radius_l4060_406084


namespace min_value_squared_sum_l4060_406027

theorem min_value_squared_sum (a b t s : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 := by
  sorry

end min_value_squared_sum_l4060_406027


namespace chord_ratio_constant_l4060_406008

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a chord of the ellipse
def chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

-- Define parallel lines
def parallel (A B M N : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (N.1 - M.1) = (B.1 - A.1) * (N.2 - M.2)

-- Define a point on a line passing through the origin
def through_origin (M N : ℝ × ℝ) : Prop :=
  M.2 * N.1 = M.1 * N.2

-- Main theorem
theorem chord_ratio_constant
  (A B M N : ℝ × ℝ)
  (h_AB : chord A B)
  (h_MN : chord M N)
  (h_parallel : parallel A B M N)
  (h_origin : through_origin M N) :
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1/4 * ((N.1 - M.1)^2 + (N.2 - M.2)^2)^2 :=
sorry

end chord_ratio_constant_l4060_406008


namespace amir_weight_l4060_406054

theorem amir_weight (bulat_weight ilnur_weight : ℝ) : 
  let amir_weight := ilnur_weight + 8
  let daniyar_weight := bulat_weight + 4
  -- The sum of the weights of the heaviest and lightest boys is 2 kg less than the sum of the weights of the other two boys
  (amir_weight + ilnur_weight = daniyar_weight + bulat_weight - 2) →
  -- All four boys together weigh 250 kg
  (amir_weight + ilnur_weight + daniyar_weight + bulat_weight = 250) →
  amir_weight = 67 := by
sorry

end amir_weight_l4060_406054


namespace nancy_crystal_sets_l4060_406018

/-- The cost of one set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of one set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends in dollars -/
def total_spent : ℕ := 29

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

theorem nancy_crystal_sets : 
  crystal_cost * crystal_sets + metal_cost * metal_sets = total_spent := by
  sorry

end nancy_crystal_sets_l4060_406018


namespace sum_x_coordinates_preserved_l4060_406061

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon from the midpoints of another polygon's sides -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_preserved (n : ℕ) (q1 q2 q3 : Polygon) :
  n = 44 →
  q1.vertices.length = n →
  sumXCoordinates q1 = 176 →
  q2 = midpointPolygon q1 →
  q3 = midpointPolygon q2 →
  sumXCoordinates q3 = 176 := by
  sorry

end sum_x_coordinates_preserved_l4060_406061


namespace tan_alpha_eq_two_implies_fraction_eq_negative_two_l4060_406000

theorem tan_alpha_eq_two_implies_fraction_eq_negative_two (α : Real) 
  (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end tan_alpha_eq_two_implies_fraction_eq_negative_two_l4060_406000


namespace sum_of_powers_l4060_406074

theorem sum_of_powers (w : ℂ) (hw : w^3 + w^2 + 1 = 0) :
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 := by
  sorry

end sum_of_powers_l4060_406074


namespace quadratic_inequality_l4060_406038

/-- A quadratic function with a negative leading coefficient -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry (b c : ℝ) : ℝ := 2

theorem quadratic_inequality (b c : ℝ) :
  f b c (axis_of_symmetry b c + 2) < f b c (axis_of_symmetry b c - 1) ∧
  f b c (axis_of_symmetry b c - 1) < f b c (axis_of_symmetry b c) :=
by sorry

end quadratic_inequality_l4060_406038


namespace distinct_values_of_c_l4060_406015

/-- Given a complex number c and distinct complex numbers r, s, and t satisfying
    (z - r)(z - s)(z - t) = (z - 2cr)(z - 2cs)(z - 2ct) for all complex z,
    there are exactly 3 distinct possible values of c. -/
theorem distinct_values_of_c (c r s t : ℂ) : 
  r ≠ s ∧ s ≠ t ∧ r ≠ t →
  (∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2*c*r) * (z - 2*c*s) * (z - 2*c*t)) →
  ∃! (values : Finset ℂ), values.card = 3 ∧ c ∈ values := by
  sorry

end distinct_values_of_c_l4060_406015


namespace somu_present_age_l4060_406060

/-- Somu's present age -/
def somu_age : ℕ := sorry

/-- Somu's father's present age -/
def father_age : ℕ := sorry

/-- Somu's age is one-third of his father's age -/
axiom current_age_ratio : somu_age = father_age / 3

/-- 9 years ago, Somu was one-fifth of his father's age -/
axiom past_age_ratio : somu_age - 9 = (father_age - 9) / 5

theorem somu_present_age : somu_age = 18 := by sorry

end somu_present_age_l4060_406060


namespace flower_count_l4060_406089

theorem flower_count (bees : ℕ) (diff : ℕ) : bees = 3 → diff = 2 → bees + diff = 5 := by
  sorry

end flower_count_l4060_406089


namespace treasury_problem_l4060_406035

theorem treasury_problem (T : ℚ) : 
  (T - T / 13 - (T - T / 13) / 17 = 150) → 
  T = 172 + 21 / 32 :=
by sorry

end treasury_problem_l4060_406035


namespace max_distance_circle_ellipse_l4060_406002

/-- The maximum distance between any point on the circle x^2 + (y-6)^2 = 2 
    and any point on the ellipse x^2/10 + y^2 = 1 is 6√2 -/
theorem max_distance_circle_ellipse : 
  ∃ (max_dist : ℝ), max_dist = 6 * Real.sqrt 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + (y₁ - 6)^2 = 2) →  -- Point on the circle
    (x₂^2 / 10 + y₂^2 = 1) →   -- Point on the ellipse
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ max_dist :=
by sorry

end max_distance_circle_ellipse_l4060_406002


namespace complex_equality_implies_ratio_one_l4060_406043

theorem complex_equality_implies_ratio_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.I : ℂ)^4 = -1 → (a + b * Complex.I)^4 = (a - b * Complex.I)^4 → b / a = 1 := by
  sorry

end complex_equality_implies_ratio_one_l4060_406043


namespace vector_calculation_l4060_406041

/-- Given vectors a, b, and c in ℝ³, prove that a + 2b - 3c equals (-7, -1, -1) -/
theorem vector_calculation (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (2, 0, 1)) 
  (hb : b = (-3, 1, -1)) 
  (hc : c = (1, 1, 0)) : 
  a + 2 • b - 3 • c = (-7, -1, -1) := by
  sorry

end vector_calculation_l4060_406041


namespace triangle_area_theorem_l4060_406078

def triangle_vertex (y : ℝ) : ℝ × ℝ := (0, y)

theorem triangle_area_theorem (y : ℝ) (h1 : y < 0) :
  let v1 : ℝ × ℝ := (8, 6)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := triangle_vertex y
  let area : ℝ := (1/2) * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)
  area = 24 → y = -4.8 := by sorry

end triangle_area_theorem_l4060_406078


namespace total_amount_is_117_l4060_406065

/-- Represents the distribution of money among three parties -/
structure Distribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total amount distributed -/
def total_amount (d : Distribution) : ℝ := d.x + d.y + d.z

/-- Theorem: Given the conditions, the total amount is 117 rupees -/
theorem total_amount_is_117 (d : Distribution) 
  (h1 : d.y = 27)  -- y's share is 27 rupees
  (h2 : d.y = 0.45 * d.x)  -- y gets 45 paisa for each rupee x gets
  (h3 : d.z = 0.50 * d.x)  -- z gets 50 paisa for each rupee x gets
  : total_amount d = 117 := by
  sorry


end total_amount_is_117_l4060_406065


namespace total_wheels_in_both_garages_l4060_406036

-- Define the types of vehicles
inductive Vehicle
| Bicycle
| Tricycle
| Unicycle
| Quadracycle

-- Define the garage contents
def first_garage : List (Vehicle × Nat) :=
  [(Vehicle.Bicycle, 5), (Vehicle.Tricycle, 6), (Vehicle.Unicycle, 9), (Vehicle.Quadracycle, 3)]

def second_garage : List (Vehicle × Nat) :=
  [(Vehicle.Bicycle, 2), (Vehicle.Tricycle, 1), (Vehicle.Unicycle, 3), (Vehicle.Quadracycle, 4)]

-- Define the number of wheels for each vehicle type
def wheels_per_vehicle (v : Vehicle) : Nat :=
  match v with
  | Vehicle.Bicycle => 2
  | Vehicle.Tricycle => 3
  | Vehicle.Unicycle => 1
  | Vehicle.Quadracycle => 4

-- Define the number of missing wheels in the second garage
def missing_wheels : Nat := 3

-- Function to calculate total wheels in a garage
def total_wheels_in_garage (garage : List (Vehicle × Nat)) : Nat :=
  garage.foldl (fun acc (v, count) => acc + wheels_per_vehicle v * count) 0

-- Theorem statement
theorem total_wheels_in_both_garages :
  total_wheels_in_garage first_garage +
  total_wheels_in_garage second_garage - missing_wheels = 72 := by
  sorry

end total_wheels_in_both_garages_l4060_406036


namespace ironman_age_l4060_406075

/-- Represents the ages of the characters in the problem -/
structure Ages where
  thor : ℕ
  captainAmerica : ℕ
  peterParker : ℕ
  ironman : ℕ

/-- The conditions of the problem -/
def problemConditions (ages : Ages) : Prop :=
  ages.thor = 13 * ages.captainAmerica ∧
  ages.captainAmerica = 7 * ages.peterParker ∧
  ages.ironman = ages.peterParker + 32 ∧
  ages.thor = 1456

/-- The theorem to be proved -/
theorem ironman_age (ages : Ages) :
  problemConditions ages → ages.ironman = 48 := by
  sorry

end ironman_age_l4060_406075


namespace custom_op_value_l4060_406011

/-- Custom operation * for non-zero integers -/
def custom_op (a b : ℤ) : ℚ := 1 / a + 1 / b

theorem custom_op_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 32) :
  custom_op a b = 3 / 8 := by
  sorry

end custom_op_value_l4060_406011


namespace journey_proof_l4060_406025

/-- Represents the distance-time relationship for a journey -/
def distance_from_destination (total_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  total_distance - speed * time

theorem journey_proof (total_distance : ℝ) (speed : ℝ) (time : ℝ) 
  (h1 : total_distance = 174)
  (h2 : speed = 60)
  (h3 : time = 1.5) :
  distance_from_destination total_distance speed time = 84 := by
  sorry

#check journey_proof

end journey_proof_l4060_406025


namespace james_college_cost_l4060_406040

/-- The cost of James's community college units over 2 semesters -/
theorem james_college_cost (units_per_semester : ℕ) (cost_per_unit : ℕ) (num_semesters : ℕ) : 
  units_per_semester = 20 → cost_per_unit = 50 → num_semesters = 2 →
  units_per_semester * cost_per_unit * num_semesters = 2000 := by
  sorry

#check james_college_cost

end james_college_cost_l4060_406040


namespace sequence_sum_divisible_by_37_l4060_406017

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Represents a sequence of four three-digit integers -/
structure FourTermSequence where
  term1 : ThreeDigitInt
  term2 : ThreeDigitInt
  term3 : ThreeDigitInt
  term4 : ThreeDigitInt
  satisfies_property : 
    term2.hundreds = term1.tens ∧ term2.tens = term1.units ∧
    term3.hundreds = term2.tens ∧ term3.tens = term2.units ∧
    term4.hundreds = term3.tens ∧ term4.tens = term3.units ∧
    term1.hundreds = term4.tens ∧ term1.tens = term4.units

/-- The sum of all terms in the sequence -/
def sequence_sum (seq : FourTermSequence) : Nat :=
  (seq.term1.hundreds * 100 + seq.term1.tens * 10 + seq.term1.units) +
  (seq.term2.hundreds * 100 + seq.term2.tens * 10 + seq.term2.units) +
  (seq.term3.hundreds * 100 + seq.term3.tens * 10 + seq.term3.units) +
  (seq.term4.hundreds * 100 + seq.term4.tens * 10 + seq.term4.units)

theorem sequence_sum_divisible_by_37 (seq : FourTermSequence) :
  37 ∣ sequence_sum seq := by
  sorry

end sequence_sum_divisible_by_37_l4060_406017


namespace minimum_cost_theorem_l4060_406063

/-- Represents the flower planting problem with given conditions -/
structure FlowerPlanting where
  costA3B4 : ℕ  -- Cost for 3 pots of A and 4 pots of B
  costA4B3 : ℕ  -- Cost for 4 pots of A and 3 pots of B
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℚ  -- Survival rate of type A flowers
  survivalRateB : ℚ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Calculates the cost per pot for type A and B flowers -/
def calculateCostPerPot (fp : FlowerPlanting) : ℚ × ℚ := sorry

/-- Calculates the minimum cost and optimal planting strategy -/
def minimumCostStrategy (fp : FlowerPlanting) : ℕ × ℕ × ℕ := sorry

/-- Theorem stating the minimum cost and optimal planting strategy -/
theorem minimum_cost_theorem (fp : FlowerPlanting) 
  (h1 : fp.costA3B4 = 330)
  (h2 : fp.costA4B3 = 300)
  (h3 : fp.totalPots = 400)
  (h4 : fp.survivalRateA = 7/10)
  (h5 : fp.survivalRateB = 9/10)
  (h6 : fp.maxReplacement = 80) :
  minimumCostStrategy fp = (200, 200, 18000) := by sorry

end minimum_cost_theorem_l4060_406063


namespace wicket_keeper_age_difference_l4060_406042

theorem wicket_keeper_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 28)
  (h3 : team_avg_age = 25)
  (h4 : (team_size * team_avg_age - captain_age - wicket_keeper_age) / (team_size - 2) = team_avg_age - 1) :
  wicket_keeper_age - captain_age = 3 :=
by sorry

end wicket_keeper_age_difference_l4060_406042


namespace x_gt_3_sufficient_not_necessary_for_x_squared_gt_9_l4060_406082

theorem x_gt_3_sufficient_not_necessary_for_x_squared_gt_9 :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ (∃ x : ℝ, x^2 > 9 ∧ x ≤ 3) := by
  sorry

end x_gt_3_sufficient_not_necessary_for_x_squared_gt_9_l4060_406082


namespace number_solution_l4060_406020

theorem number_solution (x : ℝ) (n : ℝ) (h1 : x > 0) (h2 : x / n + x / 25 = 0.06 * x) : n = 50 := by
  sorry

end number_solution_l4060_406020


namespace smallest_solution_floor_equation_l4060_406029

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 119 ∧
  (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧
  (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 19) := by
  sorry

end smallest_solution_floor_equation_l4060_406029


namespace monkey_climb_theorem_l4060_406004

/-- Calculates the time for a monkey to climb a tree given the tree height,
    hop distance, slip distance, and net climb rate per hour. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (net_climb_rate : ℕ) : ℕ :=
  (tree_height - hop_distance) / net_climb_rate + 1

theorem monkey_climb_theorem (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) :
  tree_height = 22 →
  hop_distance = 3 →
  slip_distance = 2 →
  monkey_climb_time tree_height hop_distance slip_distance (hop_distance - slip_distance) = 20 := by
  sorry

#eval monkey_climb_time 22 3 2 1

end monkey_climb_theorem_l4060_406004


namespace range_of_f_l4060_406076

-- Define the function
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ Set.Iic 8 :=
by sorry

-- Note: Set.Iic 8 represents the set (-∞, 8]

end range_of_f_l4060_406076


namespace min_value_of_f_l4060_406081

/-- The quadratic function we want to minimize -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 10

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -2/3 ∧ ∀ (x y : ℝ), f x y ≥ min := by sorry

end min_value_of_f_l4060_406081


namespace solar_system_median_moons_l4060_406031

/-- Represents the number of moons for each planet in the solar system -/
def moon_counts : List Nat := [0, 1, 1, 3, 3, 6, 8, 14, 18, 21]

/-- Calculates the median of a sorted list with an even number of elements -/
def median (l : List Nat) : Rat :=
  let n := l.length
  if n % 2 = 0 then
    let mid := n / 2
    (l.get! (mid - 1) + l.get! mid) / 2
  else
    l.get! (n / 2)

theorem solar_system_median_moons :
  median moon_counts = 4.5 := by sorry

end solar_system_median_moons_l4060_406031


namespace initial_mixture_volume_l4060_406046

/-- Given a mixture where water is 10% of the total volume, if adding 14 liters of water
    results in a new mixture with 25% water, then the initial volume of the mixture was 70 liters. -/
theorem initial_mixture_volume (V : ℝ) : 
  (0.1 * V + 14) / (V + 14) = 0.25 → V = 70 := by
  sorry

end initial_mixture_volume_l4060_406046


namespace cylinder_surface_area_l4060_406022

/-- The surface area of a cylinder with diameter 4 units and height 3 units is 20π square units. -/
theorem cylinder_surface_area : 
  let d : ℝ := 4  -- diameter
  let h : ℝ := 3  -- height
  let r : ℝ := d / 2  -- radius
  let surface_area : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  surface_area = 20 * Real.pi :=
by sorry

end cylinder_surface_area_l4060_406022


namespace jogger_distance_l4060_406066

/-- Proves that given a jogger who jogs at 12 km/hr, if jogging at 20 km/hr would result in 15 km 
    more distance covered, then the actual distance jogged is 22.5 km. -/
theorem jogger_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  actual_speed = 12 →
  faster_speed = 20 →
  faster_speed * (extra_distance / (faster_speed - actual_speed)) = 
    actual_speed * (extra_distance / (faster_speed - actual_speed)) + extra_distance →
  extra_distance = 15 →
  actual_speed * (extra_distance / (faster_speed - actual_speed)) = 22.5 :=
by sorry


end jogger_distance_l4060_406066


namespace salary_spending_problem_l4060_406080

/-- The problem statement about salaries and spending --/
theorem salary_spending_problem 
  (total_salary : ℝ)
  (a_salary : ℝ)
  (a_spend_percent : ℝ)
  (ha_total : total_salary = 6000)
  (ha_salary : a_salary = 4500)
  (ha_spend : a_spend_percent = 0.95)
  (h_equal_savings : a_salary * (1 - a_spend_percent) = (total_salary - a_salary) - ((total_salary - a_salary) * (85 / 100))) :
  (((total_salary - a_salary) - ((total_salary - a_salary) * (1 - 85 / 100))) / (total_salary - a_salary)) * 100 = 85 := by
sorry


end salary_spending_problem_l4060_406080


namespace will_old_cards_l4060_406067

/-- Calculates the number of old baseball cards Will had. -/
def old_cards (cards_per_page : ℕ) (new_cards : ℕ) (pages_used : ℕ) : ℕ :=
  cards_per_page * pages_used - new_cards

/-- Theorem stating that Will had 10 old cards. -/
theorem will_old_cards : old_cards 3 8 6 = 10 := by
  sorry

end will_old_cards_l4060_406067


namespace rectangle_cover_cost_l4060_406097

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    the total cost to cover the rectangle at $5 per square centimeter is $8000. -/
theorem rectangle_cover_cost (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) :
  5 * (l * w) = 8000 := by
  sorry

end rectangle_cover_cost_l4060_406097


namespace truncated_cone_complete_height_l4060_406047

/-- Given a truncated cone with height h, upper radius r, and lower radius R,
    the height H of the corresponding complete cone is hR / (R - r) -/
theorem truncated_cone_complete_height
  (h r R : ℝ) (h_pos : h > 0) (r_pos : r > 0) (R_pos : R > 0) (r_lt_R : r < R) :
  ∃ H : ℝ, H = h * R / (R - r) ∧ H > h := by
  sorry

#check truncated_cone_complete_height

end truncated_cone_complete_height_l4060_406047


namespace a_power_six_bounds_l4060_406045

theorem a_power_six_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := by
  sorry

end a_power_six_bounds_l4060_406045


namespace isbn_problem_l4060_406013

/-- ISBN check digit calculation -/
def isbn_check_digit (A B C D E F G H I : ℕ) : ℕ :=
  let S := 10*A + 9*B + 8*C + 7*D + 6*E + 5*F + 4*G + 3*H + 2*I
  let r := S % 11
  if r = 0 then 0
  else if r = 1 then 10  -- Represented by 'x' in the problem
  else 11 - r

/-- The problem statement -/
theorem isbn_problem (y : ℕ) : 
  isbn_check_digit 9 6 2 y 7 0 7 0 1 = 5 → y = 7 := by
sorry

end isbn_problem_l4060_406013


namespace sum_of_coefficients_l4060_406023

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (x - 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) → 
  a₀ + a₂ + a₄ = -122 := by
  sorry

end sum_of_coefficients_l4060_406023


namespace square_root_of_nine_l4060_406071

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end square_root_of_nine_l4060_406071


namespace number_of_divisors_210_l4060_406058

theorem number_of_divisors_210 : Nat.card (Nat.divisors 210) = 16 := by
  sorry

end number_of_divisors_210_l4060_406058


namespace ramu_car_price_l4060_406091

/-- Proves that given the conditions of Ramu's car purchase, repair, and sale,
    the original price he paid for the car is 42000 rupees. -/
theorem ramu_car_price :
  let repair_cost : ℝ := 12000
  let selling_price : ℝ := 64900
  let profit_percent : ℝ := 20.185185185185187
  let original_price : ℝ := 42000
  (selling_price = original_price + repair_cost + (original_price + repair_cost) * (profit_percent / 100)) →
  original_price = 42000 :=
by
  sorry

#check ramu_car_price

end ramu_car_price_l4060_406091


namespace policeman_speed_l4060_406037

/-- Proves that given the initial conditions of a chase between a policeman and a thief,
    the policeman's speed is 64 km/hr. -/
theorem policeman_speed (initial_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ) :
  initial_distance = 160 →
  thief_speed = 8 →
  thief_distance = 640 →
  ∃ (policeman_speed : ℝ), policeman_speed = 64 :=
by
  sorry


end policeman_speed_l4060_406037


namespace smallest_perfect_square_divisible_by_5_and_7_l4060_406049

theorem smallest_perfect_square_divisible_by_5_and_7 :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), n = k^2) ∧ 5 ∣ n ∧ 7 ∣ n ∧
  ∀ (m : ℕ), m > 0 → (∃ (j : ℕ), m = j^2) → 5 ∣ m → 7 ∣ m → m ≥ n :=
by
  -- The proof goes here
  sorry

end smallest_perfect_square_divisible_by_5_and_7_l4060_406049


namespace hyperbola_eccentricity_l4060_406014

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- An asymptote of the hyperbola -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) := sorry

/-- The projection of a point onto a line -/
def project (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The area of a triangle formed by three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let A := left_vertex h
  let F := right_focus h
  let asym := asymptote h
  let B := project A asym
  let Q := project F asym
  let O := (0, 0)
  triangle_area A B O / triangle_area F Q O = 1 / 2 →
  eccentricity h = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l4060_406014


namespace point_on_line_with_given_y_l4060_406024

/-- A straight line in the xy-plane with given slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_on_line_with_given_y (l : Line) (p : Point) :
  l.slope = 4 →
  l.yIntercept = 100 →
  p.y = 300 →
  pointOnLine l p →
  p.x = 50 := by
  sorry

#check point_on_line_with_given_y

end point_on_line_with_given_y_l4060_406024


namespace line_y_axis_intersection_l4060_406050

/-- The line equation 2y - 5x = 10 -/
def line_equation (x y : ℝ) : Prop := 2 * y - 5 * x = 10

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line and the y-axis -/
def intersection_point : ℝ × ℝ := (0, 5)

theorem line_y_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_y_axis x y :=
by sorry

end line_y_axis_intersection_l4060_406050


namespace max_distinct_numbers_in_circle_l4060_406033

/-- Given a circular arrangement of 2023 numbers where each number is the product of its two neighbors,
    the maximum number of distinct numbers is 1. -/
theorem max_distinct_numbers_in_circle (nums : Fin 2023 → ℝ) 
    (h : ∀ i : Fin 2023, nums i = nums (i - 1) * nums (i + 1)) : 
    Finset.card (Finset.image nums Finset.univ) = 1 := by
  sorry

end max_distinct_numbers_in_circle_l4060_406033


namespace reciprocal_equals_self_l4060_406001

theorem reciprocal_equals_self (x : ℝ) : (1 / x = x) → (x = 1 ∨ x = -1) := by
  sorry

end reciprocal_equals_self_l4060_406001


namespace problem_solution_l4060_406034

theorem problem_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a * Real.sin (π / 7) + b * Real.cos (π / 7)) / 
       (a * Real.cos (π / 7) - b * Real.sin (π / 7)) = Real.tan (10 * π / 21)) : 
  b / a = Real.sqrt 3 := by
sorry

end problem_solution_l4060_406034


namespace sum_of_roots_difference_l4060_406028

theorem sum_of_roots_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) : x + y = 25 := by
  sorry

end sum_of_roots_difference_l4060_406028


namespace trigonometric_simplification_l4060_406093

theorem trigonometric_simplification (α : ℝ) :
  (-Real.sin (π + α) + Real.sin (-α) - Real.tan (2*π + α)) /
  (Real.tan (α + π) + Real.cos (-α) + Real.cos (π - α)) = -1 := by
  sorry

end trigonometric_simplification_l4060_406093


namespace fruit_distribution_l4060_406021

theorem fruit_distribution (num_students : ℕ) 
  (h1 : 2 * num_students + 6 = num_apples)
  (h2 : 7 * num_students - 5 = num_oranges)
  (h3 : num_oranges = 3 * num_apples + 3) : 
  num_students = 26 := by
sorry

end fruit_distribution_l4060_406021


namespace speaker_arrangement_count_l4060_406010

def number_of_speakers : ℕ := 5

theorem speaker_arrangement_count :
  (number_of_speakers.factorial / 2) = 60 := by
  sorry

end speaker_arrangement_count_l4060_406010


namespace percentage_not_french_l4060_406070

def total_students : ℕ := 200
def french_and_english : ℕ := 25
def french_not_english : ℕ := 65

theorem percentage_not_french : 
  (total_students - (french_and_english + french_not_english)) * 100 / total_students = 55 := by
  sorry

end percentage_not_french_l4060_406070


namespace range_of_x_less_than_6_range_of_a_for_f_less_than_a_l4060_406073

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

-- Theorem for part I
theorem range_of_x_less_than_6 :
  ∀ x : ℝ, f x < 6 ↔ x ∈ Set.Ioo (-2 : ℝ) 4 :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_less_than_a :
  ∀ a : ℝ, (∃ x : ℝ, f x < a) ↔ a ∈ Set.Ioi 4 :=
sorry

end range_of_x_less_than_6_range_of_a_for_f_less_than_a_l4060_406073


namespace book_cost_price_l4060_406007

theorem book_cost_price (cost : ℝ) : cost = 300 :=
  by
  have h1 : 1.12 * cost + 18 = 1.18 * cost := by sorry
  sorry

end book_cost_price_l4060_406007


namespace unique_five_digit_number_l4060_406003

theorem unique_five_digit_number : ∀ N : ℕ,
  (10000 ≤ N ∧ N < 100000) →
  let P := 200000 + N
  let Q := 10 * N + 2
  Q = 3 * P →
  N = 85714 :=
by
  sorry

end unique_five_digit_number_l4060_406003


namespace simplify_expressions_l4060_406053

theorem simplify_expressions :
  ∀ (x a b : ℝ),
    (x^2 + (3*x - 5) - (4*x - 1) = x^2 - x - 4) ∧
    (7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b) := by
  sorry

end simplify_expressions_l4060_406053


namespace arithmetic_mean_of_fractions_l4060_406005

theorem arithmetic_mean_of_fractions : 
  (1 / 3 : ℚ) * ((3 / 7 : ℚ) + (5 / 9 : ℚ) + (2 / 3 : ℚ)) = 104 / 189 := by
  sorry

end arithmetic_mean_of_fractions_l4060_406005


namespace f_properties_triangle_property_l4060_406006

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f x ≤ max_value) ∧
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧
    max_value = 2 ∧
    max_set = {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} := by sorry

theorem triangle_property :
  ∀ (A B C : ℝ) (a b c : ℝ),
    a = 1 → b = Real.sqrt 3 → f A = 2 →
    (A + B + C = Real.pi) →
    (Real.sin A / a = Real.sin B / b) →
    (Real.sin B / b = Real.sin C / c) →
    (C = Real.pi / 6 ∨ C = Real.pi / 2) := by sorry

end f_properties_triangle_property_l4060_406006


namespace total_diagonals_specific_prism_l4060_406072

/-- A rectangular prism with edge lengths a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of face diagonals in a rectangular prism. -/
def face_diagonals (p : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism. -/
def space_diagonals (p : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism. -/
def total_diagonals (p : RectangularPrism) : ℕ :=
  face_diagonals p + space_diagonals p

/-- Theorem: The total number of diagonals in a rectangular prism
    with edge lengths 4, 6, and 8 is 16. -/
theorem total_diagonals_specific_prism :
  ∃ p : RectangularPrism, p.a = 4 ∧ p.b = 6 ∧ p.c = 8 ∧ total_diagonals p = 16 := by
  sorry

end total_diagonals_specific_prism_l4060_406072


namespace set_intersection_example_l4060_406087

theorem set_intersection_example :
  let A : Set ℤ := {1, 0, 3}
  let B : Set ℤ := {-1, 1, 2, 3}
  A ∩ B = {1, 3} := by
sorry

end set_intersection_example_l4060_406087
