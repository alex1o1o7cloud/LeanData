import Mathlib

namespace special_function_value_l549_54988

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 2

/-- The main theorem -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) (h₀ : f 1 = 0) : 
  f 2010 = 4018 := by
  sorry

end special_function_value_l549_54988


namespace kirill_height_difference_l549_54926

theorem kirill_height_difference (combined_height kirill_height : ℕ) 
  (h1 : combined_height = 112)
  (h2 : kirill_height = 49) :
  combined_height - kirill_height - kirill_height = 14 := by
  sorry

end kirill_height_difference_l549_54926


namespace negation_equivalence_l549_54951

theorem negation_equivalence : 
  (¬(∀ x : ℝ, |x| < 2 → x < 2)) ↔ (∀ x : ℝ, |x| ≥ 2 → x ≥ 2) := by
  sorry

end negation_equivalence_l549_54951


namespace partial_fraction_decomposition_l549_54902

theorem partial_fraction_decomposition (A B : ℝ) :
  (∀ x : ℝ, (2*x + 1) / ((x + 1) * (x + 2)) = A / (x + 1) + B / (x + 2)) →
  A = -1 ∧ B = 3 := by
sorry

end partial_fraction_decomposition_l549_54902


namespace pair_in_six_cascades_valid_coloring_exists_l549_54994

-- Define a cascade
def cascade (r : ℕ) : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≤ 12 ∧ n = k * r}

-- Part a: Existence of a pair in six cascades
theorem pair_in_six_cascades : ∃ a b : ℕ, ∃ r₁ r₂ r₃ r₄ r₅ r₆ : ℕ,
  r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
  r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
  r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
  r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
  r₅ ≠ r₆ ∧
  a ∈ cascade r₁ ∧ b ∈ cascade r₁ ∧
  a ∈ cascade r₂ ∧ b ∈ cascade r₂ ∧
  a ∈ cascade r₃ ∧ b ∈ cascade r₃ ∧
  a ∈ cascade r₄ ∧ b ∈ cascade r₄ ∧
  a ∈ cascade r₅ ∧ b ∈ cascade r₅ ∧
  a ∈ cascade r₆ ∧ b ∈ cascade r₆ := by
  sorry

-- Part b: Existence of a valid coloring function
theorem valid_coloring_exists : ∃ f : ℕ → Fin 12, ∀ r : ℕ, ∀ k₁ k₂ : ℕ,
  k₁ ≤ 12 → k₂ ≤ 12 → k₁ ≠ k₂ → f (k₁ * r) ≠ f (k₂ * r) := by
  sorry

end pair_in_six_cascades_valid_coloring_exists_l549_54994


namespace arithmetic_sequence_eighth_term_l549_54939

/-- Given an arithmetic sequence where the first term is 10/11 and the fifteenth term is 8/9,
    the eighth term is 89/99. -/
theorem arithmetic_sequence_eighth_term 
  (a : ℕ → ℚ)  -- a is the sequence
  (h1 : a 1 = 10 / 11)  -- first term is 10/11
  (h15 : a 15 = 8 / 9)  -- fifteenth term is 8/9
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  : a 8 = 89 / 99 := by
  sorry

end arithmetic_sequence_eighth_term_l549_54939


namespace kims_morning_routine_l549_54970

/-- Kim's morning routine calculation -/
theorem kims_morning_routine (coffee_time : ℕ) (status_update_time : ℕ) (payroll_update_time : ℕ) (num_employees : ℕ) :
  coffee_time = 5 →
  status_update_time = 2 →
  payroll_update_time = 3 →
  num_employees = 9 →
  coffee_time + num_employees * (status_update_time + payroll_update_time) = 50 := by
  sorry

#check kims_morning_routine

end kims_morning_routine_l549_54970


namespace sqrt_mixed_number_simplification_l549_54960

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end sqrt_mixed_number_simplification_l549_54960


namespace modified_cube_surface_area_l549_54995

/-- Represents the modified cube structure --/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCubes : Nat
  cornerSize : Nat

/-- Calculates the surface area of the modified cube structure --/
def surfaceArea (c : ModifiedCube) : Nat :=
  let remainingCubes := 27 - c.removedCubes
  let visibleCornersPerCube := 4
  let surfaceUnitsPerCorner := 3
  remainingCubes * visibleCornersPerCube * surfaceUnitsPerCorner

/-- The theorem to be proved --/
theorem modified_cube_surface_area :
  ∀ (c : ModifiedCube),
  c.initialSize = 6 ∧
  c.smallCubeSize = 2 ∧
  c.removedCubes = 7 ∧
  c.cornerSize = 1 →
  surfaceArea c = 240 := by
  sorry


end modified_cube_surface_area_l549_54995


namespace correct_guess_probability_l549_54971

/-- The number of possible digits in a phone number -/
def num_digits : ℕ := 10

/-- The length of a phone number -/
def phone_number_length : ℕ := 7

/-- The probability of correctly guessing a single unknown digit in a phone number -/
def probability_correct_guess : ℚ := 1 / num_digits

theorem correct_guess_probability : 
  probability_correct_guess = 1 / num_digits :=
by sorry

end correct_guess_probability_l549_54971


namespace crabapple_sequences_count_l549_54918

/-- The number of students in each class -/
def students_per_class : ℕ := 8

/-- The number of meetings per week for each class -/
def meetings_per_week : ℕ := 3

/-- The number of classes -/
def number_of_classes : ℕ := 2

/-- The total number of sequences of crabapple recipients for both classes in a week -/
def total_sequences : ℕ := (students_per_class ^ meetings_per_week) ^ number_of_classes

/-- Theorem stating that the total number of sequences is 262,144 -/
theorem crabapple_sequences_count : total_sequences = 262144 := by
  sorry

end crabapple_sequences_count_l549_54918


namespace two_lines_forming_30_degrees_l549_54917

/-- Represents a line in 3D space -/
structure Line3D where
  -- Define necessary properties for a line

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Define necessary properties for a plane

/-- Angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem two_lines_forming_30_degrees (a : Line3D) (α : Plane3D) (P : Point3D) :
  angle_line_plane a α = 30 →
  ∃! (s : Finset Line3D), 
    s.card = 2 ∧ 
    ∀ b ∈ s, line_passes_through b P ∧ 
              angle_between_lines a b = 30 ∧ 
              angle_line_plane b α = 30 :=
sorry

end two_lines_forming_30_degrees_l549_54917


namespace total_miles_ridden_l549_54912

-- Define the given conditions
def miles_to_school : ℕ := 6
def miles_from_school : ℕ := 7
def trips_per_week : ℕ := 5

-- Define the theorem to prove
theorem total_miles_ridden : 
  miles_to_school * trips_per_week + miles_from_school * trips_per_week = 65 := by
  sorry

end total_miles_ridden_l549_54912


namespace mary_initial_money_l549_54992

/-- The amount of money Mary had before buying the pie -/
def initial_money : ℕ := sorry

/-- The cost of the pie -/
def pie_cost : ℕ := 6

/-- The amount of money Mary has after buying the pie -/
def remaining_money : ℕ := 52

theorem mary_initial_money : 
  initial_money = remaining_money + pie_cost := by sorry

end mary_initial_money_l549_54992


namespace zero_point_implies_a_range_l549_54987

theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, a * x + 1 - 2 * a = 0) → 
  a ∈ Set.Ioo (1/3 : ℝ) 1 := by
sorry

end zero_point_implies_a_range_l549_54987


namespace subtraction_division_equality_l549_54923

theorem subtraction_division_equality : 6000 - (105 / 21.0) = 5995 := by
  sorry

end subtraction_division_equality_l549_54923


namespace inverse_of_B_cubed_l549_54903

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, -2; 0, -1]) : 
  (B^3)⁻¹ = !![27, -24; 0, -1] := by sorry

end inverse_of_B_cubed_l549_54903


namespace model_height_is_correct_l549_54968

/-- The height of the actual observatory tower in meters -/
def actual_height : ℝ := 60

/-- The volume of water the actual observatory tower can hold in liters -/
def actual_volume : ℝ := 200000

/-- The volume of water Carson's miniature model can hold in liters -/
def model_volume : ℝ := 0.2

/-- The height of Carson's miniature tower in meters -/
def model_height : ℝ := 0.6

/-- Theorem stating that the calculated model height is correct -/
theorem model_height_is_correct :
  model_height = actual_height * (model_volume / actual_volume)^(1/3) :=
by sorry

end model_height_is_correct_l549_54968


namespace vegetarian_gluten_free_fraction_is_one_twentyfifth_l549_54962

/-- Represents the menu of a restaurant --/
structure Menu where
  total_dishes : ℕ
  vegetarian_dishes : ℕ
  gluten_free_vegetarian_dishes : ℕ

/-- The fraction of dishes that are both vegetarian and gluten-free --/
def vegetarian_gluten_free_fraction (menu : Menu) : ℚ :=
  menu.gluten_free_vegetarian_dishes / menu.total_dishes

/-- Theorem stating the fraction of vegetarian and gluten-free dishes --/
theorem vegetarian_gluten_free_fraction_is_one_twentyfifth 
  (menu : Menu) 
  (h1 : menu.vegetarian_dishes = 5)
  (h2 : menu.vegetarian_dishes = menu.total_dishes / 5)
  (h3 : menu.gluten_free_vegetarian_dishes = menu.vegetarian_dishes - 4) :
  vegetarian_gluten_free_fraction menu = 1 / 25 := by
  sorry

#check vegetarian_gluten_free_fraction_is_one_twentyfifth

end vegetarian_gluten_free_fraction_is_one_twentyfifth_l549_54962


namespace geometric_sequence_property_l549_54976

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 3) ^ 2 - 6 * (a 3) + 8 = 0 ∧
  (a 15) ^ 2 - 6 * (a 15) + 8 = 0 →
  (a 1 * a 17) / a 9 = 2 * Real.sqrt 2 :=
sorry

end geometric_sequence_property_l549_54976


namespace point_position_on_line_l549_54964

/-- Given points on a line, prove the position of a point P satisfying a ratio condition -/
theorem point_position_on_line 
  (O A B C D P : ℝ) 
  (h_order : O ≤ A ∧ A ≤ B ∧ B ≤ C ∧ C ≤ D)
  (h_dist_OA : A - O = a)
  (h_dist_OB : B - O = b)
  (h_dist_OC : C - O = c)
  (h_dist_OD : D - O = d)
  (h_P_between : B ≤ P ∧ P ≤ C)
  (h_ratio : (P - A) / (D - P) = 2 * ((P - B) / (C - P))) :
  P - O = b + c - a :=
sorry

end point_position_on_line_l549_54964


namespace domain_implies_a_eq_3_odd_function_implies_a_eq_1_odd_function_solution_set_l549_54928

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 / (x - 1)) + a)

-- Define the domain condition
def domain_condition (a : ℝ) : Prop :=
  ∀ x, f a x ≠ 0 ↔ (x < 1/3 ∨ x > 1)

-- Define the odd function condition
def odd_function (a : ℝ) : Prop :=
  ∀ x, f a (-x) = -(f a x)

-- State the theorems
theorem domain_implies_a_eq_3 :
  ∃ a, domain_condition a → a = 3 :=
sorry

theorem odd_function_implies_a_eq_1 :
  ∃ a, odd_function a → a = 1 :=
sorry

theorem odd_function_solution_set :
  ∀ a, odd_function a →
    (∀ x, f a x > 0 ↔ x > 1) :=
sorry

end domain_implies_a_eq_3_odd_function_implies_a_eq_1_odd_function_solution_set_l549_54928


namespace percent_profit_calculation_l549_54909

/-- Given that the cost price of 75 articles after a 5% discount
    equals the selling price of 60 articles before a 12% sales tax,
    prove that the percent profit is 25%. -/
theorem percent_profit_calculation (CP : ℝ) (SP : ℝ) :
  75 * CP * (1 - 0.05) = 60 * SP →
  (SP - CP * (1 - 0.05)) / (CP * (1 - 0.05)) * 100 = 25 := by
  sorry

end percent_profit_calculation_l549_54909


namespace tara_spent_more_on_ice_cream_l549_54984

/-- The amount Tara spent more on ice cream than on yogurt -/
def ice_cream_yogurt_difference : ℕ :=
  let ice_cream_cartons : ℕ := 19
  let yogurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 7
  let yogurt_price : ℕ := 1
  (ice_cream_cartons * ice_cream_price) - (yogurt_cartons * yogurt_price)

/-- Theorem stating that Tara spent $129 more on ice cream than on yogurt -/
theorem tara_spent_more_on_ice_cream : ice_cream_yogurt_difference = 129 := by
  sorry

end tara_spent_more_on_ice_cream_l549_54984


namespace volume_theorem_l549_54963

noncomputable def volume_of_body : ℝ :=
  let surface1 (x y z : ℝ) := 2 * z = x^2 + y^2
  let surface2 (z : ℝ) := z = 2
  let surface3 (x : ℝ) := x = 0
  let surface4 (x y : ℝ) := y = 2 * x
  let arctan2 := Real.arctan 2
  2 * arctan2

theorem volume_theorem :
  volume_of_body = 1.704 * Real.pi := by sorry

end volume_theorem_l549_54963


namespace derivative_of_odd_is_even_l549_54955

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The derivative of an odd function is an even function -/
theorem derivative_of_odd_is_even (f : ℝ → ℝ) (hf : IsOdd f) (hf' : Differentiable ℝ f) :
  IsEven (deriv f) := by
  sorry

end derivative_of_odd_is_even_l549_54955


namespace line_AB_intersects_S₂_and_S_l549_54914

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def S₁ : Circle := { center := (0, 0), radius := 1 }
def S₂ : Circle := { center := (2, 0), radius := 1 }
def S  : Circle := { center := (1, 1), radius := 2 }
def A  : ℝ × ℝ := (1, 0)
def B  : ℝ × ℝ := (1, 2)
def O  : ℝ × ℝ := S.center

-- Define the conditions
axiom S₁_S₂_tangent : S₁.center.fst + S₁.radius = S₂.center.fst - S₂.radius
axiom O_on_S₁ : (O.fst - S₁.center.fst)^2 + (O.snd - S₁.center.snd)^2 = S₁.radius^2
axiom S₁_S_tangent_at_B : (B.fst - S₁.center.fst)^2 + (B.snd - S₁.center.snd)^2 = S₁.radius^2 ∧
                          (B.fst - S.center.fst)^2 + (B.snd - S.center.snd)^2 = S.radius^2

-- Theorem to prove
theorem line_AB_intersects_S₂_and_S :
  ∃ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B ∧
  (∃ (t : ℝ), P = (A.fst + t * (B.fst - A.fst), A.snd + t * (B.snd - A.snd))) ∧
  (P.fst - S₂.center.fst)^2 + (P.snd - S₂.center.snd)^2 = S₂.radius^2 ∧
  (P.fst - S.center.fst)^2 + (P.snd - S.center.snd)^2 = S.radius^2 :=
sorry

end line_AB_intersects_S₂_and_S_l549_54914


namespace triangle_property_l549_54982

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle angle condition
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / Real.sin A = b / Real.sin B →  -- Law of sines (partial)
  a / Real.sin A = c / Real.sin C →  -- Law of sines (partial)
  (2 * c + b) * Real.cos A + a * Real.cos B = 0 →  -- Given equation
  a = Real.sqrt 3 →  -- Given side length
  A = 2 * π / 3 ∧ Real.sqrt 3 < 2 * b + c ∧ 2 * b + c < 2 * Real.sqrt 3 := by
sorry

end triangle_property_l549_54982


namespace sum_division_problem_l549_54930

/-- Proof of the total amount in a sum division problem -/
theorem sum_division_problem (x y z total : ℚ) : 
  y = 0.45 * x →  -- For each rupee x gets, y gets 45 paisa
  z = 0.5 * x →   -- For each rupee x gets, z gets 50 paisa
  y = 18 →        -- The share of y is Rs. 18
  total = x + y + z →  -- The total is the sum of all shares
  total = 78 := by  -- The total amount is Rs. 78
sorry


end sum_division_problem_l549_54930


namespace total_colors_needed_l549_54950

/-- Represents the number of moons for each planet in the solar system -/
def moons : Fin 8 → ℕ
  | 0 => 0  -- Mercury
  | 1 => 0  -- Venus
  | 2 => 1  -- Earth
  | 3 => 2  -- Mars
  | 4 => 79 -- Jupiter
  | 5 => 82 -- Saturn
  | 6 => 27 -- Uranus
  | 7 => 14 -- Neptune

/-- The number of planets in the solar system -/
def num_planets : ℕ := 8

/-- The number of people coloring -/
def num_people : ℕ := 3

/-- The total number of celestial bodies (planets and moons) -/
def total_bodies : ℕ := num_planets + (Finset.sum Finset.univ moons)

/-- Theorem stating the total number of colors needed -/
theorem total_colors_needed : num_people * total_bodies = 639 := by
  sorry


end total_colors_needed_l549_54950


namespace min_value_expression_l549_54956

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≥ 6 / (2 * Real.sqrt 3 + Real.sqrt 6) ∧
  (∃ x₀ > 0, (x₀^2 + 3 - Real.sqrt (x₀^4 + 9)) / x₀ = 6 / (2 * Real.sqrt 3 + Real.sqrt 6)) :=
by sorry

end min_value_expression_l549_54956


namespace large_cube_surface_area_l549_54920

theorem large_cube_surface_area 
  (num_small_cubes : ℕ) 
  (small_cube_edge : ℝ) 
  (large_cube_edge : ℝ) :
  num_small_cubes = 27 →
  small_cube_edge = 4 →
  large_cube_edge = small_cube_edge * (num_small_cubes ^ (1/3 : ℝ)) →
  6 * large_cube_edge^2 = 864 := by
  sorry

end large_cube_surface_area_l549_54920


namespace vertex_locus_is_hyperbola_l549_54958

/-- The locus of the vertex of a parabola is a hyperbola -/
theorem vertex_locus_is_hyperbola 
  (a b : ℝ) 
  (h : 8 * a^2 + 4 * a * b = b^3) : 
  ∃ (x y : ℝ), x * y = 1 ∧ 
  x = -b / (2 * a) ∧ 
  y = (4 * a - b^2) / (4 * a) := by
  sorry

end vertex_locus_is_hyperbola_l549_54958


namespace log_5_125000_bounds_l549_54913

theorem log_5_125000_bounds : ∃ (a b : ℤ), 
  (a : ℝ) < Real.log 125000 / Real.log 5 ∧ 
  Real.log 125000 / Real.log 5 < (b : ℝ) ∧ 
  a = 6 ∧ 
  b = 7 ∧ 
  a + b = 13 := by
sorry

end log_5_125000_bounds_l549_54913


namespace fifth_subject_score_l549_54979

theorem fifth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℚ) :
  s1 = 50 →
  s2 = 60 →
  s3 = 70 →
  s4 = 80 →
  avg = 68 →
  (s1 + s2 + s3 + s4 : ℚ) / 4 + 80 / 5 = avg :=
by sorry

end fifth_subject_score_l549_54979


namespace square_starts_with_123456789_l549_54967

theorem square_starts_with_123456789 : ∃ (n : ℕ) (k : ℕ), 
  (123456789 : ℕ) * 10^k ≤ n^2 ∧ n^2 < (123456790 : ℕ) * 10^k :=
sorry

end square_starts_with_123456789_l549_54967


namespace original_blueberry_count_l549_54954

/-- Represents the number of blueberry jelly beans Camilla originally had -/
def blueberry : ℕ := sorry

/-- Represents the number of cherry jelly beans Camilla originally had -/
def cherry : ℕ := sorry

/-- Theorem stating the original number of blueberry jelly beans -/
theorem original_blueberry_count : blueberry = 30 := by
  have h1 : blueberry = 3 * cherry := sorry
  have h2 : blueberry - 20 = 2 * (cherry - 5) := sorry
  sorry


end original_blueberry_count_l549_54954


namespace coefficient_of_expansion_l549_54944

theorem coefficient_of_expansion (x : ℝ) : 
  ∃ a b c d e : ℝ, (2*x + 1)^5 = a + b*(x+1) + c*(x+1)^2 + d*(x+1)^3 + e*(x+1)^4 + (-5)*(x+1)^5 := by
  sorry

end coefficient_of_expansion_l549_54944


namespace base10_157_equals_base12_B21_l549_54922

/-- Converts a base 12 number to base 10 -/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 12 + d) 0

/-- Represents 'B' as 11 in base 12 -/
def baseB : Nat := 11

theorem base10_157_equals_base12_B21 :
  157 = base12ToBase10 [baseB, 2, 1] := by
  sorry

end base10_157_equals_base12_B21_l549_54922


namespace assembly_line_theorem_l549_54916

/-- Represents the assembly line production --/
structure AssemblyLine where
  initial_rate : ℕ
  initial_order : ℕ
  increased_rate : ℕ
  second_order : ℕ

/-- Calculates the overall average output of the assembly line --/
def average_output (line : AssemblyLine) : ℚ :=
  let total_cogs := line.initial_order + line.second_order
  let total_time := (line.initial_order : ℚ) / line.initial_rate + (line.second_order : ℚ) / line.increased_rate
  total_cogs / total_time

/-- Theorem stating that the average output for the given conditions is 40 cogs per hour --/
theorem assembly_line_theorem (line : AssemblyLine) 
    (h1 : line.initial_rate = 30)
    (h2 : line.initial_order = 60)
    (h3 : line.increased_rate = 60)
    (h4 : line.second_order = 60) :
  average_output line = 40 := by
  sorry

end assembly_line_theorem_l549_54916


namespace parallelogram_vertex_sum_l549_54929

structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def is_valid_parallelogram (p : Parallelogram) : Prop :=
  p.A.1 = 2 ∧ p.A.2 = -3 ∧
  p.B.1 = 7 ∧ p.B.2 = 0 ∧
  p.D.1 = -2 ∧ p.D.2 = 5 ∧
  (p.A.1 + p.D.1) / 2 = (p.B.1 + p.C.1) / 2 ∧
  (p.A.2 + p.D.2) / 2 = (p.B.2 + p.C.2) / 2

theorem parallelogram_vertex_sum (p : Parallelogram) 
  (h : is_valid_parallelogram p) : p.C.1 + p.C.2 = -5 := by
  sorry

end parallelogram_vertex_sum_l549_54929


namespace sum_is_non_horizontal_line_l549_54969

-- Define the original parabola
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define f(x) as the reflection and left translation of the original parabola
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c (x + 5)

-- Define g(x) as the reflection and right translation of the original parabola
def g (a b c : ℝ) (x : ℝ) : ℝ := -original_parabola a b c (x - 5)

-- Define the sum function (f + g)(x)
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

-- Theorem: The sum function (f + g)(x) is a non-horizontal line
theorem sum_is_non_horizontal_line (a b c : ℝ) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x : ℝ, f_plus_g a b c x = m * x + k :=
sorry

end sum_is_non_horizontal_line_l549_54969


namespace total_points_is_24_l549_54911

/-- Calculates points earned based on pounds recycled and points per set of pounds -/
def calculatePoints (pounds : ℕ) (poundsPerSet : ℕ) (pointsPerSet : ℕ) : ℕ :=
  (pounds / poundsPerSet) * pointsPerSet

/-- Represents the recycling problem and calculates total points -/
def recyclingProblem : ℕ :=
  let gwenPoints := calculatePoints 12 4 2
  let lisaPoints := calculatePoints 25 5 3
  let jackPoints := calculatePoints 21 7 1
  gwenPoints + lisaPoints + jackPoints

/-- Theorem stating that the total points earned is 24 -/
theorem total_points_is_24 : recyclingProblem = 24 := by
  sorry


end total_points_is_24_l549_54911


namespace chocolate_chip_cookies_l549_54986

theorem chocolate_chip_cookies (cookies_per_bag : ℕ) (baggies : ℕ) (oatmeal_cookies : ℕ) :
  cookies_per_bag = 5 →
  baggies = 7 →
  oatmeal_cookies = 2 →
  cookies_per_bag * baggies - oatmeal_cookies = 33 :=
by sorry

end chocolate_chip_cookies_l549_54986


namespace point_q_coordinates_l549_54977

/-- A point on the unit circle --/
structure PointOnUnitCircle where
  x : ℝ
  y : ℝ
  on_circle : x^2 + y^2 = 1

/-- The arc length between two points on the unit circle --/
def arcLength (p q : PointOnUnitCircle) : ℝ := sorry

theorem point_q_coordinates :
  ∀ (p q : PointOnUnitCircle),
  p.x = 1 ∧ p.y = 0 →
  arcLength p q = π / 3 →
  q.x = 1 / 2 ∧ q.y = Real.sqrt 3 / 2 := by sorry

end point_q_coordinates_l549_54977


namespace maplewood_population_estimate_l549_54940

theorem maplewood_population_estimate :
  ∀ (avg_population : ℝ),
  (25 : ℝ) > 0 →
  6200 ≤ avg_population →
  avg_population ≤ 6800 →
  ∃ (total_population : ℝ),
  total_population = 25 * avg_population ∧
  total_population = 162500 :=
by sorry

end maplewood_population_estimate_l549_54940


namespace student_average_equals_actual_average_l549_54927

theorem student_average_equals_actual_average 
  (w x y z : ℤ) (h : w < x ∧ x < y ∧ y < z) :
  (((w + x) / 2 + (y + z) / 2) / 2 : ℚ) = ((w + x + y + z) / 4 : ℚ) := by
  sorry

end student_average_equals_actual_average_l549_54927


namespace last_digit_n_power_9999_minus_5555_l549_54980

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_n_power_9999_minus_5555 (n : ℕ) : 
  last_digit (n^9999 - n^5555) = 0 :=
sorry

end last_digit_n_power_9999_minus_5555_l549_54980


namespace gcd_228_1995_l549_54934

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l549_54934


namespace jessie_friends_l549_54961

/-- The number of friends Jessie invited -/
def num_friends (total_muffins : ℕ) (muffins_per_person : ℕ) : ℕ :=
  total_muffins / muffins_per_person - 1

/-- Theorem stating that Jessie invited 4 friends -/
theorem jessie_friends : num_friends 20 4 = 4 := by
  sorry

end jessie_friends_l549_54961


namespace intersection_implies_m_equals_one_l549_54946

theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
sorry

end intersection_implies_m_equals_one_l549_54946


namespace multiple_of_p_plus_q_l549_54924

theorem multiple_of_p_plus_q (p q : ℚ) (h : p / q = 3 / 11) :
  ∃ m : ℤ, m * p + q = 17 ∧ m = 2 := by
  sorry

end multiple_of_p_plus_q_l549_54924


namespace equal_distribution_problem_l549_54947

theorem equal_distribution_problem (earnings : Fin 5 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 27)
  (h3 : earnings 2 = 30)
  (h4 : earnings 3 = 35)
  (h5 : earnings 4 = 50) :
  50 - (earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5 = 18 := by
  sorry

end equal_distribution_problem_l549_54947


namespace jinyoung_has_fewest_l549_54915

/-- Represents the number of marbles each person has -/
structure Marbles where
  seonho : ℕ
  minjeong : ℕ
  jinyoung : ℕ
  joohwan : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  m.seonho = m.minjeong + 1 ∧
  m.jinyoung = m.joohwan - 3 ∧
  m.minjeong = 6 ∧
  m.joohwan = 7

/-- Jinyoung has the fewest marbles -/
theorem jinyoung_has_fewest (m : Marbles) (h : marble_conditions m) :
  m.jinyoung ≤ m.seonho ∧ m.jinyoung ≤ m.minjeong ∧ m.jinyoung ≤ m.joohwan :=
by sorry

end jinyoung_has_fewest_l549_54915


namespace equality_of_expressions_l549_54966

theorem equality_of_expressions (x : ℝ) (hx : x > 0) :
  x^(x+1) + x^(x+1) = 2*x^(x+1) ∧
  x^(x+1) + x^(x+1) ≠ x^(2*x+2) ∧
  x^(x+1) + x^(x+1) ≠ (2*x)^(x+1) ∧
  x^(x+1) + x^(x+1) ≠ (2*x)^(2*x+2) :=
by sorry

end equality_of_expressions_l549_54966


namespace one_fourth_of_six_point_eight_l549_54997

theorem one_fourth_of_six_point_eight : (1 / 4 : ℚ) * (68 / 10 : ℚ) = 17 / 10 := by
  sorry

end one_fourth_of_six_point_eight_l549_54997


namespace intersection_when_a_neg_two_subset_condition_l549_54900

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: When a = -2, A ∩ B = {x | -5 ≤ x < -1}
theorem intersection_when_a_neg_two :
  A (-2) ∩ B = {x : ℝ | -5 ≤ x ∧ x < -1} :=
sorry

-- Theorem 2: A ⊆ B if and only if a ≤ -4 or a ≥ 3
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 3 :=
sorry

end intersection_when_a_neg_two_subset_condition_l549_54900


namespace three_mn_odd_l549_54983

theorem three_mn_odd (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  (3 * m * n) % 2 = 1 := by
  sorry

end three_mn_odd_l549_54983


namespace perfect_square_factors_count_l549_54973

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_perfect_square_factors (a b c d : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1) * (d + 1)

theorem perfect_square_factors_count :
  count_perfect_square_factors 6 7 8 4 = 2520 := by sorry

end perfect_square_factors_count_l549_54973


namespace inequality_proof_l549_54931

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) + a + b + c > 3 := by
sorry

end inequality_proof_l549_54931


namespace f_monotone_increasing_interval_l549_54901

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) := x^2 + 2*x + 1

/-- The monotonically increasing interval of f(x) is [-1, +∞) -/
theorem f_monotone_increasing_interval :
  ∀ x y : ℝ, x ≥ -1 → y ≥ -1 → x < y → f x < f y :=
by sorry

end f_monotone_increasing_interval_l549_54901


namespace simplify_and_rationalize_l549_54907

theorem simplify_and_rationalize (x : ℝ) (h : x^3 = 3) : 
  1 / (1 + 1 / (x + 1)) = 9 / 13 := by
  sorry

end simplify_and_rationalize_l549_54907


namespace tony_midpoint_age_l549_54996

/-- Represents Tony's age and earnings over a 60-day period --/
structure TonyEarnings where
  daysWorked : Nat
  hoursPerDay : Nat
  hourlyRateMultiplier : Rat
  startAge : Nat
  midAge : Nat
  endAge : Nat
  totalEarnings : Rat

/-- Calculates Tony's earnings based on his age and work details --/
def calculateEarnings (t : TonyEarnings) : Rat :=
  let firstHalfDays := t.daysWorked / 2
  let secondHalfDays := t.daysWorked - firstHalfDays
  (t.hoursPerDay * t.hourlyRateMultiplier * t.startAge * firstHalfDays : Rat) +
  (t.hoursPerDay * t.hourlyRateMultiplier * t.endAge * secondHalfDays : Rat)

/-- Theorem stating that Tony's age at the midpoint was 11 --/
theorem tony_midpoint_age (t : TonyEarnings) 
  (h1 : t.daysWorked = 60)
  (h2 : t.hoursPerDay = 3)
  (h3 : t.hourlyRateMultiplier = 3/4)
  (h4 : t.startAge = 10)
  (h5 : t.endAge = 12)
  (h6 : t.totalEarnings = 1125)
  (h7 : calculateEarnings t = t.totalEarnings) :
  t.midAge = 11 := by
  sorry

end tony_midpoint_age_l549_54996


namespace ball_distribution_ratio_l549_54981

/-- The number of ways to distribute n identical objects into k distinct bins --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical objects into k distinct bins,
    where each bin receives a specified number of objects --/
def distributeSpecific (n k : ℕ) (bins : Fin k → ℕ) : ℕ := sorry

theorem ball_distribution_ratio : 
  let total_balls : ℕ := 15
  let num_bins : ℕ := 4
  let pattern1 : Fin num_bins → ℕ := ![3, 6, 3, 3]
  let pattern2 : Fin num_bins → ℕ := ![3, 2, 3, 7]
  
  (distributeSpecific total_balls num_bins pattern1) / 
  (distributeSpecific total_balls num_bins pattern2) = 560 := by sorry

end ball_distribution_ratio_l549_54981


namespace betty_sugar_purchase_l549_54972

def min_sugar_purchase (s : ℕ) : Prop :=
  ∃ (f : ℝ),
    f ≥ 4 + s / 3 ∧
    f ≤ 3 * s ∧
    2 * s + 3 * f ≤ 36 ∧
    ∀ (s' : ℕ), s' < s → ¬∃ (f' : ℝ),
      f' ≥ 4 + s' / 3 ∧
      f' ≤ 3 * s' ∧
      2 * s' + 3 * f' ≤ 36

theorem betty_sugar_purchase : min_sugar_purchase 4 :=
sorry

end betty_sugar_purchase_l549_54972


namespace loan_principal_calculation_l549_54974

/-- Calculates the principal amount given the simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: If a loan at 12% annual simple interest generates Rs. 1500 interest in 10 years, 
    then the principal amount was Rs. 1250 -/
theorem loan_principal_calculation :
  let interest : ℚ := 1500
  let rate : ℚ := 12
  let time : ℚ := 10
  calculate_principal interest rate time = 1250 := by
sorry

end loan_principal_calculation_l549_54974


namespace volunteer_event_arrangements_l549_54953

/-- The number of ways to arrange volunteers for a 5-day event -/
def volunteer_arrangements (total_days : ℕ) (consecutive_days : ℕ) (total_people : ℕ) : ℕ :=
  (total_days - consecutive_days + 1) * (Nat.factorial (total_people - 1))

/-- Theorem: The number of arrangements for the volunteer event is 24 -/
theorem volunteer_event_arrangements :
  volunteer_arrangements 5 2 4 = 24 := by
  sorry

end volunteer_event_arrangements_l549_54953


namespace x_squared_plus_x_is_quadratic_binomial_l549_54985

/-- A quadratic binomial is a polynomial of degree 2 with two terms. -/
def is_quadratic_binomial (p : Polynomial ℝ) : Prop :=
  p.degree = 2 ∧ p.support.card = 2

/-- x^2 + x is a quadratic binomial -/
theorem x_squared_plus_x_is_quadratic_binomial :
  is_quadratic_binomial (X^2 + X : Polynomial ℝ) := by
  sorry

end x_squared_plus_x_is_quadratic_binomial_l549_54985


namespace arithmetic_sequence_sum_constant_l549_54937

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ+, S n = n^2 + 2*n + (S 1 - 3)) →
  S 1 - 3 = 0 := by
  sorry

end arithmetic_sequence_sum_constant_l549_54937


namespace min_value_squared_sum_l549_54906

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 16) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 64 ∧ 
  ∃ p q r s t u v w : ℝ, p * q * r * s = 16 ∧ t * u * v * w = 16 ∧ 
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 = 64 := by
  sorry

end min_value_squared_sum_l549_54906


namespace least_prime_factor_of_5_5_minus_5_4_l549_54998

theorem least_prime_factor_of_5_5_minus_5_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
sorry

end least_prime_factor_of_5_5_minus_5_4_l549_54998


namespace triangle_inequality_l549_54990

theorem triangle_inequality (a b c : ℝ) (α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hα : 0 < α ∧ α < π) (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
  (h_cosine : a^2 = b^2 + c^2 - 2*b*c*(Real.cos α)) :
  (2*b*c*(Real.cos α))/(b + c) < b + c - a ∧ b + c - a < (2*b*c)/a := by
  sorry

end triangle_inequality_l549_54990


namespace problem_solution_l549_54938

noncomputable section

-- Define the function f
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f a 2 x)

theorem problem_solution (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  -- Part 1: k = 2 if f is an odd function
  (∀ x, f a 2 x = -(f a 2 (-x))) →
  -- Part 2: If f(1) < 0, then f is decreasing and the inequality holds iff -3 < t < 5
  (f a 2 1 < 0 →
    (∀ x y, x < y → f a 2 x > f a 2 y) ∧
    (∀ t, (∀ x, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ -3 < t ∧ t < 5)) ∧
  -- Part 3: If f(1) = 3/2 and g has min value -2 on [1,+∞), then m = 2
  (f a 2 1 = 3/2 →
    (∃ m, (∀ x ≥ 1, g a m x ≥ -2) ∧
          (∃ x ≥ 1, g a m x = -2)) →
    m = 2) :=
by sorry

end problem_solution_l549_54938


namespace expected_remainder_mod_64_l549_54959

/-- The expected value of (a + 2b + 4c + 8d + 16e + 32f) mod 64, where a, b, c, d, e, f 
    are independently and uniformly randomly selected integers from {1,2,...,100} -/
theorem expected_remainder_mod_64 : 
  let S := Finset.range 100
  let M (a b c d e f : ℕ) := a + 2*b + 4*c + 8*d + 16*e + 32*f
  (S.sum (λ a => S.sum (λ b => S.sum (λ c => S.sum (λ d => S.sum (λ e => 
    S.sum (λ f => (M a b c d e f) % 64))))))) / S.card^6 = 63/2 := by
  sorry

end expected_remainder_mod_64_l549_54959


namespace max_squares_covered_two_inch_card_l549_54948

/-- Represents a square card with a given side length -/
structure SquareCard where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a checkerboard with squares of a given side length -/
structure Checkerboard where
  square_side_length : ℝ
  square_side_length_pos : square_side_length > 0

/-- The maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : SquareCard) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a 2-inch square card can cover at most 9 one-inch squares on a checkerboard -/
theorem max_squares_covered_two_inch_card :
  ∀ (card : SquareCard) (board : Checkerboard),
    card.side_length = 2 →
    board.square_side_length = 1 →
    max_squares_covered card board = 9 :=
  sorry

end max_squares_covered_two_inch_card_l549_54948


namespace greatest_four_digit_divisible_by_15_25_40_75_l549_54952

theorem greatest_four_digit_divisible_by_15_25_40_75 : ∃ n : ℕ,
  n ≤ 9999 ∧
  n ≥ 1000 ∧
  n % 15 = 0 ∧
  n % 25 = 0 ∧
  n % 40 = 0 ∧
  n % 75 = 0 ∧
  ∀ m : ℕ, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 15 = 0 ∧ m % 25 = 0 ∧ m % 40 = 0 ∧ m % 75 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

#eval 9600 -- Expected output: 9600

end greatest_four_digit_divisible_by_15_25_40_75_l549_54952


namespace intersection_implies_a_value_l549_54957

def M (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def P (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (M a) ∩ (P a) = {-3} → a = -1 := by
sorry

end intersection_implies_a_value_l549_54957


namespace inverse_variation_cube_l549_54989

/-- Given two positive real numbers x and y that vary inversely with respect to x^3,
    if y = 8 when x = 2, then x = 1 when y = 64. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ (k : ℝ), ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (x^3 * y)) :
  y = 64 → x = 1 := by
sorry

end inverse_variation_cube_l549_54989


namespace frog_edge_probability_l549_54943

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center : Position
| Edge : Position

/-- Represents the number of hops -/
def MaxHops : ℕ := 3

/-- The probability of reaching an edge from the center in n hops -/
def probability_reach_edge (n : ℕ) : ℚ :=
  sorry

/-- The 4x4 grid with wrapping and movement rules -/
structure Grid :=
  (size : ℕ := 4)
  (wrap : Bool := true)
  (diagonal_moves : Bool := false)

/-- Theorem: The probability of reaching an edge within 3 hops is 13/16 -/
theorem frog_edge_probability (g : Grid) : 
  probability_reach_edge MaxHops = 13/16 :=
sorry

end frog_edge_probability_l549_54943


namespace count_complementary_sets_l549_54905

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 4
  color : Fin 4
  shade : Fin 4

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet : Type := Finset Card

/-- Checks if a set of three cards is complementary -/
def isComplementary (set : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def ComplementarySets : Finset ThreeCardSet := sorry

theorem count_complementary_sets :
  Finset.card ComplementarySets = 360 := by sorry

end count_complementary_sets_l549_54905


namespace even_function_solution_set_l549_54908

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f x < 0}

theorem even_function_solution_set
  (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_zero : f (-4) = 0 ∧ f 2 = 0)
  (h_decreasing : ∀ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, x < y → f x > f y)
  (h_increasing : ∀ x ∈ Set.Ici 3, ∀ y ∈ Set.Ici 3, x < y → f x < f y) :
  solution_set f = Set.union (Set.union (Set.Iio (-4)) (Set.Ioo (-2) 0)) (Set.Ioo 2 4) :=
sorry

end even_function_solution_set_l549_54908


namespace z_in_second_quadrant_l549_54925

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition z / (1 + i) = 2i
axiom z_condition : z / (1 + Complex.I) = 2 * Complex.I

-- Define the second quadrant
def second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

-- Theorem statement
theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end z_in_second_quadrant_l549_54925


namespace investment_rate_problem_l549_54935

theorem investment_rate_problem (total_investment : ℝ) (first_investment : ℝ) (second_rate : ℝ) (total_interest : ℝ) :
  total_investment = 10000 →
  first_investment = 6000 →
  second_rate = 0.09 →
  total_interest = 840 →
  ∃ (r : ℝ),
    r * first_investment + second_rate * (total_investment - first_investment) = total_interest ∧
    r = 0.08 :=
by sorry

end investment_rate_problem_l549_54935


namespace total_time_in_hours_l549_54936

def laundry_time : ℕ := 30
def bathroom_cleaning_time : ℕ := 15
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40

def minutes_per_hour : ℕ := 60

theorem total_time_in_hours :
  (laundry_time + bathroom_cleaning_time + room_cleaning_time + homework_time) / minutes_per_hour = 2 := by
  sorry

end total_time_in_hours_l549_54936


namespace power_equality_l549_54942

theorem power_equality (m : ℕ) : 16^6 = 4^m → m = 12 := by
  sorry

end power_equality_l549_54942


namespace area_of_region_is_24_l549_54933

/-- The region in the plane defined by the given inequality -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (|p.1| + |3 * p.2| - 6) * (|3 * p.1| + |p.2| - 6) ≤ 0}

/-- The area of the region -/
def AreaOfRegion : ℝ := sorry

/-- Theorem stating that the area of the region is 24 -/
theorem area_of_region_is_24 : AreaOfRegion = 24 := by sorry

end area_of_region_is_24_l549_54933


namespace other_number_proof_l549_54975

/-- Given two positive integers with specified HCF and LCM, prove that if one number is 36, the other is 176. -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.gcd A B = 16 →
  Nat.lcm A B = 396 →
  A = 36 →
  B = 176 := by
  sorry

end other_number_proof_l549_54975


namespace complex_equation_solution_l549_54919

theorem complex_equation_solution (z : ℂ) : 3 + 2 * Complex.I * z = 7 - 4 * Complex.I * z ↔ z = -2 * Complex.I / 3 := by
  sorry

end complex_equation_solution_l549_54919


namespace amusement_park_cost_per_trip_l549_54910

/-- The cost per trip to an amusement park given the following conditions:
  - Two season passes are purchased
  - Each pass costs 100 units of currency
  - One person uses their pass 35 times
  - Another person uses their pass 15 times
-/
theorem amusement_park_cost_per_trip 
  (pass_cost : ℝ) 
  (trips_person1 : ℕ) 
  (trips_person2 : ℕ) : 
  pass_cost = 100 ∧ 
  trips_person1 = 35 ∧ 
  trips_person2 = 15 → 
  (2 * pass_cost) / (trips_person1 + trips_person2 : ℝ) = 4 := by
  sorry

#check amusement_park_cost_per_trip

end amusement_park_cost_per_trip_l549_54910


namespace man_double_son_age_l549_54999

/-- Represents the number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  2

/-- Theorem stating that it takes 2 years for the man's age to be twice his son's age -/
theorem man_double_son_age (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 24) 
  (h2 : age_difference = 26) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

#check man_double_son_age

end man_double_son_age_l549_54999


namespace tangent_slope_determines_point_l549_54993

/-- Given a curve y = 2x^2 + 4x, prove that if the slope of the tangent line
    at point P is 16, then the coordinates of P are (3, 30). -/
theorem tangent_slope_determines_point :
  ∀ x y : ℝ,
  (y = 2 * x^2 + 4 * x) →  -- Curve equation
  ((4 * x + 4) = 16) →     -- Slope of tangent line is 16
  (x = 3 ∧ y = 30)         -- Coordinates of point P
  := by sorry

end tangent_slope_determines_point_l549_54993


namespace frequency_of_fifth_group_l549_54978

theorem frequency_of_fifth_group 
  (total_students : ℕ) 
  (group1 group2 group3 group4 : ℕ) 
  (h1 : total_students = 40)
  (h2 : group1 = 12)
  (h3 : group2 = 10)
  (h4 : group3 = 6)
  (h5 : group4 = 8) :
  total_students - (group1 + group2 + group3 + group4) = 4 := by
  sorry

end frequency_of_fifth_group_l549_54978


namespace two_numbers_product_cube_sum_l549_54941

theorem two_numbers_product_cube_sum : ∃ (a b : ℚ), 
  (∃ (x : ℚ), a + (a * b) = x^3) ∧ 
  (∃ (y : ℚ), b + (a * b) = y^3) ∧ 
  a = 112/13 ∧ 
  b = 27/169 := by
sorry

end two_numbers_product_cube_sum_l549_54941


namespace cost_difference_formula_option_A_cheaper_at_50_l549_54949

/-- The number of teachers -/
def num_teachers : ℕ := 5

/-- The full ticket price -/
def full_price : ℕ := 40

/-- Cost calculation for Option A -/
def cost_A (x : ℕ) : ℕ := 20 * x + 200

/-- Cost calculation for Option B -/
def cost_B (x : ℕ) : ℕ := 24 * x + 120

/-- The cost difference between Option B and Option A -/
def cost_difference (x : ℕ) : ℤ := (cost_B x : ℤ) - (cost_A x : ℤ)

theorem cost_difference_formula (x : ℕ) : 
  cost_difference x = 4 * x - 80 :=
sorry

theorem option_A_cheaper_at_50 : 
  cost_A 50 < cost_B 50 :=
sorry

end cost_difference_formula_option_A_cheaper_at_50_l549_54949


namespace octagon_area_theorem_l549_54932

-- Define the rectangle BDEF
structure Rectangle :=
  (B D E F : ℝ × ℝ)

-- Define the octagon
structure Octagon :=
  (vertices : Fin 8 → ℝ × ℝ)

-- Define the condition AB = BC = 2
def side_length : ℝ := 2

-- Define the function to calculate the area of the octagon
noncomputable def octagon_area (rect : Rectangle) (side : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem octagon_area_theorem (rect : Rectangle) :
  octagon_area rect side_length = 16 + 8 * Real.sqrt 2 :=
sorry

end octagon_area_theorem_l549_54932


namespace value_of_N_l549_54991

theorem value_of_N : ∃ N : ℝ, (0.25 * N = 0.55 * 3010) ∧ (N = 6622) := by
  sorry

end value_of_N_l549_54991


namespace expand_and_simplify_l549_54921

theorem expand_and_simplify (y : ℝ) : 5 * (4 * y^2 - 3 * y + 2) = 20 * y^2 - 15 * y + 10 := by
  sorry

end expand_and_simplify_l549_54921


namespace average_of_multiples_of_four_is_even_l549_54965

theorem average_of_multiples_of_four_is_even (m n : ℤ) : 
  ∃ k : ℤ, (4*m + 4*n) / 2 = 2*k := by
  sorry

end average_of_multiples_of_four_is_even_l549_54965


namespace underpaid_amount_is_correct_l549_54945

/-- Represents the time it takes for the minute hand to coincide with the hour hand once on an accurate clock (in minutes) -/
def accurate_clock_time : ℚ := 60 + 60 / 11

/-- Represents the time it takes for the minute hand to coincide with the hour hand once on the inaccurate clock (in minutes) -/
def inaccurate_clock_time : ℚ := 69

/-- Represents the hourly wage rate (in yuan) -/
def hourly_wage : ℚ := 6

/-- Represents the nominal workday length (in hours) -/
def nominal_workday : ℚ := 8

/-- Calculates the actual working time in a day (in hours) -/
def actual_working_time : ℚ :=
  nominal_workday * (inaccurate_clock_time / accurate_clock_time)

/-- Calculates the excess time worked (in hours) -/
def excess_time : ℚ := actual_working_time - nominal_workday

/-- Calculates the amount underpaid to each worker per day (in yuan) -/
def underpaid_amount : ℚ := hourly_wage * excess_time

theorem underpaid_amount_is_correct :
  underpaid_amount = 26 / 10 := by sorry

end underpaid_amount_is_correct_l549_54945


namespace zero_not_in_range_of_g_l549_54904

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- This value won't be used, it's just to make the function total

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
by sorry

end zero_not_in_range_of_g_l549_54904
