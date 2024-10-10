import Mathlib

namespace line_circle_intersection_l3209_320903

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  (2*k+1)*x + (k-1)*y - (4*k-1) = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- Define the minimum |AB| line
def min_AB_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := y = (5/12)*x + (28/12)

theorem line_circle_intersection :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y ∧ circle_C x y → 
    (∀ x' y' : ℝ, min_AB_line x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - x')^2 + (y - y')^2)) ∧
  (∃ x y : ℝ, min_AB_line x y ∧ circle_C x y ∧
    ∃ x' y' : ℝ, min_AB_line x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 8) ∧
  (∀ x y : ℝ, (tangent_line_1 x ∨ tangent_line_2 x y) →
    (x - 4)^2 + (y - 4)^2 = ((x - 2)^2 + (y - 1)^2 - 4)^2 / ((x - 2)^2 + (y - 1)^2)) :=
by sorry

end line_circle_intersection_l3209_320903


namespace stock_price_after_two_years_l3209_320967

/-- The final stock price after a 150% increase followed by a 30% decrease, given an initial price of $120 -/
theorem stock_price_after_two_years (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) :
  initial_price = 120 →
  first_year_increase = 150 / 100 →
  second_year_decrease = 30 / 100 →
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 210 := by
  sorry

end stock_price_after_two_years_l3209_320967


namespace linear_polynomial_impossibility_l3209_320964

theorem linear_polynomial_impossibility (a b : ℝ) : 
  ¬(∃ (f : ℝ → ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (|f 0 - 1| < 1) ∧ 
    (|f 1 - 3| < 1) ∧ 
    (|f 2 - 9| < 1)) := by
  sorry

end linear_polynomial_impossibility_l3209_320964


namespace det_B_equals_two_l3209_320944

open Matrix

theorem det_B_equals_two (x y : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  (B + 2 * B⁻¹ = 0) → det B = 2 := by
  sorry

end det_B_equals_two_l3209_320944


namespace difference_of_squares_l3209_320926

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end difference_of_squares_l3209_320926


namespace inverse_proportion_problem_l3209_320956

/-- Given that x and y are inversely proportional, prove that y = -16.875 when x = -10 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 30 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) : 
  x = -10 → y = -16.875 := by
  sorry

end inverse_proportion_problem_l3209_320956


namespace school_tournament_games_l3209_320984

/-- The number of games in a round-robin tournament for n teams -/
def roundRobinGames (n : ℕ) : ℕ := n.choose 2

/-- The total number of games in a multi-grade round-robin tournament -/
def totalGames (grade1 grade2 grade3 : ℕ) : ℕ :=
  roundRobinGames grade1 + roundRobinGames grade2 + roundRobinGames grade3

theorem school_tournament_games :
  totalGames 5 8 3 = 41 := by sorry

end school_tournament_games_l3209_320984


namespace solution_set_l3209_320983

theorem solution_set (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Icc (-4) (-3/2) :=
by sorry

end solution_set_l3209_320983


namespace special_cone_volume_l3209_320910

/-- A cone with inscribed and circumscribed spheres having the same center -/
structure SpecialCone where
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ
  /-- The inscribed and circumscribed spheres have the same center -/
  spheres_same_center : Bool

/-- The volume of a SpecialCone -/
noncomputable def volume (cone : SpecialCone) : ℝ := sorry

/-- Theorem: The volume of a SpecialCone with inscribed radius 1 is 2π -/
theorem special_cone_volume (cone : SpecialCone) 
  (h1 : cone.inscribed_radius = 1) 
  (h2 : cone.spheres_same_center = true) : 
  volume cone = 2 * Real.pi := by sorry

end special_cone_volume_l3209_320910


namespace f_max_value_l3209_320925

noncomputable def f (x : ℝ) : ℝ := Real.log 2 * Real.log 5 - Real.log (2 * x) * Real.log (5 * x)

theorem f_max_value :
  ∃ (max : ℝ), (∀ (x : ℝ), x > 0 → f x ≤ max) ∧ (∃ (x : ℝ), x > 0 ∧ f x = max) ∧ max = 1/4 := by
  sorry

end f_max_value_l3209_320925


namespace inequality_proof_l3209_320968

theorem inequality_proof (x y k : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x ≠ y) 
  (h4 : k > 0) 
  (h5 : k < 2) : 
  ((x + y) / 2) ^ k > (Real.sqrt (x * y)) ^ k ∧ 
  (Real.sqrt (x * y)) ^ k > (2 * x * y / (x + y)) ^ k := by
sorry

end inequality_proof_l3209_320968


namespace circle_center_is_3_neg4_l3209_320931

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 8*y - 12 = 0

/-- The center of a circle -/
def circle_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (x^2 - 6*x + y^2 + 8*y - 12 + 37) / 2

theorem circle_center_is_3_neg4 : circle_center 3 (-4) := by
  sorry

end circle_center_is_3_neg4_l3209_320931


namespace cost_price_per_metre_values_l3209_320920

-- Define the cloth types
inductive ClothType
  | A
  | B
  | C

-- Define the properties for each cloth type
def metres_sold (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 200
  | ClothType.B => 150
  | ClothType.C => 100

def selling_price (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 10000
  | ClothType.B => 6000
  | ClothType.C => 4000

def loss (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 1000
  | ClothType.B => 450
  | ClothType.C => 200

-- Define the cost price per metre function
def cost_price_per_metre (t : ClothType) : ℚ :=
  (selling_price t + loss t : ℚ) / metres_sold t

-- State the theorem
theorem cost_price_per_metre_values :
  cost_price_per_metre ClothType.A = 55 ∧
  cost_price_per_metre ClothType.B = 43 ∧
  cost_price_per_metre ClothType.C = 42 := by
  sorry


end cost_price_per_metre_values_l3209_320920


namespace bag_original_price_l3209_320936

theorem bag_original_price (sale_price : ℝ) (discount_percent : ℝ) (original_price : ℝ) : 
  sale_price = 120 → 
  discount_percent = 50 → 
  sale_price = original_price * (1 - discount_percent / 100) → 
  original_price = 240 := by
sorry

end bag_original_price_l3209_320936


namespace sin_sixty_degrees_l3209_320954

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_sixty_degrees_l3209_320954


namespace rabbit_distribution_count_l3209_320902

/-- Represents the number of stores -/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits -/
def num_parents : ℕ := 2

/-- Represents the number of child rabbits -/
def num_children : ℕ := 4

/-- Represents the total number of rabbits -/
def total_rabbits : ℕ := num_parents + num_children

/-- 
Represents the number of ways to distribute rabbits to stores 
such that no store has both a parent and a child 
-/
def distribution_ways : ℕ := sorry

theorem rabbit_distribution_count : distribution_ways = 380 := by sorry

end rabbit_distribution_count_l3209_320902


namespace sue_nuts_count_l3209_320965

theorem sue_nuts_count (bill_nuts harry_nuts sue_nuts : ℕ) : 
  bill_nuts = 6 * harry_nuts →
  harry_nuts = 2 * sue_nuts →
  bill_nuts + harry_nuts = 672 →
  sue_nuts = 48 := by
sorry

end sue_nuts_count_l3209_320965


namespace triangle_rectangle_ratio_l3209_320966

theorem triangle_rectangle_ratio : 
  ∀ (triangle_leg : ℝ) (rect_short_side : ℝ),
  triangle_leg > 0 ∧ rect_short_side > 0 →
  2 * triangle_leg + Real.sqrt 2 * triangle_leg = 48 →
  2 * (rect_short_side + 2 * rect_short_side) = 48 →
  (Real.sqrt 2 * triangle_leg) / rect_short_side = 3 * (2 * Real.sqrt 2 - 2) :=
by sorry

end triangle_rectangle_ratio_l3209_320966


namespace water_balloon_problem_l3209_320940

theorem water_balloon_problem (janice randy cynthia : ℕ) : 
  cynthia = 4 * randy →
  randy = janice / 2 →
  cynthia + randy = janice + 12 →
  janice = 8 := by
sorry

end water_balloon_problem_l3209_320940


namespace fraction_subtraction_l3209_320907

theorem fraction_subtraction : (18 : ℚ) / 45 - 3 / 8 = 1 / 40 := by
  sorry

end fraction_subtraction_l3209_320907


namespace solve_for_k_l3209_320978

theorem solve_for_k : ∃ k : ℚ, 
  (let x : ℚ := -3
   k * (x - 2) - 4 = k - 2 * x) ∧ 
  k = -5/3 := by
  sorry

end solve_for_k_l3209_320978


namespace number_with_remainder_36_mod_45_l3209_320945

theorem number_with_remainder_36_mod_45 (k : ℤ) :
  k % 45 = 36 → ∃ (n : ℕ), k = 45 * n + 36 :=
by sorry

end number_with_remainder_36_mod_45_l3209_320945


namespace circle_diameter_from_area_l3209_320962

theorem circle_diameter_from_area :
  ∀ (A r d : ℝ),
  A = 81 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 18 := by sorry

end circle_diameter_from_area_l3209_320962


namespace polygon_three_sides_l3209_320943

/-- A polygon with n sides where the sum of interior angles is less than the sum of exterior angles. -/
structure Polygon (n : ℕ) where
  interior_sum : ℝ
  exterior_sum : ℝ
  interior_less : interior_sum < exterior_sum
  exterior_360 : exterior_sum = 360

/-- Theorem: If a polygon's interior angle sum is less than its exterior angle sum (which is 360°), then it has 3 sides. -/
theorem polygon_three_sides {n : ℕ} (p : Polygon n) : n = 3 := by
  sorry

end polygon_three_sides_l3209_320943


namespace cubic_decomposition_sum_l3209_320905

theorem cubic_decomposition_sum :
  ∃ (a b c d e : ℝ),
    (∀ x : ℝ, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧
    (a + b + c + d + e = 60) := by
  sorry

end cubic_decomposition_sum_l3209_320905


namespace collinear_points_m_value_l3209_320941

/-- Two non-collinear vectors in a vector space -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  e₁ : V
  e₂ : V
  not_collinear : ∃ (a b : ℝ), a • e₁ + b • e₂ ≠ 0

/-- Three collinear points in a vector space -/
structure CollinearPoints (V : Type*) [AddCommGroup V] [Module ℝ V] where
  A : V
  B : V
  C : V
  collinear : ∃ (t : ℝ), C - A = t • (B - A)

/-- Theorem: If e₁ and e₂ are non-collinear vectors, AB = 2e₁ + me₂, BC = e₁ + 3e₂,
    and points A, B, C are collinear, then m = 6 -/
theorem collinear_points_m_value
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (ncv : NonCollinearVectors V)
  (cp : CollinearPoints V)
  (h₁ : cp.B - cp.A = 2 • ncv.e₁ + m • ncv.e₂)
  (h₂ : cp.C - cp.B = ncv.e₁ + 3 • ncv.e₂)
  : m = 6 := by
  sorry

end collinear_points_m_value_l3209_320941


namespace train_average_speed_l3209_320975

theorem train_average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 225) (h2 : d2 = 370) (h3 : t1 = 3.5) (h4 : t2 = 5) :
  (d1 + d2) / (t1 + t2) = 70 :=
by
  sorry

end train_average_speed_l3209_320975


namespace power_equation_solution_l3209_320914

theorem power_equation_solution : ∃ y : ℕ, (2^10 + 2^10 + 2^10 + 2^10 : ℕ) = 4^y ∧ y = 6 := by
  sorry

end power_equation_solution_l3209_320914


namespace point_six_units_from_negative_three_l3209_320953

theorem point_six_units_from_negative_three (x : ℝ) : 
  (|x - (-3)| = 6) ↔ (x = 3 ∨ x = -9) := by sorry

end point_six_units_from_negative_three_l3209_320953


namespace f_order_l3209_320961

def f (x : ℝ) : ℝ := -x^2 + 2

theorem f_order : f (-2) < f 1 ∧ f 1 < f 0 :=
  by sorry

end f_order_l3209_320961


namespace stratified_sampling_problem_l3209_320934

theorem stratified_sampling_problem (total_sample : ℕ) (school_A : ℕ) (school_B : ℕ) (school_C : ℕ) 
  (h1 : total_sample = 60)
  (h2 : school_A = 180)
  (h3 : school_B = 140)
  (h4 : school_C = 160) :
  (total_sample * school_C) / (school_A + school_B + school_C) = 20 :=
by sorry

end stratified_sampling_problem_l3209_320934


namespace ear_muffs_before_december_count_l3209_320958

/-- The number of ear muffs bought before December -/
def ear_muffs_before_december (total : ℕ) (during_december : ℕ) : ℕ :=
  total - during_december

/-- Theorem stating that the number of ear muffs bought before December is 1346 -/
theorem ear_muffs_before_december_count :
  ear_muffs_before_december 7790 6444 = 1346 := by
  sorry

end ear_muffs_before_december_count_l3209_320958


namespace raja_savings_l3209_320946

def monthly_income : ℝ := 37500

def household_percentage : ℝ := 35
def clothes_percentage : ℝ := 20
def medicines_percentage : ℝ := 5

def total_expenditure_percentage : ℝ := household_percentage + clothes_percentage + medicines_percentage

def savings_percentage : ℝ := 100 - total_expenditure_percentage

theorem raja_savings : (savings_percentage / 100) * monthly_income = 15000 := by
  sorry

end raja_savings_l3209_320946


namespace circle_tangent_to_line_l3209_320979

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 5

-- Define the line's equation
def line_equation (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- State the theorem
theorem circle_tangent_to_line :
  -- The circle has center (-1, 2)
  ∃ (x₀ y₀ : ℝ), x₀ = -1 ∧ y₀ = 2 ∧
  -- The circle is tangent to the line
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y ∧
  -- Any point satisfying both equations is unique (tangency condition)
  ∀ (x' y' : ℝ), circle_equation x' y' ∧ line_equation x' y' → x' = x ∧ y' = y :=
sorry

end circle_tangent_to_line_l3209_320979


namespace M_equals_m_plus_one_l3209_320947

/-- Given natural numbers n, m, h, and b, where n ≥ h(m+1) and h ≥ 1,
    M_{(n, n m, b)} represents a certain combinatorial property. -/
def M (n m h b : ℕ) : ℕ := sorry

/-- Theorem stating that M_{(n, n m, b)} = m + 1 under given conditions -/
theorem M_equals_m_plus_one (n m h b : ℕ) (h1 : n ≥ h * (m + 1)) (h2 : h ≥ 1) :
  M n m h b = m + 1 := by
  sorry

end M_equals_m_plus_one_l3209_320947


namespace june_election_win_l3209_320982

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) (june_male_vote_percentage : ℚ) 
  (h_total : total_students = 200)
  (h_boy : boy_percentage = 3/5)
  (h_june_male : june_male_vote_percentage = 27/40)
  : ∃ (min_female_vote_percentage : ℚ), 
    min_female_vote_percentage ≥ 1/4 ∧ 
    (boy_percentage * june_male_vote_percentage + (1 - boy_percentage) * min_female_vote_percentage) * total_students > total_students / 2 := by
  sorry

end june_election_win_l3209_320982


namespace min_squared_distance_to_origin_l3209_320924

/-- The minimum value of x^2 + y^2 for points on the line x + y - 4 = 0 is 8 -/
theorem min_squared_distance_to_origin (x y : ℝ) : 
  x + y - 4 = 0 → (∀ a b : ℝ, a + b - 4 = 0 → x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 8 := by
  sorry

end min_squared_distance_to_origin_l3209_320924


namespace sector_max_area_l3209_320974

/-- Given a sector with circumference 20, prove that its area is maximized when the central angle is 2 radians. -/
theorem sector_max_area (r : ℝ) (l : ℝ) (α : ℝ) :
  l + 2 * r = 20 →  -- Circumference condition
  l = r * α →       -- Arc length formula
  α = 2 →           -- Proposed maximum angle
  ∀ (r' : ℝ) (l' : ℝ) (α' : ℝ),
    l' + 2 * r' = 20 →
    l' = r' * α' →
    (1/2) * r * l ≥ (1/2) * r' * l' :=
by sorry

end sector_max_area_l3209_320974


namespace fraction_to_decimal_l3209_320994

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l3209_320994


namespace choose_with_mandatory_l3209_320976

theorem choose_with_mandatory (n m k : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 1) :
  (Nat.choose (n - k) (m - k)) = 84 :=
sorry

end choose_with_mandatory_l3209_320976


namespace probability_a_b_same_area_l3209_320981

def total_employees : ℕ := 4
def employees_per_area : ℕ := 2
def num_areas : ℕ := 2

def probability_same_area (total : ℕ) (per_area : ℕ) (areas : ℕ) : ℚ :=
  if total = total_employees ∧ per_area = employees_per_area ∧ areas = num_areas then
    1 / 3
  else
    0

theorem probability_a_b_same_area :
  probability_same_area total_employees employees_per_area num_areas = 1 / 3 := by
  sorry

end probability_a_b_same_area_l3209_320981


namespace min_colors_for_four_color_rect_l3209_320969

/-- Represents a coloring of an n × n board using k colors. -/
structure Coloring (n k : ℕ) :=
  (colors : Fin n → Fin n → Fin k)
  (all_used : ∀ c : Fin k, ∃ i j : Fin n, colors i j = c)

/-- Checks if four cells at the intersections of two rows and two columns have different colors. -/
def hasFourColorRect (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ i₁ i₂ j₁ j₂ : Fin n, i₁ ≠ i₂ ∧ j₁ ≠ j₂ ∧
    c.colors i₁ j₁ ≠ c.colors i₁ j₂ ∧
    c.colors i₁ j₁ ≠ c.colors i₂ j₁ ∧
    c.colors i₁ j₁ ≠ c.colors i₂ j₂ ∧
    c.colors i₁ j₂ ≠ c.colors i₂ j₁ ∧
    c.colors i₁ j₂ ≠ c.colors i₂ j₂ ∧
    c.colors i₂ j₁ ≠ c.colors i₂ j₂

/-- The main theorem stating that 2n is the smallest number of colors
    that guarantees a four-color rectangle in any coloring. -/
theorem min_colors_for_four_color_rect (n : ℕ) (h : n ≥ 2) :
  (∀ k : ℕ, k ≥ 2*n → ∀ c : Coloring n k, hasFourColorRect n k c) ∧
  (∃ c : Coloring n (2*n - 1), ¬hasFourColorRect n (2*n - 1) c) :=
sorry

end min_colors_for_four_color_rect_l3209_320969


namespace difference_of_squares_73_47_l3209_320948

theorem difference_of_squares_73_47 : 73^2 - 47^2 = 3120 := by
  sorry

end difference_of_squares_73_47_l3209_320948


namespace corner_triangles_area_l3209_320970

/-- Given a square with side length 16 units, if we remove four isosceles right triangles 
    from its corners, where the leg of each triangle is 1/4 of the square's side length, 
    the total area of the removed triangles is 32 square units. -/
theorem corner_triangles_area (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 16 → 
  triangle_leg = square_side / 4 → 
  4 * (1/2 * triangle_leg^2) = 32 :=
by sorry

end corner_triangles_area_l3209_320970


namespace strawberry_weight_calculation_l3209_320971

def total_fruit_weight : ℕ := 10
def apple_weight : ℕ := 3
def orange_weight : ℕ := 1
def grape_weight : ℕ := 3

theorem strawberry_weight_calculation :
  total_fruit_weight - (apple_weight + orange_weight + grape_weight) = 3 :=
by
  sorry

end strawberry_weight_calculation_l3209_320971


namespace mod_congruence_l3209_320912

theorem mod_congruence (m : ℕ) : 
  (65 * 90 * 111 ≡ m [ZMOD 20]) → 
  (0 ≤ m ∧ m < 20) → 
  m = 10 := by
  sorry

end mod_congruence_l3209_320912


namespace hyperbola_eccentricity_l3209_320996

/-- Given a hyperbola and a parabola, if the asymptote of the hyperbola
    intersects the parabola at only one point, then the eccentricity
    of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℝ, ∀ x y : ℝ,
    (x^2/a^2 - y^2/b^2 = 1 → y = k*x) ∧
    (x^2 = y - 1 → y = k*x) →
    (∀ z : ℝ, z ≠ x → x^2 = z - 1 → y ≠ k*z)) →
  let c := Real.sqrt (a^2 + b^2)
  c/a = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l3209_320996


namespace smallest_odd_five_primes_proof_l3209_320901

/-- The smallest odd number with five different prime factors -/
def smallest_odd_five_primes : ℕ := 15015

/-- The list of prime factors of the smallest odd number with five different prime factors -/
def prime_factors : List ℕ := [3, 5, 7, 11, 13]

theorem smallest_odd_five_primes_proof :
  (smallest_odd_five_primes % 2 = 1) ∧
  (List.length prime_factors = 5) ∧
  (List.all prime_factors Nat.Prime) ∧
  (List.prod prime_factors = smallest_odd_five_primes) ∧
  (∀ n : ℕ, n < smallest_odd_five_primes →
    n % 2 = 1 →
    (∃ factors : List ℕ, List.all factors Nat.Prime ∧
      List.prod factors = n ∧
      List.length factors < 5)) :=
by sorry

end smallest_odd_five_primes_proof_l3209_320901


namespace seashells_given_to_sam_proof_l3209_320916

/-- The number of seashells Joan initially found -/
def initial_seashells : ℕ := 70

/-- The number of seashells Joan has left -/
def remaining_seashells : ℕ := 27

/-- The number of seashells Joan gave to Sam -/
def seashells_given_to_sam : ℕ := initial_seashells - remaining_seashells

theorem seashells_given_to_sam_proof :
  seashells_given_to_sam = 43 := by sorry

end seashells_given_to_sam_proof_l3209_320916


namespace difference_of_prime_squares_can_be_perfect_square_l3209_320904

theorem difference_of_prime_squares_can_be_perfect_square :
  ∃ (p q : ℕ) (n : ℕ), Prime p ∧ Prime q ∧ p^2 - q^2 = n^2 := by
  sorry

end difference_of_prime_squares_can_be_perfect_square_l3209_320904


namespace expression_simplification_l3209_320927

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x / (x - 2) - x / (x + 2)) / (4 * x / (x - 2)) = 1 / 5 := by
  sorry

end expression_simplification_l3209_320927


namespace susans_purchase_l3209_320921

/-- Given Susan's purchase scenario, prove the number of 50-cent items -/
theorem susans_purchase (x y z : ℕ) : 
  x + y + z = 50 →  -- total number of items
  50 * x + 300 * y + 500 * z = 10000 →  -- total price in cents
  x = 40  -- number of 50-cent items
:= by sorry

end susans_purchase_l3209_320921


namespace f_composition_of_one_l3209_320950

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 3 else 4 * x + 1

theorem f_composition_of_one : f (f (f (f 1))) = 341 := by
  sorry

end f_composition_of_one_l3209_320950


namespace g_in_M_l3209_320929

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the function g
def g : ℝ → ℝ := λ x ↦ x^2 + 2*x - 1

-- Theorem statement
theorem g_in_M : g ∈ M := by
  sorry

end g_in_M_l3209_320929


namespace gcd_lcm_product_l3209_320973

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  (Nat.gcd a b) * (Nat.lcm a b) = 56700 := by
  sorry

end gcd_lcm_product_l3209_320973


namespace cube_root_simplification_l3209_320918

theorem cube_root_simplification :
  (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by sorry

end cube_root_simplification_l3209_320918


namespace last_digit_of_sum_l3209_320913

theorem last_digit_of_sum (n : ℕ) : 
  (54^2020 + 28^2022) % 10 = 0 := by sorry

end last_digit_of_sum_l3209_320913


namespace peach_difference_l3209_320932

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 87 →
  steven_peaches = jill_peaches + 18 →
  jake_peaches = jill_peaches + 13 →
  steven_peaches - jake_peaches = 5 := by
sorry

end peach_difference_l3209_320932


namespace quadratic_equation_completion_square_l3209_320935

theorem quadratic_equation_completion_square (x : ℝ) : 
  16 * x^2 - 32 * x - 512 = 0 → ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by
  sorry

end quadratic_equation_completion_square_l3209_320935


namespace forgotten_angle_measure_l3209_320928

theorem forgotten_angle_measure (n : ℕ) (sum_without_one : ℝ) : 
  n ≥ 3 → 
  sum_without_one = 2070 → 
  (n - 2) * 180 - sum_without_one = 90 := by
  sorry

end forgotten_angle_measure_l3209_320928


namespace quadratic_always_real_roots_k_value_when_x_is_two_l3209_320939

/-- The quadratic equation x^2 - kx + k - 1 = 0 -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + k - 1

theorem quadratic_always_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic k x = 0 :=
sorry

theorem k_value_when_x_is_two :
  ∃ k : ℝ, quadratic k 2 = 0 ∧ k = 3 :=
sorry

end quadratic_always_real_roots_k_value_when_x_is_two_l3209_320939


namespace chocolate_price_proof_l3209_320986

/-- Proves that if a chocolate's price is reduced by 57 cents and the resulting price is $1.43, then the original price was $2.00. -/
theorem chocolate_price_proof (original_price : ℝ) : 
  (original_price - 0.57 = 1.43) → original_price = 2.00 := by
  sorry

end chocolate_price_proof_l3209_320986


namespace min_sum_squares_l3209_320922

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 1 →
    x^2 + y^2 + z^2 ≥ m ∧ m ≤ a^2 + b^2 + c^2 := by
  sorry

end min_sum_squares_l3209_320922


namespace cone_height_from_circular_sector_l3209_320985

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let circumference := 2 * Real.pi * r
  let sector_arc_length := circumference / 3
  let base_radius := sector_arc_length / (2 * Real.pi)
  let height := Real.sqrt (r^2 - base_radius^2)
  height = 20 * Real.sqrt 2 / 3 := by sorry

end cone_height_from_circular_sector_l3209_320985


namespace game_ends_in_58_rounds_l3209_320923

/-- Represents the state of the game at any point --/
structure GameState where
  playerA : Nat
  playerB : Nat
  playerC : Nat

/-- Simulates one round of the game --/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended --/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends --/
def countRounds (state : GameState) : Nat :=
  sorry

/-- Theorem stating that the game ends after 58 rounds --/
theorem game_ends_in_58_rounds :
  let initialState : GameState := { playerA := 20, playerB := 18, playerC := 15 }
  countRounds initialState = 58 := by
  sorry

end game_ends_in_58_rounds_l3209_320923


namespace angle_bisector_length_l3209_320917

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 9 ∧ BC = 12 ∧ AC = 15

-- Define the angle bisector CD
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop :=
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  BD / AD = 12 / 15

-- Theorem statement
theorem angle_bisector_length (A B C D : ℝ × ℝ) :
  triangle_ABC A B C → is_angle_bisector A B C D →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4 * Real.sqrt 10 :=
by
  sorry


end angle_bisector_length_l3209_320917


namespace card_selection_count_l3209_320988

def total_cards : ℕ := 12
def red_cards : ℕ := 4
def yellow_cards : ℕ := 4
def blue_cards : ℕ := 4
def cards_to_select : ℕ := 3
def max_red_cards : ℕ := 1

theorem card_selection_count :
  (Nat.choose (yellow_cards + blue_cards) cards_to_select) +
  (Nat.choose red_cards max_red_cards * Nat.choose (yellow_cards + blue_cards) (cards_to_select - max_red_cards)) = 168 := by
  sorry

end card_selection_count_l3209_320988


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3209_320995

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∨ a = 9) ∧  -- One side is either 4 or 9
    (b = a) ∧          -- The triangle is isosceles
    (c = if a = 4 then 9 else 4) ∧  -- The third side is whichever of 4 or 9 that a is not
    (a + b + c = 22)   -- The perimeter is 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof :
  ∃ a b c, isosceles_triangle_perimeter a b c :=
by
  sorry  -- The proof is omitted as per instructions

#check isosceles_triangle_perimeter
#check isosceles_triangle_perimeter_proof

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3209_320995


namespace hyperbola_focus_distance_l3209_320955

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define a point on the left branch of the hyperbola
def left_branch_point (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ P.1 < 0

-- Define the distance from a point to the left focus
def dist_to_left_focus (P : ℝ × ℝ) : ℝ := 10

-- Theorem statement
theorem hyperbola_focus_distance (P : ℝ × ℝ) :
  left_branch_point P → dist_to_left_focus P = 10 →
  ∃ (dist_to_right_focus : ℝ), dist_to_right_focus = 18 :=
by sorry

end hyperbola_focus_distance_l3209_320955


namespace max_area_circular_sector_l3209_320900

/-- Theorem: Maximum area of a circular sector with perimeter 16 --/
theorem max_area_circular_sector (r θ : ℝ) : 
  r > 0 → 
  θ > 0 → 
  2 * r + θ * r = 16 → 
  (1/2) * θ * r^2 ≤ 16 ∧ 
  (∃ (r₀ θ₀ : ℝ), r₀ > 0 ∧ θ₀ > 0 ∧ 2 * r₀ + θ₀ * r₀ = 16 ∧ (1/2) * θ₀ * r₀^2 = 16) :=
by sorry

end max_area_circular_sector_l3209_320900


namespace condition_analysis_l3209_320991

theorem condition_analysis :
  (∃ a b : ℝ, (1 / a > 1 / b ∧ a ≥ b) ∨ (1 / a ≤ 1 / b ∧ a < b)) ∧
  (∀ A B : Set α, A = ∅ → A ∩ B = ∅) ∧
  (∃ A B : Set α, A ∩ B = ∅ ∧ A ≠ ∅) ∧
  (∀ a b : ℝ, a^2 + b^2 ≠ 0 ↔ |a| + |b| ≠ 0) ∧
  (∃ a b : ℝ, ∃ n : ℕ, n ≥ 2 ∧ (a^n > b^n ∧ ¬(a > b ∧ b > 0))) :=
by sorry

end condition_analysis_l3209_320991


namespace part_one_part_two_l3209_320957

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 4*m*x - 5*m^2 < 0}

-- Part 1: Prove that when B = {x | -5 < x < 1}, m = 1
theorem part_one : 
  (B 1 = {x | -5 < x ∧ x < 1}) → 1 = 1 := by sorry

-- Part 2: Prove that when A ⊆ B, m ≤ -1 or m ≥ 4
theorem part_two (m : ℝ) : 
  A ⊆ B m → m ≤ -1 ∨ m ≥ 4 := by sorry

end part_one_part_two_l3209_320957


namespace min_value_fraction_l3209_320915

theorem min_value_fraction (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧ ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
sorry

end min_value_fraction_l3209_320915


namespace home_theater_savings_l3209_320906

def in_store_price : ℝ := 320
def in_store_discount : ℝ := 0.05
def website_monthly_payment : ℝ := 62
def website_num_payments : ℕ := 5
def website_shipping : ℝ := 10

theorem home_theater_savings :
  let website_total := website_monthly_payment * website_num_payments + website_shipping
  let in_store_discounted := in_store_price * (1 - in_store_discount)
  website_total - in_store_discounted = 16 := by sorry

end home_theater_savings_l3209_320906


namespace intersection_equality_l3209_320949

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 1/3 ≤ x ∧ x < 16}

-- State the theorem
theorem intersection_equality : M ∩ N = intersection_set := by
  sorry

end intersection_equality_l3209_320949


namespace unseen_corner_color_code_l3209_320977

/-- Represents the colors of a Rubik's Cube -/
inductive Color
  | White
  | Yellow
  | Green
  | Blue
  | Orange
  | Red

/-- Represents a corner piece of a Rubik's Cube -/
structure Corner :=
  (c1 c2 c3 : Color)

/-- Assigns a numeric code to each color -/
def color_code (c : Color) : ℕ :=
  match c with
  | Color.White => 1
  | Color.Yellow => 2
  | Color.Green => 3
  | Color.Blue => 4
  | Color.Orange => 5
  | Color.Red => 6

/-- Represents the state of a Rubik's Cube -/
structure RubiksCube :=
  (corners : List Corner)

/-- Represents a solved Rubik's Cube -/
def solved_cube : RubiksCube := sorry

/-- Represents a scrambled Rubik's Cube with 7 visible corners -/
def scrambled_cube : RubiksCube := sorry

theorem unseen_corner_color_code :
  ∀ (cube : RubiksCube),
    (cube.corners.length = 8) →
    (∃ (visible_corners : List Corner), visible_corners.length = 7 ∧ visible_corners ⊆ cube.corners) →
    ∃ (unseen_corner : Corner),
      unseen_corner ∈ cube.corners ∧
      unseen_corner ∉ (visible_corners : List Corner) ∧
      color_code (unseen_corner.c1) = 1 :=
by sorry

end unseen_corner_color_code_l3209_320977


namespace new_member_amount_l3209_320999

theorem new_member_amount (group : Finset ℕ) (group_sum : ℕ) (new_member : ℕ) : 
  Finset.card group = 7 →
  group_sum / 7 = 20 →
  (group_sum + new_member) / 8 = 14 →
  new_member = 756 := by
sorry

end new_member_amount_l3209_320999


namespace mary_extra_flour_l3209_320960

/-- Given a recipe that calls for a certain amount of flour and the actual amount used,
    calculate the extra amount of flour used. -/
def extra_flour (recipe_amount : ℝ) (actual_amount : ℝ) : ℝ :=
  actual_amount - recipe_amount

/-- Theorem stating that Mary used 2 extra cups of flour -/
theorem mary_extra_flour :
  let recipe_amount : ℝ := 7.0
  let actual_amount : ℝ := 9.0
  extra_flour recipe_amount actual_amount = 2 := by
  sorry

end mary_extra_flour_l3209_320960


namespace evaluate_f_l3209_320909

/-- The function f(x) = 2x^2 - 4x + 9 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

/-- Theorem stating that 2f(3) + 3f(-3) = 147 -/
theorem evaluate_f : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end evaluate_f_l3209_320909


namespace train_length_l3209_320998

/-- The length of a train given specific passing times -/
theorem train_length : ∃ (L : ℝ), 
  (∀ (V : ℝ), V = L / 24 → V = (L + 650) / 89) → L = 240 :=
by
  sorry

end train_length_l3209_320998


namespace mother_daughter_age_ratio_l3209_320963

/-- Given a mother who is 27 years older than her daughter and is currently 55 years old,
    prove that the ratio of their ages one year ago was 2:1. -/
theorem mother_daughter_age_ratio : 
  ∀ (mother_age daughter_age : ℕ),
  mother_age = 55 →
  mother_age = daughter_age + 27 →
  (mother_age - 1) / (daughter_age - 1) = 2 := by
sorry

end mother_daughter_age_ratio_l3209_320963


namespace first_sequence_6th_7th_terms_l3209_320930

def first_sequence : ℕ → ℕ
  | 0 => 3
  | n + 1 => 2 * first_sequence n + 1

theorem first_sequence_6th_7th_terms :
  first_sequence 5 = 127 ∧ first_sequence 6 = 255 := by
  sorry

end first_sequence_6th_7th_terms_l3209_320930


namespace completing_square_result_l3209_320997

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) ↔ ((x - 2)^2 = 2) := by
  sorry

end completing_square_result_l3209_320997


namespace rubber_duck_race_l3209_320993

theorem rubber_duck_race (regular_price large_price large_count total : ℕ) :
  regular_price = 3 →
  large_price = 5 →
  large_count = 185 →
  total = 1588 →
  ∃ regular_count : ℕ, 
    regular_count * regular_price + large_count * large_price = total ∧
    regular_count = 221 := by
  sorry

end rubber_duck_race_l3209_320993


namespace problem_solution_l3209_320911

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by sorry

end problem_solution_l3209_320911


namespace code_cracker_combinations_l3209_320937

/-- The number of different colors of pegs in the CodeCracker game -/
def num_colors : ℕ := 6

/-- The number of slots for pegs in the CodeCracker game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in the CodeCracker game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes in the CodeCracker game is 7776 -/
theorem code_cracker_combinations : total_codes = 7776 := by
  sorry

end code_cracker_combinations_l3209_320937


namespace hash_2_3_4_l3209_320952

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c + b

-- Theorem statement
theorem hash_2_3_4 : hash 2 3 4 = -20 := by sorry

end hash_2_3_4_l3209_320952


namespace sum_of_coefficients_equals_one_l3209_320933

theorem sum_of_coefficients_equals_one (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 := by
sorry

end sum_of_coefficients_equals_one_l3209_320933


namespace subsets_of_three_element_set_l3209_320990

theorem subsets_of_three_element_set : 
  Finset.card (Finset.powerset {1, 2, 3}) = 8 := by sorry

end subsets_of_three_element_set_l3209_320990


namespace condition_type_l3209_320908

theorem condition_type (a : ℝ) : 
  (∀ x : ℝ, x > 2 → x^2 > 2*x) ∧ 
  (∃ y : ℝ, y ≤ 2 ∧ y^2 > 2*y) :=
by sorry

end condition_type_l3209_320908


namespace solution_value_l3209_320919

theorem solution_value (x a : ℝ) : x = 2 ∧ 2 * x + a = 3 → a = -1 := by
  sorry

end solution_value_l3209_320919


namespace stock_price_theorem_l3209_320942

/-- The stock price after three years of changes -/
def stock_price_after_three_years (initial_price : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + 0.8)
  let price_after_second_year := price_after_first_year * (1 - 0.3)
  let price_after_third_year := price_after_second_year * (1 + 0.5)
  price_after_third_year

/-- Theorem stating that the stock price after three years is $226.8 -/
theorem stock_price_theorem :
  stock_price_after_three_years 120 = 226.8 := by
  sorry

end stock_price_theorem_l3209_320942


namespace courtyard_paving_l3209_320972

-- Define the courtyard dimensions in meters
def courtyard_length : ℝ := 42
def courtyard_width : ℝ := 22

-- Define the brick dimensions in centimeters
def brick_length : ℝ := 16
def brick_width : ℝ := 10

-- Define the conversion factor from square meters to square centimeters
def sq_m_to_sq_cm : ℝ := 10000

-- Theorem statement
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width sq_m_to_sq_cm : ℝ) :
  courtyard_length = 42 →
  courtyard_width = 22 →
  brick_length = 16 →
  brick_width = 10 →
  sq_m_to_sq_cm = 10000 →
  (courtyard_length * courtyard_width * sq_m_to_sq_cm) / (brick_length * brick_width) = 57750 :=
by
  sorry


end courtyard_paving_l3209_320972


namespace marble_theorem_l3209_320938

def marble_problem (wolfgang_marbles : ℕ) : Prop :=
  let ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
  let total_wolfgang_ludo : ℕ := wolfgang_marbles + ludo_marbles
  let michael_marbles : ℕ := (2 * total_wolfgang_ludo) / 3
  let total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles
  wolfgang_marbles = 16 →
  total_marbles / 3 = 20

theorem marble_theorem : marble_problem 16 := by
  sorry

end marble_theorem_l3209_320938


namespace digimon_pack_cost_is_445_l3209_320987

/-- The cost of a pack of Digimon cards -/
def digimon_pack_cost : ℝ := 4.45

/-- The number of Digimon card packs bought -/
def num_digimon_packs : ℕ := 4

/-- The cost of the baseball card deck -/
def baseball_deck_cost : ℝ := 6.06

/-- The total amount spent on cards -/
def total_spent : ℝ := 23.86

/-- Theorem stating that the cost of each Digimon card pack is $4.45 -/
theorem digimon_pack_cost_is_445 :
  digimon_pack_cost * num_digimon_packs + baseball_deck_cost = total_spent :=
by sorry

end digimon_pack_cost_is_445_l3209_320987


namespace arrangement_count_l3209_320959

/-- The number of white pieces -/
def white_pieces : ℕ := 5

/-- The number of black pieces -/
def black_pieces : ℕ := 10

/-- The number of different arrangements of white and black pieces
    satisfying the given conditions -/
def num_arrangements : ℕ := Nat.choose black_pieces white_pieces

theorem arrangement_count :
  num_arrangements = 252 :=
by sorry

end arrangement_count_l3209_320959


namespace cos_555_degrees_l3209_320951

theorem cos_555_degrees : 
  Real.cos (555 * Real.pi / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_555_degrees_l3209_320951


namespace interest_percentage_approx_l3209_320980

def purchase_price : ℚ := 2345
def down_payment : ℚ := 385
def num_monthly_payments : ℕ := 18
def monthly_payment : ℚ := 125

def total_paid : ℚ := down_payment + num_monthly_payments * monthly_payment

def interest_paid : ℚ := total_paid - purchase_price

def interest_percentage : ℚ := (interest_paid / purchase_price) * 100

theorem interest_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.1 ∧ |interest_percentage - 12.4| < ε :=
sorry

end interest_percentage_approx_l3209_320980


namespace work_completion_time_l3209_320989

/-- The time it takes to complete a work with two workers working sequentially -/
def total_work_time (mahesh_full_time : ℕ) (mahesh_work_time : ℕ) (rajesh_finish_time : ℕ) : ℕ :=
  mahesh_work_time + rajesh_finish_time

/-- Theorem stating that under given conditions, the total work time is 50 days -/
theorem work_completion_time :
  total_work_time 45 20 30 = 50 := by
  sorry

end work_completion_time_l3209_320989


namespace intersection_chord_length_l3209_320992

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop :=
  2 * x - 2 * Real.sqrt 3 * y + 2 * Real.sqrt 3 - 1 = 0

/-- The circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2

/-- The theorem stating that the length of the chord formed by the intersection of line l and circle C is √5/2 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 5 / 2 :=
sorry

end intersection_chord_length_l3209_320992
