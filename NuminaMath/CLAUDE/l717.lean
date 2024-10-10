import Mathlib

namespace circles_relationship_l717_71797

/-- The positional relationship between two circles -/
theorem circles_relationship (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ) : 
  (C1.1 + 1)^2 + (C1.2 + 1)^2 = 4 →
  (C2.1 - 2)^2 + (C2.2 - 1)^2 = 4 →
  r1 = 2 →
  r2 = 2 →
  C1 = (-1, -1) →
  C2 = (2, 1) →
  (r1 - r2)^2 < (C1.1 - C2.1)^2 + (C1.2 - C2.2)^2 ∧ 
  (C1.1 - C2.1)^2 + (C1.2 - C2.2)^2 < (r1 + r2)^2 :=
by sorry


end circles_relationship_l717_71797


namespace negative_fraction_comparison_l717_71779

theorem negative_fraction_comparison : -3/5 > -5/7 := by
  sorry

end negative_fraction_comparison_l717_71779


namespace ball_placement_theorem_l717_71767

/-- The number of ways to place 4 different balls into 4 different boxes --/
def placeBalls (emptyBoxes : Nat) : Nat :=
  if emptyBoxes = 1 then 144
  else if emptyBoxes = 2 then 84
  else 0

theorem ball_placement_theorem :
  (placeBalls 1 = 144) ∧ (placeBalls 2 = 84) := by
  sorry

#eval placeBalls 1  -- Expected output: 144
#eval placeBalls 2  -- Expected output: 84

end ball_placement_theorem_l717_71767


namespace power_of_eight_sum_equals_power_of_two_l717_71726

theorem power_of_eight_sum_equals_power_of_two : 8^17 + 8^17 + 8^17 + 8^17 = 2^53 := by
  sorry

end power_of_eight_sum_equals_power_of_two_l717_71726


namespace jam_jar_max_theorem_l717_71747

/-- Represents the initial state of jam jars --/
structure JamJars :=
  (carlson_weight : ℕ)
  (baby_weight : ℕ)
  (carlson_min_jar : ℕ)

/-- Conditions for the jam jar problem --/
def valid_jam_jars (j : JamJars) : Prop :=
  j.carlson_weight = 13 * j.baby_weight ∧
  j.carlson_weight - j.carlson_min_jar = 8 * (j.baby_weight + j.carlson_min_jar)

/-- The maximum number of jars Carlson could have initially --/
def max_carlson_jars : ℕ := 23

/-- Theorem stating the maximum number of jars Carlson could have initially --/
theorem jam_jar_max_theorem (j : JamJars) (h : valid_jam_jars j) :
  (j.carlson_weight / j.carlson_min_jar : ℚ) ≤ max_carlson_jars :=
sorry

end jam_jar_max_theorem_l717_71747


namespace bowling_ball_weight_l717_71706

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 8 * b = 5 * c)  -- 8 bowling balls weigh the same as 5 canoes
  (h2 : 3 * c = 135)    -- 3 canoes weigh 135 pounds
  : b = 28.125 :=       -- One bowling ball weighs 28.125 pounds
by
  sorry

end bowling_ball_weight_l717_71706


namespace expand_and_simplify_l717_71716

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := by
  sorry

end expand_and_simplify_l717_71716


namespace b_plus_c_equals_three_l717_71752

/-- A function f: ℝ → ℝ defined as f(x) = x^3 + bx^2 + cx -/
def f (b c : ℝ) : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x

/-- The derivative of f -/
def f_deriv (b c : ℝ) : ℝ → ℝ := λ x ↦ 3*x^2 + 2*b*x + c

/-- A function g: ℝ → ℝ defined as g(x) = f(x) - f'(x) -/
def g (b c : ℝ) : ℝ → ℝ := λ x ↦ f b c x - f_deriv b c x

/-- A predicate stating that a function is odd -/
def is_odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

/-- The main theorem -/
theorem b_plus_c_equals_three (b c : ℝ) :
  is_odd_function (g b c) → b + c = 3 := by sorry

end b_plus_c_equals_three_l717_71752


namespace zarnin_battle_station_staffing_l717_71737

/-- The number of ways to fill positions from a pool of candidates -/
def fill_positions (total_candidates : Nat) (positions_to_fill : Nat) : Nat :=
  List.range positions_to_fill
  |>.map (fun i => total_candidates - i)
  |>.prod

/-- The problem statement -/
theorem zarnin_battle_station_staffing :
  fill_positions 20 5 = 1860480 := by
  sorry

end zarnin_battle_station_staffing_l717_71737


namespace cubic_equation_solution_sum_l717_71769

theorem cubic_equation_solution_sum (r s t : ℝ) : 
  r^3 - 5*r^2 + 6*r = 9 →
  s^3 - 5*s^2 + 6*s = 9 →
  t^3 - 5*t^2 + 6*t = 9 →
  r*s/t + s*t/r + t*r/s = -6 := by
  sorry

end cubic_equation_solution_sum_l717_71769


namespace ball_max_height_l717_71789

/-- The height function of the ball's path -/
def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 21

/-- Theorem stating that the maximum height of the ball is 121 feet -/
theorem ball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 121 := by
  sorry

end ball_max_height_l717_71789


namespace congruence_solution_l717_71787

theorem congruence_solution (n : ℤ) : 13 * n ≡ 8 [ZMOD 47] ↔ n ≡ 29 [ZMOD 47] := by
  sorry

end congruence_solution_l717_71787


namespace mass_of_man_on_boat_l717_71724

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : Real) : Real :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : Real := 4
  let boat_breadth : Real := 3
  let boat_sink_height : Real := 0.01
  let water_density : Real := 1000
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 120 := by
  sorry

#check mass_of_man_on_boat

end mass_of_man_on_boat_l717_71724


namespace largest_fraction_sum_l717_71756

theorem largest_fraction_sum : 
  let a := (3 : ℚ) / 10 + (2 : ℚ) / 20
  let b := (1 : ℚ) / 6 + (1 : ℚ) / 8
  let c := (1 : ℚ) / 5 + (2 : ℚ) / 15
  let d := (1 : ℚ) / 7 + (4 : ℚ) / 21
  let e := (2 : ℚ) / 9 + (3 : ℚ) / 18
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by sorry

end largest_fraction_sum_l717_71756


namespace total_bathing_suits_l717_71795

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end total_bathing_suits_l717_71795


namespace problem_statement_l717_71794

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * (Real.sin (ω * x / 2))^2

theorem problem_statement 
  (ω : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω (x + 3 * Real.pi) = f ω x)
  (a b c A B C : ℝ)
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_b : b = 2)
  (h_fA : f ω A = Real.sqrt 3 - 1)
  (h_sides : Real.sqrt 3 * a = 2 * b * Real.sin A) :
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, ∀ y ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω y ≥ f ω x) ∧ 
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, ∀ y ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω y ≤ f ω x) ∧
  (1/2 * a * b * Real.sin C = (3 + Real.sqrt 3) / 3) := by
sorry

end problem_statement_l717_71794


namespace inequality_proof_l717_71784

theorem inequality_proof (a : ℝ) (h : a ≠ -1) :
  (1 + a^3) / ((1 + a)^3) ≥ (1 : ℝ) / 4 := by
  sorry

end inequality_proof_l717_71784


namespace triangle_triplets_characterization_l717_71707

def is_valid_triplet (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧
  (∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r) ∧
  (a = 100 ∨ c = 100)

def valid_triplets : Set (ℕ × ℕ × ℕ) :=
  {(49,70,100), (64,80,100), (81,90,100), (100,100,100), (100,110,121),
   (100,120,144), (100,130,169), (100,140,196), (100,150,225), (100,160,256)}

theorem triangle_triplets_characterization :
  {(a, b, c) | is_valid_triplet a b c} = valid_triplets :=
by sorry

end triangle_triplets_characterization_l717_71707


namespace annie_brownies_left_l717_71749

/-- Calculates the number of brownies Annie has left after sharing -/
def brownies_left (initial : ℕ) (to_simon : ℕ) : ℕ :=
  let to_admin := initial / 2
  let after_admin := initial - to_admin
  let to_carl := after_admin / 2
  let after_carl := after_admin - to_carl
  after_carl - to_simon

/-- Proves that Annie has 3 brownies left after sharing -/
theorem annie_brownies_left :
  brownies_left 20 2 = 3 := by
sorry

end annie_brownies_left_l717_71749


namespace marks_fruit_purchase_l717_71746

/-- The total cost of Mark's fruit purchase --/
def total_cost (tomato_price apple_price orange_price : ℝ)
                (tomato_weight apple_weight orange_weight : ℝ)
                (apple_discount : ℝ) : ℝ :=
  tomato_price * tomato_weight +
  apple_price * apple_weight * (1 - apple_discount) +
  orange_price * orange_weight

/-- Theorem stating the total cost of Mark's fruit purchase --/
theorem marks_fruit_purchase :
  total_cost 4.50 3.25 2.75 3 7 4 0.1 = 44.975 := by
  sorry

#eval total_cost 4.50 3.25 2.75 3 7 4 0.1

end marks_fruit_purchase_l717_71746


namespace rectangle_division_count_l717_71717

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a division of a large rectangle into smaller rectangles --/
structure RectangleDivision where
  large : Rectangle
  small : Rectangle
  divisions : List (List ℕ)

/-- Counts the number of ways to divide a rectangle --/
def countDivisions (r : RectangleDivision) : ℕ :=
  r.divisions.length

/-- The main rectangle --/
def mainRectangle : Rectangle :=
  { width := 24, height := 20 }

/-- The sub-rectangle --/
def subRectangle : Rectangle :=
  { width := 5, height := 4 }

/-- The division of the main rectangle into sub-rectangles --/
def rectangleDivision : RectangleDivision :=
  { large := mainRectangle
    small := subRectangle
    divisions := [[4, 4, 4, 4, 4, 4], [4, 5, 5, 5, 5], [5, 4, 5, 5, 5], [5, 5, 4, 5, 5], [5, 5, 5, 4, 5], [5, 5, 5, 5, 4]] }

/-- Theorem stating that the number of ways to divide the rectangle is 6 --/
theorem rectangle_division_count : countDivisions rectangleDivision = 6 := by
  sorry

end rectangle_division_count_l717_71717


namespace smallest_candy_count_l717_71753

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n + 5) % 8 = 0 ∧ 
  (n - 8) % 5 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m + 5) % 8 = 0 ∧ (m - 8) % 5 = 0 → n ≤ m) ∧
  n = 123 :=
by sorry

end smallest_candy_count_l717_71753


namespace additional_peaches_l717_71738

theorem additional_peaches (initial_peaches total_peaches : ℕ) 
  (h1 : initial_peaches = 20)
  (h2 : total_peaches = 45) :
  total_peaches - initial_peaches = 25 := by
  sorry

end additional_peaches_l717_71738


namespace no_real_roots_l717_71748

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 1) + 2 = 0 := by
  sorry

end no_real_roots_l717_71748


namespace ratio_of_numbers_l717_71763

theorem ratio_of_numbers (a b : ℕ) (h1 : a = 45) (h2 : b = 60) (h3 : Nat.lcm a b = 180) :
  (a : ℚ) / b = 3 / 4 := by
  sorry

end ratio_of_numbers_l717_71763


namespace triangle_area_angle_l717_71744

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a² + b² - c²), then C = π/3 -/
theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)
  S = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) →
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π/3 :=
by sorry

end triangle_area_angle_l717_71744


namespace guppies_ratio_l717_71739

/-- The number of guppies Haylee has -/
def haylee_guppies : ℕ := 36

/-- The number of guppies Jose has -/
def jose_guppies : ℕ := haylee_guppies / 2

/-- The number of guppies Charliz has -/
def charliz_guppies : ℕ := jose_guppies / 3

/-- The total number of guppies all four friends have -/
def total_guppies : ℕ := 84

/-- The number of guppies Nicolai has -/
def nicolai_guppies : ℕ := total_guppies - (haylee_guppies + jose_guppies + charliz_guppies)

/-- Theorem stating that the ratio of Nicolai's guppies to Charliz's guppies is 4:1 -/
theorem guppies_ratio : nicolai_guppies / charliz_guppies = 4 := by sorry

end guppies_ratio_l717_71739


namespace product_remainder_by_10_l717_71761

theorem product_remainder_by_10 : (8623 * 2475 * 56248 * 1234) % 10 = 0 := by
  sorry

end product_remainder_by_10_l717_71761


namespace triangle_max_area_l717_71786

theorem triangle_max_area (a b c : ℝ) (h1 : a = 75) (h2 : c = 2 * b) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (∀ x : ℝ, x > 0 → area ≤ 1100) ∧ (∃ x : ℝ, x > 0 ∧ area = 1100) :=
by sorry

end triangle_max_area_l717_71786


namespace flute_cost_l717_71712

/-- Calculates the cost of a flute given the total amount spent and the costs of other items --/
theorem flute_cost (total_spent music_stand_cost song_book_cost : ℚ) : 
  total_spent = 158.35 ∧ music_stand_cost = 8.89 ∧ song_book_cost = 7 →
  total_spent - (music_stand_cost + song_book_cost) = 142.46 := by
  sorry

end flute_cost_l717_71712


namespace divisibility_implication_l717_71701

theorem divisibility_implication (x y : ℤ) : 
  (23 ∣ (3 * x + 2 * y)) → (23 ∣ (17 * x + 19 * y)) := by
  sorry

end divisibility_implication_l717_71701


namespace article_cost_price_l717_71728

theorem article_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.75 * 1250)
  (h2 : selling_price_increased = selling_price + 500)
  (h3 : selling_price_increased = 1.15 * 1250) : 1250 = 1250 := by
  sorry

end article_cost_price_l717_71728


namespace quadratic_single_intersection_l717_71754

/-- 
A quadratic function f(x) = ax^2 - ax + 3x + 1 intersects the x-axis at only one point 
if and only if a = 1 or a = 9.
-/
theorem quadratic_single_intersection (a : ℝ) : 
  (∃! x, a * x^2 - a * x + 3 * x + 1 = 0) ↔ (a = 1 ∨ a = 9) := by
  sorry

end quadratic_single_intersection_l717_71754


namespace no_single_common_tangent_for_equal_circles_l717_71721

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a function to count common tangents between two circles
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem no_single_common_tangent_for_equal_circles (c1 c2 : Circle) :
  c1.radius = c2.radius → c1 ≠ c2 → countCommonTangents c1 c2 ≠ 1 := by
  sorry

end no_single_common_tangent_for_equal_circles_l717_71721


namespace negation_of_existence_l717_71740

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 2^x₀ * (x₀ - a) > 1) ↔
  (∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1) :=
by sorry

end negation_of_existence_l717_71740


namespace unique_modular_solution_l717_71727

theorem unique_modular_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -567 [ZMOD 13] ∧ n = 5 := by
  sorry

end unique_modular_solution_l717_71727


namespace t_shape_perimeter_l717_71776

/-- The perimeter of a T shape formed by a vertical rectangle and a horizontal rectangle -/
def t_perimeter (v_width v_height h_width h_height : ℝ) : ℝ :=
  2 * v_height + 2 * h_width + h_height

/-- Theorem: The perimeter of the T shape is 22 inches -/
theorem t_shape_perimeter :
  t_perimeter 2 6 3 2 = 22 := by
  sorry

end t_shape_perimeter_l717_71776


namespace intersection_point_is_unique_l717_71759

/-- A line in 3D space defined by the equation (x-2)/2 = (y-2)/(-1) = (z-4)/3 -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + 2*t, 2 - t, 4 + 3*t)

/-- A plane in 3D space defined by the equation x + 3y + 5z - 42 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  x + 3*y + 5*z - 42 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (4, 1, 7)

theorem intersection_point_is_unique :
  ∃! t : ℝ, line t = intersection_point ∧ plane (line t) :=
sorry

end intersection_point_is_unique_l717_71759


namespace log_domain_intersection_l717_71703

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem log_domain_intersection :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end log_domain_intersection_l717_71703


namespace number_ratio_problem_l717_71729

theorem number_ratio_problem (N : ℝ) : 
  (1/3 : ℝ) * (2/5 : ℝ) * N = 15 ∧ (40/100 : ℝ) * N = 180 → 
  15 / N = 1 / 7.5 :=
by sorry

end number_ratio_problem_l717_71729


namespace line_contains_point_l717_71745

/-- A line in the xy-plane is represented by the equation 2 - kx = -4y for some real number k. -/
def line (k : ℝ) (x y : ℝ) : Prop := 2 - k * x = -4 * y

/-- The point (2, -1) lies on the line. -/
def point_on_line (k : ℝ) : Prop := line k 2 (-1)

/-- The value of k for which the line contains the point (2, -1) is -1. -/
theorem line_contains_point : ∃! k : ℝ, point_on_line k ∧ k = -1 := by sorry

end line_contains_point_l717_71745


namespace geometric_relations_l717_71719

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelLL : Line → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularLL : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem geometric_relations 
  (a b c : Line) (α β γ : Plane) :
  -- Proposition 2
  (skew a b ∧ 
   contains α a ∧ 
   contains β b ∧ 
   parallelLP a β ∧ 
   parallelLP b α → 
   parallel α β) ∧
  -- Proposition 3
  (intersect α β a ∧ 
   intersect β γ b ∧ 
   intersect γ α c ∧ 
   parallelLL a b → 
   parallelLP c β) ∧
  -- Proposition 4
  (skew a b ∧ 
   parallelLP a α ∧ 
   parallelLP b α ∧ 
   perpendicularLL c a ∧ 
   perpendicularLL c b → 
   perpendicularLP c α) :=
sorry

end geometric_relations_l717_71719


namespace circumradius_inradius_ratio_rational_l717_71774

/-- Given a triangle with rational side lengths, prove that the ratio of its circumradius to inradius is rational. -/
theorem circumradius_inradius_ratio_rational 
  (a b c : ℚ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p : ℚ := (a + b + c) / 2
  ∃ (q : ℚ), q = a * b * c / (4 * (p - a) * (p - b) * (p - c)) :=
sorry

end circumradius_inradius_ratio_rational_l717_71774


namespace simple_interest_rate_percent_l717_71772

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 720)
  (h2 : interest = 180)
  (h3 : time = 4)
  : (interest * 100) / (principal * time) = 6.25 := by
  sorry

end simple_interest_rate_percent_l717_71772


namespace hexagon_intersection_collinearity_l717_71735

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Represents a hexagon -/
structure Hexagon :=
  (A B C D E F : Point)

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

/-- Returns the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Theorem: Collinearity of intersections in a hexagon with specific conditions -/
theorem hexagon_intersection_collinearity 
  (ABCDEF : Hexagon)
  (diagonalIntersection : Point)
  (hDiagonals : intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection ∧ 
                intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection ∧ 
                intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection)
  (A' : Point) (hA' : A' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (B' : Point) (hB' : B' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (C' : Point) (hC' : C' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (D' E' F' : Point)
  : collinear 
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0)) :=
by sorry

end hexagon_intersection_collinearity_l717_71735


namespace problem_statement_l717_71758

theorem problem_statement (a c : ℤ) : 
  (∃ (x : ℤ), x^2 = 2*a - 1 ∧ (x = 3 ∨ x = -3)) → 
  c = ⌊Real.sqrt 17⌋ → 
  a + c = 9 :=
by sorry

end problem_statement_l717_71758


namespace curve_to_line_equation_l717_71720

/-- The curve parameterized by (x,y) = (3t + 6, 5t - 7) can be expressed as y = (5/3)x - 17 --/
theorem curve_to_line_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 → y = (5/3) * x - 17 := by
  sorry

end curve_to_line_equation_l717_71720


namespace min_distance_circle_to_line_l717_71718

/-- The minimum distance from a point on the circle x^2 + y^2 = 1 to the line 3x + 4y - 25 = 0 is 4 -/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y - 25 = 0}
  ∃ (d : ℝ), d = 4 ∧ ∀ (p : ℝ × ℝ), p ∈ circle →
    ∀ (q : ℝ × ℝ), q ∈ line →
      d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end min_distance_circle_to_line_l717_71718


namespace expand_expression_l717_71723

theorem expand_expression (x y : ℝ) : (x + 10) * (2 * y + 10) = 2 * x * y + 10 * x + 20 * y + 100 := by
  sorry

end expand_expression_l717_71723


namespace function_value_problem_l717_71711

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f (x / 2 - 1) = 2 * x + 3) →
  f m = 6 →
  m = -1/4 := by
    sorry

end function_value_problem_l717_71711


namespace tangent_line_to_parabola_l717_71722

theorem tangent_line_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x → (∃! x₀ y₀ : ℝ, y₀ = 3*x₀ + c ∧ y₀^2 = 12*x₀)) → 
  c = 1 := by
sorry

end tangent_line_to_parabola_l717_71722


namespace roots_of_polynomial_l717_71775

theorem roots_of_polynomial : ∃ (a b c d : ℂ), 
  (a = 2 ∧ b = -2 ∧ c = 2*I ∧ d = -2*I) ∧ 
  (∀ x : ℂ, x^4 + 4*x^3 - 2*x^2 - 20*x + 24 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
sorry

end roots_of_polynomial_l717_71775


namespace time_to_hospital_l717_71709

/-- Proves that given a distance of 0.09 kilometers to the hospital and a speed of 3 meters per 4 seconds, it takes 120 seconds for Ayeon to reach the hospital. -/
theorem time_to_hospital (distance_km : ℝ) (speed_m : ℝ) (speed_s : ℝ) : 
  distance_km = 0.09 →
  speed_m = 3 →
  speed_s = 4 →
  (distance_km * 1000) / (speed_m / speed_s) = 120 := by
sorry

end time_to_hospital_l717_71709


namespace f_always_negative_iff_a_in_range_l717_71782

/-- A quadratic function f(x) = ax^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

/-- The property that f(x) is always less than 0 on ℝ -/
def always_negative (a : ℝ) : Prop := ∀ x, f a x < 0

/-- Theorem stating that f(x) is always negative if and only if a is in the interval (-4, 0] -/
theorem f_always_negative_iff_a_in_range :
  ∀ a : ℝ, always_negative a ↔ -4 < a ∧ a ≤ 0 := by sorry

end f_always_negative_iff_a_in_range_l717_71782


namespace area_is_two_l717_71732

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (A B C : Circle) : Prop :=
  A.radius = 1 ∧
  B.radius = 1 ∧
  C.radius = 1 ∧
  -- A and B are tangent
  dist A.center B.center = 2 ∧
  -- C is tangent to the midpoint of AB
  C.center.1 = (A.center.1 + B.center.1) / 2 ∧
  C.center.2 = (A.center.2 + B.center.2) / 2 + 1

-- Define the area function
def area_inside_C_outside_AB (A B C : Circle) : ℝ := sorry

-- Theorem statement
theorem area_is_two (A B C : Circle) :
  problem_setup A B C → area_inside_C_outside_AB A B C = 2 := by sorry

end area_is_two_l717_71732


namespace equation_solution_l717_71793

theorem equation_solution : ∃! x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ∧ x ≠ 4 :=
  by sorry

end equation_solution_l717_71793


namespace cubic_equation_solution_l717_71741

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 21 = 0) → 
  (3 * q^2 - 5 * q - 21 = 0) → 
  p ≠ q →
  (9 * p^3 - 9 * q^3) / (p - q) = 88 := by
sorry

end cubic_equation_solution_l717_71741


namespace fraction_of_fraction_l717_71755

theorem fraction_of_fraction (a b c d e f : ℚ) :
  a = 2 → b = 9 → c = 5 → d = 6 → e = 3 → f = 4 →
  (a/b * c/d) / (e/f) = 20/81 := by sorry

end fraction_of_fraction_l717_71755


namespace combined_height_l717_71783

theorem combined_height (kirill_height brother_height : ℕ) : 
  kirill_height = 49 →
  brother_height = kirill_height + 14 →
  kirill_height + brother_height = 112 := by
sorry

end combined_height_l717_71783


namespace apps_deleted_l717_71762

/-- Proves that Dave deleted 8 apps given the initial conditions -/
theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) : 
  initial_apps = 16 →
  remaining_apps = initial_apps / 2 →
  initial_apps - remaining_apps = 8 := by
sorry

end apps_deleted_l717_71762


namespace harper_consumption_l717_71708

/-- Represents the mineral water consumption problem -/
structure MineralWaterConsumption where
  bottles_per_case : ℕ
  cost_per_case : ℚ
  total_spent : ℚ
  days_supply : ℕ

/-- Calculates the daily mineral water consumption given the problem parameters -/
def daily_consumption (m : MineralWaterConsumption) : ℚ :=
  (m.total_spent / m.cost_per_case * m.bottles_per_case) / m.days_supply

/-- Theorem stating that Harper's daily mineral water consumption is 0.5 bottles -/
theorem harper_consumption :
  ∃ (m : MineralWaterConsumption),
    m.bottles_per_case = 24 ∧
    m.cost_per_case = 12 ∧
    m.total_spent = 60 ∧
    m.days_supply = 240 ∧
    daily_consumption m = 1/2 := by
  sorry

end harper_consumption_l717_71708


namespace complex_number_subtraction_l717_71733

theorem complex_number_subtraction : (5 * Complex.I) - (2 + 2 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end complex_number_subtraction_l717_71733


namespace negation_of_proposition_l717_71751

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l717_71751


namespace soccer_camp_ratio_l717_71799

/-- Proves the ratio of kids going to soccer camp in the morning to the total number of kids going to soccer camp -/
theorem soccer_camp_ratio (total_kids : ℕ) (soccer_kids : ℕ) (afternoon_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : soccer_kids = total_kids / 2)
  (h3 : afternoon_kids = 750) :
  (soccer_kids - afternoon_kids) / soccer_kids = 1 / 4 := by
  sorry

#check soccer_camp_ratio

end soccer_camp_ratio_l717_71799


namespace units_digit_of_sum_of_factorials_49_l717_71757

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_sum_of_factorials_49 :
  (sum_of_factorials 49) % 10 = 3 := by
  sorry

end units_digit_of_sum_of_factorials_49_l717_71757


namespace power_sum_seven_l717_71765

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^7 + b^7 = 29 -/
theorem power_sum_seven (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h6 : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^7 + b^7 = 29 := by
sorry

end power_sum_seven_l717_71765


namespace right_triangle_from_equation_l717_71705

theorem right_triangle_from_equation (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 + 338 = 10*a + 24*b + 26*c) : 
  a^2 + b^2 = c^2 := by
sorry

end right_triangle_from_equation_l717_71705


namespace M_intersect_N_eq_one_l717_71743

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

theorem M_intersect_N_eq_one : M ∩ N = {1} := by sorry

end M_intersect_N_eq_one_l717_71743


namespace constant_ratio_l717_71715

/-- Two arithmetic sequences with sums of first n terms S_n and T_n -/
def arithmetic_sequences (S T : ℕ → ℝ) : Prop :=
  ∃ (a₁ d_a b₁ d_b : ℝ),
    ∀ n : ℕ, 
      S n = n / 2 * (2 * a₁ + (n - 1) * d_a) ∧
      T n = n / 2 * (2 * b₁ + (n - 1) * d_b)

/-- The product of sums equals n^3 - n for all positive n -/
def product_condition (S T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ+, S n * T n = (n : ℝ)^3 - n

/-- The main theorem: if the conditions are satisfied, then S_n / T_n is constant -/
theorem constant_ratio 
  (S T : ℕ → ℝ) 
  (h1 : arithmetic_sequences S T) 
  (h2 : product_condition S T) : 
  ∃ c : ℝ, ∀ n : ℕ+, S n / T n = c :=
sorry

end constant_ratio_l717_71715


namespace not_equal_to_three_halves_l717_71771

theorem not_equal_to_three_halves : ∃ x : ℚ, x ≠ (3/2 : ℚ) ∧
  (x = (5/3 : ℚ)) ∧
  ((9/6 : ℚ) = (3/2 : ℚ)) ∧
  ((3/2 : ℚ) = (3/2 : ℚ)) ∧
  ((7/4 : ℚ) = (3/2 : ℚ)) ∧
  ((9/6 : ℚ) = (3/2 : ℚ)) :=
by sorry

end not_equal_to_three_halves_l717_71771


namespace standard_flowchart_property_l717_71730

/-- Represents a flowchart --/
structure Flowchart where
  start_points : Nat
  end_points : Nat

/-- A flowchart is standard if it has exactly one start point and at least one end point --/
def is_standard (f : Flowchart) : Prop :=
  f.start_points = 1 ∧ f.end_points ≥ 1

/-- Theorem stating that a standard flowchart has exactly one start point and at least one end point --/
theorem standard_flowchart_property (f : Flowchart) (h : is_standard f) :
  f.start_points = 1 ∧ f.end_points ≥ 1 := by
  sorry

end standard_flowchart_property_l717_71730


namespace positive_real_inequalities_l717_71714

theorem positive_real_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + 3*b^2 ≥ 2*b*(a + b)) ∧ (a^3 + b^3 ≥ a*b^2 + a^2*b) := by
  sorry

end positive_real_inequalities_l717_71714


namespace seaweed_for_human_consumption_l717_71778

/-- Given that:
  - 400 pounds of seaweed are harvested
  - 50% of seaweed is used for starting fires
  - 150 pounds are fed to livestock
Prove that 25% of the remaining seaweed after starting fires can be eaten by humans -/
theorem seaweed_for_human_consumption 
  (total_seaweed : ℝ) 
  (fire_seaweed_percentage : ℝ) 
  (livestock_seaweed : ℝ) 
  (h1 : total_seaweed = 400)
  (h2 : fire_seaweed_percentage = 0.5)
  (h3 : livestock_seaweed = 150) :
  let remaining_seaweed := total_seaweed * (1 - fire_seaweed_percentage)
  let human_seaweed := remaining_seaweed - livestock_seaweed
  human_seaweed / remaining_seaweed = 0.25 := by
sorry

end seaweed_for_human_consumption_l717_71778


namespace marble_group_size_l717_71790

theorem marble_group_size : 
  ∀ (x : ℕ), 
  (144 / x : ℚ) = (144 / (x + 2) : ℚ) + 1 → 
  x = 16 := by
sorry

end marble_group_size_l717_71790


namespace factor_implies_h_value_l717_71713

theorem factor_implies_h_value (m h : ℝ) : 
  (∃ k : ℝ, m^2 - h*m - 24 = (m - 8) * k) → h = 5 := by
sorry

end factor_implies_h_value_l717_71713


namespace smallest_integer_satisfying_conditions_l717_71791

theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ 
  (∃ (k : ℕ), (5 * n) / 3 = k + 2/3) ∧
  (∃ (k : ℕ), (7 * n) / 5 = k + 2/5) ∧
  (∃ (k : ℕ), (9 * n) / 7 = k + 2/7) ∧
  (∃ (k : ℕ), (11 * n) / 9 = k + 2/9) ∧
  (∀ (m : ℕ), m > 1 → 
    ((∃ (k : ℕ), (5 * m) / 3 = k + 2/3) ∧
     (∃ (k : ℕ), (7 * m) / 5 = k + 2/5) ∧
     (∃ (k : ℕ), (9 * m) / 7 = k + 2/7) ∧
     (∃ (k : ℕ), (11 * m) / 9 = k + 2/9)) → m ≥ n) ∧
  n = 316 :=
by sorry

end smallest_integer_satisfying_conditions_l717_71791


namespace two_points_same_color_distance_l717_71770

-- Define a type for colors
inductive Color
| Yellow
| Red

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem two_points_same_color_distance (x : ℝ) (h : x > 0) :
  ∃ (c : Color) (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry

end two_points_same_color_distance_l717_71770


namespace proportional_relationship_l717_71766

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 7 when z = 16, prove that x = 7/9 when z = 144 -/
theorem proportional_relationship (k m : ℝ) (h1 : k > 0) (h2 : m > 0) : 
  (∀ x y z : ℝ, x = k * y^2 ∧ y = m / Real.sqrt z → 
    (x = 7 ∧ z = 16 → x * z = 112) ∧
    (z = 144 → x = 7/9)) := by
  sorry

end proportional_relationship_l717_71766


namespace nanning_gdp_scientific_notation_l717_71792

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem nanning_gdp_scientific_notation :
  let gdp : ℝ := 1060 * 10^9  -- 1060 billion
  let scientific_form := toScientificNotation gdp
  scientific_form.coefficient = 1.06 ∧ scientific_form.exponent = 11 :=
by sorry

end nanning_gdp_scientific_notation_l717_71792


namespace hyperbola_intersecting_line_l717_71768

/-- Given a hyperbola and an ellipse with specific properties, prove the equation of a line intersecting the hyperbola. -/
theorem hyperbola_intersecting_line 
  (a : ℝ) 
  (h_a_pos : a > 0)
  (C : Set (ℝ × ℝ)) 
  (h_C : C = {(x, y) | x^2/a^2 - y^2/4 = 1})
  (E : Set (ℝ × ℝ))
  (h_E : E = {(x, y) | x^2/16 + y^2/8 = 1})
  (h_foci : {(-4, 0), (4, 0)} ⊆ C)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ C ∧ B ∈ C)
  (h_midpoint : (A.1 + B.1)/2 = 6 ∧ (A.2 + B.2)/2 = 1) :
  ∃ (k m : ℝ), k * A.1 + m * A.2 = 1 ∧ k * B.1 + m * B.2 = 1 ∧ k = 2 ∧ m = -1 :=
sorry

end hyperbola_intersecting_line_l717_71768


namespace line_symmetry_l717_71764

-- Define the original line
def original_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the line of symmetry
def symmetry_line (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_line x₀ y₀ ∧ symmetry_line x₀) →
  (symmetric_line x y ↔ 
    ∃ (x₁ y₁ : ℝ), original_line x₁ y₁ ∧ 
    x - x₀ = x₀ - x₁ ∧ 
    y - y₀ = y₀ - y₁ ∧
    symmetry_line x₀) :=
by sorry

end line_symmetry_l717_71764


namespace arithmetic_geometric_properties_l717_71702

/-- Given an arithmetic progression {a_n} with common difference d,
    where a_3, a_4, and a_8 form a geometric progression,
    prove certain properties about the sequence and its sum. -/
theorem arithmetic_geometric_properties
  (a : ℕ → ℝ)  -- The sequence a_n
  (d : ℝ)      -- Common difference
  (h1 : d ≠ 0) -- d is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d)  -- Arithmetic progression property
  (h3 : (a 4) ^ 2 = a 3 * a 8)     -- Geometric progression property
  : a 1 * d < 0 ∧ 
    d * (4 * a 1 + 6 * d) < 0 ∧
    (a 4 / a 3 = 4) :=
by sorry

end arithmetic_geometric_properties_l717_71702


namespace exponent_multiplication_l717_71796

theorem exponent_multiplication (a : ℝ) : 2 * a^2 * a^4 = 2 * a^6 := by
  sorry

end exponent_multiplication_l717_71796


namespace two_digit_number_digit_difference_l717_71704

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 27, prove that the difference 
between the two digits of the number is 3.
-/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 27 → x - y = 3 := by
  sorry

end two_digit_number_digit_difference_l717_71704


namespace triangle_abc_properties_l717_71734

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  a = 2 →
  c = 3 →
  B = π / 3 ∧ Real.sin C = (3 * Real.sqrt 14) / 14 := by
sorry

end triangle_abc_properties_l717_71734


namespace bread_theorem_l717_71731

def bread_problem (slices_per_loaf : ℕ) (num_friends : ℕ) (num_loaves : ℕ) : ℕ :=
  (slices_per_loaf * num_loaves) / num_friends

theorem bread_theorem :
  bread_problem 15 10 4 = 6 := by
  sorry

end bread_theorem_l717_71731


namespace non_adjacent_arrangements_l717_71773

def number_of_people : ℕ := 6

def number_of_gaps (n : ℕ) : ℕ := n + 1

def permutations (n : ℕ) : ℕ := n.factorial

def arrangements_with_gaps (n : ℕ) : ℕ :=
  permutations (n - 2) * (number_of_gaps (n - 2)).choose 2

theorem non_adjacent_arrangements :
  arrangements_with_gaps number_of_people = 480 := by sorry

end non_adjacent_arrangements_l717_71773


namespace arithmetic_sequence_problem_l717_71710

/-- Given an arithmetic sequence {aₙ}, prove that if a₅ + a₁₁ = 30 and a₄ = 7, then a₁₂ = 23 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_sum : a 5 + a 11 = 30)
  (h_a4 : a 4 = 7) :
  a 12 = 23 := by
sorry

end arithmetic_sequence_problem_l717_71710


namespace inequality_range_inequality_solution_l717_71788

-- Part 1
theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (1 - a) * x + a - 2 ≥ -2) ↔ a ∈ Set.Ici (1/3) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x < 1 }
  else if a > 0 then { x | -1/a < x ∧ x < 1 }
  else if a = -1 then { x | x ≠ 1 }
  else if a < -1 then { x | x > 1 ∨ x < -1/a }
  else { x | x < 1 ∨ x > -1/a }

theorem inequality_solution (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 + (1 - a) * x + a - 2 < a - 1 :=
sorry

end inequality_range_inequality_solution_l717_71788


namespace binomial_coefficient_divisibility_equivalence_l717_71798

theorem binomial_coefficient_divisibility_equivalence 
  (n : ℕ) (p : ℕ) (h_prime : Prime p) : 
  (∀ k : ℕ, k ≤ n → ¬(p ∣ Nat.choose n k)) ↔ 
  (∃ s m : ℕ, s > 0 ∧ m < p ∧ n = p^s * m - 1) :=
sorry

end binomial_coefficient_divisibility_equivalence_l717_71798


namespace cube_coverage_l717_71725

/-- Represents a paper strip of size 3 × 1 -/
structure PaperStrip :=
  (length : Nat := 3)
  (width : Nat := 1)

/-- Represents a cube of size n × n × n -/
structure Cube (n : Nat) :=
  (side_length : Nat := n)

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : Nat) : Prop := n % 3 = 0

/-- Predicate to check if it's possible to cover three sides of a cube with paper strips -/
def can_cover_sides (c : Cube n) (p : PaperStrip) : Prop :=
  divisible_by_three n

/-- Theorem stating the condition for covering three sides of a cube with paper strips -/
theorem cube_coverage (n : Nat) :
  ∀ (c : Cube n) (p : PaperStrip),
    can_cover_sides c p ↔ divisible_by_three n :=
by sorry

end cube_coverage_l717_71725


namespace geometric_sequence_ratio_l717_71777

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

/-- Given conditions for the geometric sequence -/
def GeometricSequenceConditions (a : ℕ → ℝ) : Prop :=
  GeometricSequence a ∧ (a 5 * a 11 = 4) ∧ (a 3 + a 13 = 5)

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h : GeometricSequenceConditions a) : 
  (a 14 / a 4 = 4) ∨ (a 14 / a 4 = 1/4) :=
sorry

end geometric_sequence_ratio_l717_71777


namespace loan_duration_C_l717_71736

/-- Calculates simple interest -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_duration_C (principal_B principal_C total_interest : ℚ) 
  (time_B : ℚ) (rate : ℚ) :
  principal_B = 4000 →
  principal_C = 2000 →
  time_B = 2 →
  rate = 13.75 →
  total_interest = 2200 →
  simpleInterest principal_B rate time_B + simpleInterest principal_C rate (4 : ℚ) = total_interest :=
by sorry

end loan_duration_C_l717_71736


namespace point_above_line_l717_71742

/-- A point is above a line if its y-coordinate is greater than the y-coordinate of the point on the line with the same x-coordinate. -/
def IsAboveLine (x y : ℝ) (a b c : ℝ) : Prop :=
  y > (a * x + c) / b

/-- The theorem states that for a point P(-2, t) to be above the line 2x - 3y + 6 = 0, t must be greater than 2/3. -/
theorem point_above_line (t : ℝ) :
  IsAboveLine (-2) t 2 (-3) 6 ↔ t > 2/3 := by
  sorry

#check point_above_line

end point_above_line_l717_71742


namespace largest_number_with_equal_quotient_and_remainder_l717_71781

theorem largest_number_with_equal_quotient_and_remainder (A B C : ℕ) 
  (h1 : A = 8 * B + C) 
  (h2 : B = C) 
  (h3 : C < 8) : 
  A ≤ 63 ∧ ∃ (A' : ℕ), A' = 63 ∧ ∃ (B' C' : ℕ), A' = 8 * B' + C' ∧ B' = C' ∧ C' < 8 :=
sorry

end largest_number_with_equal_quotient_and_remainder_l717_71781


namespace middle_term_of_arithmetic_sequence_l717_71760

def arithmetic_sequence (a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (a₂ - a₁) = (a₃ - a₂) ∧ (a₃ - a₂) = (a₄ - a₃) ∧ (a₄ - a₃) = (a₅ - a₄)

theorem middle_term_of_arithmetic_sequence (x z : ℝ) :
  arithmetic_sequence 23 x 38 z 53 → 38 = (23 + 53) / 2 := by
  sorry

end middle_term_of_arithmetic_sequence_l717_71760


namespace multiplication_problem_l717_71785

theorem multiplication_problem : ∃ x : ℕ, 72516 * x = 724797420 ∧ x = 10001 := by sorry

end multiplication_problem_l717_71785


namespace sum_of_fractions_nonnegative_l717_71700

theorem sum_of_fractions_nonnegative (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + 
  (33 * b^2 - b) / (33 * b^2 + 1) + 
  (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 := by
  sorry

end sum_of_fractions_nonnegative_l717_71700


namespace units_digit_of_expression_l717_71780

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the expression (35)^7 + (93)^45 -/
def expression : ℕ := 35^7 + 93^45

/-- Theorem stating that the units digit of (35)^7 + (93)^45 is 8 -/
theorem units_digit_of_expression : unitsDigit expression = 8 := by
  sorry

end units_digit_of_expression_l717_71780


namespace inequality_range_l717_71750

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2*Real.sqrt 3 := by
sorry

end inequality_range_l717_71750
