import Mathlib

namespace fourth_circle_radius_l1717_171761

/-- Represents a configuration of seven consecutively tangent circles between two parallel lines -/
structure CircleConfiguration where
  radii : Fin 7 → ℝ
  largest_radius : radii 6 = 24
  smallest_radius : radii 0 = 6
  tangent : ∀ i : Fin 6, radii i < radii (i.succ)

/-- The theorem stating that the radius of the fourth circle is 12√2 -/
theorem fourth_circle_radius (config : CircleConfiguration) : config.radii 3 = 12 * Real.sqrt 2 := by
  sorry

end fourth_circle_radius_l1717_171761


namespace fraction_equality_l1717_171762

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_equality_l1717_171762


namespace quadratic_roots_properties_l1717_171700

-- Define the quadratic equation and its roots
def quadratic_eq (p q x : ℝ) := x^2 + p*x + q = 0

theorem quadratic_roots_properties (p q : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic_eq p q x₁) (h₂ : quadratic_eq p q x₂) (h₃ : x₁ ≠ x₂) :
  (1/x₁ + 1/x₂ = -p/q) ∧
  (1/x₁^2 + 1/x₂^2 = (p^2 - 2*q)/q^2) ∧
  (x₁^3 + x₂^3 = -p^3 + 3*p*q) ∧
  (1/(x₁ + p)^2 + 1/(x₂ + p)^2 = (p^2 - 2*q)/q^2) :=
by sorry

end quadratic_roots_properties_l1717_171700


namespace games_to_sell_l1717_171738

def playstation_cost : ℝ := 500
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5

theorem games_to_sell : 
  ⌈(playstation_cost - (birthday_money + christmas_money)) / game_price⌉ = 20 := by sorry

end games_to_sell_l1717_171738


namespace pen_pencil_cost_l1717_171777

/-- Given a pen and pencil where the pen costs twice as much as the pencil and the pen costs $4,
    prove that the total cost of the pen and pencil is $6. -/
theorem pen_pencil_cost (pen_cost pencil_cost : ℝ) : 
  pen_cost = 4 → pen_cost = 2 * pencil_cost → pen_cost + pencil_cost = 6 := by
sorry

end pen_pencil_cost_l1717_171777


namespace function_composition_equality_l1717_171739

theorem function_composition_equality (f g : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = x / 6 + 2) → 
  (∀ x, g x = 5 - 2 * x) → 
  f (g b) = 4 → 
  b = -7 / 2 := by
sorry

end function_composition_equality_l1717_171739


namespace set_union_problem_l1717_171791

theorem set_union_problem (M N : Set ℕ) (a : ℕ) :
  M = {a, 0} ∧ N = {1, 2} ∧ M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end set_union_problem_l1717_171791


namespace custom_op_value_l1717_171768

-- Define the custom operation *
def custom_op (a b : ℤ) : ℚ := 1 / a + 1 / b

-- State the theorem
theorem custom_op_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 9) (prod_eq : a * b = 20) : 
  custom_op a b = 9 / 20 := by
  sorry

end custom_op_value_l1717_171768


namespace max_garden_area_l1717_171799

/-- Given 420 feet of fencing to enclose a rectangular garden on three sides
    (with the fourth side against a wall), the maximum area that can be achieved
    is 22050 square feet. -/
theorem max_garden_area (fencing : ℝ) (h : fencing = 420) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * l + w = fencing ∧
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → 2 * l' + w' = fencing →
  l * w ≥ l' * w' ∧ l * w = 22050 := by
  sorry

end max_garden_area_l1717_171799


namespace factorization_equality_l1717_171735

theorem factorization_equality (x y : ℝ) : 2 * x^2 - 4 * x * y = 2 * x * (x - 2 * y) := by
  sorry

end factorization_equality_l1717_171735


namespace min_horizontal_distance_l1717_171795

/-- Parabola equation: y = x^2 - x - 2 -/
def parabola (x : ℝ) : ℝ := x^2 - x - 2

/-- Point P has y-coordinate 10 -/
def point_P : Set ℝ := {x : ℝ | parabola x = 10}

/-- Point Q has y-coordinate 0 -/
def point_Q : Set ℝ := {x : ℝ | parabola x = 0}

/-- The horizontal distance between two x-coordinates -/
def horizontal_distance (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem min_horizontal_distance :
  ∃ (p q : ℝ), p ∈ point_P ∧ q ∈ point_Q ∧
  ∀ (p' q' : ℝ), p' ∈ point_P → q' ∈ point_Q →
  horizontal_distance p q ≤ horizontal_distance p' q' ∧
  horizontal_distance p q = 2 :=
sorry

end min_horizontal_distance_l1717_171795


namespace cos_double_angle_special_point_l1717_171704

/-- Given that the terminal side of angle α passes through point (1,2), prove that cos 2α = -3/5 -/
theorem cos_double_angle_special_point (α : ℝ) :
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = 2) →
  Real.cos (2 * α) = -3/5 := by
  sorry

end cos_double_angle_special_point_l1717_171704


namespace ln_inequality_l1717_171765

theorem ln_inequality (x : ℝ) (h : x > 1) : 2 * Real.log x < x - 1 / x := by
  sorry

end ln_inequality_l1717_171765


namespace teachers_survey_l1717_171784

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 80)
  (h_heart_trouble : heart_trouble = 50)
  (h_both : both = 30) :
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 100/3 := by
sorry

end teachers_survey_l1717_171784


namespace water_pouring_proof_l1717_171770

/-- Calculates the fraction of water remaining after n rounds -/
def water_remaining (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 1/2
  | 2 => 1/3
  | k + 3 => water_remaining (k + 2) * (2 * (k + 3)) / (2 * (k + 3) + 1)

/-- The number of rounds needed to reach exactly 1/5 of the original water -/
def rounds_to_one_fifth : ℕ := 6

theorem water_pouring_proof :
  water_remaining rounds_to_one_fifth = 1/5 :=
sorry

end water_pouring_proof_l1717_171770


namespace elevator_weight_problem_l1717_171780

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (final_people : ℕ) (final_avg_weight : ℝ) :
  initial_people = 6 ∧ 
  initial_avg_weight = 160 ∧ 
  final_people = 7 ∧ 
  final_avg_weight = 151 →
  (final_people * final_avg_weight) - (initial_people * initial_avg_weight) = 97 := by
sorry

end elevator_weight_problem_l1717_171780


namespace points_per_bag_l1717_171786

theorem points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) :
  total_bags = 17 →
  unrecycled_bags = 8 →
  total_points = 45 →
  (total_points / (total_bags - unrecycled_bags) : ℚ) = 5 := by
  sorry

end points_per_bag_l1717_171786


namespace no_solution_for_sock_problem_l1717_171752

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1) : ℚ) = 1/2 := by
  sorry

end no_solution_for_sock_problem_l1717_171752


namespace regular_polygon_sides_l1717_171727

theorem regular_polygon_sides (n : ℕ) (angle_OAB : ℝ) : 
  n > 0 → 
  angle_OAB = 72 → 
  (360 : ℝ) / angle_OAB = n → 
  n = 5 := by
  sorry

end regular_polygon_sides_l1717_171727


namespace sqrt_sum_squares_ge_sqrt2_sum_l1717_171769

theorem sqrt_sum_squares_ge_sqrt2_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end sqrt_sum_squares_ge_sqrt2_sum_l1717_171769


namespace f_is_even_and_decreasing_l1717_171753

def f (x : ℝ) := -x^2 + 1

theorem f_is_even_and_decreasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y → f y < f x) :=
by sorry

end f_is_even_and_decreasing_l1717_171753


namespace janet_pay_calculation_l1717_171708

/-- Represents Janet's work parameters and calculates her pay per post. -/
def janet_pay_per_post (check_time : ℕ) (hourly_rate : ℚ) : ℚ :=
  let seconds_per_hour : ℕ := 3600
  let posts_per_hour : ℕ := seconds_per_hour / check_time
  hourly_rate / posts_per_hour

/-- Proves that Janet's pay per post is $0.25 given the specified conditions. -/
theorem janet_pay_calculation :
  janet_pay_per_post 10 90 = 1/4 := by
  sorry

#eval janet_pay_per_post 10 90

end janet_pay_calculation_l1717_171708


namespace interior_point_is_center_of_gravity_l1717_171778

/-- A lattice point represented by its x and y coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle represented by its three vertices -/
structure LatticeTriangle where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint

/-- Checks if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Checks if a point is in the interior of a triangle -/
def isInterior (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Calculates the center of gravity of a triangle -/
def centerOfGravity (t : LatticeTriangle) : LatticePoint := sorry

/-- The main theorem -/
theorem interior_point_is_center_of_gravity 
  (t : LatticeTriangle) 
  (h1 : t.v1 = ⟨0, 0⟩) 
  (h2 : ∀ p : LatticePoint, p ≠ t.v1 ∧ p ≠ t.v2 ∧ p ≠ t.v3 → ¬isOnBoundary p t) 
  (p : LatticePoint) 
  (h3 : isInterior p t) 
  (h4 : ∀ q : LatticePoint, q ≠ p → ¬isInterior q t) : 
  p = centerOfGravity t := by
  sorry

end interior_point_is_center_of_gravity_l1717_171778


namespace equal_remainders_theorem_l1717_171793

theorem equal_remainders_theorem (p : ℕ) (x : ℕ) (h_prime : Nat.Prime p) (h_pos : x > 0) :
  (∃ r : ℕ, x % p = r ∧ p^2 % x = r) →
  ((x = p ∧ p % x = 0) ∨ (x = p^2 ∧ p^2 % x = 0) ∨ (x = p + 1 ∧ p^2 % x = 1)) :=
sorry

end equal_remainders_theorem_l1717_171793


namespace committee_combinations_l1717_171785

theorem committee_combinations : Nat.choose 8 5 = 56 := by
  sorry

end committee_combinations_l1717_171785


namespace b_value_l1717_171781

theorem b_value (b : ℚ) (h : b + b/4 - 1 = 3/2) : b = 2 := by
  sorry

end b_value_l1717_171781


namespace two_digit_number_ending_with_zero_l1717_171763

/-- A two-digit number -/
structure TwoDigitNumber where
  value : ℕ
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Reverse the digits of a two-digit number -/
def reverse_digits (n : TwoDigitNumber) : ℕ :=
  (n.value % 10) * 10 + (n.value / 10)

/-- Check if a natural number is a perfect fourth power -/
def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

theorem two_digit_number_ending_with_zero (N : TwoDigitNumber) :
  (N.value - reverse_digits N > 0) →
  is_perfect_fourth_power (N.value - reverse_digits N) →
  N.value % 10 = 0 := by sorry

end two_digit_number_ending_with_zero_l1717_171763


namespace mixture_alcohol_percentage_l1717_171743

/-- Represents the properties of an alcohol solution -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the volume of alcohol in a solution -/
def alcoholVolume (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

/-- Theorem: Adding 50 mL of 30% alcohol solution to 200 mL of 10% alcohol solution results in 14% alcohol solution -/
theorem mixture_alcohol_percentage 
  (x : Solution) 
  (y : Solution) 
  (h1 : x.volume = 200)
  (h2 : x.alcoholPercentage = 0.1)
  (h3 : y.volume = 50)
  (h4 : y.alcoholPercentage = 0.3) :
  let finalSolution : Solution := {
    volume := x.volume + y.volume,
    alcoholPercentage := (alcoholVolume x + alcoholVolume y) / (x.volume + y.volume)
  }
  finalSolution.alcoholPercentage = 0.14 := by
  sorry

#check mixture_alcohol_percentage

end mixture_alcohol_percentage_l1717_171743


namespace rectangle_area_change_l1717_171712

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let L' := 1.3 * L
  let B' := 0.75 * B
  let A := L * B
  let A' := L' * B'
  A' / A = 0.975 := by sorry

end rectangle_area_change_l1717_171712


namespace circle_area_l1717_171782

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area (θ : Real) (r : Real → Real) :
  (r = fun θ ↦ 3 * Real.cos θ - 4 * Real.sin θ) →
  (∀ θ, ∃ x y : Real, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
  (∃ c : Real × Real, ∃ radius : Real, ∀ x y : Real,
    (x - c.1)^2 + (y - c.2)^2 = radius^2 ↔ ∃ θ : Real, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
  (π * (5/2)^2 : Real) = 25*π/4 := by
sorry

end circle_area_l1717_171782


namespace remainder_3012_div_97_l1717_171705

theorem remainder_3012_div_97 : 3012 % 97 = 5 := by
  sorry

end remainder_3012_div_97_l1717_171705


namespace polygon_sides_l1717_171722

theorem polygon_sides (n : ℕ) (angle_sum : ℝ) (excluded_angle : ℝ) :
  angle_sum = 2970 ∧
  angle_sum = (n - 2) * 180 - 2 * excluded_angle ∧
  excluded_angle > 0 ∧
  excluded_angle < 180 →
  n = 19 :=
by sorry

end polygon_sides_l1717_171722


namespace positive_real_solution_floor_product_l1717_171702

theorem positive_real_solution_floor_product (x : ℝ) : 
  x > 0 → x * ⌊x⌋ = 72 → x = 9 := by sorry

end positive_real_solution_floor_product_l1717_171702


namespace import_tax_calculation_l1717_171741

theorem import_tax_calculation (total_value : ℝ) (tax_rate : ℝ) (threshold : ℝ) (tax_amount : ℝ) : 
  total_value = 2580 →
  tax_rate = 0.07 →
  threshold = 1000 →
  tax_amount = (total_value - threshold) * tax_rate →
  tax_amount = 110.60 := by
sorry

end import_tax_calculation_l1717_171741


namespace oplus_equation_solution_l1717_171747

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 4 * a + 2 * b

-- Theorem statement
theorem oplus_equation_solution :
  ∃ y : ℝ, oplus 3 (oplus 4 y) = -14 ∧ y = -14.5 := by
sorry

end oplus_equation_solution_l1717_171747


namespace cistern_emptying_l1717_171725

/-- Represents the rate at which a pipe can empty a cistern -/
structure EmptyingRate where
  fraction : ℚ
  time : ℕ

/-- Represents the operation of pipes emptying a cistern -/
def empty_cistern (pipe1 : EmptyingRate) (pipe2 : EmptyingRate) (time1 : ℕ) (time2 : ℕ) : ℚ :=
  sorry

theorem cistern_emptying :
  let pipe1 : EmptyingRate := ⟨3/4, 12⟩
  let pipe2 : EmptyingRate := ⟨1/2, 15⟩
  empty_cistern pipe1 pipe2 4 10 = 7/12 :=
by sorry

end cistern_emptying_l1717_171725


namespace initial_girls_count_l1717_171714

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 18) = b) →
  (4 * (b - 36) = g - 18) →
  g = 31 :=
by sorry

end initial_girls_count_l1717_171714


namespace inequalities_comparison_l1717_171794

theorem inequalities_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  ((1/2 : ℝ)^a > (1/2 : ℝ)^b) ∧
  (1/a > 1/b) ∧
  (b^2 > a^2) ∧
  (¬(Real.log a > Real.log b)) := by
  sorry

end inequalities_comparison_l1717_171794


namespace fold_and_cut_square_l1717_171715

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents a folding operation -/
def Fold := Point → Point

/-- Checks if a line intersects all four 1x1 squares in a 2x2 square -/
def intersectsAllSquares (l : Line) : Prop :=
  ∃ (p1 p2 p3 p4 : Point),
    p1.x + p1.y = 1 ∧ 
    p2.x + p2.y = 3 ∧ 
    p3.x - p3.y = 1 ∧ 
    p4.x - p4.y = -1 ∧
    l.a * p1.x + l.b * p1.y + l.c = 0 ∧
    l.a * p2.x + l.b * p2.y + l.c = 0 ∧
    l.a * p3.x + l.b * p3.y + l.c = 0 ∧
    l.a * p4.x + l.b * p4.y + l.c = 0

/-- The main theorem stating that it's possible to fold and cut a 2x2 square into four 1x1 squares -/
theorem fold_and_cut_square : 
  ∃ (f1 f2 : Fold) (l : Line),
    intersectsAllSquares l :=
sorry

end fold_and_cut_square_l1717_171715


namespace f_neg_two_value_l1717_171771

-- Define f as a function from R to R
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (2 * x) + x^2

-- State the theorem
theorem f_neg_two_value (h1 : f 2 = 2) (h2 : ∀ x, g x = -g (-x)) : f (-2) = -4 := by
  sorry

end f_neg_two_value_l1717_171771


namespace wharf_length_l1717_171737

/-- The length of the wharf in meters -/
def L_wharf : ℝ := 64

/-- The average speed in meters per second -/
def V_avg : ℝ := 2

/-- The travel time in seconds -/
def T_travel : ℝ := 16

/-- Theorem: The length of the wharf is 64 meters -/
theorem wharf_length : L_wharf = 2 * V_avg * T_travel := by
  sorry

end wharf_length_l1717_171737


namespace convex_hull_of_37gons_has_at_least_37_sides_l1717_171745

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of regular polygons -/
def SetOfPolygons (n : ℕ) := Set (RegularPolygon n)

/-- The convex hull of a set of points in ℝ² -/
def ConvexHull (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The number of sides in the convex hull of a set of points -/
def NumSides (S : Set (ℝ × ℝ)) : ℕ := sorry

/-- The vertices of all polygons in a set -/
def AllVertices (S : SetOfPolygons n) : Set (ℝ × ℝ) := sorry

/-- Theorem: The convex hull of any set of regular 37-gons has at least 37 sides -/
theorem convex_hull_of_37gons_has_at_least_37_sides (S : SetOfPolygons 37) :
  NumSides (ConvexHull (AllVertices S)) ≥ 37 := by sorry

end convex_hull_of_37gons_has_at_least_37_sides_l1717_171745


namespace banana_theorem_l1717_171757

/-- The number of pounds of bananas purchased by a grocer -/
def banana_problem (buy_rate : ℚ) (sell_rate : ℚ) (total_profit : ℚ) : ℚ :=
  total_profit / (sell_rate - buy_rate)

theorem banana_theorem :
  banana_problem (1/6) (1/4) 11 = 132 := by
  sorry

#eval banana_problem (1/6) (1/4) 11

end banana_theorem_l1717_171757


namespace area_code_digits_l1717_171749

/-- Represents the set of allowed digits -/
def allowed_digits : Finset ℕ := {2, 3, 4}

/-- Calculates the number of valid area codes for a given number of digits -/
def valid_codes (n : ℕ) : ℕ := 3^n - 1

/-- The actual number of valid codes as per the problem statement -/
def actual_valid_codes : ℕ := 26

/-- The theorem stating that the number of digits in each area code is 3 -/
theorem area_code_digits :
  ∃ (n : ℕ), n > 0 ∧ valid_codes n = actual_valid_codes ∧ n = 3 := by
  sorry


end area_code_digits_l1717_171749


namespace no_real_solutions_for_abs_equation_l1717_171723

theorem no_real_solutions_for_abs_equation :
  ∀ x : ℝ, |2*x - 6| ≠ x^2 - x + 2 := by
sorry

end no_real_solutions_for_abs_equation_l1717_171723


namespace screen_width_calculation_l1717_171792

theorem screen_width_calculation (height width diagonal : ℝ) : 
  height / width = 3 / 4 →
  height^2 + width^2 = diagonal^2 →
  diagonal = 36 →
  width = 28.8 :=
by sorry

end screen_width_calculation_l1717_171792


namespace seven_balls_two_boxes_at_least_two_in_first_l1717_171710

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes -/
def total_distributions (n : ℕ) : ℕ := 2^n

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes
    where the first box contains exactly k balls -/
def distributions_with_k_in_first_box (n k : ℕ) : ℕ := n.choose k

theorem seven_balls_two_boxes_at_least_two_in_first : 
  total_distributions 7 - (distributions_with_k_in_first_box 7 0 + distributions_with_k_in_first_box 7 1) = 120 := by
  sorry

end seven_balls_two_boxes_at_least_two_in_first_l1717_171710


namespace car_speed_problem_l1717_171716

/-- Proves that the speed at which a car takes 15 seconds less to travel 1 kilometer
    compared to traveling at 48 km/h is 60 km/h. -/
theorem car_speed_problem (v : ℝ) : v > 0 →
  (1 / (48 / 3600) = 1 / (v / 3600) + 15) → v = 60 := by
  sorry

end car_speed_problem_l1717_171716


namespace smallest_number_of_eggs_l1717_171706

theorem smallest_number_of_eggs (total_eggs : ℕ) (containers : ℕ) : 
  total_eggs > 130 →
  total_eggs = 15 * containers - 3 →
  (∀ n : ℕ, n > 130 ∧ ∃ m : ℕ, n = 15 * m - 3 → n ≥ total_eggs) →
  total_eggs = 132 :=
by sorry

end smallest_number_of_eggs_l1717_171706


namespace choir_size_l1717_171731

/-- Represents a choir split into three groups -/
structure Choir :=
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)

/-- The total number of members in the choir -/
def Choir.total (c : Choir) : ℕ :=
  c.group1 + c.group2 + c.group3

/-- Theorem stating that a choir with the given group sizes has 70 members -/
theorem choir_size (c : Choir) 
  (h1 : c.group1 = 25) 
  (h2 : c.group2 = 30) 
  (h3 : c.group3 = 15) : 
  c.total = 70 := by
  sorry

/-- The specific choir instance from the problem -/
def problem_choir : Choir :=
  { group1 := 25
    group2 := 30
    group3 := 15 }

#eval Choir.total problem_choir

end choir_size_l1717_171731


namespace symmetric_function_sum_zero_l1717_171728

theorem symmetric_function_sum_zero 
  (v : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc (-1.75) 1.75, v (-x) = -v x) : 
  v (-1.75) + v (-0.5) + v 0.5 + v 1.75 = 0 := by
  sorry

end symmetric_function_sum_zero_l1717_171728


namespace total_revenue_proof_l1717_171748

def planned_daily_sales : ℕ := 100

def sales_data : List ℤ := [7, -5, -3, 13, -6, 12, 5]

def selling_price : ℚ := 5.5

def shipping_cost : ℚ := 2

def net_income_per_kg : ℚ := selling_price - shipping_cost

def total_planned_sales : ℕ := planned_daily_sales * 7

def actual_sales : ℤ := total_planned_sales + (sales_data.sum)

theorem total_revenue_proof :
  (actual_sales : ℚ) * net_income_per_kg = 2530.5 := by sorry

end total_revenue_proof_l1717_171748


namespace problem_statement_l1717_171796

def f (a x : ℝ) : ℝ := |x - 1| + |x - a|

theorem problem_statement (a : ℝ) (h1 : a > 1) :
  (∀ x, f a x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2) →
  a = 2 ∧ 
  (∀ x, f a x + |x - 1| ≥ 1 → a ∈ Set.Ici 2) :=
by sorry

end problem_statement_l1717_171796


namespace cycle_price_calculation_l1717_171787

theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1020)
  (h2 : loss_percentage = 15) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1200 := by
  sorry

end cycle_price_calculation_l1717_171787


namespace one_fourth_of_hundred_equals_ten_percent_of_250_l1717_171707

theorem one_fourth_of_hundred_equals_ten_percent_of_250 : 
  (1 / 4 : ℚ) * 100 = (10 / 100 : ℚ) * 250 := by
  sorry

end one_fourth_of_hundred_equals_ten_percent_of_250_l1717_171707


namespace closest_to_division_l1717_171740

def options : List ℝ := [0.2, 2, 20, 200, 2000]

theorem closest_to_division (x y : ℝ) (h1 : y ≠ 0) :
  ∃ z ∈ options, ∀ w ∈ options, |x / y - z| ≤ |x / y - w| :=
sorry

end closest_to_division_l1717_171740


namespace inequalities_proof_l1717_171758

theorem inequalities_proof (a b c d : ℝ) : 
  ((a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧ 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) := by
  sorry

end inequalities_proof_l1717_171758


namespace line_y_intercept_l1717_171711

/-- A straight line in the xy-plane with slope 2 and passing through (259, 520) has y-intercept 2 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x y, f y = 2 * x + f 0) →  -- slope is 2
  f 520 = 2 * 259 + f 0 →      -- point (259, 520) lies on the line
  f 0 = 2 := by               -- y-intercept is 2
sorry

end line_y_intercept_l1717_171711


namespace sum_with_twenty_equals_thirty_l1717_171718

theorem sum_with_twenty_equals_thirty (x : ℝ) : 20 + x = 30 → x = 10 := by
  sorry

end sum_with_twenty_equals_thirty_l1717_171718


namespace correct_division_l1717_171797

theorem correct_division (x : ℝ) (h : x / 1.5 = 3.8) : x / 6 = 0.95 := by
  sorry

end correct_division_l1717_171797


namespace words_with_A_count_l1717_171759

def letter_set : Finset Char := {'A', 'B', 'C', 'D', 'E'}

/-- The number of 4-letter words using letters A, B, C, D, E with repetition allowed -/
def total_words : ℕ := (Finset.card letter_set) ^ 4

/-- The number of 4-letter words using only B, C, D, E with repetition allowed -/
def words_without_A : ℕ := ((Finset.card letter_set) - 1) ^ 4

/-- The number of 4-letter words using A, B, C, D, E with repetition, containing at least one A -/
def words_with_A : ℕ := total_words - words_without_A

theorem words_with_A_count : words_with_A = 369 := by sorry

end words_with_A_count_l1717_171759


namespace not_all_even_numbers_representable_l1717_171760

theorem not_all_even_numbers_representable :
  ∃ k : ℕ, k > 1000 ∧ k % 2 = 0 ∧
  ∀ m n : ℕ, k ≠ n * (n + 1) * (n + 2) - m * (m + 1) :=
by sorry

end not_all_even_numbers_representable_l1717_171760


namespace stock_price_calculation_l1717_171724

/-- Given an income, dividend rate, and investment amount, calculate the price of a stock. -/
theorem stock_price_calculation (income investment : ℚ) (dividend_rate : ℚ) : 
  income = 650 →
  dividend_rate = 1/10 →
  investment = 6240 →
  (investment / (income / dividend_rate)) * 100 = 96 := by
  sorry

end stock_price_calculation_l1717_171724


namespace quadratic_inequality_range_l1717_171751

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + (a - 1) * x + 1 > 0) ↔ (1 ≤ a ∧ a < 5) :=
by sorry

end quadratic_inequality_range_l1717_171751


namespace sticker_difference_l1717_171773

/-- Given two people with the same initial number of stickers, if one person uses 15 stickers
    and the other buys 18 stickers, the difference in their final number of stickers is 33. -/
theorem sticker_difference (initial_stickers : ℕ) : 
  (initial_stickers + 18) - (initial_stickers - 15) = 33 := by
  sorry

end sticker_difference_l1717_171773


namespace kate_red_balloons_l1717_171730

/-- Given Kate's balloon scenario, prove she initially had 2 red balloons -/
theorem kate_red_balloons (R : ℕ) : 
  (R + 2) / (R + 8 : ℚ) = 2/5 → R = 2 := by
  sorry

end kate_red_balloons_l1717_171730


namespace smallest_prime_q_l1717_171701

theorem smallest_prime_q (p : ℕ) : 
  Prime p → Prime (13 * p + 2) → (13 * p + 2) ≥ 41 := by
  sorry

end smallest_prime_q_l1717_171701


namespace unique_solution_l1717_171756

/-- The sequence x_n defined by x_n = n / (n + 2016) -/
def x (n : ℕ) : ℚ := n / (n + 2016)

/-- Theorem stating the unique solution for m and n -/
theorem unique_solution :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ x 2016 = x m * x n ∧ m = 6048 ∧ n = 4032 := by
  sorry

end unique_solution_l1717_171756


namespace partition_product_property_l1717_171732

theorem partition_product_property (S : Finset ℕ) (h : S = Finset.range (3^5 - 2) ∪ {3^5}) :
  ∀ (A B : Finset ℕ), A ∪ B = S → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c) :=
by sorry

end partition_product_property_l1717_171732


namespace soccer_team_size_l1717_171774

/-- The number of players prepared for a soccer game -/
def players_prepared (starting_players : ℕ) (first_half_subs : ℕ) (second_half_subs : ℕ) (non_playing_players : ℕ) : ℕ :=
  starting_players + first_half_subs + non_playing_players

theorem soccer_team_size :
  let starting_players : ℕ := 11
  let first_half_subs : ℕ := 2
  let second_half_subs : ℕ := 2 * first_half_subs
  let non_playing_players : ℕ := 7
  players_prepared starting_players first_half_subs second_half_subs non_playing_players = 20 := by
  sorry

end soccer_team_size_l1717_171774


namespace downstream_distance_l1717_171772

theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : travel_time = 3) : 
  boat_speed + stream_speed * travel_time = 84 := by
sorry

end downstream_distance_l1717_171772


namespace smallest_solutions_l1717_171720

def is_solution (k : ℕ) : Prop :=
  Real.cos (k^2 + 8^2 : ℝ) ^ 2 = 1

theorem smallest_solutions :
  (∀ k : ℕ, k > 0 ∧ k < 48 → ¬ is_solution k) ∧
  is_solution 48 ∧
  (∀ k : ℕ, k > 48 ∧ k < 53 → ¬ is_solution k) ∧
  is_solution 53 :=
sorry

end smallest_solutions_l1717_171720


namespace largest_eight_digit_even_digits_proof_l1717_171734

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  10000000 ≤ n ∧ n ≤ 99999999

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_even_digits_proof :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ m : Nat, is_eight_digit m ∧ contains_all_even_digits m →
    m ≤ largest_eight_digit_with_even_digits :=
by sorry

end largest_eight_digit_even_digits_proof_l1717_171734


namespace unique_integer_for_complex_sixth_power_l1717_171744

theorem unique_integer_for_complex_sixth_power : 
  ∃! (n : ℤ), ∃ (m : ℤ), (n + Complex.I) ^ 6 = m := by sorry

end unique_integer_for_complex_sixth_power_l1717_171744


namespace vanessa_saves_three_weeks_l1717_171750

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_needed := dress_cost - initial_savings
  let net_weekly_savings := weekly_allowance - weekly_spending
  (additional_needed + net_weekly_savings - 1) / net_weekly_savings

/-- Proves that Vanessa needs 3 weeks to save for the dress -/
theorem vanessa_saves_three_weeks :
  weeks_to_save 80 20 30 10 = 3 := by
  sorry

end vanessa_saves_three_weeks_l1717_171750


namespace triangle_ratio_sine_relation_l1717_171755

theorem triangle_ratio_sine_relation (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6 →
  (Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) + 
   Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) /
  Real.sin (Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))) = 2 := by
sorry

end triangle_ratio_sine_relation_l1717_171755


namespace total_sides_is_75_l1717_171775

/-- Represents the number of sides for each shape --/
def sides_of_shape (shape : String) : ℕ :=
  match shape with
  | "triangle" => 3
  | "square" => 4
  | "hexagon" => 6
  | "octagon" => 8
  | "circle" => 0
  | "pentagon" => 5
  | _ => 0

/-- Calculates the total number of sides for a given shape and quantity --/
def total_sides (shape : String) (quantity : ℕ) : ℕ :=
  (sides_of_shape shape) * quantity

/-- Represents the cookie cutter drawer --/
structure CookieCutterDrawer :=
  (top_layer : ℕ)
  (middle_layer_squares : ℕ)
  (middle_layer_hexagons : ℕ)
  (bottom_layer_octagons : ℕ)
  (bottom_layer_circles : ℕ)
  (bottom_layer_pentagons : ℕ)

/-- Calculates the total number of sides for all cookie cutters in the drawer --/
def total_sides_in_drawer (drawer : CookieCutterDrawer) : ℕ :=
  total_sides "triangle" drawer.top_layer +
  total_sides "square" drawer.middle_layer_squares +
  total_sides "hexagon" drawer.middle_layer_hexagons +
  total_sides "octagon" drawer.bottom_layer_octagons +
  total_sides "circle" drawer.bottom_layer_circles +
  total_sides "pentagon" drawer.bottom_layer_pentagons

/-- The cookie cutter drawer described in the problem --/
def emery_drawer : CookieCutterDrawer :=
  { top_layer := 6,
    middle_layer_squares := 4,
    middle_layer_hexagons := 2,
    bottom_layer_octagons := 3,
    bottom_layer_circles := 5,
    bottom_layer_pentagons := 1 }

theorem total_sides_is_75 :
  total_sides_in_drawer emery_drawer = 75 := by
  sorry

end total_sides_is_75_l1717_171775


namespace fourth_day_temperature_l1717_171754

/-- Given three temperatures and a four-day average, calculate the fourth temperature --/
theorem fourth_day_temperature 
  (temp1 temp2 temp3 : ℤ) 
  (average : ℚ) 
  (h1 : temp1 = -36)
  (h2 : temp2 = -15)
  (h3 : temp3 = -10)
  (h4 : average = -12)
  : (4 : ℚ) * average - (temp1 + temp2 + temp3 : ℚ) = 13 := by
  sorry

#check fourth_day_temperature

end fourth_day_temperature_l1717_171754


namespace average_children_in_families_with_children_l1717_171713

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : average_children = 3)
  (h3 : childless_families = 3)
  : (total_families * average_children) / (total_families - childless_families) = 4 := by
  sorry

end average_children_in_families_with_children_l1717_171713


namespace fraction_equals_zero_l1717_171717

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (4 * x^2 - 1) = 0 ↔ x = 5 := by sorry

end fraction_equals_zero_l1717_171717


namespace greatest_integer_problem_l1717_171733

theorem greatest_integer_problem : 
  ∃ (m : ℕ), m < 150 ∧ 
  (∃ (k : ℕ), m = 9 * k - 2) ∧ 
  (∃ (j : ℕ), m = 11 * j - 4) ∧
  (∀ (n : ℕ), n < 150 → 
    (∃ (k' : ℕ), n = 9 * k' - 2) → 
    (∃ (j' : ℕ), n = 11 * j' - 4) → 
    n ≤ m) ∧
  m = 142 := by
sorry

end greatest_integer_problem_l1717_171733


namespace impossible_filling_l1717_171776

/-- Represents a 7 × 3 table filled with 0s and 1s -/
def Table := Fin 7 → Fin 3 → Bool

/-- Checks if a 2 × 2 submatrix in the table has all the same values -/
def has_same_2x2_submatrix (t : Table) : Prop :=
  ∃ (i j : Fin 7) (k l : Fin 3), i < j ∧ k < l ∧
    t i k = t i l ∧ t i k = t j k ∧ t i k = t j l

/-- Theorem stating that any 7 × 3 table filled with 0s and 1s
    always has a 2 × 2 submatrix with all the same values -/
theorem impossible_filling :
  ∀ (t : Table), has_same_2x2_submatrix t :=
sorry

end impossible_filling_l1717_171776


namespace sodium_hydroxide_combined_l1717_171742

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between acetic acid and sodium hydroxide -/
structure Reaction where
  acetic_acid : Moles
  sodium_hydroxide : Moles
  sodium_acetate : Moles

/-- The reaction occurs in a 1:1 molar ratio -/
axiom reaction_ratio (r : Reaction) : r.acetic_acid = r.sodium_hydroxide

/-- The number of moles of sodium acetate formed equals the number of moles of acetic acid used -/
axiom sodium_acetate_formation (r : Reaction) : r.sodium_acetate = r.acetic_acid

theorem sodium_hydroxide_combined (r : Reaction) :
  r.sodium_hydroxide = r.sodium_acetate :=
by sorry

end sodium_hydroxide_combined_l1717_171742


namespace p_sufficient_not_necessary_l1717_171729

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := 0 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ 
  (∃ x, q x ∧ ¬(p x)) :=
sorry

end p_sufficient_not_necessary_l1717_171729


namespace childs_movie_ticket_cost_l1717_171709

/-- Proves that the cost of a child's movie ticket is $3 given the specified conditions. -/
theorem childs_movie_ticket_cost (total_money : ℚ) (adult_ticket_cost : ℚ) (num_children : ℕ) 
  (h1 : total_money = 35)
  (h2 : adult_ticket_cost = 8)
  (h3 : num_children = 9) :
  ∃ (child_ticket_cost : ℚ), 
    child_ticket_cost = 3 ∧ 
    adult_ticket_cost + num_children * child_ticket_cost ≤ total_money :=
by sorry

end childs_movie_ticket_cost_l1717_171709


namespace binomial_series_expansion_l1717_171766

theorem binomial_series_expansion (x : ℝ) (n : ℕ) (h : |x| < 1) :
  (1 / (1 - x))^n = 1 + ∑' k, (n + k - 1).choose (n - 1) * x^k :=
sorry

end binomial_series_expansion_l1717_171766


namespace count_integer_solutions_l1717_171779

/-- The number of integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
def integerSolutionCount : ℕ := 6

/-- The quadratic equation in question -/
def hasIntegerSolution (a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 + a*x + 9*a = 0

/-- The theorem stating that there are exactly 6 integer values of a for which
    the equation x^2 + ax + 9a = 0 has integer solutions for x -/
theorem count_integer_solutions :
  (∃! (s : Finset ℤ), s.card = integerSolutionCount ∧ ∀ a : ℤ, a ∈ s ↔ hasIntegerSolution a) :=
sorry

end count_integer_solutions_l1717_171779


namespace salmon_count_l1717_171719

theorem salmon_count (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261) 
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end salmon_count_l1717_171719


namespace ellipse_circle_tangent_l1717_171746

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def line_equation (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.m

def perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

theorem ellipse_circle_tangent (e : Ellipse) (a : Point) (l : Line) :
  ellipse_equation e a ∧ 
  a.x = 2 ∧ a.y = Real.sqrt 2 ∧
  ∃ (p q : Point),
    ellipse_equation e p ∧
    ellipse_equation e q ∧
    line_equation l p ∧
    line_equation l q ∧
    perpendicular p q →
  ∃ (r : ℝ), r = Real.sqrt (8/3) ∧
    ∀ (x y : ℝ), x^2 + y^2 = r^2 →
    ∃ (t : Point), line_equation l t ∧ t.x^2 + t.y^2 = r^2 :=
sorry

end ellipse_circle_tangent_l1717_171746


namespace computer_price_increase_l1717_171721

/-- Given a computer with original price x dollars, where 2x = 540,
    prove that after a 30% increase, the new price is 351 dollars. -/
theorem computer_price_increase (x : ℝ) (h1 : 2 * x = 540) :
  x * 1.3 = 351 := by
  sorry

end computer_price_increase_l1717_171721


namespace water_consumption_l1717_171736

theorem water_consumption (initial_water : ℚ) : 
  initial_water > 0 →
  let remaining_day1 := initial_water / 2
  let remaining_day2 := remaining_day1 * 2 / 3
  let remaining_day3 := remaining_day2 / 2
  remaining_day3 = 250 →
  initial_water = 1500 := by
sorry

end water_consumption_l1717_171736


namespace rectangle_triangle_length_l1717_171788

/-- Given a rectangle ABCD with side lengths and a triangle DEF inside it, 
    proves that EF has a specific length when certain conditions are met. -/
theorem rectangle_triangle_length (AB BC DE DF EF : ℝ) : 
  AB = 8 → 
  BC = 10 → 
  DE = DF → 
  (1/2 * DE * DF) = (1/3 * AB * BC) → 
  EF = (16 * Real.sqrt 15) / 3 := by
  sorry

end rectangle_triangle_length_l1717_171788


namespace reciprocal_inequality_l1717_171789

theorem reciprocal_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  1 / (b - c) > 1 / (a - c) := by
  sorry

end reciprocal_inequality_l1717_171789


namespace proposition_evaluation_l1717_171764

theorem proposition_evaluation : 
  let p : Prop := (2 + 4 = 7)
  let q : Prop := (∀ x : ℝ, x = 1 → x^2 ≠ 1)
  ¬(p ∧ q) ∧ (p ∨ q) := by
sorry

end proposition_evaluation_l1717_171764


namespace min_value_theorem_l1717_171703

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min :=
sorry

end min_value_theorem_l1717_171703


namespace triangle_ratio_theorem_l1717_171798

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : t.A = 60 * π / 180)  -- A = 60°
  (h2 : t.a = 3)             -- a = 3
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B)  -- Law of Sines
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C)  -- Law of Sines
  : (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 3 := by
  sorry

end triangle_ratio_theorem_l1717_171798


namespace quadratic_has_two_real_roots_root_greater_than_three_implies_k_greater_than_one_l1717_171790

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 - (2*k + 2)*x + 2*k + 1

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0 :=
sorry

-- Theorem 2: If one root is greater than 3, then k > 1
theorem root_greater_than_three_implies_k_greater_than_one (k : ℝ) :
  (∃ x : ℝ, quadratic_equation k x = 0 ∧ x > 3) → k > 1 :=
sorry

end quadratic_has_two_real_roots_root_greater_than_three_implies_k_greater_than_one_l1717_171790


namespace no_number_with_specific_digit_sums_l1717_171726

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: No natural number exists with sum of digits 1000 and sum of square's digits 1000000 -/
theorem no_number_with_specific_digit_sums :
  ¬ ∃ n : ℕ, sumOfDigits n = 1000 ∧ sumOfDigits (n^2) = 1000000 := by sorry

end no_number_with_specific_digit_sums_l1717_171726


namespace geometric_sequence_inequality_l1717_171767

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_q_positive : q > 0
  h_q_not_one : q ≠ 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- 
For a geometric sequence with positive terms and common ratio q where q > 0 and q ≠ 1,
a_n + a_{n+3} > a_{n+1} + a_{n+2} for all n
-/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  ∀ n, seq.a n + seq.a (n + 3) > seq.a (n + 1) + seq.a (n + 2) :=
sorry

end geometric_sequence_inequality_l1717_171767


namespace hyperbola_point_distance_to_origin_l1717_171783

/-- Given points F₁ and F₂ on the x-axis, and a point P satisfying the hyperbola equation,
    prove that the distance from P to the origin is √6/2 when P's y-coordinate is 1/2. -/
theorem hyperbola_point_distance_to_origin :
  ∀ (P : ℝ × ℝ),
  let F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 2, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  P.2 = 1/2 →
  dist P F₂ - dist P F₁ = 2 →
  dist P (0, 0) = Real.sqrt 6 / 2 :=
by sorry

end hyperbola_point_distance_to_origin_l1717_171783
