import Mathlib

namespace inequality_relationship_l1558_155828

theorem inequality_relationship (a b : ℝ) (ha : a > 0) (hb : -1 < b ∧ b < 0) :
  a * b < a * b^2 ∧ a * b^2 < a := by
  sorry

end inequality_relationship_l1558_155828


namespace meeting_percentage_is_35_percent_l1558_155850

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 35

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 2 * first_meeting_minutes

/-- Represents the duration of the third meeting in minutes -/
def third_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes

/-- Represents the percentage of the work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_35_percent : meeting_percentage = 35 := by
  sorry

end meeting_percentage_is_35_percent_l1558_155850


namespace complex_equation_solution_l1558_155899

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + b + 5*I = 9 + a*I → b = 4 := by
sorry

end complex_equation_solution_l1558_155899


namespace closest_point_l1558_155841

/-- The curve y = 3 - x^2 for x > 0 -/
def curve (x : ℝ) : ℝ := 3 - x^2

/-- The fixed point P(0, 2) -/
def P : ℝ × ℝ := (0, 2)

/-- A point Q on the curve -/
def Q (x : ℝ) : ℝ × ℝ := (x, curve x)

/-- The squared distance between P and Q -/
def distance_squared (x : ℝ) : ℝ := (x - P.1)^2 + (curve x - P.2)^2

/-- The theorem stating that (√2/2, 5/2) is the closest point to P on the curve -/
theorem closest_point :
  ∃ (x : ℝ), x > 0 ∧ 
  ∀ (y : ℝ), y > 0 → distance_squared x ≤ distance_squared y ∧
  Q x = (Real.sqrt 2 / 2, 5 / 2) :=
sorry

end closest_point_l1558_155841


namespace binomial_expansion_properties_l1558_155821

/-- Given that the binomial coefficients of the third and seventh terms
    in the expansion of (x+2)^n are equal, prove that n = 8 and
    the coefficient of the (k+1)th term is maximum when k = 5 or k = 6 -/
theorem binomial_expansion_properties (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 6) →
  (n = 8 ∧ 
   (∀ j : ℕ, j ≠ 5 ∧ j ≠ 6 → 
     Nat.choose 8 5 * 2^5 ≥ Nat.choose 8 j * 2^j ∧
     Nat.choose 8 6 * 2^6 ≥ Nat.choose 8 j * 2^j)) :=
by sorry

end binomial_expansion_properties_l1558_155821


namespace wheel_radii_problem_l1558_155823

theorem wheel_radii_problem (x : ℝ) : 
  (2 * x > 0) →  -- Ensure positive radii
  (1500 / (2 * Real.pi * x + 5) = 1875 / (4 * Real.pi * x - 5)) → 
  (x = 15 / (2 * Real.pi) ∧ 2 * x = 15 / Real.pi) :=
by sorry

end wheel_radii_problem_l1558_155823


namespace stratified_sampling_total_l1558_155852

theorem stratified_sampling_total (sample_size : ℕ) (model_a_count : ℕ) (total_model_b : ℕ) :
  sample_size = 80 →
  model_a_count = 50 →
  total_model_b = 1800 →
  (sample_size - model_a_count) * 60 = total_model_b →
  sample_size * 60 = 4800 := by
sorry

end stratified_sampling_total_l1558_155852


namespace bisecting_angle_tangent_l1558_155813

/-- A triangle with side lengths 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13

/-- A line that bisects both the perimeter and area of the triangle -/
structure BisectingLine where
  x : ℝ
  y : ℝ

/-- The two bisecting lines of the triangle -/
def bisecting_lines (t : RightTriangle) : Prod BisectingLine BisectingLine :=
  ⟨⟨10, -5⟩, ⟨7.5, -7.5⟩⟩

/-- The acute angle between the two bisecting lines -/
def bisecting_angle (t : RightTriangle) : ℝ := sorry

theorem bisecting_angle_tangent (t : RightTriangle) :
  let lines := bisecting_lines t
  let φ := bisecting_angle t
  Real.tan φ = 
    let v1 := lines.1
    let v2 := lines.2
    let dot_product := v1.x * v2.x + v1.y * v2.y
    let mag1 := Real.sqrt (v1.x^2 + v1.y^2)
    let mag2 := Real.sqrt (v2.x^2 + v2.y^2)
    let cos_φ := dot_product / (mag1 * mag2)
    Real.sqrt (1 - cos_φ^2) / cos_φ := by sorry

end bisecting_angle_tangent_l1558_155813


namespace expression_values_l1558_155896

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a^2 - b*c = b^2 - a*c) (h2 : b^2 - a*c = c^2 - a*b) :
  (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b) = 7/2) ∨
  (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b) = -7) := by
  sorry

#check expression_values

end expression_values_l1558_155896


namespace binomial_expansion_constant_term_l1558_155812

/-- 
Given that the third term of the expansion of (3x - 2/x)^n is a constant term,
prove that n = 8.
-/
theorem binomial_expansion_constant_term (n : ℕ) : 
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → 
    (Nat.choose n 2 * (3 * x - 2 / x)^(n - 2) * (-2 / x)^2 = c)) → 
  n = 8 := by
sorry

end binomial_expansion_constant_term_l1558_155812


namespace carrie_profit_calculation_l1558_155836

/-- Calculates Carrie's profit from making a wedding cake --/
theorem carrie_profit_calculation :
  let weekday_hours : ℕ := 5 * 4
  let weekend_hours : ℕ := 3 * 4
  let weekday_rate : ℚ := 35
  let weekend_rate : ℚ := 45
  let supply_cost : ℚ := 180
  let supply_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.07

  let total_earnings : ℚ := weekday_hours * weekday_rate + weekend_hours * weekend_rate
  let discounted_supply_cost : ℚ := supply_cost * (1 - supply_discount)
  let sales_tax : ℚ := total_earnings * sales_tax_rate
  let profit : ℚ := total_earnings - discounted_supply_cost - sales_tax

  profit = 991.20 := by sorry

end carrie_profit_calculation_l1558_155836


namespace total_unique_polygons_l1558_155856

/-- Represents a regular polyhedron --/
inductive RegularPolyhedron
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Returns the number of unique non-planar polygons for a given regular polyhedron --/
def num_unique_polygons (p : RegularPolyhedron) : Nat :=
  match p with
  | .Tetrahedron => 1
  | .Cube => 1
  | .Octahedron => 3
  | .Dodecahedron => 2
  | .Icosahedron => 3

/-- The list of all regular polyhedra --/
def all_polyhedra : List RegularPolyhedron :=
  [RegularPolyhedron.Tetrahedron, RegularPolyhedron.Cube, RegularPolyhedron.Octahedron,
   RegularPolyhedron.Dodecahedron, RegularPolyhedron.Icosahedron]

/-- Theorem stating that the total number of unique non-planar polygons for all regular polyhedra is 10 --/
theorem total_unique_polygons :
  (all_polyhedra.map num_unique_polygons).sum = 10 := by
  sorry

#eval (all_polyhedra.map num_unique_polygons).sum

end total_unique_polygons_l1558_155856


namespace probability_one_second_class_l1558_155867

/-- The probability of drawing exactly one second-class product from a batch of products -/
theorem probability_one_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (drawn : ℕ) :
  total = first_class + second_class →
  total = 100 →
  first_class = 90 →
  second_class = 10 →
  drawn = 4 →
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose total drawn =
    Nat.choose second_class 1 * Nat.choose first_class 3 / Nat.choose total drawn :=
by sorry

end probability_one_second_class_l1558_155867


namespace guessing_game_solution_l1558_155889

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem guessing_game_solution :
  ∃! n : ℕ,
    1 ≤ n ∧ n ≤ 99 ∧
    is_perfect_square n ∧
    ¬(n < 5) ∧
    (n < 7 ∨ n < 10 ∨ n ≥ 100) ∧
    n = 9 :=
by sorry

end guessing_game_solution_l1558_155889


namespace negation_of_universal_non_negative_square_l1558_155847

theorem negation_of_universal_non_negative_square (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end negation_of_universal_non_negative_square_l1558_155847


namespace range_of_f_l1558_155874

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = 4 ∧
  (∀ y, (∃ x, x ∈ [0, 3] ∧ f x = y) ↔ y ∈ [a, b]) :=
sorry

end range_of_f_l1558_155874


namespace square_sum_reciprocal_l1558_155830

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 3) : x^2 + (1 / x^2) = 7 := by
  sorry

end square_sum_reciprocal_l1558_155830


namespace mitch_weekly_earnings_is_118_80_l1558_155879

/-- Mitch's weekly earnings after expenses and taxes -/
def mitchWeeklyEarnings : ℝ :=
  let monToWedEarnings := 3 * 5 * 3
  let thuFriEarnings := 2 * 6 * 4
  let satEarnings := 4 * 6
  let sunEarnings := 5 * 8
  let totalEarnings := monToWedEarnings + thuFriEarnings + satEarnings + sunEarnings
  let afterExpenses := totalEarnings - 25
  let taxAmount := afterExpenses * 0.1
  afterExpenses - taxAmount

/-- Theorem stating that Mitch's weekly earnings after expenses and taxes is $118.80 -/
theorem mitch_weekly_earnings_is_118_80 :
  mitchWeeklyEarnings = 118.80 := by sorry

end mitch_weekly_earnings_is_118_80_l1558_155879


namespace square_difference_divided_l1558_155816

theorem square_difference_divided : (147^2 - 133^2) / 14 = 280 := by
  sorry

end square_difference_divided_l1558_155816


namespace monotonic_cubic_range_l1558_155814

/-- The function f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ -/
def is_monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ, then a ∈ [-√3, √3] -/
theorem monotonic_cubic_range (a : ℝ) :
  is_monotonic (fun x => -x^3 + a*x^2 - x - 1) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_cubic_range_l1558_155814


namespace train_distance_problem_l1558_155822

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 16) (h2 : v2 = 21) 
  (h3 : v1 > 0) (h4 : v2 > 0) (h5 : d > 0) : 
  (∃ (t : ℝ), t > 0 ∧ v1 * t + v2 * t = v1 * t + d + v2 * t) → 
  v1 * t + v2 * t = 444 :=
sorry

end train_distance_problem_l1558_155822


namespace set_intersection_theorem_l1558_155846

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- State the theorem
theorem set_intersection_theorem :
  (A ∩ B = {x | 0 < x ∧ x ≤ 2}) ∧
  (A ∩ (U \ B) = {x | x > 2}) := by sorry

end set_intersection_theorem_l1558_155846


namespace expression_evaluation_l1558_155833

theorem expression_evaluation : (3 * 4 * 6) * (1/3 + 1/4 + 1/6) = 54 := by
  sorry

end expression_evaluation_l1558_155833


namespace three_parallel_lines_theorem_l1558_155891

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Checks if three lines are coplanar -/
def are_coplanar (l1 l2 l3 : Line3D) : Prop := sorry

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- The number of planes determined by three lines -/
def planes_from_lines (l1 l2 l3 : Line3D) : ℕ := sorry

/-- The number of parts the space is divided into by these planes -/
def space_divisions (planes : ℕ) : ℕ := sorry

theorem three_parallel_lines_theorem (a b c : Line3D) 
  (h_parallel_ab : are_parallel a b)
  (h_parallel_bc : are_parallel b c)
  (h_parallel_ac : are_parallel a c)
  (h_not_coplanar : ¬ are_coplanar a b c) :
  planes_from_lines a b c = 3 ∧ space_divisions (planes_from_lines a b c) = 7 := by
  sorry

end three_parallel_lines_theorem_l1558_155891


namespace smallest_multiple_with_last_four_digits_l1558_155849

theorem smallest_multiple_with_last_four_digits (n : ℕ) : 
  (n % 10000 = 2020) → (n % 77 = 0) → (∀ m : ℕ, m < n → (m % 10000 ≠ 2020 ∨ m % 77 ≠ 0)) → n = 722020 :=
by sorry

end smallest_multiple_with_last_four_digits_l1558_155849


namespace farmer_plot_allocation_l1558_155864

theorem farmer_plot_allocation (x y : ℕ) (h : x ≠ y) :
  ∃ (a b : ℕ), a^2 + b^2 = 2 * (x^2 + y^2) :=
by
  sorry

end farmer_plot_allocation_l1558_155864


namespace vector_on_line_l1558_155895

/-- Given distinct vectors a and b in a real vector space, 
    prove that the vector (1/4)*a + (3/4)*b lies on the line passing through a and b. -/
theorem vector_on_line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/4 : ℝ) • a + (3/4 : ℝ) • b = a + t • (b - a) := by
  sorry

end vector_on_line_l1558_155895


namespace quadratic_solution_sum_l1558_155826

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + 15 = 0 ↔ x = (a : ℂ) + b * I ∨ x = (a : ℂ) - b * I) →
  a + b^2 = 79/25 := by
  sorry

end quadratic_solution_sum_l1558_155826


namespace multiplication_puzzle_solution_l1558_155875

theorem multiplication_puzzle_solution : 
  (78346 * 346 = 235038) ∧ (9374 * 82 = 768668) := by
  sorry

end multiplication_puzzle_solution_l1558_155875


namespace power_mod_thirteen_l1558_155854

theorem power_mod_thirteen : 6^4032 ≡ 1 [ZMOD 13] := by
  sorry

end power_mod_thirteen_l1558_155854


namespace emma_share_l1558_155892

theorem emma_share (total : ℕ) (ratio_daniel ratio_emma ratio_fiona : ℕ) (h1 : total = 153) (h2 : ratio_daniel = 3) (h3 : ratio_emma = 5) (h4 : ratio_fiona = 9) : 
  (ratio_emma * total) / (ratio_daniel + ratio_emma + ratio_fiona) = 45 := by
sorry

end emma_share_l1558_155892


namespace workshop_average_salary_l1558_155839

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 12)
  (h2 : technicians = 6)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 9000 :=
by
  sorry

#check workshop_average_salary

end workshop_average_salary_l1558_155839


namespace tagged_fish_in_second_catch_l1558_155807

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 250) 
  (h2 : initial_tagged = 50) 
  (h3 : second_catch = 50) :
  (initial_tagged : ℚ) / total_fish = (initial_tagged : ℚ) / second_catch → 
  (initial_tagged : ℚ) * second_catch / total_fish = 10 :=
by sorry

end tagged_fish_in_second_catch_l1558_155807


namespace negation_of_at_least_three_l1558_155805

-- Define a proposition for "at least n"
def at_least (n : ℕ) : Prop := sorry

-- Define a proposition for "at most n"
def at_most (n : ℕ) : Prop := sorry

-- State the given condition
axiom negation_rule : ∀ n : ℕ, ¬(at_least n) ↔ at_most (n - 1)

-- State the theorem to be proved
theorem negation_of_at_least_three : ¬(at_least 3) ↔ at_most 2 := by sorry

end negation_of_at_least_three_l1558_155805


namespace canvas_cost_l1558_155824

/-- Proves that the cost of canvases is $40.00 given the specified conditions -/
theorem canvas_cost (total_spent easel_cost paintbrush_cost canvas_cost : ℚ) : 
  total_spent = 90 ∧ 
  easel_cost = 15 ∧ 
  paintbrush_cost = 15 ∧ 
  total_spent = canvas_cost + (1/2 * canvas_cost) + easel_cost + paintbrush_cost →
  canvas_cost = 40 := by
sorry

end canvas_cost_l1558_155824


namespace expression_evaluation_l1558_155897

theorem expression_evaluation : (20 - 16) * (12 + 8) / 4 = 20 := by
  sorry

end expression_evaluation_l1558_155897


namespace T_is_three_rays_with_common_endpoint_l1558_155802

/-- The set T of points in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≤ 5) ∨
               (5 = y - 2 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≥ x + 3)}

/-- The common endpoint of the three rays -/
def common_endpoint : ℝ × ℝ := (2, 7)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 7}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 7 ∧ p.1 ≤ 2}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 5 ∧ p.1 ≤ 2}

/-- Theorem stating that T consists of three rays with a common endpoint -/
theorem T_is_three_rays_with_common_endpoint :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_endpoint ∈ ray1 ∧
  common_endpoint ∈ ray2 ∧
  common_endpoint ∈ ray3 :=
sorry

end T_is_three_rays_with_common_endpoint_l1558_155802


namespace museum_time_per_student_l1558_155876

theorem museum_time_per_student 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (time_per_group : ℕ) 
  (h1 : total_students = 18)
  (h2 : num_groups = 3)
  (h3 : time_per_group = 24)
  (h4 : total_students % num_groups = 0) -- Ensures equal division
  : (time_per_group * num_groups) / total_students = 4 := by
  sorry

end museum_time_per_student_l1558_155876


namespace function_properties_l1558_155878

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2*b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b (-1)) ∧
    (f a b (-1) = 2) ∧
    (a = 2 ∧ b = 1) ∧
    (∀ x ∈ Set.Icc (-1) 1, f a b x ≤ 6) ∧
    (∀ x ∈ Set.Icc (-1) 1, f a b x ≥ 50/27) ∧
    (∃ x ∈ Set.Icc (-1) 1, f a b x = 6) ∧
    (∃ x ∈ Set.Icc (-1) 1, f a b x = 50/27) :=
by
  sorry


end function_properties_l1558_155878


namespace max_value_of_objective_function_l1558_155868

def objective_function (x₁ x₂ : ℝ) : ℝ := 4 * x₁ + 6 * x₂

def feasible_region (x₁ x₂ : ℝ) : Prop :=
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ + x₂ ≤ 18 ∧ 0.5 * x₁ + x₂ ≤ 12 ∧ 2 * x₁ ≤ 24 ∧ 2 * x₂ ≤ 18

theorem max_value_of_objective_function :
  ∃ (x₁ x₂ : ℝ), feasible_region x₁ x₂ ∧
    ∀ (y₁ y₂ : ℝ), feasible_region y₁ y₂ →
      objective_function x₁ x₂ ≥ objective_function y₁ y₂ ∧
      objective_function x₁ x₂ = 84 :=
by sorry

end max_value_of_objective_function_l1558_155868


namespace M_binary_op_result_l1558_155820

def M : Set ℕ := {2, 3}

def binary_op (A : Set ℕ) : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem M_binary_op_result : binary_op M = {4, 5, 6} := by sorry

end M_binary_op_result_l1558_155820


namespace election_result_proof_l1558_155842

theorem election_result_proof (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 5500 →
  candidate_percentage = 35 / 100 →
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -1650 := by
  sorry

end election_result_proof_l1558_155842


namespace sqrt_15_factorial_simplification_l1558_155803

theorem sqrt_15_factorial_simplification :
  ∃ (a b : ℕ+) (q : ℚ),
    (a:ℝ) * Real.sqrt b = Real.sqrt (Nat.factorial 15) ∧
    q * (Nat.factorial 15 : ℚ) = (a * b : ℚ) ∧
    q = 1 / 30240 := by sorry

end sqrt_15_factorial_simplification_l1558_155803


namespace alfred_maize_storage_l1558_155810

/-- Calculates the total amount of maize Alfred has after 2 years of storage, theft, and donation -/
theorem alfred_maize_storage (
  monthly_storage : ℕ)  -- Amount of maize stored each month
  (storage_period : ℕ)   -- Storage period in years
  (stolen : ℕ)           -- Amount of maize stolen
  (donation : ℕ)         -- Amount of maize donated
  (h1 : monthly_storage = 1)
  (h2 : storage_period = 2)
  (h3 : stolen = 5)
  (h4 : donation = 8) :
  monthly_storage * (storage_period * 12) - stolen + donation = 27 :=
by
  sorry


end alfred_maize_storage_l1558_155810


namespace vector_triangle_inequality_l1558_155890

variable {V : Type*} [NormedAddCommGroup V]

theorem vector_triangle_inequality (a b : V) : ‖a + b‖ ≤ ‖a‖ + ‖b‖ := by
  sorry

end vector_triangle_inequality_l1558_155890


namespace triangle_angle_not_all_less_than_60_l1558_155831

theorem triangle_angle_not_all_less_than_60 : 
  ¬ (∀ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) → 
    (a + b + c = 180) → 
    (a < 60 ∧ b < 60 ∧ c < 60)) := by
  sorry

end triangle_angle_not_all_less_than_60_l1558_155831


namespace range_of_a_l1558_155869

theorem range_of_a (p q : Prop) (h_p : p ↔ ∀ x ∈ Set.Icc (1/2) 1, 1/x - a ≥ 0)
  (h_q : q ↔ ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) (h_pq : p ∧ q) :
  a ∈ Set.Iic (-2) ∪ {1} :=
sorry

end range_of_a_l1558_155869


namespace unique_prime_sum_of_squares_and_divisibility_l1558_155881

theorem unique_prime_sum_of_squares_and_divisibility (p m n : ℤ) : 
  Prime p → 
  p = m^2 + n^2 → 
  p ∣ m^3 + n^3 - 4 → 
  p = 2 :=
by sorry

end unique_prime_sum_of_squares_and_divisibility_l1558_155881


namespace number_equation_solution_l1558_155884

theorem number_equation_solution :
  ∃ N : ℝ, N - (1002 / 20.04) = 2450 ∧ N = 2500 := by sorry

end number_equation_solution_l1558_155884


namespace quadratic_distinct_roots_l1558_155882

theorem quadratic_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 1 = 0 ∧ x₂^2 + m*x₂ - 1 = 0 :=
sorry

end quadratic_distinct_roots_l1558_155882


namespace f_is_quadratic_l1558_155877

/-- A function f: ℝ → ℝ is quadratic if it can be expressed as f(x) = ax² + bx + c for some real constants a, b, and c, where a ≠ 0. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = (x-1)(x-2) -/
def f (x : ℝ) : ℝ := (x - 1) * (x - 2)

/-- Theorem: The function f(x) = (x-1)(x-2) is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f :=
sorry

end f_is_quadratic_l1558_155877


namespace soap_cost_theorem_l1558_155883

/-- The cost of soap for a year given the duration of a bar, its cost, and months in a year -/
def soap_cost_for_year (months_per_bar : ℚ) (cost_per_bar : ℚ) (months_in_year : ℕ) : ℚ :=
  (months_in_year / months_per_bar) * cost_per_bar

/-- Theorem: The cost of soap for a year is $48.00 given the specified conditions -/
theorem soap_cost_theorem :
  soap_cost_for_year 2 8 12 = 48 :=
by sorry

end soap_cost_theorem_l1558_155883


namespace cube_inequality_iff_l1558_155804

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end cube_inequality_iff_l1558_155804


namespace decimal_to_fraction_l1558_155838

theorem decimal_to_fraction : 
  ∃ (n d : ℕ), n / d = (38 : ℚ) / 100 ∧ gcd n d = 1 := by sorry

end decimal_to_fraction_l1558_155838


namespace parabola_directrix_l1558_155829

/-- The equation of the directrix of the parabola y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/8 * x^2) → (∃ p : ℝ, p = 4 ∧ y = p/2) :=
sorry

end parabola_directrix_l1558_155829


namespace length_of_BC_l1558_155855

-- Define the triangles and their properties
def triangle_ABC (AB AC BC : ℝ) : Prop :=
  AB^2 + AC^2 = BC^2 ∧ AB > 0 ∧ AC > 0 ∧ BC > 0

def triangle_ABD (AB AD BD : ℝ) : Prop :=
  AB^2 + AD^2 = BD^2 ∧ AB > 0 ∧ AD > 0 ∧ BD > 0

-- State the theorem
theorem length_of_BC :
  ∀ AB AC BC AD BD,
  triangle_ABC AB AC BC →
  triangle_ABD AB AD BD →
  AB = 12 →
  AC = 16 →
  AD = 30 →
  BC = 20 :=
sorry

end length_of_BC_l1558_155855


namespace card_shop_problem_l1558_155861

theorem card_shop_problem :
  ∃ (x y : ℕ), 1.25 * (x : ℝ) + 1.75 * (y : ℝ) = 18 := by
  sorry

end card_shop_problem_l1558_155861


namespace min_value_theorem_l1558_155827

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (2^x) + Real.log (8^y) = Real.log 2) : 
  1/x + 1/(3*y) ≥ 3 := by
  sorry

end min_value_theorem_l1558_155827


namespace greatest_divisor_with_remainders_l1558_155843

theorem greatest_divisor_with_remainders : Nat.gcd (1557 - 7) (2037 - 5) = 2 := by
  sorry

end greatest_divisor_with_remainders_l1558_155843


namespace chess_tournament_l1558_155886

theorem chess_tournament (n : ℕ) (k : ℚ) : n > 2 →
  (8 : ℚ) + n * k = (n + 2) * (n + 1) / 2 →
  (∀ m : ℕ, m > 2 → (8 : ℚ) + m * k ≠ (m + 2) * (m + 1) / 2 → m ≠ n) →
  n = 7 ∨ n = 14 := by
sorry

end chess_tournament_l1558_155886


namespace town_shoppers_count_l1558_155857

/-- Represents the shopping scenario in the town. -/
structure ShoppingScenario where
  stores : Nat
  total_visits : Nat
  double_visitors : Nat
  max_visits_per_person : Nat

/-- The specific shopping scenario described in the problem. -/
def town_scenario : ShoppingScenario :=
  { stores := 8
  , total_visits := 22
  , double_visitors := 8
  , max_visits_per_person := 3 }

/-- The number of people who went shopping given a shopping scenario. -/
def shoppers (s : ShoppingScenario) : Nat :=
  s.double_visitors + (s.total_visits - 2 * s.double_visitors) / s.max_visits_per_person

/-- Theorem stating that the number of shoppers in the town scenario is 10. -/
theorem town_shoppers_count :
  shoppers town_scenario = 10 := by
  sorry

end town_shoppers_count_l1558_155857


namespace percentage_failed_hindi_l1558_155887

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_english = 70)
  (h2 : failed_both = 10)
  (h3 : passed_both = 20) :
  ∃ failed_hindi : ℝ, failed_hindi = 20 ∧ 
    passed_both + (failed_hindi + failed_english - failed_both) = 100 :=
by sorry

end percentage_failed_hindi_l1558_155887


namespace snack_pack_distribution_l1558_155837

theorem snack_pack_distribution (pretzels : ℕ) (suckers : ℕ) (kids : ℕ) :
  pretzels = 64 →
  suckers = 32 →
  kids = 16 →
  (pretzels + 4 * pretzels + suckers) / kids = 22 := by
  sorry

end snack_pack_distribution_l1558_155837


namespace ratio_problem_l1558_155832

theorem ratio_problem (first_number second_number : ℚ) : 
  (first_number / second_number = 15) → 
  (first_number = 150) → 
  (second_number = 10) := by
sorry

end ratio_problem_l1558_155832


namespace range_of_x_l1558_155844

-- Define the sets
def S1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def S2 : Set ℝ := {x | x < 1 ∨ x > 4}

-- Define the condition
def condition (x : ℝ) : Prop := ¬(x ∈ S1 ∨ x ∈ S2)

-- State the theorem
theorem range_of_x : 
  ∀ x : ℝ, condition x → x ∈ {y : ℝ | 1 ≤ y ∧ y < 2} :=
sorry

end range_of_x_l1558_155844


namespace tangent_segment_length_l1558_155885

-- Define the circle and points
variable (circle : Type) (A B C P Q R : ℝ × ℝ)

-- Define the properties of tangents and points
def is_tangent (point : ℝ × ℝ) (touch_point : ℝ × ℝ) : Prop := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem tangent_segment_length 
  (h1 : is_tangent A B)
  (h2 : is_tangent A C)
  (h3 : is_tangent P Q)
  (h4 : is_tangent R Q)
  (h5 : distance P B = distance P R)
  (h6 : distance A B = 24) :
  distance P Q = 12 := by sorry

end tangent_segment_length_l1558_155885


namespace base4_multiplication_division_l1558_155825

-- Define a function to convert base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

-- Define a function to convert decimal to base 4
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the numbers in base 4 as lists of digits
def num1 : List Nat := [1, 3, 2]  -- 231₄
def num2 : List Nat := [1, 2]     -- 21₄
def num3 : List Nat := [3]        -- 3₄
def result : List Nat := [3, 3, 0, 2]  -- 2033₄

-- State the theorem
theorem base4_multiplication_division :
  decimalToBase4 ((base4ToDecimal num1 * base4ToDecimal num2) / base4ToDecimal num3) = result := by
  sorry

end base4_multiplication_division_l1558_155825


namespace sqrt_of_four_l1558_155894

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem stating that the square root of 4 is ±2
theorem sqrt_of_four : sqrt 4 = {2, -2} := by sorry

end sqrt_of_four_l1558_155894


namespace art_club_participation_l1558_155872

theorem art_club_participation (total : ℕ) (painting : ℕ) (sculpting : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : painting = 80)
  (h3 : sculpting = 60)
  (h4 : both = 20) :
  total - (painting + sculpting - both) = 30 := by
  sorry

end art_club_participation_l1558_155872


namespace rug_design_inner_length_l1558_155873

theorem rug_design_inner_length : 
  ∀ (y : ℝ), 
  let inner_area := 2 * y
  let middle_area := 6 * y + 24
  let outer_area := 10 * y + 80
  (middle_area - inner_area = outer_area - middle_area) →
  y = 4 := by
sorry

end rug_design_inner_length_l1558_155873


namespace intersection_A_B_intersection_complement_A_B_l1558_155835

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for (∁ₐA) ∩ B
theorem intersection_complement_A_B : (Aᶜ) ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

end intersection_A_B_intersection_complement_A_B_l1558_155835


namespace intersection_P_Q_l1558_155870

def P : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x > 4 ∨ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} := by sorry

end intersection_P_Q_l1558_155870


namespace isosceles_triangle_base_length_l1558_155863

/-- An isosceles triangle with a median dividing its perimeter -/
structure IsoscelesTriangleWithMedian where
  /-- Length of each leg of the isosceles triangle -/
  leg : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : leg > 0
  /-- The median on one leg divides the perimeter into parts of 6cm and 12cm -/
  medianDivision : leg + leg / 2 = 12 ∧ leg / 2 + base = 6
  /-- Triangle inequality -/
  triangleInequality : 2 * leg > base ∧ base > 0

/-- Theorem: The base of the isosceles triangle is 2cm -/
theorem isosceles_triangle_base_length
  (t : IsoscelesTriangleWithMedian) : t.base = 2 := by
  sorry

end isosceles_triangle_base_length_l1558_155863


namespace sum_of_digits_of_greatest_prime_divisor_l1558_155866

def number : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor number) = 19 := by sorry

end sum_of_digits_of_greatest_prime_divisor_l1558_155866


namespace brownie_solution_l1558_155817

/-- Represents the brownie distribution problem --/
def brownie_problem (total_brownies : ℕ) (total_cost : ℚ) (faculty_fraction : ℚ) 
  (faculty_price_increase : ℚ) (carl_fraction : ℚ) (simon_brownies : ℕ) 
  (friends_fraction : ℚ) (num_friends : ℕ) : Prop :=
  let original_price := total_cost / total_brownies
  let faculty_brownies := (faculty_fraction * total_brownies).floor
  let faculty_price := original_price + faculty_price_increase
  let remaining_after_faculty := total_brownies - faculty_brownies
  let carl_brownies := (carl_fraction * remaining_after_faculty).floor
  let remaining_after_carl := remaining_after_faculty - carl_brownies - simon_brownies
  let friends_brownies := (friends_fraction * remaining_after_carl).floor
  let annie_brownies := remaining_after_carl - friends_brownies
  let annie_cost := annie_brownies * original_price
  let faculty_cost := faculty_brownies * faculty_price
  annie_cost = 5.1 ∧ faculty_cost = 45

/-- Theorem stating the solution to the brownie problem --/
theorem brownie_solution : 
  brownie_problem 150 45 (3/5) 0.2 (1/4) 3 (2/3) 5 := by
  sorry

end brownie_solution_l1558_155817


namespace sqrt_x_over_5_increase_l1558_155851

theorem sqrt_x_over_5_increase (x : ℝ) (hx : x > 0) :
  let x_new := x * 1.69
  let original := Real.sqrt (x / 5)
  let new_value := Real.sqrt (x_new / 5)
  (new_value - original) / original * 100 = 30 := by
  sorry

end sqrt_x_over_5_increase_l1558_155851


namespace max_uncovered_sections_specific_case_l1558_155806

/-- Represents a corridor with carpet strips -/
structure CarpetedCorridor where
  corridorLength : ℕ
  numStrips : ℕ
  totalStripLength : ℕ

/-- Calculates the maximum number of uncovered sections in a carpeted corridor -/
def maxUncoveredSections (c : CarpetedCorridor) : ℕ :=
  sorry

/-- Theorem stating the maximum number of uncovered sections for the given problem -/
theorem max_uncovered_sections_specific_case :
  let c : CarpetedCorridor := {
    corridorLength := 100,
    numStrips := 20,
    totalStripLength := 1000
  }
  maxUncoveredSections c = 11 :=
by sorry

end max_uncovered_sections_specific_case_l1558_155806


namespace percentage_problem_l1558_155888

theorem percentage_problem (P : ℝ) : 
  (20 / 100) * 680 = (P / 100) * 140 + 80 → P = 40 := by
  sorry

end percentage_problem_l1558_155888


namespace rational_coefficient_terms_count_l1558_155801

theorem rational_coefficient_terms_count :
  let expansion := (fun (x y : ℝ) => (x * Real.rpow 2 (1/3) + y * Real.sqrt 3) ^ 500)
  let total_terms := 501
  let is_rational_coeff (k : ℕ) := (k % 3 = 0) ∧ ((500 - k) % 2 = 0)
  (Finset.filter is_rational_coeff (Finset.range total_terms)).card = 84 :=
sorry

end rational_coefficient_terms_count_l1558_155801


namespace number_division_l1558_155800

theorem number_division (x : ℝ) : x - 17 = 55 → x / 9 = 8 := by
  sorry

end number_division_l1558_155800


namespace value_of_x_when_y_is_two_l1558_155815

theorem value_of_x_when_y_is_two (x y : ℚ) : 
  y = 1 / (5 * x + 2) → y = 2 → x = -3 / 10 := by sorry

end value_of_x_when_y_is_two_l1558_155815


namespace angle_measure_with_special_supplement_complement_l1558_155834

theorem angle_measure_with_special_supplement_complement : 
  ∀ x : ℝ, 
    (0 < x) ∧ (x < 90) →
    (180 - x = 7 * (90 - x)) → 
    x = 75 := by
  sorry

end angle_measure_with_special_supplement_complement_l1558_155834


namespace a_bounds_l1558_155840

theorem a_bounds (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3) 
  (sum_squares_eq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end a_bounds_l1558_155840


namespace max_bouquet_size_l1558_155811

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- Represents a valid bouquet of tulips -/
structure Bouquet where
  yellow : ℕ
  red : ℕ
  odd_total : Odd (yellow + red)
  color_diff : (yellow = red + 1) ∨ (red = yellow + 1)
  within_budget : yellow * yellow_cost + red * red_cost ≤ max_budget

/-- The maximum number of tulips in a bouquet -/
def max_tulips : ℕ := 15

/-- Theorem stating that the maximum number of tulips in a valid bouquet is 15 -/
theorem max_bouquet_size :
  ∀ b : Bouquet, b.yellow + b.red ≤ max_tulips ∧
  ∃ b' : Bouquet, b'.yellow + b'.red = max_tulips :=
sorry

end max_bouquet_size_l1558_155811


namespace sector_central_angle_l1558_155859

/-- Given a circular sector with radius 10 cm and perimeter 45 cm, 
    its central angle is 2.5 radians. -/
theorem sector_central_angle : 
  ∀ (r p l α : ℝ), 
    r = 10 → 
    p = 45 → 
    l = p - 2 * r → 
    α = l / r → 
    α = 2.5 := by
  sorry

end sector_central_angle_l1558_155859


namespace negative_sixty_four_to_two_thirds_l1558_155809

theorem negative_sixty_four_to_two_thirds : (-64 : ℝ) ^ (2/3) = 16 := by
  sorry

end negative_sixty_four_to_two_thirds_l1558_155809


namespace box_comparison_l1558_155845

-- Define a structure for a box with three dimensions
structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the comparison operation for boxes
def Box.lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∧
  (a.x < b.x ∨ a.y < b.y ∨ a.z < b.z)

-- Define boxes A, B, and C
def A : Box := ⟨6, 5, 3⟩
def B : Box := ⟨5, 4, 1⟩
def C : Box := ⟨3, 2, 2⟩

-- Theorem to prove A > B and C < A
theorem box_comparison :
  (Box.lt B A) ∧ (Box.lt C A) := by
  sorry

end box_comparison_l1558_155845


namespace problem_statement_l1558_155860

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : 1/a + 1/b + 1/c = 1) : 
  (∃ (min : ℝ), min = 36 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 1 → x + 4*y + 9*z ≥ min) ∧ 
  ((b+c)/Real.sqrt a + (a+c)/Real.sqrt b + (a+b)/Real.sqrt c ≥ 2 * Real.sqrt (a*b*c)) :=
sorry

end problem_statement_l1558_155860


namespace angle_with_special_supplementary_complementary_relation_l1558_155893

theorem angle_with_special_supplementary_complementary_relation :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 90 →
  (180 - x = 3 * (90 - x)) →
  x = 45 :=
by sorry

end angle_with_special_supplementary_complementary_relation_l1558_155893


namespace women_not_french_approx_97_14_percent_l1558_155819

/-- Represents the composition of employees in a company -/
structure Company where
  total_employees : ℕ
  men_percentage : ℚ
  men_french_percentage : ℚ
  total_french_percentage : ℚ

/-- Calculates the percentage of women who do not speak French in the company -/
def women_not_french_percentage (c : Company) : ℚ :=
  let women_percentage := 1 - c.men_percentage
  let men_french := c.men_percentage * c.men_french_percentage
  let women_french := c.total_french_percentage - men_french
  let women_not_french := women_percentage - women_french
  women_not_french / women_percentage

/-- Theorem stating that for a company with the given percentages,
    the percentage of women who do not speak French is approximately 97.14% -/
theorem women_not_french_approx_97_14_percent 
  (c : Company) 
  (h1 : c.men_percentage = 65/100)
  (h2 : c.men_french_percentage = 60/100)
  (h3 : c.total_french_percentage = 40/100) :
  ∃ ε > 0, |women_not_french_percentage c - 9714/10000| < ε :=
sorry

end women_not_french_approx_97_14_percent_l1558_155819


namespace not_p_neither_sufficient_nor_necessary_for_not_q_l1558_155880

theorem not_p_neither_sufficient_nor_necessary_for_not_q : ∃ (a : ℝ),
  (¬(a > 0) ∧ ¬(a^2 > a)) ∨ (¬(a > 0) ∧ (a^2 > a)) ∨ ((a > 0) ∧ ¬(a^2 > a)) :=
by sorry

end not_p_neither_sufficient_nor_necessary_for_not_q_l1558_155880


namespace volunteers_assignment_l1558_155848

/-- The number of ways to assign volunteers to service points -/
def assign_volunteers (n_volunteers : ℕ) (n_points : ℕ) : ℕ :=
  sorry

/-- Theorem stating that assigning 4 volunteers to 3 service points results in 36 ways -/
theorem volunteers_assignment :
  assign_volunteers 4 3 = 36 :=
sorry

end volunteers_assignment_l1558_155848


namespace sin_cos_relation_in_triangle_l1558_155898

theorem sin_cos_relation_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A < Real.cos B) :=
sorry

end sin_cos_relation_in_triangle_l1558_155898


namespace sum_first_15_odd_integers_l1558_155871

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2*n + 1) = 225 := by
  sorry

end sum_first_15_odd_integers_l1558_155871


namespace problem_solution_l1558_155808

def second_order_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem problem_solution :
  (second_order_det 3 (-2) 4 (-3) = -1) ∧
  (∀ x : ℝ, second_order_det (2*x-3) (x+2) 2 4 = 6*x - 16) ∧
  (second_order_det 5 6 2 4 = 8) := by
  sorry

end problem_solution_l1558_155808


namespace multiplier_is_three_l1558_155853

theorem multiplier_is_three (n : ℝ) (h1 : 3 * n = (20 - n) + 20) (h2 : n = 10) : 3 = 3 := by
  sorry

end multiplier_is_three_l1558_155853


namespace queen_mary_heads_l1558_155858

/-- The number of heads on the luxury liner Queen Mary II -/
def total_heads : ℕ := by sorry

/-- The number of legs on the luxury liner Queen Mary II -/
def total_legs : ℕ := 41

/-- The number of cats on the ship -/
def num_cats : ℕ := 5

/-- The number of legs each cat has -/
def cat_legs : ℕ := 4

/-- The number of legs each sailor or cook has -/
def crew_legs : ℕ := 2

/-- The number of legs the captain has -/
def captain_legs : ℕ := 1

/-- The number of sailors and cooks combined -/
def num_crew : ℕ := by sorry

theorem queen_mary_heads :
  total_heads = num_cats + num_crew + 1 ∧
  total_legs = num_cats * cat_legs + num_crew * crew_legs + captain_legs ∧
  total_heads = 16 := by sorry

end queen_mary_heads_l1558_155858


namespace equation_properties_l1558_155818

-- Define the equation
def equation (x p : ℝ) : ℝ := (x - 3) * (x - 2) - p^2

-- Define the property of having two distinct real roots
def has_two_distinct_real_roots (p : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ p = 0 ∧ equation x₂ p = 0

-- Define the condition for the roots
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 3 * x₁ * x₂

-- Theorem statement
theorem equation_properties :
  (∀ p : ℝ, has_two_distinct_real_roots p) ∧
  (∀ p x₁ x₂ : ℝ, equation x₁ p = 0 → equation x₂ p = 0 → 
    roots_condition x₁ x₂ → p = 1 ∨ p = -1) :=
sorry

end equation_properties_l1558_155818


namespace volume_63_ounces_l1558_155865

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assertion that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_63_ounces (s : Substance) 
  (h : volume s 112 = 48) : volume s 63 = 27 := by
  sorry

end volume_63_ounces_l1558_155865


namespace bicycle_car_speed_problem_l1558_155862

theorem bicycle_car_speed_problem (distance : ℝ) (delay : ℝ) 
  (h_distance : distance = 10) 
  (h_delay : delay = 1/3) : 
  ∃ (x : ℝ), x > 0 ∧ distance / x = distance / (2 * x) + delay → x = 15 := by
  sorry

end bicycle_car_speed_problem_l1558_155862
