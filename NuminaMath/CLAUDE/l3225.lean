import Mathlib

namespace circles_common_chord_l3225_322530

/-- Two circles with equations x² + y² - 2x + 2y - 2 = 0 and x² + y² - 2mx = 0 (m > 0) 
    have a common chord of length 2 if and only if m = √6/2 -/
theorem circles_common_chord (m : ℝ) (hm : m > 0) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + 2*y - 2 = 0 ∧ x^2 + y^2 - 2*m*x = 0) ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*x₁ + 2*y₁ - 2 = 0 ∧
    x₁^2 + y₁^2 - 2*m*x₁ = 0 ∧
    x₂^2 + y₂^2 - 2*x₂ + 2*y₂ - 2 = 0 ∧
    x₂^2 + y₂^2 - 2*m*x₂ = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) ↔ 
  m = Real.sqrt 6 / 2 := by
sorry

end circles_common_chord_l3225_322530


namespace target_breaking_orders_l3225_322512

/-- The number of targets in the first column -/
def column_A : ℕ := 4

/-- The number of targets in the second column -/
def column_B : ℕ := 3

/-- The number of targets in the third column -/
def column_C : ℕ := 3

/-- The total number of targets -/
def total_targets : ℕ := column_A + column_B + column_C

/-- The number of different orders to break the targets -/
def break_orders : ℕ := (Nat.factorial total_targets) / 
  (Nat.factorial column_A * Nat.factorial column_B * Nat.factorial column_C)

theorem target_breaking_orders : break_orders = 4200 := by
  sorry

end target_breaking_orders_l3225_322512


namespace two_triangles_exist_l3225_322549

/-- Given a side length, ratio of other sides, and circumradius, prove existence of two triangles -/
theorem two_triangles_exist (a : ℝ) (k : ℝ) (R : ℝ) 
    (h_a : a > 0) (h_k : k > 0) (h_R : R > 0) (h_aR : a < 2*R) : 
  ∃ (b₁ c₁ b₂ c₂ : ℝ), 
    (b₁ > 0 ∧ c₁ > 0 ∧ b₂ > 0 ∧ c₂ > 0) ∧ 
    (b₁/c₁ = k ∧ b₂/c₂ = k) ∧
    (a + b₁ > c₁ ∧ b₁ + c₁ > a ∧ c₁ + a > b₁) ∧
    (a + b₂ > c₂ ∧ b₂ + c₂ > a ∧ c₂ + a > b₂) ∧
    (b₁ ≠ b₂ ∨ c₁ ≠ c₂) ∧
    (4 * R * R * (a + b₁ + c₁) = (a * b₁ * c₁) / R) ∧
    (4 * R * R * (a + b₂ + c₂) = (a * b₂ * c₂) / R) :=
by sorry

end two_triangles_exist_l3225_322549


namespace vertex_x_is_three_l3225_322590

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : (2 : ℝ)^2 * a + 2 * b + c = 8
  point2 : (4 : ℝ)^2 * a + 4 * b + c = 8
  point3 : c = 3

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x (f : QuadraticFunction) : ℝ := sorry

/-- Theorem stating that the x-coordinate of the vertex is 3 -/
theorem vertex_x_is_three (f : QuadraticFunction) : vertex_x f = 3 := by sorry

end vertex_x_is_three_l3225_322590


namespace cloth_cost_price_l3225_322523

/-- Calculates the cost price per meter of cloth given total meters, selling price, and profit per meter -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Proves that the cost price of one meter of cloth is 5 Rs. given the problem conditions -/
theorem cloth_cost_price :
  cost_price_per_meter 66 660 5 = 5 := by
  sorry

end cloth_cost_price_l3225_322523


namespace regression_validity_l3225_322552

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents the sample means of x and y -/
structure SampleMeans where
  x : ℝ
  y : ℝ

/-- Checks if the linear regression equation is valid for the given sample means -/
def isValidRegression (reg : LinearRegression) (means : SampleMeans) : Prop :=
  means.y = reg.slope * means.x + reg.intercept

/-- Theorem stating that the given linear regression is valid for the provided sample means -/
theorem regression_validity (means : SampleMeans) 
    (h_corr : 0 < 0.4) -- Positive correlation between x and y
    (h_means_x : means.x = 3)
    (h_means_y : means.y = 3.5) :
    isValidRegression ⟨0.4, 2.3⟩ means := by
  sorry

end regression_validity_l3225_322552


namespace parabola_line_intersection_l3225_322598

/-- Parabola defined by y² = 8x -/
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (2, 0)

/-- Line passing through the focus with slope k -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 2)

/-- Point M -/
def M : ℝ × ℝ := (-2, 2)

/-- Intersection points of the line and the parabola -/
def Intersects (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  Parabola A.1 A.2 ∧ Parabola B.1 B.2 ∧
  Line k A.1 A.2 ∧ Line k B.1 B.2

/-- Vector dot product -/
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem parabola_line_intersection (k : ℝ) :
  ∃ A B : ℝ × ℝ, Intersects k A B ∧
  DotProduct (A.1 + 2, A.2 - 2) (B.1 + 2, B.2 - 2) = 0 →
  k = 2 :=
sorry

end parabola_line_intersection_l3225_322598


namespace cyclist_final_speed_l3225_322506

/-- Calculates the final speed of a cyclist given initial speed, acceleration, and time. -/
def final_speed (initial_speed : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initial_speed + acceleration * time

/-- Converts speed from m/s to km/h. -/
def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * 3.6

theorem cyclist_final_speed :
  let initial_speed := 16 -- m/s
  let acceleration := 0.5 -- m/s²
  let time := 2 * 3600 -- 2 hours in seconds
  let final_speed_ms := final_speed initial_speed acceleration time
  let final_speed_kmh := ms_to_kmh final_speed_ms
  final_speed_kmh = 13017.6 := by sorry

end cyclist_final_speed_l3225_322506


namespace triangle_similarity_theorem_l3225_322557

/-- Two triangles are similar if they have the same shape but not necessarily the same size. -/
def are_similar (t1 t2 : Triangle) : Prop := sorry

/-- An equilateral triangle has all sides equal and all angles equal to 60°. -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- An isosceles triangle with a 120° angle has two equal sides and two base angles of 30°. -/
def is_isosceles_120 (t : Triangle) : Prop := sorry

/-- Two triangles are congruent if they have the same shape and size. -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- A right triangle has one angle of 90°. -/
def is_right_triangle (t : Triangle) : Prop := sorry

theorem triangle_similarity_theorem :
  ∀ t1 t2 : Triangle,
  (is_equilateral t1 ∧ is_equilateral t2) → are_similar t1 t2 ∧
  (is_isosceles_120 t1 ∧ is_isosceles_120 t2) → are_similar t1 t2 ∧
  are_congruent t1 t2 → are_similar t1 t2 ∧
  ∃ t3 t4 : Triangle, is_right_triangle t3 ∧ is_right_triangle t4 ∧ ¬ are_similar t3 t4 :=
by sorry

end triangle_similarity_theorem_l3225_322557


namespace johnny_works_four_and_half_hours_l3225_322532

/-- Represents Johnny's dog walking business --/
structure DogWalker where
  dogs_per_walk : ℕ
  pay_30min : ℕ
  pay_60min : ℕ
  long_walks_per_day : ℕ
  work_days_per_week : ℕ
  weekly_earnings : ℕ

/-- Calculates the number of hours Johnny works per day --/
def hours_worked_per_day (dw : DogWalker) : ℚ :=
  let long_walk_earnings := dw.pay_60min * (dw.long_walks_per_day / dw.dogs_per_walk)
  let weekly_long_walk_earnings := long_walk_earnings * dw.work_days_per_week
  let weekly_short_walk_earnings := dw.weekly_earnings - weekly_long_walk_earnings
  let short_walks_per_week := weekly_short_walk_earnings / dw.pay_30min
  let short_walks_per_day := short_walks_per_week / dw.work_days_per_week
  let short_walk_sets_per_day := short_walks_per_day / dw.dogs_per_walk
  ((dw.long_walks_per_day / dw.dogs_per_walk) * 60 + short_walk_sets_per_day * 30) / 60

/-- Theorem stating that Johnny works 4.5 hours per day --/
theorem johnny_works_four_and_half_hours
  (johnny : DogWalker)
  (h1 : johnny.dogs_per_walk = 3)
  (h2 : johnny.pay_30min = 15)
  (h3 : johnny.pay_60min = 20)
  (h4 : johnny.long_walks_per_day = 6)
  (h5 : johnny.work_days_per_week = 5)
  (h6 : johnny.weekly_earnings = 1500) :
  hours_worked_per_day johnny = 4.5 := by
  sorry

end johnny_works_four_and_half_hours_l3225_322532


namespace subtraction_preserves_inequality_l3225_322540

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end subtraction_preserves_inequality_l3225_322540


namespace remaining_macaroons_formula_l3225_322548

/-- The number of remaining macaroons after Fran eats some -/
def remaining_macaroons (k : ℚ) : ℚ :=
  let red := 50
  let green := 40
  let blue := 30
  let yellow := 20
  let orange := 10
  let total_baked := red + green + blue + yellow + orange
  let eaten_green := k
  let eaten_red := 2 * k
  let eaten_blue := 3 * k
  let eaten_yellow := (1 / 2) * k * yellow
  let eaten_orange := (1 / 5) * k
  let total_eaten := eaten_green + eaten_red + eaten_blue + eaten_yellow + eaten_orange
  total_baked - total_eaten

theorem remaining_macaroons_formula (k : ℚ) :
  remaining_macaroons k = 150 - (81 * k / 5) := by
  sorry

end remaining_macaroons_formula_l3225_322548


namespace beans_remaining_fraction_l3225_322554

/-- Given a jar and coffee beans, where:
  1. The weight of the jar is 10% of the total weight when filled with beans.
  2. After removing some beans, the weight of the jar and remaining beans is 60% of the original total weight.
  Prove that the fraction of beans remaining in the jar is 5/9. -/
theorem beans_remaining_fraction (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (remaining_beans_weight : ℝ) : 
  (jar_weight = 0.1 * (jar_weight + full_beans_weight)) →
  (jar_weight + remaining_beans_weight = 0.6 * (jar_weight + full_beans_weight)) →
  (remaining_beans_weight / full_beans_weight = 5 / 9) :=
by sorry

end beans_remaining_fraction_l3225_322554


namespace triangle_side_ratio_l3225_322565

theorem triangle_side_ratio (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ x y, (x = a ∧ y = b) ∨ (x = a ∧ y = c) ∨ (x = b ∧ y = c) ∧
  ((Real.sqrt 5 - 1) / 2 ≤ x / y) ∧ (x / y ≤ (Real.sqrt 5 + 1) / 2) := by
  sorry

end triangle_side_ratio_l3225_322565


namespace lg_sum_equals_two_l3225_322583

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by
  sorry

end lg_sum_equals_two_l3225_322583


namespace total_digits_of_powers_l3225_322596

theorem total_digits_of_powers : ∃ m n : ℕ,
  (10^(m-1) < 2^2019 ∧ 2^2019 < 10^m) ∧
  (10^(n-1) < 5^2019 ∧ 5^2019 < 10^n) ∧
  m + n = 2020 :=
by sorry

end total_digits_of_powers_l3225_322596


namespace valid_solutions_are_only_solutions_l3225_322502

/-- A structure representing a solution to the system of equations -/
structure Solution :=
  (x y z t : ℕ)

/-- The set of all valid solutions -/
def valid_solutions : Set Solution :=
  { ⟨1,1,2,3⟩, ⟨3,2,1,1⟩, ⟨4,1,3,1⟩, ⟨1,3,4,1⟩ }

/-- Predicate to check if a solution satisfies the equations -/
def satisfies_equations (s : Solution) : Prop :=
  ∃ a b : ℕ,
    s.x^2 + s.y^2 = a ∧
    s.z^2 + s.t^2 = b ∧
    (s.x^2 + s.t^2) * (s.z^2 + s.y^2) = 50

/-- Theorem stating that the valid solutions are the only ones satisfying the equations -/
theorem valid_solutions_are_only_solutions :
  ∀ s : Solution, satisfies_equations s ↔ s ∈ valid_solutions :=
sorry

end valid_solutions_are_only_solutions_l3225_322502


namespace unique_solution_geometric_series_l3225_322589

theorem unique_solution_geometric_series :
  ∃! x : ℝ, x = x^3 * (1 / (1 + x)) ∧ |x| < 1 :=
by
  -- The unique solution is (√5 - 1) / 2
  sorry

end unique_solution_geometric_series_l3225_322589


namespace min_fence_length_l3225_322503

theorem min_fence_length (area : ℝ) (h : area = 64) :
  ∃ (length width : ℝ), length > 0 ∧ width > 0 ∧
  length * width = area ∧
  ∀ (l w : ℝ), l > 0 → w > 0 → l * w = area →
  2 * (length + width) ≤ 2 * (l + w) ∧
  2 * (length + width) = 32 :=
sorry

end min_fence_length_l3225_322503


namespace cd_purchase_remaining_money_l3225_322562

theorem cd_purchase_remaining_money 
  (total_money : ℚ) 
  (total_cds : ℚ) 
  (cd_price : ℚ) 
  (h1 : cd_price > 0) 
  (h2 : total_money > 0) 
  (h3 : total_cds > 0) 
  (h4 : total_money / 5 = cd_price * total_cds / 3) :
  total_money - cd_price * total_cds = 2 * total_money / 5 := by
sorry

end cd_purchase_remaining_money_l3225_322562


namespace complex_fraction_l3225_322535

theorem complex_fraction (z : ℂ) (h : z = 1 - 2*I) :
  (z + 2) / (z - 1) = 1 + (3/2)*I := by sorry

end complex_fraction_l3225_322535


namespace min_total_cost_both_measures_l3225_322507

/-- Represents the cost and effectiveness of a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given the initial probability, potential loss,
    and a list of implemented preventive measures -/
def totalCost (initialProb : ℝ) (potentialLoss : ℝ) (measures : List PreventiveMeasure) : ℝ :=
  let measuresCost := measures.foldl (fun acc m => acc + m.cost) 0
  let finalProb := measures.foldl (fun acc m => acc * (1 - m.effectiveness)) initialProb
  measuresCost + finalProb * potentialLoss

/-- Theorem stating that the minimum total cost is achieved by implementing both measures -/
theorem min_total_cost_both_measures
  (initialProb : ℝ)
  (potentialLoss : ℝ)
  (measureA : PreventiveMeasure)
  (measureB : PreventiveMeasure)
  (h_initialProb : initialProb = 0.3)
  (h_potentialLoss : potentialLoss = 400)
  (h_measureA : measureA = { cost := 0.45, effectiveness := 0.9 })
  (h_measureB : measureB = { cost := 0.3, effectiveness := 0.85 }) :
  (totalCost initialProb potentialLoss [measureA, measureB] ≤ 
   min (totalCost initialProb potentialLoss [])
      (min (totalCost initialProb potentialLoss [measureA])
           (totalCost initialProb potentialLoss [measureB]))) ∧
  (totalCost initialProb potentialLoss [measureA, measureB] = 81) := by
  sorry

#check min_total_cost_both_measures

end min_total_cost_both_measures_l3225_322507


namespace smaller_factor_of_4582_l3225_322529

theorem smaller_factor_of_4582 :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    a * b = 4582 ∧
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 4582 → min x y = 21) :=
by sorry

end smaller_factor_of_4582_l3225_322529


namespace ellipse_k_range_l3225_322519

/-- Represents an ellipse with equation kx^2 + y^2 = 2 and focus on x-axis -/
structure Ellipse (k : ℝ) where
  equation : ∀ (x y : ℝ), k * x^2 + y^2 = 2
  focus_on_x_axis : True  -- This is a placeholder for the focus condition

/-- The range of k for an ellipse with equation kx^2 + y^2 = 2 and focus on x-axis is (0, 1) -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end ellipse_k_range_l3225_322519


namespace coins_divisible_by_six_l3225_322567

theorem coins_divisible_by_six (n : ℕ) : 
  (∃ (a b c : ℕ), n = 2*a + 2*b + 2*c) ∧ 
  (∃ (x y : ℕ), n = 3*x + 3*y) → 
  ∃ (z : ℕ), n = 6*z :=
sorry

end coins_divisible_by_six_l3225_322567


namespace acid_mixture_percentage_l3225_322597

theorem acid_mixture_percentage (a w : ℚ) :
  a > 0 ∧ w > 0 →
  (a + 1) / (a + w + 1) = 1/4 →
  (a + 1) / (a + w + 2) = 1/5 →
  a / (a + w) = 2/11 :=
by sorry

end acid_mixture_percentage_l3225_322597


namespace factorization_equality_l3225_322533

theorem factorization_equality (x : ℝ) : 
  (3 * x^3 + 48 * x^2 - 14) - (-9 * x^3 + 2 * x^2 - 14) = 2 * x^2 * (6 * x + 23) := by
  sorry

end factorization_equality_l3225_322533


namespace cos_96_cos_24_minus_sin_96_sin_24_l3225_322591

theorem cos_96_cos_24_minus_sin_96_sin_24 : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (24 * π / 180) = -1/2 := by
  sorry

end cos_96_cos_24_minus_sin_96_sin_24_l3225_322591


namespace rakesh_distance_rakesh_walked_approx_distance_l3225_322558

/-- Proves that Rakesh walked approximately 28.29 kilometers given the conditions of the problem. -/
theorem rakesh_distance (hiro_distance : ℝ) : ℝ :=
  let rakesh_distance := 4 * hiro_distance - 10
  let sanjay_distance := 2 * hiro_distance + 3
  have total_distance : hiro_distance + rakesh_distance + sanjay_distance = 60 := by sorry
  have hiro_calc : hiro_distance = 67 / 7 := by sorry
  rakesh_distance

/-- The approximate distance Rakesh walked -/
def rakesh_approx_distance : ℝ := 28.29

theorem rakesh_walked_approx_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |rakesh_distance (67 / 7) - rakesh_approx_distance| < ε :=
by sorry

end rakesh_distance_rakesh_walked_approx_distance_l3225_322558


namespace line_plane_perpendicularity_l3225_322595

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) 
  (h1 : parallel l α) 
  (h2 : perpendicular l β) : 
  plane_perpendicular α β :=
sorry

end line_plane_perpendicularity_l3225_322595


namespace arithmetic_sequence_fifth_term_l3225_322504

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 20th term is 15 
    and the 21st term is 18, the 5th term is -30. -/
theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) (h : ArithmeticSequence a) 
  (h20 : a 20 = 15) (h21 : a 21 = 18) : 
  a 5 = -30 := by
  sorry


end arithmetic_sequence_fifth_term_l3225_322504


namespace village_population_theorem_l3225_322531

theorem village_population_theorem (total_population : ℕ) 
  (h1 : total_population = 800) 
  (h2 : ∃ (part : ℕ), 4 * part = total_population) 
  (h3 : ∃ (male_population : ℕ), male_population = 2 * (total_population / 4)) :
  ∃ (male_population : ℕ), male_population = 400 := by
sorry

end village_population_theorem_l3225_322531


namespace largest_n_divisibility_l3225_322580

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 926 → ¬(n + 10 ∣ n^3 + 64) ∧ (926 + 10 ∣ 926^3 + 64) := by
  sorry

end largest_n_divisibility_l3225_322580


namespace inequality_and_system_solution_l3225_322527

theorem inequality_and_system_solution :
  (∀ x : ℝ, 2 * (-3 + x) > 3 * (x + 2) ↔ x < -12) ∧
  (∀ x : ℝ, (1/2 * (x + 1) < 2 ∧ (x + 2)/2 ≥ (x + 3)/3) ↔ 0 ≤ x ∧ x < 3) := by
  sorry

end inequality_and_system_solution_l3225_322527


namespace root_sum_theorem_l3225_322588

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 30*a^2 + 65*a - 42 = 0 → 
  b^3 - 30*b^2 + 65*b - 42 = 0 → 
  c^3 - 30*c^2 + 65*c - 42 = 0 → 
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 770/43 := by
sorry

end root_sum_theorem_l3225_322588


namespace alpha_beta_range_l3225_322576

theorem alpha_beta_range (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -12 < α * (-abs β) ∧ α * (-abs β) < -2 := by
  sorry

end alpha_beta_range_l3225_322576


namespace union_of_M_and_N_l3225_322550

def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

theorem union_of_M_and_N : M ∪ N = {x | x < 3} := by sorry

end union_of_M_and_N_l3225_322550


namespace no_solution_inequality_l3225_322536

theorem no_solution_inequality : ¬∃ (x : ℝ), 2 - 3*x + 2*x^2 ≤ 0 := by sorry

end no_solution_inequality_l3225_322536


namespace absolute_value_equation_solution_difference_l3225_322559

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁ > x₂) ∧ 
  (|2 * x₁ - 3| = 14) ∧ 
  (|2 * x₂ - 3| = 14) ∧ 
  (x₁ - x₂ = 14) := by
  sorry

end absolute_value_equation_solution_difference_l3225_322559


namespace tens_digit_of_13_pow_3007_l3225_322566

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define the sequence of last two digits of 13^n
def lastTwoDigitsOf13Pow (n : ℕ) : ℕ :=
  match n % 10 with
  | 0 => 49
  | 1 => 37
  | 2 => 81
  | 3 => 53
  | 4 => 89
  | 5 => 57
  | 6 => 41
  | 7 => 17
  | 8 => 21
  | 9 => 73
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem tens_digit_of_13_pow_3007 :
  (lastTwoDigitsOf13Pow 3007) / 10 = 1 := by
  sorry


end tens_digit_of_13_pow_3007_l3225_322566


namespace sum_of_roots_quadratic_l3225_322517

theorem sum_of_roots_quadratic (x : ℝ) : 
  (4 * x + 3) * (3 * x - 8) = 0 → 
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 23 / 12 ∧ 
    ((4 * x₁ + 3) * (3 * x₁ - 8) = 0) ∧ 
    ((4 * x₂ + 3) * (3 * x₂ - 8) = 0) :=
by sorry

end sum_of_roots_quadratic_l3225_322517


namespace system_solution_l3225_322568

theorem system_solution :
  ∃ (x y : ℚ), 
    (5 * x - 3 * y = -7) ∧ 
    (4 * x + 6 * y = 34) ∧ 
    (x = 10 / 7) ∧ 
    (y = 33 / 7) := by
  sorry

end system_solution_l3225_322568


namespace quadratic_two_distinct_nonnegative_solutions_l3225_322575

theorem quadratic_two_distinct_nonnegative_solutions (a : ℝ) :
  (6 - 3 * a > 0) →
  (a > 0) →
  (3 * a^2 + a - 2 ≥ 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ≥ 0 ∧ x₂ ≥ 0 ∧
    3 * x₁^2 - 3 * a * x₁ + a = 0 ∧
    3 * x₂^2 - 3 * a * x₂ + a = 0) ↔
  (2/3 ≤ a ∧ a < 5/3) ∨ (5/3 < a ∧ a < 2) :=
by sorry

end quadratic_two_distinct_nonnegative_solutions_l3225_322575


namespace equation_solutions_count_l3225_322500

open Real

theorem equation_solutions_count (a : ℝ) (h : a < 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, -π < x ∧ x < π ∧ 
    (a - 1) * (sin (2 * x) + cos x) + (a - 1) * (sin x - cos (2 * x)) = 0) ∧
  (∀ x, -π < x → x < π → 
    (a - 1) * (sin (2 * x) + cos x) + (a - 1) * (sin x - cos (2 * x)) = 0 → x ∈ s) :=
sorry

end equation_solutions_count_l3225_322500


namespace original_price_calculation_l3225_322509

theorem original_price_calculation (discount_percentage : ℝ) (selling_price : ℝ) 
  (h1 : discount_percentage = 20)
  (h2 : selling_price = 14) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percentage / 100) = selling_price ∧ 
    original_price = 17.5 := by
  sorry

end original_price_calculation_l3225_322509


namespace parallelogram_area_and_magnitude_l3225_322561

/-- Given a complex number z with positive real part, if the parallelogram formed by 
    0, z, z², and z + z² has area 20/29, then the smallest possible value of |z² + z| 
    is (r² + r), where r³|sin θ| = 20/29 and z = r(cos θ + i sin θ). -/
theorem parallelogram_area_and_magnitude (z : ℂ) (r θ : ℝ) (h1 : z.re > 0) 
  (h2 : z = r * Complex.exp (θ * Complex.I)) 
  (h3 : r > 0) 
  (h4 : r^3 * |Real.sin θ| = 20/29) 
  (h5 : Complex.abs (z * z - z) = 20/29) : 
  ∃ (d : ℝ), d = r^2 + r ∧ 
  ∀ (w : ℂ), Complex.abs (w^2 + w) ≥ d := by
  sorry

end parallelogram_area_and_magnitude_l3225_322561


namespace larger_rhombus_side_length_l3225_322508

/-- Two similar rhombi sharing a diagonal -/
structure SimilarRhombi where
  small_area : ℝ
  large_area : ℝ
  shared_diagonal : ℝ
  similar : small_area > 0 ∧ large_area > 0

/-- The side length of a rhombus -/
def side_length (r : SimilarRhombi) : ℝ → ℝ := sorry

/-- Theorem: The side length of the larger rhombus is √15 -/
theorem larger_rhombus_side_length (r : SimilarRhombi) 
  (h1 : r.small_area = 1) 
  (h2 : r.large_area = 9) : 
  side_length r r.large_area = Real.sqrt 15 := by
  sorry

end larger_rhombus_side_length_l3225_322508


namespace cereal_cost_l3225_322546

theorem cereal_cost (total_spent groceries_cost milk_cost cereal_boxes banana_cost banana_count
                     apple_cost apple_count cookie_cost_multiplier cookie_boxes : ℚ) :
  groceries_cost = 25 →
  milk_cost = 3 →
  cereal_boxes = 2 →
  banana_cost = 0.25 →
  banana_count = 4 →
  apple_cost = 0.5 →
  apple_count = 4 →
  cookie_cost_multiplier = 2 →
  cookie_boxes = 2 →
  (groceries_cost - (milk_cost + banana_cost * banana_count + apple_cost * apple_count +
   cookie_cost_multiplier * milk_cost * cookie_boxes)) / cereal_boxes = 3.5 := by
sorry

end cereal_cost_l3225_322546


namespace cost_per_bottle_l3225_322592

/-- Given that 3 bottles cost €1.50 and 4 bottles cost €2, prove that the cost per bottle is €0.50 -/
theorem cost_per_bottle (cost_three : ℝ) (cost_four : ℝ) 
  (h1 : cost_three = 1.5) 
  (h2 : cost_four = 2) : 
  cost_three / 3 = 0.5 ∧ cost_four / 4 = 0.5 := by
  sorry


end cost_per_bottle_l3225_322592


namespace negation_equivalence_l3225_322514

theorem negation_equivalence : 
  (¬∃ x₀ : ℝ, x₀^2 - 2*x₀ + 4 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end negation_equivalence_l3225_322514


namespace complex_roots_theorem_l3225_322581

theorem complex_roots_theorem (x y : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (x + 3 * Complex.I) * (x + 3 * Complex.I) - (13 + 12 * Complex.I) * (x + 3 * Complex.I) + (15 + 72 * Complex.I) = 0 ∧
  (y + 6 * Complex.I) * (y + 6 * Complex.I) - (13 + 12 * Complex.I) * (y + 6 * Complex.I) + (15 + 72 * Complex.I) = 0 →
  x = 11 ∧ y = 2 := by
sorry

end complex_roots_theorem_l3225_322581


namespace largest_solution_of_equation_l3225_322584

theorem largest_solution_of_equation :
  let f : ℝ → ℝ := λ x => (3*x)/7 + 2/(7*x)
  ∃ x : ℝ, x > 0 ∧ f x = 3/4 ∧ ∀ y : ℝ, y > 0 → f y = 3/4 → y ≤ x ∧ x = (21 + Real.sqrt 345) / 12 :=
by sorry

end largest_solution_of_equation_l3225_322584


namespace annas_age_at_marriage_l3225_322577

/-- Proves Anna's age at marriage given the conditions of the problem -/
theorem annas_age_at_marriage
  (josh_age_at_marriage : ℕ)
  (years_of_marriage : ℕ)
  (combined_age_factor : ℕ)
  (h1 : josh_age_at_marriage = 22)
  (h2 : years_of_marriage = 30)
  (h3 : combined_age_factor = 5)
  (h4 : josh_age_at_marriage + years_of_marriage + (josh_age_at_marriage + years_of_marriage + anna_age_at_marriage) = combined_age_factor * josh_age_at_marriage) :
  anna_age_at_marriage = 28 :=
by
  sorry

#check annas_age_at_marriage

end annas_age_at_marriage_l3225_322577


namespace problem_1997_2000_l3225_322599

theorem problem_1997_2000 : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = 0 := by
  sorry

end problem_1997_2000_l3225_322599


namespace specific_plot_fencing_cost_l3225_322556

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a given rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem specific_plot_fencing_cost :
  let plot : RectangularPlot := {
    length := 56,
    breadth := 56 - 12,
    fencing_cost_per_meter := 26.5
  }
  total_fencing_cost plot = 5300 := by
  sorry


end specific_plot_fencing_cost_l3225_322556


namespace chili_paste_can_difference_l3225_322526

def large_can_size : ℕ := 25
def small_can_size : ℕ := 15
def large_cans_needed : ℕ := 45

theorem chili_paste_can_difference :
  (large_cans_needed * large_can_size) / small_can_size - large_cans_needed = 30 :=
by sorry

end chili_paste_can_difference_l3225_322526


namespace floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one_l3225_322518

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- Theorem statement
theorem floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) := by
  sorry

end floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one_l3225_322518


namespace fifth_root_over_sixth_root_of_eleven_l3225_322563

theorem fifth_root_over_sixth_root_of_eleven (x : ℝ) :
  (11 ^ (1/5)) / (11 ^ (1/6)) = 11 ^ (1/30) :=
sorry

end fifth_root_over_sixth_root_of_eleven_l3225_322563


namespace cinema_lineup_ways_l3225_322511

def number_of_people : ℕ := 8
def number_of_windows : ℕ := 2

theorem cinema_lineup_ways :
  (2 ^ number_of_people) * (Nat.factorial number_of_people) = 10321920 := by
  sorry

end cinema_lineup_ways_l3225_322511


namespace haley_trees_l3225_322539

theorem haley_trees (initial_trees : ℕ) : 
  (((initial_trees - 5) - 8) - 3 = 12) → initial_trees = 28 := by
  sorry

end haley_trees_l3225_322539


namespace problem_1_problem_2_problem_3_problem_4_l3225_322578

-- Problem 1
theorem problem_1 : -7 + (-3) - 4 - |(-8)| = -22 := by sorry

-- Problem 2
theorem problem_2 : (1/2 - 5/9 + 7/12) * (-36) = -19 := by sorry

-- Problem 3
theorem problem_3 : -3^2 + 16 / (-2) * (1/2) - (-1)^2023 = -14 := by sorry

-- Problem 4
theorem problem_4 (a b : ℝ) : 3*a^2 - 2*a*b - a^2 + 5*a*b = 2*a^2 + 3*a*b := by sorry

end problem_1_problem_2_problem_3_problem_4_l3225_322578


namespace sumAreaVolume_specific_l3225_322579

/-- Represents a point in 3D space with integer coordinates -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D
  base4 : Point3D
  height : Int

/-- Calculates the sum of surface area and volume of a parallelepiped -/
def sumAreaVolume (p : Parallelepiped) : Int :=
  sorry -- Actual calculation would go here

/-- The specific parallelepiped from the problem -/
def specificParallelepiped : Parallelepiped :=
  { base1 := { x := 0, y := 0, z := 0 },
    base2 := { x := 3, y := 4, z := 0 },
    base3 := { x := 7, y := 0, z := 0 },
    base4 := { x := 10, y := 4, z := 0 },
    height := 5 }

theorem sumAreaVolume_specific : sumAreaVolume specificParallelepiped = 365 := by
  sorry

end sumAreaVolume_specific_l3225_322579


namespace seminar_handshakes_l3225_322515

/-- The number of people attending the seminar -/
def n : ℕ := 12

/-- The number of pairs of people who don't shake hands -/
def excluded_pairs : ℕ := 1

/-- The total number of handshakes in the seminar -/
def total_handshakes : ℕ := n.choose 2 - excluded_pairs

/-- Theorem stating the total number of handshakes in the seminar -/
theorem seminar_handshakes : total_handshakes = 65 := by
  sorry

end seminar_handshakes_l3225_322515


namespace percentage_problem_l3225_322537

theorem percentage_problem (x : ℝ) (h : (32 / 100) * x = 115.2) : x = 360 := by
  sorry

end percentage_problem_l3225_322537


namespace solution_ratio_l3225_322551

theorem solution_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end solution_ratio_l3225_322551


namespace power_function_through_point_l3225_322534

/-- A power function passing through the point (33, 3) has exponent 3 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x^α) →  -- f is a power function with exponent α
  f 33 = 3 →          -- f passes through the point (33, 3)
  α = 3 :=             -- the exponent α is equal to 3
by
  sorry


end power_function_through_point_l3225_322534


namespace water_needed_to_fill_tanks_l3225_322586

/-- Proves that the total amount of water needed to fill three tanks with equal capacity is 1593 liters, 
    given the specified conditions. -/
theorem water_needed_to_fill_tanks (capacity : ℝ) 
  (h1 : capacity * 0.45 = 450)
  (h2 : capacity > 0) : 
  (capacity - 300) + (capacity - 450) + (capacity - (capacity * 0.657)) = 1593 := by
  sorry

end water_needed_to_fill_tanks_l3225_322586


namespace line_parallel_plane_iff_no_common_points_l3225_322594

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for planes in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define what it means for a line to be parallel to a plane
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define what it means for a line and a plane to have no common points
def no_common_points (l : Line3D) (p : Plane3D) : Prop :=
  ∀ t : ℝ, 
    (l.point.x + t * l.direction.x - p.point.x) * p.normal.x +
    (l.point.y + t * l.direction.y - p.point.y) * p.normal.y +
    (l.point.z + t * l.direction.z - p.point.z) * p.normal.z ≠ 0

-- State the theorem
theorem line_parallel_plane_iff_no_common_points (l : Line3D) (p : Plane3D) :
  is_parallel l p ↔ no_common_points l p :=
sorry

end line_parallel_plane_iff_no_common_points_l3225_322594


namespace white_balls_count_l3225_322585

/-- Given a bag with 10 balls where the probability of drawing a white ball is 30%,
    prove that the number of white balls in the bag is 3. -/
theorem white_balls_count (total_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) :
  total_balls = 10 →
  prob_white = 3/10 →
  white_balls = (total_balls : ℚ) * prob_white →
  white_balls = 3 :=
by sorry

end white_balls_count_l3225_322585


namespace chessboard_determinability_l3225_322569

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Represents the chessboard and the game state -/
structure Chessboard (n : Nat) where
  selected : Set Square
  adjacent_counts : Square → Nat

/-- Defines when a number is "beautiful" (remainder 1 when divided by 3) -/
def is_beautiful (k : Nat) : Bool :=
  k % 3 = 1

/-- Defines when a square is beautiful -/
def beautiful_square (s : Square) : Bool :=
  is_beautiful s.row ∧ is_beautiful s.col

/-- Defines when Bianka can uniquely determine Aranka's selection -/
def can_determine (n : Nat) (board : Chessboard n) : Prop :=
  ∀ (alt_board : Chessboard n),
    (∀ (s : Square), board.adjacent_counts s = alt_board.adjacent_counts s) →
    board.selected = alt_board.selected

/-- The main theorem to be proved -/
theorem chessboard_determinability (n : Nat) :
  (∃ (k : Nat), n = 3 * k + 1) → (∀ (board : Chessboard n), can_determine n board) ∧
  (∃ (k : Nat), n = 3 * k + 2) → (∃ (board : Chessboard n), ¬can_determine n board) :=
sorry

end chessboard_determinability_l3225_322569


namespace tennis_ball_price_is_6_l3225_322560

/-- The price of a tennis ball in yuan -/
def tennis_ball_price : ℝ := 6

/-- The price of a tennis racket in yuan -/
def tennis_racket_price : ℝ := tennis_ball_price + 83

/-- The total cost of 2 tennis rackets and 7 tennis balls in yuan -/
def total_cost : ℝ := 220

theorem tennis_ball_price_is_6 :
  (2 * tennis_racket_price + 7 * tennis_ball_price = total_cost) ∧
  (tennis_racket_price = tennis_ball_price + 83) →
  tennis_ball_price = 6 :=
by sorry

end tennis_ball_price_is_6_l3225_322560


namespace parallelogram_area_example_l3225_322593

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 24 16 = 384 := by
  sorry

end parallelogram_area_example_l3225_322593


namespace opposite_of_negative_nine_l3225_322528

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_nine : opposite (-9) = 9 := by
  -- The proof goes here
  sorry

end opposite_of_negative_nine_l3225_322528


namespace hyperbola_eccentricity_l3225_322501

/-- The eccentricity of the hyperbola 9y² - 16x² = 144 is 5/4 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 5/4 ∧ 
  ∀ (x y : ℝ), 9*y^2 - 16*x^2 = 144 → 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    y^2/a^2 - x^2/b^2 = 1 ∧
    c^2 = a^2 + b^2 ∧
    e = c/a :=
sorry

end hyperbola_eccentricity_l3225_322501


namespace plywood_cutting_result_l3225_322564

/-- Represents the cutting of a square plywood into smaller squares. -/
structure PlywoodCutting where
  side : ℝ
  small_square_side : ℝ
  large_square_side : ℝ
  total_cut_length : ℝ

/-- Calculates the total number of squares obtained from cutting the plywood. -/
def total_squares (cut : PlywoodCutting) : ℕ :=
  sorry

/-- Theorem stating that for the given plywood cutting specifications, 
    the total number of squares obtained is 16. -/
theorem plywood_cutting_result : 
  let cut := PlywoodCutting.mk 50 10 20 280
  total_squares cut = 16 := by
  sorry

end plywood_cutting_result_l3225_322564


namespace hemisphere_surface_area_l3225_322555

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 64 * π) :
  2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end hemisphere_surface_area_l3225_322555


namespace cheaper_candy_price_l3225_322538

/-- Proves that the price of the cheaper candy is $2.00 per pound given the conditions of the mixture problem. -/
theorem cheaper_candy_price (total_weight : ℝ) (mixture_price : ℝ) (cheaper_weight : ℝ) (expensive_price : ℝ) 
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheaper_weight = 64)
  (h4 : expensive_price = 3.00) :
  ∃ (cheaper_price : ℝ), 
    cheaper_price * cheaper_weight + expensive_price * (total_weight - cheaper_weight) = 
    mixture_price * total_weight ∧ cheaper_price = 2.00 := by
  sorry

end cheaper_candy_price_l3225_322538


namespace function_property_l3225_322516

theorem function_property (f : ℕ+ → ℕ) 
  (h1 : ∀ (x y : ℕ+), f (x * y) = f x + f y)
  (h2 : f 10 = 16)
  (h3 : f 40 = 26)
  (h4 : f 8 = 12) :
  f 1000 = 48 := by
  sorry

end function_property_l3225_322516


namespace math_club_team_selection_l3225_322521

def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8
def boys_in_team : ℕ := 5
def girls_in_team : ℕ := 3

theorem math_club_team_selection :
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 55440 := by
  sorry

end math_club_team_selection_l3225_322521


namespace parallelogram_area_l3225_322570

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- Length of side AB
  a : ℝ
  -- Height of the parallelogram
  v : ℝ
  -- Ensures a and v are positive
  a_pos : 0 < a
  v_pos : 0 < v
  -- When F is 1/5 of BD from D, shaded area is 1 cm² greater than when F is 2/5 of BD from D
  area_difference : (17/50 - 13/50) * (a * v) = 1

/-- The area of a parallelogram with the given properties is 12.5 cm² -/
theorem parallelogram_area (p : Parallelogram) : p.a * p.v = 12.5 := by
  sorry

end parallelogram_area_l3225_322570


namespace equation_solution_l3225_322522

theorem equation_solution :
  ∃ x : ℚ, (4 * x^2 + 3 * x + 2) / (x + 2) = 4 * x + 3 ∧ x = -1/2 := by
  sorry

end equation_solution_l3225_322522


namespace counterexample_exists_l3225_322572

theorem counterexample_exists : ∃ n : ℕ, 
  (∀ m : ℕ, m * m ≠ n) ∧ ¬(Nat.Prime (n + 4)) := by
  sorry

end counterexample_exists_l3225_322572


namespace joe_savings_l3225_322573

def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

theorem joe_savings : 
  flight_cost + hotel_cost + food_cost + money_left = 6000 := by
  sorry

end joe_savings_l3225_322573


namespace function_inequality_implies_upper_bound_l3225_322541

/-- Given a real number a, we define a function f on (-∞, a] such that f(x) = x + 1.
    We assume that for all x and y in (-∞, a], f(x+y) ≤ 2f(x) - 3f(y).
    This theorem states that under these conditions, a must be less than or equal to -2. -/
theorem function_inequality_implies_upper_bound (a : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, x ≤ a → f x = x + 1)
  (h2 : ∀ x y, x ≤ a → y ≤ a → f (x + y) ≤ 2 * f x - 3 * f y) :
  a ≤ -2 :=
sorry

end function_inequality_implies_upper_bound_l3225_322541


namespace abs_equation_solution_l3225_322520

theorem abs_equation_solution :
  ∀ x : ℝ, |2*x + 6| = 3*x - 1 ↔ x = 7 := by sorry

end abs_equation_solution_l3225_322520


namespace sufficient_condition_for_sum_of_roots_l3225_322505

theorem sufficient_condition_for_sum_of_roots 
  (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) 
  (hroots : x₁ * x₁ + a⁻¹ * b * x₁ + a⁻¹ * c = 0 ∧ 
            x₂ * x₂ + a⁻¹ * b * x₂ + a⁻¹ * c = 0) :
  x₁ + x₂ = -b / a := by
  sorry


end sufficient_condition_for_sum_of_roots_l3225_322505


namespace intersection_implies_m_value_subset_implies_m_range_l3225_322525

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 2)*(x - m - 2) ≤ 0}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 :=
sorry

-- Theorem 2
theorem subset_implies_m_range :
  ∀ m : ℝ, A ⊆ (Set.univ : Set ℝ) \ B m → m < -3 ∨ m > 5 :=
sorry

end intersection_implies_m_value_subset_implies_m_range_l3225_322525


namespace coin_problem_l3225_322587

/-- Represents the number and denomination of coins -/
structure CoinCount where
  twenties : Nat
  fifteens : Nat

/-- Calculates the total value of coins in kopecks -/
def totalValue (coins : CoinCount) : Nat :=
  20 * coins.twenties + 15 * coins.fifteens

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  coins : CoinCount
  moreTwenties : coins.twenties > coins.fifteens
  fifthSpentWithTwoCoins : ∃ (a b : Nat), a + b = 2 ∧ 
    (a * 20 + b * 15 = totalValue coins / 5)
  halfRemainingSpentWithThreeCoins : ∃ (c d : Nat), c + d = 3 ∧ 
    (c * 20 + d * 15 = (4 * totalValue coins / 5) / 2)

/-- The theorem to be proved -/
theorem coin_problem (conditions : ProblemConditions) : 
  conditions.coins = CoinCount.mk 6 2 := by
  sorry


end coin_problem_l3225_322587


namespace quadratic_negative_range_l3225_322553

/-- The quadratic function f(x) = ax^2 + 2ax + m -/
def f (a m : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + m

theorem quadratic_negative_range (a m : ℝ) (h1 : a < 0) (h2 : f a m 2 = 0) :
  {x : ℝ | f a m x < 0} = {x : ℝ | x < -4 ∨ x > 2} := by
  sorry

end quadratic_negative_range_l3225_322553


namespace cylinder_volume_theorem_l3225_322542

/-- Represents the dimensions of a rectangle formed by unrolling a cylinder's lateral surface -/
structure UnrolledCylinder where
  side1 : ℝ
  side2 : ℝ

/-- Calculates the possible volumes of a cylinder given its unrolled lateral surface dimensions -/
def possible_cylinder_volumes (uc : UnrolledCylinder) : Set ℝ :=
  let v1 := (uc.side1 / (2 * Real.pi)) ^ 2 * Real.pi * uc.side2
  let v2 := (uc.side2 / (2 * Real.pi)) ^ 2 * Real.pi * uc.side1
  {v1, v2}

/-- Theorem stating that a cylinder with unrolled lateral surface of 8π and 4π has volume 32π² or 64π² -/
theorem cylinder_volume_theorem (uc : UnrolledCylinder) 
    (h1 : uc.side1 = 8 * Real.pi) (h2 : uc.side2 = 4 * Real.pi) : 
    possible_cylinder_volumes uc = {32 * Real.pi ^ 2, 64 * Real.pi ^ 2} := by
  sorry

#check cylinder_volume_theorem

end cylinder_volume_theorem_l3225_322542


namespace rational_absolute_difference_sum_l3225_322574

theorem rational_absolute_difference_sum (a b : ℚ) : 
  |a - b| = a + b → a ≥ 0 ∧ b ≥ 0 := by sorry

end rational_absolute_difference_sum_l3225_322574


namespace max_a_for_increasing_f_l3225_322543

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

-- State the theorem
theorem max_a_for_increasing_f :
  ∃ (a : ℝ), a = 1 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a → f x₁ < f x₂) ∧
  (∀ a' : ℝ, a' > a → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a' ∧ f x₁ ≥ f x₂) :=
sorry

end max_a_for_increasing_f_l3225_322543


namespace hyperbola_eccentricity_sqrt_6_l3225_322582

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Represents a hyperbola with equation x²/a² - y² = 1 -/
def Hyperbola (a : ℝ) := {p : Point | p.x^2 / a^2 - p.y^2 = 1}

/-- The directrix of the parabola y² = 4x -/
def directrix : Set Point := {p : Point | p.x = -1}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Predicate to check if three points form a right-angled triangle -/
def isRightTriangle (p q r : Point) : Prop := sorry

/-- The eccentricity of a hyperbola -/
def hyperbolaEccentricity (a : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_sqrt_6 (a : ℝ) (A B : Point) :
  A ∈ Hyperbola a →
  B ∈ Hyperbola a →
  A ∈ directrix →
  B ∈ directrix →
  isRightTriangle A B focus →
  hyperbolaEccentricity a = Real.sqrt 6 := by
  sorry

end hyperbola_eccentricity_sqrt_6_l3225_322582


namespace probability_product_72_l3225_322545

/-- A function representing the possible outcomes of rolling a standard die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := (standardDie.card) ^ 3

/-- The number of favorable outcomes (combinations that multiply to 72) -/
def favorableOutcomes : ℕ := 6

/-- The probability of rolling three dice such that their product is 72 -/
def probabilityProductIs72 : ℚ := favorableOutcomes / totalOutcomes

theorem probability_product_72 : probabilityProductIs72 = 1 / 36 := by
  sorry

end probability_product_72_l3225_322545


namespace square_side_prime_l3225_322524

/-- Given an integer 'a' representing the side length of a square, if it's impossible to construct 
    a rectangle with the same area as the square where both sides of the rectangle are integers 
    greater than 1, then 'a' must be a prime number. -/
theorem square_side_prime (a : ℕ) (h : a > 1) : 
  (∀ m n : ℕ, m > 1 → n > 1 → m * n ≠ a * a) → Nat.Prime a := by
  sorry

end square_side_prime_l3225_322524


namespace problem_statement_l3225_322547

theorem problem_statement (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) :
  x^2 * y - x * y^2 = -6 := by
  sorry

end problem_statement_l3225_322547


namespace expansion_terms_count_l3225_322571

theorem expansion_terms_count (G1 G2 : Finset (Fin 4)) 
  (hG1 : G1.card = 4) (hG2 : G2.card = 4) :
  (G1.product G2).card = 16 := by
  sorry

end expansion_terms_count_l3225_322571


namespace prime_factors_of_n_l3225_322544

theorem prime_factors_of_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : ∃ k : ℕ, 14 * n = 60 * k) :
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ∣ n ∧ q ∣ n ∧ r ∣ n :=
sorry

end prime_factors_of_n_l3225_322544


namespace quadratic_zero_condition_l3225_322510

/-- A quadratic function f(x) = x^2 - 2x + m has a zero in (-1, 0) if and only if -3 < m < 0 -/
theorem quadratic_zero_condition (m : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ x^2 - 2*x + m = 0) ↔ -3 < m ∧ m < 0 :=
by sorry

end quadratic_zero_condition_l3225_322510


namespace triangle_isosceles_or_right_angled_l3225_322513

/-- A triangle with sides a, b, and c is either isosceles or right-angled if (a-b)(a^2+b^2-c^2) = 0 --/
theorem triangle_isosceles_or_right_angled 
  {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : (a - b) * (a^2 + b^2 - c^2) = 0) : 
  (a = b ∨ a = c ∨ b = c) ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
sorry

end triangle_isosceles_or_right_angled_l3225_322513
