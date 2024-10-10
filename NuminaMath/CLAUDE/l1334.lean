import Mathlib

namespace hyperbola_equation_l1334_133440

/-- Given a hyperbola with asymptotes y = ± 2(x-1) and one focus at (1+2√5, 0),
    prove that its equation is (x - 1)²/5 - y²/20 = 1 -/
theorem hyperbola_equation 
  (asymptotes : ℝ → ℝ → Prop)
  (focus : ℝ × ℝ)
  (h_asymptotes : ∀ x y, asymptotes x y ↔ y = 2*(x-1) ∨ y = -2*(x-1))
  (h_focus : focus = (1 + 2*Real.sqrt 5, 0)) :
  ∀ x y, ((x - 1)^2 / 5 - y^2 / 20 = 1) ↔ 
    (∃ a b c : ℝ, a*(x-1)^2 + b*y^2 + c = 0 ∧ 
    (∀ x' y', asymptotes x' y' → a*(x'-1)^2 + b*y'^2 + c = 0) ∧
    a*(focus.1-1)^2 + b*focus.2^2 + c = 0) :=
by sorry

end hyperbola_equation_l1334_133440


namespace range_of_a_l1334_133421

def sequence_a (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

theorem range_of_a (a : ℝ) :
  (∀ n, sequence_a a n < sequence_a a (n + 1)) ↔ a ≥ 0 := by sorry

end range_of_a_l1334_133421


namespace existence_of_even_odd_composition_l1334_133495

theorem existence_of_even_odd_composition :
  ∃ (p q : ℝ → ℝ),
    (∀ x, p x = p (-x)) ∧
    (∀ x, p (q x) = -(p (q (-x)))) ∧
    (∃ x, p (q x) ≠ 0) := by
  sorry

end existence_of_even_odd_composition_l1334_133495


namespace max_k_value_l1334_133439

theorem max_k_value (k : ℤ) : 
  (∀ x : ℝ, x > 1 → x * Real.log x - k * x > 3) → k ≤ -3 :=
sorry

end max_k_value_l1334_133439


namespace total_fish_l1334_133414

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7

theorem total_fish : gold_fish + blue_fish = 22 := by
  sorry

end total_fish_l1334_133414


namespace square_value_l1334_133457

theorem square_value (square : ℚ) (h : (1:ℚ)/9 + (1:ℚ)/18 = (1:ℚ)/square) : square = 6 := by
  sorry

end square_value_l1334_133457


namespace sum_of_squares_roots_l1334_133404

theorem sum_of_squares_roots (a : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0 ∧ x^2 + y^2 = 21) ↔ a = -3 :=
sorry

end sum_of_squares_roots_l1334_133404


namespace simplification_value_at_3_value_at_negative_3_even_function_l1334_133429

-- Define the original expression
def original_expression (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2)

-- Define the simplified expression
def simplified_expression (x : ℝ) : ℝ :=
  2 * x^2 + 2

-- Theorem stating that the original expression simplifies to the simplified expression
theorem simplification : 
  ∀ x : ℝ, original_expression x = simplified_expression x :=
sorry

-- Theorem stating that the simplified expression equals 20 when x = 3
theorem value_at_3 : simplified_expression 3 = 20 :=
sorry

-- Theorem stating that the simplified expression equals 20 when x = -3
theorem value_at_negative_3 : simplified_expression (-3) = 20 :=
sorry

-- Theorem stating that the simplified expression is an even function
theorem even_function :
  ∀ x : ℝ, simplified_expression x = simplified_expression (-x) :=
sorry

end simplification_value_at_3_value_at_negative_3_even_function_l1334_133429


namespace negation_of_implication_l1334_133494

theorem negation_of_implication (a b : ℝ) :
  ¬(a^2 > b^2 → a > b) ↔ (a^2 ≤ b^2 → a ≤ b) := by sorry

end negation_of_implication_l1334_133494


namespace min_value_n_over_2_plus_50_over_n_l1334_133449

theorem min_value_n_over_2_plus_50_over_n (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 2 + 50 / n ≥ 10 ∧
  ((n : ℝ) / 2 + 50 / n = 10 ↔ n = 10) := by
  sorry

end min_value_n_over_2_plus_50_over_n_l1334_133449


namespace remainder_problem_l1334_133498

theorem remainder_problem (y : ℤ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end remainder_problem_l1334_133498


namespace equilateral_cone_central_angle_l1334_133419

/-- An equilateral cone is a cone whose cross-section is an equilateral triangle -/
structure EquilateralCone where
  radius : ℝ
  slant_height : ℝ
  slant_height_eq : slant_height = 2 * radius

/-- The central angle of the sector of an equilateral cone is π radians -/
theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  (2 * π * cone.radius) / cone.slant_height = π :=
sorry

end equilateral_cone_central_angle_l1334_133419


namespace square_side_length_l1334_133490

-- Define the perimeter of the square
def perimeter : ℝ := 34.8

-- Theorem: The length of one side of a square with perimeter 34.8 cm is 8.7 cm
theorem square_side_length : 
  perimeter / 4 = 8.7 := by
  sorry

end square_side_length_l1334_133490


namespace jana_walking_distance_l1334_133407

/-- Jana's walking pattern and distance traveled -/
theorem jana_walking_distance :
  let usual_pace : ℚ := 1 / 30  -- miles per minute
  let half_pace : ℚ := usual_pace / 2
  let double_pace : ℚ := usual_pace * 2
  let first_15_min_distance : ℚ := half_pace * 15
  let next_5_min_distance : ℚ := double_pace * 5
  first_15_min_distance + next_5_min_distance = 7 / 12 := by
  sorry

end jana_walking_distance_l1334_133407


namespace quadrilateral_rod_count_quadrilateral_rod_count_is_17_l1334_133416

theorem quadrilateral_rod_count : ℕ → Prop :=
  fun n =>
    let rods : Finset ℕ := Finset.range 30
    let used_rods : Finset ℕ := {3, 7, 15}
    let valid_rods : Finset ℕ := 
      rods.filter (fun x => 
        x > 5 ∧ x < 25 ∧ x ∉ used_rods)
    n = valid_rods.card

theorem quadrilateral_rod_count_is_17 :
  quadrilateral_rod_count 17 := by sorry

end quadrilateral_rod_count_quadrilateral_rod_count_is_17_l1334_133416


namespace ellipse_symmetric_points_m_bound_l1334_133435

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = 4 * x + m

/-- Two points are symmetric with respect to a line -/
def symmetric_points (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 ∧ y₂ - y₁ = -4 * (x₂ - x₁)

/-- The theorem statement -/
theorem ellipse_symmetric_points_m_bound :
  ∀ m : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∃ x₀ y₀ : ℝ,
      line x₀ y₀ m ∧
      symmetric_points x₁ y₁ x₂ y₂ x₀ y₀)) →
  -2 * Real.sqrt 3 / 13 < m ∧ m < 2 * Real.sqrt 3 / 13 :=
sorry

end ellipse_symmetric_points_m_bound_l1334_133435


namespace problem_solid_surface_area_l1334_133481

/-- Represents a solid formed by unit cubes --/
structure CubeSolid where
  base_layer : Nat
  second_layer : Nat
  third_layer : Nat
  top_layer : Nat

/-- Calculates the surface area of a CubeSolid --/
def surface_area (solid : CubeSolid) : Nat :=
  sorry

/-- The specific solid described in the problem --/
def problem_solid : CubeSolid :=
  { base_layer := 4
  , second_layer := 4
  , third_layer := 3
  , top_layer := 1 }

/-- Theorem stating that the surface area of the problem_solid is 28 --/
theorem problem_solid_surface_area :
  surface_area problem_solid = 28 :=
sorry

end problem_solid_surface_area_l1334_133481


namespace polynomial_equality_l1334_133488

theorem polynomial_equality (m : ℝ) : (2 * m^2 + 3 * m - 4) + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1 := by
  sorry

end polynomial_equality_l1334_133488


namespace daisy_taller_than_reese_l1334_133428

/-- The heights of three people and their relationships -/
structure Heights where
  daisy : ℝ
  parker : ℝ
  reese : ℝ
  parker_shorter : parker = daisy - 4
  reese_height : reese = 60
  average_height : (daisy + parker + reese) / 3 = 64

/-- Daisy is 8 inches taller than Reese -/
theorem daisy_taller_than_reese (h : Heights) : h.daisy - h.reese = 8 := by
  sorry

end daisy_taller_than_reese_l1334_133428


namespace incorrect_calculation_correction_l1334_133417

theorem incorrect_calculation_correction (x : ℝ) (h : x * 7 = 115.15) : 
  115.15 / 49 = 2.35 := by
sorry

end incorrect_calculation_correction_l1334_133417


namespace min_value_x2_minus_xy_plus_y2_l1334_133410

theorem min_value_x2_minus_xy_plus_y2 :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end min_value_x2_minus_xy_plus_y2_l1334_133410


namespace crate_dimensions_for_largest_tank_l1334_133400

/-- Represents a rectangular crate with length, width, and height -/
structure RectangularCrate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank with radius and height -/
structure CylindricalTank where
  radius : ℝ
  height : ℝ

/-- The tank fits in the crate when standing upright -/
def tankFitsInCrate (tank : CylindricalTank) (crate : RectangularCrate) : Prop :=
  2 * tank.radius ≤ min crate.length crate.width ∧ tank.height ≤ crate.height

theorem crate_dimensions_for_largest_tank (crate : RectangularCrate) 
    (h : ∃ tank : CylindricalTank, tank.radius = 10 ∧ tankFitsInCrate tank crate) :
    crate.length ≥ 20 ∧ crate.width ≥ 20 := by
  sorry

end crate_dimensions_for_largest_tank_l1334_133400


namespace negation_equivalence_l1334_133415

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by
  sorry

end negation_equivalence_l1334_133415


namespace cubic_roots_sum_l1334_133467

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -18 := by
  sorry

end cubic_roots_sum_l1334_133467


namespace unique_positive_number_l1334_133496

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 8 = 128 / x := by
  sorry

end unique_positive_number_l1334_133496


namespace shortest_path_length_l1334_133424

/-- Represents a frustum of a right circular cone -/
structure ConeFrustum where
  lower_circumference : ℝ
  upper_circumference : ℝ
  inclination_angle : ℝ

/-- The shortest path from a point on the lower base to the upper base and back -/
def shortest_return_path (cf : ConeFrustum) : ℝ := sorry

theorem shortest_path_length (cf : ConeFrustum) 
  (h1 : cf.lower_circumference = 8)
  (h2 : cf.upper_circumference = 6)
  (h3 : cf.inclination_angle = π / 3) :
  shortest_return_path cf = 4 * Real.sqrt 3 / π := by sorry

end shortest_path_length_l1334_133424


namespace relationship_abc_l1334_133437

-- Define a, b, and c
def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

-- Theorem stating the relationship between a, b, and c
theorem relationship_abc : c > a ∧ a > b := by
  sorry

end relationship_abc_l1334_133437


namespace car_insurance_cost_l1334_133465

theorem car_insurance_cost (nancy_percentage : ℝ) (nancy_annual_payment : ℝ) :
  nancy_percentage = 0.40 →
  nancy_annual_payment = 384 →
  (nancy_annual_payment / nancy_percentage) / 12 = 80 := by
sorry

end car_insurance_cost_l1334_133465


namespace population_is_all_scores_l1334_133413

/-- Represents a math exam with participants and their scores -/
structure MathExam where
  participants : ℕ
  scores : Finset ℝ

/-- Represents a statistical analysis of a math exam -/
structure StatisticalAnalysis where
  exam : MathExam
  sample_size : ℕ

/-- The definition of population in the context of this statistical analysis -/
def population (analysis : StatisticalAnalysis) : Finset ℝ :=
  analysis.exam.scores

/-- Theorem stating that the population in this statistical analysis
    is the set of all participants' scores -/
theorem population_is_all_scores
  (exam : MathExam)
  (analysis : StatisticalAnalysis)
  (h1 : exam.participants = 40000)
  (h2 : analysis.sample_size = 400)
  (h3 : analysis.exam = exam)
  (h4 : exam.scores.card = exam.participants) :
  population analysis = exam.scores :=
sorry

end population_is_all_scores_l1334_133413


namespace quadratic_factorization_l1334_133456

theorem quadratic_factorization (p q : ℤ) :
  (∀ x, 20 * x^2 - 110 * x - 120 = (5 * x + p) * (4 * x + q)) →
  p + 2 * q = -8 := by
  sorry

end quadratic_factorization_l1334_133456


namespace banking_problem_l1334_133475

/-- Calculates the final amount after deposit growth and withdrawal fee --/
def finalAmount (initialDeposit : ℝ) (growthRate : ℝ) (feeRate : ℝ) : ℝ :=
  initialDeposit * (1 + growthRate) * (1 - feeRate)

/-- Represents the banking problem with Vlad and Dima's deposits --/
theorem banking_problem (initialDeposit : ℝ) 
  (h_initial : initialDeposit = 3000) 
  (vladGrowthRate dimaGrowthRate vladFeeRate dimaFeeRate : ℝ)
  (h_vlad_growth : vladGrowthRate = 0.2)
  (h_vlad_fee : vladFeeRate = 0.1)
  (h_dima_growth : dimaGrowthRate = 0.4)
  (h_dima_fee : dimaFeeRate = 0.2) :
  finalAmount initialDeposit dimaGrowthRate dimaFeeRate - 
  finalAmount initialDeposit vladGrowthRate vladFeeRate = 120 := by
  sorry


end banking_problem_l1334_133475


namespace david_zachary_pushup_difference_l1334_133487

/-- Given that David did 62 push-ups and Zachary did 47 push-ups,
    prove that David did 15 more push-ups than Zachary. -/
theorem david_zachary_pushup_difference :
  let david_pushups : ℕ := 62
  let zachary_pushups : ℕ := 47
  david_pushups - zachary_pushups = 15 := by
  sorry

end david_zachary_pushup_difference_l1334_133487


namespace sphere_radius_from_shadows_l1334_133426

/-- Given a sphere and a stick under parallel sun rays, prove the radius of the sphere -/
theorem sphere_radius_from_shadows
  (shadow_sphere : ℝ)  -- Length of the sphere's shadow
  (height_stick : ℝ)   -- Height of the stick
  (shadow_stick : ℝ)   -- Length of the stick's shadow
  (h_shadow_sphere : shadow_sphere = 20)
  (h_height_stick : height_stick = 1)
  (h_shadow_stick : shadow_stick = 4)
  : ∃ (radius : ℝ), radius = 5 ∧ (radius / shadow_sphere = height_stick / shadow_stick) :=
sorry

end sphere_radius_from_shadows_l1334_133426


namespace temperature_difference_product_of_N_values_l1334_133469

theorem temperature_difference (N : ℤ) : 
  (∃ D M : ℤ, 
    M = D + N ∧ 
    abs ((M - 8) - (D + 5)) = 3) → 
  (N = 10 ∨ N = 16) :=
by sorry

theorem product_of_N_values : 
  (∀ N : ℤ, (∃ D M : ℤ, 
    M = D + N ∧ 
    abs ((M - 8) - (D + 5)) = 3) → 
  (N = 10 ∨ N = 16)) → 
  (10 * 16 = 160) :=
by sorry

end temperature_difference_product_of_N_values_l1334_133469


namespace tangent_line_circle_l1334_133442

theorem tangent_line_circle (r : ℝ) (hr : r > 0) :
  (∀ x y : ℝ, x + y = r → x^2 + y^2 = r → (∀ x' y' : ℝ, x' + y' = r → x'^2 + y'^2 ≤ r)) →
  r = 2 := by
sorry

end tangent_line_circle_l1334_133442


namespace tank_insulation_cost_l1334_133489

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

/-- Proves that the cost to insulate a rectangular tank with given dimensions is $1240 -/
theorem tank_insulation_cost :
  insulationCost 5 3 2 20 = 1240 := by
  sorry

end tank_insulation_cost_l1334_133489


namespace inequality_of_powers_l1334_133423

theorem inequality_of_powers (a n k : ℕ) (ha : a > 1) (hnk : 0 < n ∧ n < k) :
  (a^n - 1) / n < (a^k - 1) / k := by
  sorry

end inequality_of_powers_l1334_133423


namespace derivative_sin_pi_sixth_l1334_133472

theorem derivative_sin_pi_sixth (h : Real.sin (π / 6) = (1 : ℝ) / 2) : 
  deriv (λ _ : ℝ => Real.sin (π / 6)) = 0 := by
  sorry

end derivative_sin_pi_sixth_l1334_133472


namespace linear_equation_solution_l1334_133468

/-- Given that (a - 3)x^(|a| - 2) + 6 = 0 is a linear equation in terms of x,
    prove that the solution is x = 1 -/
theorem linear_equation_solution (a : ℝ) :
  (∀ x, ∃ k m, (a - 3) * x^(|a| - 2) + 6 = k * x + m) →
  ∃! x, (a - 3) * x^(|a| - 2) + 6 = 0 ∧ x = 1 :=
by sorry

end linear_equation_solution_l1334_133468


namespace car_sale_price_l1334_133461

/-- The final sale price of a car after multiple discounts and tax --/
theorem car_sale_price (original_price : ℝ) (discount1 discount2 discount3 tax_rate : ℝ) :
  original_price = 20000 ∧
  discount1 = 0.12 ∧
  discount2 = 0.10 ∧
  discount3 = 0.05 ∧
  tax_rate = 0.08 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 + tax_rate)) = 16251.84 := by
  sorry

#eval (20000 : ℝ) * (1 - 0.12) * (1 - 0.10) * (1 - 0.05) * (1 + 0.08)

end car_sale_price_l1334_133461


namespace original_fraction_l1334_133446

theorem original_fraction (x y : ℚ) : 
  (1.2 * x) / (0.9 * y) = 20 / 21 → x / y = 5 / 7 := by
  sorry

end original_fraction_l1334_133446


namespace collinear_points_k_value_l1334_133441

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The main theorem stating that if the given points are collinear, then k = 24. -/
theorem collinear_points_k_value (k : ℝ) :
  collinear 1 (-2) 3 2 6 (k/3) → k = 24 := by
  sorry

#check collinear_points_k_value

end collinear_points_k_value_l1334_133441


namespace sum_of_binary_numbers_l1334_133478

/-- Convert a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + (if c = '1' then 1 else 0)) 0

/-- Convert a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
    let rec aux (m : ℕ) : String :=
      if m = 0 then "" else aux (m / 2) ++ (if m % 2 = 1 then "1" else "0")
    aux n

theorem sum_of_binary_numbers :
  let a := binary_to_nat "1100"
  let b := binary_to_nat "101"
  let c := binary_to_nat "11"
  let d := binary_to_nat "11011"
  let e := binary_to_nat "100"
  nat_to_binary (a + b + c + d + e) = "1000101" := by
  sorry

end sum_of_binary_numbers_l1334_133478


namespace largest_multiple_of_5_and_6_under_1000_l1334_133445

theorem largest_multiple_of_5_and_6_under_1000 :
  ∃ n : ℕ, n = 990 ∧
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ n) :=
by sorry

end largest_multiple_of_5_and_6_under_1000_l1334_133445


namespace infinitely_many_y_greater_than_sqrt_n_l1334_133477

theorem infinitely_many_y_greater_than_sqrt_n
  (x y : ℕ → ℕ+)
  (h : ∀ n : ℕ, n ≥ 1 → (y (n + 1) : ℚ) / (x (n + 1) : ℚ) > (y n : ℚ) / (x n : ℚ)) :
  Set.Infinite {n : ℕ | (y n : ℝ) > Real.sqrt n} :=
sorry

end infinitely_many_y_greater_than_sqrt_n_l1334_133477


namespace quadratic_factorization_l1334_133406

theorem quadratic_factorization (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end quadratic_factorization_l1334_133406


namespace bridge_length_l1334_133458

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 225 :=
by
  sorry


end bridge_length_l1334_133458


namespace function_composition_problem_l1334_133453

theorem function_composition_problem (a : ℝ) : 
  let f (x : ℝ) := x / 4 + 2
  let g (x : ℝ) := 5 - x
  f (g a) = 4 → a = -3 := by
sorry

end function_composition_problem_l1334_133453


namespace discount_ratio_proof_l1334_133497

/-- Proves that given a 15% discount on an item, if a person with $500 still needs $95 more to purchase it, the ratio of the additional money needed to the initial amount is 19:100. -/
theorem discount_ratio_proof (initial_amount : ℝ) (additional_needed : ℝ) (discount_rate : ℝ) :
  initial_amount = 500 →
  additional_needed = 95 →
  discount_rate = 0.15 →
  (additional_needed / initial_amount) = (19 / 100) :=
by sorry

end discount_ratio_proof_l1334_133497


namespace greatest_divisor_of_fourth_power_difference_l1334_133434

/-- The function that reverses the digits of a positive integer -/
noncomputable def reverse_digits (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that 99 is the greatest integer that always divides n^4 - f(n)^4 -/
theorem greatest_divisor_of_fourth_power_difference (n : ℕ+) : 
  (∃ (k : ℕ), k > 99 ∧ ∀ (m : ℕ+), k ∣ (m^4 - (reverse_digits m)^4)) → False :=
sorry

end greatest_divisor_of_fourth_power_difference_l1334_133434


namespace min_b_over_a_is_one_minus_e_l1334_133433

/-- Given two real functions f and g, if f(x) ≤ g(x) for all x > 0,
    then the minimum value of b/a is 1 - e. -/
theorem min_b_over_a_is_one_minus_e (a b : ℝ)
    (f : ℝ → ℝ) (g : ℝ → ℝ)
    (hf : ∀ x, x > 0 → f x = Real.log x + a)
    (hg : ∀ x, g x = a * x + b + 1)
    (h_le : ∀ x, x > 0 → f x ≤ g x) :
    ∃ m, m = 1 - Real.exp 1 ∧ ∀ k, (b / a ≥ k → k ≥ m) :=
  sorry

end min_b_over_a_is_one_minus_e_l1334_133433


namespace pie_chart_shows_percentage_relation_l1334_133499

/-- Represents different types of statistical graphs -/
inductive StatGraph
  | PieChart
  | BarGraph
  | LineGraph
  | Histogram

/-- Defines the property of showing percentage of a part in relation to the whole -/
def shows_percentage_relation (g : StatGraph) : Prop :=
  match g with
  | StatGraph.PieChart => true
  | _ => false

/-- Theorem stating that the Pie chart is the graph that shows percentage relation -/
theorem pie_chart_shows_percentage_relation :
  ∀ (g : StatGraph), shows_percentage_relation g ↔ g = StatGraph.PieChart :=
by
  sorry

end pie_chart_shows_percentage_relation_l1334_133499


namespace lateral_edge_length_for_specific_pyramid_l1334_133486

/-- A regular quadrilateral pyramid with given base side length and volume -/
structure RegularQuadPyramid where
  base_side : ℝ
  volume : ℝ

/-- The length of the lateral edge of a regular quadrilateral pyramid -/
def lateral_edge_length (p : RegularQuadPyramid) : ℝ :=
  sorry

theorem lateral_edge_length_for_specific_pyramid :
  let p : RegularQuadPyramid := { base_side := 2, volume := 4 * Real.sqrt 3 / 3 }
  lateral_edge_length p = Real.sqrt 5 := by
    sorry

end lateral_edge_length_for_specific_pyramid_l1334_133486


namespace marble_exchange_ratio_l1334_133479

theorem marble_exchange_ratio : 
  ∀ (ben_initial john_initial ben_final john_final marbles_given : ℕ),
    ben_initial = 18 →
    john_initial = 17 →
    ben_final = ben_initial - marbles_given →
    john_final = john_initial + marbles_given →
    john_final = ben_final + 17 →
    marbles_given * 2 = ben_initial :=
by
  sorry

end marble_exchange_ratio_l1334_133479


namespace square_root_2023_plus_2_squared_minus_4_times_plus_5_l1334_133450

theorem square_root_2023_plus_2_squared_minus_4_times_plus_5 :
  let m : ℝ := Real.sqrt 2023 + 2
  m^2 - 4*m + 5 = 2024 := by
sorry

end square_root_2023_plus_2_squared_minus_4_times_plus_5_l1334_133450


namespace total_apple_and_cherry_pies_l1334_133455

def apple_pies : ℕ := 6
def pecan_pies : ℕ := 9
def pumpkin_pies : ℕ := 8
def cherry_pies : ℕ := 5
def blueberry_pies : ℕ := 3

theorem total_apple_and_cherry_pies : apple_pies + cherry_pies = 11 := by
  sorry

end total_apple_and_cherry_pies_l1334_133455


namespace large_box_height_is_four_l1334_133451

-- Define the dimensions of the larger box
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.6
def small_box_width : ℝ := 0.5
def small_box_height : ℝ := 0.4

-- Define the maximum number of small boxes
def max_small_boxes : ℕ := 1000

-- Theorem statement
theorem large_box_height_is_four :
  ∃ (h : ℝ), 
    h = 4 ∧ 
    large_box_length * large_box_width * h = 
      (max_small_boxes : ℝ) * small_box_length * small_box_width * small_box_height :=
by sorry

end large_box_height_is_four_l1334_133451


namespace discount_per_shirt_calculation_l1334_133448

theorem discount_per_shirt_calculation (num_shirts : ℕ) (total_cost discount_percentage : ℚ) 
  (h1 : num_shirts = 3)
  (h2 : total_cost = 60)
  (h3 : discount_percentage = 40/100) :
  let discount_amount := total_cost * discount_percentage
  let discounted_total := total_cost - discount_amount
  let price_per_shirt := discounted_total / num_shirts
  price_per_shirt = 12 := by
sorry

end discount_per_shirt_calculation_l1334_133448


namespace inequality_proof_l1334_133427

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a ≤ 3 * a * b * c) :
  Real.sqrt ((a^2 + b^2) / (a + b)) + Real.sqrt ((b^2 + c^2) / (b + c)) +
  Real.sqrt ((c^2 + a^2) / (c + a)) + 3 ≤
  Real.sqrt 2 * (Real.sqrt (a + b) + Real.sqrt (b + c) + Real.sqrt (c + a)) := by
  sorry

end inequality_proof_l1334_133427


namespace average_weight_problem_l1334_133462

theorem average_weight_problem (a b c : ℝ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 41 →
  b = 27 →
  (a + b + c) / 3 = 45 := by
sorry

end average_weight_problem_l1334_133462


namespace wheel_diameter_l1334_133402

theorem wheel_diameter (radius : ℝ) (h : radius = 7) : 2 * radius = 14 := by
  sorry

end wheel_diameter_l1334_133402


namespace jumping_contest_l1334_133444

theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 14 →
  frog_jump = grasshopper_jump + 37 →
  mouse_jump = frog_jump - 16 →
  mouse_jump - grasshopper_jump = 21 :=
by sorry

end jumping_contest_l1334_133444


namespace house_transaction_result_l1334_133430

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : ℝ
  hasHouse : Bool

/-- Represents the state of the house -/
structure HouseState where
  value : ℝ
  owner : String

def initial_mr_a : FinancialState := { cash := 15000, hasHouse := true }
def initial_mr_b : FinancialState := { cash := 20000, hasHouse := false }
def initial_house : HouseState := { value := 15000, owner := "A" }

def house_sale_price : ℝ := 20000
def depreciation_rate : ℝ := 0.15

theorem house_transaction_result :
  let first_transaction_mr_a : FinancialState :=
    { cash := initial_mr_a.cash + house_sale_price, hasHouse := false }
  let first_transaction_mr_b : FinancialState :=
    { cash := initial_mr_b.cash - house_sale_price, hasHouse := true }
  let depreciated_house_value : ℝ := initial_house.value * (1 - depreciation_rate)
  let final_mr_a : FinancialState :=
    { cash := first_transaction_mr_a.cash - depreciated_house_value, hasHouse := true }
  let final_mr_b : FinancialState :=
    { cash := first_transaction_mr_b.cash + depreciated_house_value, hasHouse := false }
  let mr_a_net_gain : ℝ := final_mr_a.cash + depreciated_house_value - (initial_mr_a.cash + initial_house.value)
  let mr_b_net_gain : ℝ := final_mr_b.cash - initial_mr_b.cash
  mr_a_net_gain = 5000 ∧ mr_b_net_gain = -7250 := by
  sorry

end house_transaction_result_l1334_133430


namespace arithmetic_mean_of_fractions_l1334_133474

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 7
  let b := (6 : ℚ) / 11
  (a + b) / 2 = 75 / 154 := by
sorry

end arithmetic_mean_of_fractions_l1334_133474


namespace colored_points_segment_existence_l1334_133485

/-- Represents a color --/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents a colored point on a line --/
structure ColoredPoint where
  position : ℝ
  color : Color

/-- The main theorem --/
theorem colored_points_segment_existence
  (n : ℕ)
  (h_n : n ≥ 4)
  (points : Fin n → ColoredPoint)
  (h_distinct : ∀ i j, i ≠ j → (points i).position ≠ (points j).position)
  (h_all_colors : ∀ c : Color, ∃ i, (points i).color = c) :
  ∃ (a b : ℝ), a < b ∧
    (∃ (c₁ c₂ : Color), c₁ ≠ c₂ ∧
      (∃! i, a ≤ (points i).position ∧ (points i).position ≤ b ∧ (points i).color = c₁) ∧
      (∃! j, a ≤ (points j).position ∧ (points j).position ≤ b ∧ (points j).color = c₂)) ∧
    (∃ (c₃ c₄ : Color), c₃ ≠ c₄ ∧ c₃ ≠ c₁ ∧ c₃ ≠ c₂ ∧ c₄ ≠ c₁ ∧ c₄ ≠ c₂ ∧
      (∃ i, a ≤ (points i).position ∧ (points i).position ≤ b ∧ (points i).color = c₃) ∧
      (∃ j, a ≤ (points j).position ∧ (points j).position ≤ b ∧ (points j).color = c₄)) :=
by
  sorry


end colored_points_segment_existence_l1334_133485


namespace ponderosa_price_calculation_l1334_133412

/-- The price of each ponderosa pine tree -/
def ponderosa_price : ℕ := 225

/-- The total number of trees -/
def total_trees : ℕ := 850

/-- The number of trees bought of one kind -/
def trees_of_one_kind : ℕ := 350

/-- The price of each Douglas fir tree -/
def douglas_price : ℕ := 300

/-- The total amount paid for all trees -/
def total_paid : ℕ := 217500

theorem ponderosa_price_calculation :
  ponderosa_price = 225 ∧
  total_trees = 850 ∧
  trees_of_one_kind = 350 ∧
  douglas_price = 300 ∧
  total_paid = 217500 →
  ∃ (douglas_count ponderosa_count : ℕ),
    douglas_count + ponderosa_count = total_trees ∧
    (douglas_count = trees_of_one_kind ∨ ponderosa_count = trees_of_one_kind) ∧
    douglas_count * douglas_price + ponderosa_count * ponderosa_price = total_paid :=
by sorry

end ponderosa_price_calculation_l1334_133412


namespace ratio_problem_l1334_133480

theorem ratio_problem (a b : ℝ) (h1 : a / b = 150 / 1) (h2 : a = 300) : b = 2 := by
  sorry

end ratio_problem_l1334_133480


namespace area_FDBG_is_155_l1334_133425

/-- Triangle ABC with given properties -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : Real)
  (AC : Real)
  (area : Real)
  (h_AB : AB = 60)
  (h_AC : AC = 15)
  (h_area : area = 180)

/-- Point D on AB -/
def D (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point E on AC -/
def E (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point F on DE and angle bisector of BAC -/
def F (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point G on BC and angle bisector of BAC -/
def G (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Length of AD -/
def AD (t : Triangle) : Real :=
  20

/-- Length of DB -/
def DB (t : Triangle) : Real :=
  40

/-- Length of AE -/
def AE (t : Triangle) : Real :=
  5

/-- Length of EC -/
def EC (t : Triangle) : Real :=
  10

/-- Area of quadrilateral FDBG -/
def area_FDBG (t : Triangle) : Real :=
  sorry

/-- Main theorem: Area of FDBG is 155 -/
theorem area_FDBG_is_155 (t : Triangle) :
  area_FDBG t = 155 := by
  sorry

end area_FDBG_is_155_l1334_133425


namespace inequality_proof_l1334_133484

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l1334_133484


namespace f_properties_l1334_133403

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x * Real.exp x else x^2 - 2*x + 1/2

theorem f_properties :
  (∀ x, x ≤ 0 → (deriv f) x = 2 * (1 + x) * Real.exp x) ∧
  (∀ x, x > 0 → (deriv f) x = 2*x - 2) ∧
  ((deriv f) (-2) = -2 / Real.exp 2) ∧
  (∀ x, f x ≥ -2 / Real.exp 1) ∧
  (∃ x, f x = -2 / Real.exp 1) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) :=
by
  sorry

end f_properties_l1334_133403


namespace complement_of_union_is_two_l1334_133447

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 4}

-- Define set B
def B : Set ℕ := {3, 4}

-- Theorem statement
theorem complement_of_union_is_two :
  (U \ (A ∪ B)) = {2} := by sorry

end complement_of_union_is_two_l1334_133447


namespace special_prime_sum_of_squares_l1334_133420

theorem special_prime_sum_of_squares (n : ℕ) : 
  (∃ a b : ℤ, n = a^2 + b^2 ∧ Int.gcd a b = 1) →
  (∀ p : ℕ, Nat.Prime p → p ≤ Nat.sqrt n → ∃ k : ℤ, k * p = a * b) →
  n = 5 ∨ n = 13 := by sorry

end special_prime_sum_of_squares_l1334_133420


namespace divide_fractions_l1334_133463

theorem divide_fractions : (3 : ℚ) / 4 / ((7 : ℚ) / 8) = 6 / 7 := by
  sorry

end divide_fractions_l1334_133463


namespace ellipse_equation_max_distance_max_distance_point_l1334_133401

/-- Definition of an ellipse with eccentricity 1/2 passing through (0, √3) -/
def Ellipse (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2) / a^2 = 1/4 ∧ b^2 = 3

/-- The equation of the ellipse is x²/4 + y²/3 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  Ellipse x y ↔ x^2/4 + y^2/3 = 1 :=
sorry

/-- The maximum distance from a point on the ellipse to (0, √3) is 2√3 -/
theorem max_distance :
  ∃ (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧
  ∀ (x y : ℝ), Ellipse x y →
  (x₀^2 + (y₀ - Real.sqrt 3)^2) ≥ (x^2 + (y - Real.sqrt 3)^2) ∧
  Real.sqrt (x₀^2 + (y₀ - Real.sqrt 3)^2) = 2 * Real.sqrt 3 :=
sorry

/-- The point that maximizes the distance has coordinates (-√3, 0) -/
theorem max_distance_point :
  ∃! (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧
  ∀ (x y : ℝ), Ellipse x y →
  (x₀^2 + (y₀ - Real.sqrt 3)^2) ≥ (x^2 + (y - Real.sqrt 3)^2) ∧
  x₀ = -Real.sqrt 3 ∧ y₀ = 0 :=
sorry

end ellipse_equation_max_distance_max_distance_point_l1334_133401


namespace inequality_solution_set_l1334_133436

theorem inequality_solution_set (a : ℝ) : 
  ((3 - a) / 2 - 2 = 2) → 
  {x : ℝ | (2 - a / 5) < (1 / 3) * x} = {x : ℝ | x > 9} := by
  sorry

end inequality_solution_set_l1334_133436


namespace robin_female_fraction_l1334_133473

theorem robin_female_fraction (total_birds : ℝ) (h1 : total_birds > 0) : 
  let robins : ℝ := (2/5) * total_birds
  let bluejays : ℝ := (3/5) * total_birds
  let female_bluejays : ℝ := (2/3) * bluejays
  let male_birds : ℝ := (7/15) * total_birds
  let female_robins : ℝ := (1/3) * robins
  female_robins + female_bluejays = total_birds - male_birds :=
by
  sorry

#check robin_female_fraction

end robin_female_fraction_l1334_133473


namespace race_track_cost_l1334_133405

theorem race_track_cost (initial_amount : ℚ) (num_cars : ℕ) (car_cost : ℚ) (remaining : ℚ) : 
  initial_amount = 17.80 ∧ 
  num_cars = 4 ∧ 
  car_cost = 0.95 ∧ 
  remaining = 8 → 
  initial_amount - (↑num_cars * car_cost) - remaining = 6 := by
sorry

end race_track_cost_l1334_133405


namespace relative_error_comparison_l1334_133443

/-- Given two measurements and their respective errors, this theorem states that
    the relative error of the second measurement is less than that of the first. -/
theorem relative_error_comparison
  (measurement1 : ℝ) (error1 : ℝ) (measurement2 : ℝ) (error2 : ℝ)
  (h1 : measurement1 = 0.15)
  (h2 : error1 = 0.03)
  (h3 : measurement2 = 125)
  (h4 : error2 = 0.25)
  : error2 / measurement2 < error1 / measurement1 := by
  sorry

end relative_error_comparison_l1334_133443


namespace tangent_line_equality_l1334_133466

noncomputable def f (x : ℝ) : ℝ := Real.log x

def g (a x : ℝ) : ℝ := a * x^2 - a

theorem tangent_line_equality (a : ℝ) : 
  (∀ x, deriv f x = deriv (g a) x) → a = 1/2 := by
  sorry

end tangent_line_equality_l1334_133466


namespace octal_subtraction_l1334_133482

/-- Converts a base 8 number represented as a list of digits to a natural number -/
def octalToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base 8 representation as a list of digits -/
def natToOctal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else go (m / 8) ((m % 8) :: acc)
    go n []

theorem octal_subtraction :
  let a := [1, 3, 5, 2]
  let b := [0, 6, 7, 4]
  let result := [1, 4, 5, 6]
  octalToNat a - octalToNat b = octalToNat result := by
  sorry

end octal_subtraction_l1334_133482


namespace simplify_expression_1_simplify_expression_2_l1334_133431

-- Define variables
variable (x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - 3/2 * x^2 * y) + x^2 * y^2) = -x^2 * y^2 := by sorry

end simplify_expression_1_simplify_expression_2_l1334_133431


namespace f_simplification_f_value_in_second_quadrant_l1334_133452

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (7 * Real.pi - α) * Real.cos (α + 3 * Real.pi / 2) * Real.cos (3 * Real.pi + α)) /
  (Real.sin (α - 3 * Real.pi / 2) * Real.cos (α + 5 * Real.pi / 2) * Real.tan (α - 5 * Real.pi))

theorem f_simplification (α : ℝ) : f α = Real.cos α := by sorry

theorem f_value_in_second_quadrant (α : ℝ) 
  (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : Real.cos (3 * Real.pi / 2 + α) = 1 / 7) : 
  f α = -4 * Real.sqrt 3 / 7 := by sorry

end f_simplification_f_value_in_second_quadrant_l1334_133452


namespace hexagon_area_in_triangle_l1334_133460

/-- The area of a regular hexagon inscribed in a square, which is inscribed in a circle, 
    which is in turn inscribed in a triangle with side length 6 cm, is 27√3 cm². -/
theorem hexagon_area_in_triangle (s : ℝ) (h : s = 6) : 
  let r := s / 2 * Real.sqrt 3 / 3
  let square_side := 2 * r
  let hexagon_side := r
  let hexagon_area := 3 * Real.sqrt 3 / 2 * hexagon_side ^ 2
  hexagon_area = 27 * Real.sqrt 3 := by sorry

end hexagon_area_in_triangle_l1334_133460


namespace soda_cost_l1334_133492

theorem soda_cost (burger_cost soda_cost : ℚ) : 
  (3 * burger_cost + 2 * soda_cost = 360) →
  (4 * burger_cost + 3 * soda_cost = 490) →
  soda_cost = 30 :=
by
  sorry

end soda_cost_l1334_133492


namespace justine_paper_usage_l1334_133454

theorem justine_paper_usage 
  (total_sheets : ℕ) 
  (num_binders : ℕ) 
  (sheets_per_binder : ℕ) 
  (justine_binder : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : num_binders = 5)
  (h3 : sheets_per_binder = total_sheets / num_binders)
  (h4 : justine_binder = sheets_per_binder / 2) :
  justine_binder = 245 := by
  sorry

end justine_paper_usage_l1334_133454


namespace bankers_gain_calculation_l1334_133422

/-- Banker's gain calculation -/
theorem bankers_gain_calculation 
  (time : ℝ) 
  (rate : ℝ) 
  (true_discount : ℝ) 
  (ε : ℝ) 
  (h1 : time = 1) 
  (h2 : rate = 12) 
  (h3 : true_discount = 55) 
  (h4 : ε > 0) : 
  ∃ (bankers_gain : ℝ), 
    abs (bankers_gain - 6.60) < ε ∧ 
    bankers_gain = 
      (((true_discount * 100) / (rate * time) + true_discount) * rate * time) / 100 - 
      true_discount :=
sorry

end bankers_gain_calculation_l1334_133422


namespace power_function_through_point_power_function_at_4_l1334_133491

/-- A power function that passes through the point (3, √3) -/
def f (x : ℝ) : ℝ := x^(1/2)

theorem power_function_through_point : f 3 = Real.sqrt 3 := by sorry

theorem power_function_at_4 : f 4 = 2 := by sorry

end power_function_through_point_power_function_at_4_l1334_133491


namespace ferris_wheel_seats_l1334_133408

/-- The number of people that can ride the Ferris wheel at the same time -/
def total_riders : ℕ := 4

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := 2

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := total_riders / people_per_seat

theorem ferris_wheel_seats : num_seats = 2 := by
  sorry

end ferris_wheel_seats_l1334_133408


namespace table_length_proof_l1334_133464

theorem table_length_proof (table_width : ℝ) (sheet_width sheet_height : ℝ) 
  (h1 : table_width = 80)
  (h2 : sheet_width = 8)
  (h3 : sheet_height = 5)
  (h4 : ∃ n : ℕ, n * 1 = table_width - sheet_width ∧ n * 1 = table_width - sheet_height) :
  ∃ x : ℝ, x = 77 ∧ x = table_width - (sheet_width - sheet_height) := by
sorry

end table_length_proof_l1334_133464


namespace kayak_rental_cost_l1334_133432

theorem kayak_rental_cost 
  (canoe_cost : ℝ) 
  (canoe_kayak_ratio : ℚ) 
  (total_revenue : ℝ) 
  (canoe_kayak_difference : ℕ) :
  canoe_cost = 12 →
  canoe_kayak_ratio = 3 / 2 →
  total_revenue = 504 →
  canoe_kayak_difference = 7 →
  ∃ (kayak_cost : ℝ) (num_canoes num_kayaks : ℕ),
    num_canoes = num_kayaks + canoe_kayak_difference ∧
    (num_canoes : ℚ) / num_kayaks = canoe_kayak_ratio ∧
    total_revenue = canoe_cost * num_canoes + kayak_cost * num_kayaks ∧
    kayak_cost = 18 :=
by sorry

end kayak_rental_cost_l1334_133432


namespace gcf_of_75_and_105_l1334_133483

theorem gcf_of_75_and_105 : Nat.gcd 75 105 = 15 := by
  sorry

end gcf_of_75_and_105_l1334_133483


namespace tangent_line_to_curve_l1334_133418

/-- A line y = x - 2a is tangent to the curve y = x ln x - x if and only if a = e/2 -/
theorem tangent_line_to_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (x₀ - 2*a = x₀ * Real.log x₀ - x₀) ∧ 
    (1 = Real.log x₀)) ↔ 
  a = Real.exp 1 / 2 := by
sorry

end tangent_line_to_curve_l1334_133418


namespace basketball_probability_l1334_133470

theorem basketball_probability (adam beth jack jill sandy : ℚ)
  (h_adam : adam = 1/5)
  (h_beth : beth = 2/9)
  (h_jack : jack = 1/6)
  (h_jill : jill = 1/7)
  (h_sandy : sandy = 1/8) :
  (1 - adam) * beth * (1 - jack) * jill * sandy = 1/378 :=
by sorry

end basketball_probability_l1334_133470


namespace cos_arcsin_eight_seventeenths_l1334_133459

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
  sorry

end cos_arcsin_eight_seventeenths_l1334_133459


namespace larger_number_proof_l1334_133493

theorem larger_number_proof (x y : ℤ) : 
  x + y = 84 → y = x + 12 → y = 48 := by
  sorry

end larger_number_proof_l1334_133493


namespace hyperbola_distance_theorem_l1334_133411

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- A point is on a hyperbola if the absolute difference of its distances to the foci is constant -/
def IsOnHyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), |distance p h.F₁ - distance p h.F₂| = k

theorem hyperbola_distance_theorem (h : Hyperbola) (p : ℝ × ℝ) :
  IsOnHyperbola h p → distance p h.F₁ = 12 →
  distance p h.F₂ = 22 ∨ distance p h.F₂ = 2 := by sorry

end hyperbola_distance_theorem_l1334_133411


namespace coefficient_relation_l1334_133438

/-- A polynomial function with specific properties -/
def g (a b c d e : ℝ) (x : ℝ) : ℝ := a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- Theorem stating the relationship between coefficients a and b -/
theorem coefficient_relation (a b c d e : ℝ) :
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  (g a b c d e 0 = 3) →
  b = -2*a := by sorry

end coefficient_relation_l1334_133438


namespace nancy_money_l1334_133409

def five_dollar_bills : ℕ := 9
def ten_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 7

def total_money : ℕ := five_dollar_bills * 5 + ten_dollar_bills * 10 + one_dollar_bills * 1

theorem nancy_money : total_money = 92 := by
  sorry

end nancy_money_l1334_133409


namespace river_distance_l1334_133476

theorem river_distance (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) := by
  sorry

end river_distance_l1334_133476


namespace exponential_function_properties_l1334_133471

/-- A function f(x) = b * a^x with specific properties -/
structure ExponentialFunction where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_neq_one : a ≠ 1
  f_def : ℝ → ℝ
  f_eq : ∀ x, f_def x = b * a^x
  f_at_one : f_def 1 = 27
  f_at_neg_one : f_def (-1) = 3

/-- The main theorem capturing the properties of the exponential function -/
theorem exponential_function_properties (f : ExponentialFunction) :
  (f.a = 3 ∧ f.b = 9) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f.a^x + f.b^x ≥ m) ↔ m ≤ 12) := by
  sorry

end exponential_function_properties_l1334_133471
