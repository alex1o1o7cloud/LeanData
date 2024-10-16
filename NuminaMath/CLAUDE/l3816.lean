import Mathlib

namespace NUMINAMATH_CALUDE_f_value_at_11pi_over_6_l3816_381689

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, (is_periodic f q ∧ q > 0) → p ≤ q

theorem f_value_at_11pi_over_6 (f : ℝ → ℝ) :
  is_odd f →
  smallest_positive_period f π →
  (∀ x ∈ Set.Ioo 0 (π/2), f x = 2 * Real.sin x) →
  f (11*π/6) = -1 := by sorry

end NUMINAMATH_CALUDE_f_value_at_11pi_over_6_l3816_381689


namespace NUMINAMATH_CALUDE_function_satisfying_lcm_gcd_condition_l3816_381621

theorem function_satisfying_lcm_gcd_condition :
  ∀ (f : ℕ → ℕ),
    (∀ (m n : ℕ), m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)) →
    ∃ (k : ℕ), k > 0 ∧ ∀ (x : ℕ), f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_lcm_gcd_condition_l3816_381621


namespace NUMINAMATH_CALUDE_q_polynomial_form_l3816_381654

/-- The function q satisfying the given equation -/
noncomputable def q : ℝ → ℝ := fun x => 4*x^4 + 16*x^3 + 36*x^2 + 10*x + 4 - (2*x^6 + 5*x^4 + 11*x^2 + 6*x)

/-- Theorem stating that q has the specified polynomial form -/
theorem q_polynomial_form : q = fun x => -2*x^6 - x^4 + 16*x^3 + 25*x^2 + 4*x + 4 := by sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l3816_381654


namespace NUMINAMATH_CALUDE_cindy_jump_rope_time_l3816_381634

/-- Cindy's jump rope time in minutes -/
def cindy_time : ℕ := 12

/-- Betsy's jump rope time in minutes -/
def betsy_time : ℕ := cindy_time / 2

/-- Tina's jump rope time in minutes -/
def tina_time : ℕ := 3 * betsy_time

theorem cindy_jump_rope_time :
  cindy_time = 12 ∧
  betsy_time = cindy_time / 2 ∧
  tina_time = 3 * betsy_time ∧
  tina_time = cindy_time + 6 :=
by sorry

end NUMINAMATH_CALUDE_cindy_jump_rope_time_l3816_381634


namespace NUMINAMATH_CALUDE_area_enclosed_by_line_and_parabola_l3816_381692

/-- The area of the region enclosed by y = (a/6)x and y = x^2, 
    where a is the constant term in (x + 2/x)^n and 
    the sum of coefficients in the expansion is 81 -/
theorem area_enclosed_by_line_and_parabola (n : ℕ) (a : ℝ) : 
  (3 : ℝ)^n = 81 →
  (∃ k, (Nat.choose 4 2) * 2^2 = k ∧ k = a) →
  (∫ x in (0)..(a/4), (a/6 * x - x^2)) = 32/3 := by
sorry

end NUMINAMATH_CALUDE_area_enclosed_by_line_and_parabola_l3816_381692


namespace NUMINAMATH_CALUDE_g_difference_l3816_381667

noncomputable def g (n : ℤ) : ℝ :=
  (2 + Real.sqrt 2) / 4 * ((1 + Real.sqrt 2) / 2) ^ n + 
  (2 - Real.sqrt 2) / 4 * ((1 - Real.sqrt 2) / 2) ^ n

theorem g_difference (n : ℤ) : g (n + 1) - g (n - 1) = (Real.sqrt 2 / 2) * g n := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l3816_381667


namespace NUMINAMATH_CALUDE_even_function_implies_b_zero_solution_set_inequality_l3816_381661

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

-- Theorem 1: If f is even, then b = 0
theorem even_function_implies_b_zero (b : ℝ) :
  (∀ x : ℝ, f b x = f b (-x)) → b = 0 := by sorry

-- Define the specific function f with b = 0
def f_zero (x : ℝ) : ℝ := x^2 + 1

-- Theorem 2: Solution set of f(x-1) < |x|
theorem solution_set_inequality :
  {x : ℝ | f_zero (x - 1) < |x|} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_even_function_implies_b_zero_solution_set_inequality_l3816_381661


namespace NUMINAMATH_CALUDE_popsicle_melting_speed_l3816_381644

theorem popsicle_melting_speed (n : ℕ) (a : ℕ → ℝ) :
  n = 6 →
  (∀ i, 1 ≤ i → i < n → a (i + 1) = 2 * a i) →
  a n = 32 * a 1 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_melting_speed_l3816_381644


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3816_381697

theorem quadratic_root_problem (a : ℝ) (k : ℝ) :
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = a + 3*Complex.I) →
  k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3816_381697


namespace NUMINAMATH_CALUDE_patty_weight_factor_l3816_381608

/-- Given:
  - Robbie weighs 100 pounds
  - Patty was initially x times as heavy as Robbie
  - Patty lost 235 pounds
  - After weight loss, Patty weighs 115 pounds more than Robbie
Prove that x = 4.5 -/
theorem patty_weight_factor (x : ℝ) 
  (robbie_weight : ℝ) (patty_weight_loss : ℝ) (patty_final_difference : ℝ)
  (h1 : robbie_weight = 100)
  (h2 : patty_weight_loss = 235)
  (h3 : patty_final_difference = 115)
  (h4 : x * robbie_weight - patty_weight_loss = robbie_weight + patty_final_difference) :
  x = 4.5 := by
sorry

end NUMINAMATH_CALUDE_patty_weight_factor_l3816_381608


namespace NUMINAMATH_CALUDE_part1_part2_part3_l3816_381695

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

-- Part 1
theorem part1 (m : ℝ) : (∀ x, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∀ x, f m x ≥ (m + 1) * x) ↔
  (m = -1 ∧ ∀ x, x ≥ 1) ∨
  (m > -1 ∧ ∀ x, x ≤ (m - 1) / (m + 1) ∨ x ≥ 1) ∨
  (m < -1 ∧ ∀ x, 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) :=
sorry

-- Part 3
theorem part3 (m : ℝ) : (∀ x ∈ Set.Icc (-1/2) (1/2), f m x ≥ 0) ↔ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l3816_381695


namespace NUMINAMATH_CALUDE_reading_pages_solution_l3816_381606

/-- The number of pages Xiao Ming's father reads per day -/
def father_pages : ℕ := sorry

/-- The number of pages Xiao Ming reads per day -/
def xiao_ming_pages : ℕ := sorry

/-- Xiao Ming reads 5 pages more than his father every day -/
axiom pages_difference : xiao_ming_pages = father_pages + 5

/-- The time it takes for Xiao Ming to read 100 pages is equal to the time it takes for his father to read 80 pages -/
axiom reading_time_equality : (100 : ℚ) / xiao_ming_pages = (80 : ℚ) / father_pages

theorem reading_pages_solution :
  father_pages = 20 ∧ xiao_ming_pages = 25 :=
sorry

end NUMINAMATH_CALUDE_reading_pages_solution_l3816_381606


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3816_381686

theorem quadratic_root_difference (p : ℝ) : 
  let a := 1
  let b := -(2*p + 1)
  let c := p^2 - 5
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / (2*a)
  root_difference = Real.sqrt (2*p^2 + 4*p + 11) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3816_381686


namespace NUMINAMATH_CALUDE_robin_photos_count_l3816_381631

/-- Given that each page holds six photos and Robin can fill 122 full pages,
    prove that Robin has 732 photos in total. -/
theorem robin_photos_count :
  let photos_per_page : ℕ := 6
  let full_pages : ℕ := 122
  photos_per_page * full_pages = 732 :=
by sorry

end NUMINAMATH_CALUDE_robin_photos_count_l3816_381631


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l3816_381681

theorem complex_subtraction_simplification :
  (5 - 3*I) - (2 + 7*I) = 3 - 10*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l3816_381681


namespace NUMINAMATH_CALUDE_circle_c_equation_l3816_381660

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ

-- Define the equation of a circle
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 
    (c.passes_through.1 - c.center.1)^2 + (c.passes_through.2 - c.center.2)^2

-- Theorem statement
theorem circle_c_equation :
  let c : Circle := { center := (1, 1), passes_through := (0, 0) }
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_c_equation_l3816_381660


namespace NUMINAMATH_CALUDE_regression_line_equation_specific_regression_line_equation_l3816_381652

/-- The regression line equation given the y-intercept and a point it passes through -/
theorem regression_line_equation (a : ℝ) (x_center y_center : ℝ) :
  let b := (y_center - a) / x_center
  (∀ x y, y = b * x + a) → (y_center = b * x_center + a) :=
by
  sorry

/-- The specific regression line equation for the given problem -/
theorem specific_regression_line_equation :
  let a := 0.2
  let x_center := 4
  let y_center := 5
  let b := (y_center - a) / x_center
  (∀ x y, y = b * x + a) ∧ (y_center = b * x_center + a) ∧ (b = 1.2) :=
by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_specific_regression_line_equation_l3816_381652


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3816_381609

theorem solve_linear_equation :
  ∃ x : ℝ, 45 - 3 * x = 12 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3816_381609


namespace NUMINAMATH_CALUDE_set_relation_proof_l3816_381637

theorem set_relation_proof (M P : Set α) (h_nonempty : M.Nonempty) 
  (h_not_subset : ¬(M ⊆ P)) : 
  (∃ x ∈ M, x ∉ P) ∧ ¬(∀ x ∈ M, x ∈ P) := by
  sorry

end NUMINAMATH_CALUDE_set_relation_proof_l3816_381637


namespace NUMINAMATH_CALUDE_impossibility_of_measuring_one_liter_l3816_381651

theorem impossibility_of_measuring_one_liter :
  ¬ ∃ (k l : ℤ), k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_measuring_one_liter_l3816_381651


namespace NUMINAMATH_CALUDE_half_angle_in_third_quadrant_l3816_381699

theorem half_angle_in_third_quadrant (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (|Real.sin (θ / 2)| = -Real.sin (θ / 2)) →  -- |sin(θ/2)| = -sin(θ/2)
  (π < θ / 2 ∧ θ / 2 < 3 * π / 2) -- θ/2 is in the third quadrant
  := by sorry

end NUMINAMATH_CALUDE_half_angle_in_third_quadrant_l3816_381699


namespace NUMINAMATH_CALUDE_ed_length_l3816_381647

/-- Given five points in a plane with specific distances between them, prove that ED = 74 -/
theorem ed_length (A B C D E : EuclideanSpace ℝ (Fin 2)) 
  (h_AB : dist A B = 12)
  (h_BC : dist B C = 50)
  (h_CD : dist C D = 38)
  (h_AD : dist A D = 100)
  (h_BE : dist B E = 30)
  (h_CE : dist C E = 40) :
  dist E D = 74 := by
  sorry

end NUMINAMATH_CALUDE_ed_length_l3816_381647


namespace NUMINAMATH_CALUDE_red_balls_count_l3816_381683

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → prob = 2/35 → ∃ r : ℕ, r = 6 ∧ 
  (r : ℚ) / total * ((r : ℚ) - 1) / (total - 1) = prob := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3816_381683


namespace NUMINAMATH_CALUDE_cherry_sales_analysis_l3816_381629

/-- Represents the daily sales of cherries -/
structure CherrySales where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ

/-- The specific cherry sales scenario -/
def cherry_scenario : CherrySales where
  purchase_price := 20
  min_selling_price := 20
  max_selling_price := 40
  sales_function := λ x => -2 * x + 160
  profit_function := λ x => (x - 20) * (-2 * x + 160)

theorem cherry_sales_analysis (c : CherrySales) 
  (h1 : c.purchase_price = 20)
  (h2 : c.min_selling_price = 20)
  (h3 : c.max_selling_price = 40)
  (h4 : c.sales_function 25 = 110)
  (h5 : c.sales_function 30 = 100)
  (h6 : ∀ x, c.min_selling_price ≤ x ∧ x ≤ c.max_selling_price → 
    c.sales_function x = -2 * x + 160)
  (h7 : ∀ x, c.profit_function x = (x - c.purchase_price) * (c.sales_function x)) :
  (∀ x, c.sales_function x = -2 * x + 160) ∧ 
  (∃ x, c.min_selling_price ≤ x ∧ x ≤ c.max_selling_price ∧ c.profit_function x = 1000 ∧ x = 30) ∧
  (∃ x, c.min_selling_price ≤ x ∧ x ≤ c.max_selling_price ∧ 
    ∀ y, c.min_selling_price ≤ y ∧ y ≤ c.max_selling_price → c.profit_function x ≥ c.profit_function y) ∧
  (∃ x, c.profit_function x = 1600 ∧ x = 40) := by
  sorry

#check cherry_sales_analysis cherry_scenario

end NUMINAMATH_CALUDE_cherry_sales_analysis_l3816_381629


namespace NUMINAMATH_CALUDE_probability_is_75_1024_l3816_381638

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of moving in any direction -/
def directionProbability : ℚ := 1/4

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The target point -/
def target : Point := ⟨3, 3⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 8

/-- Calculates the probability of reaching the target point from the start point
    in at most maxSteps steps -/
def probabilityToReachTarget (start : Point) (target : Point) (maxSteps : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem probability_is_75_1024 :
  probabilityToReachTarget start target maxSteps = 75/1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_75_1024_l3816_381638


namespace NUMINAMATH_CALUDE_amy_balloon_count_l3816_381657

/-- Given that James has 1222 balloons and 709 more balloons than Amy,
    prove that Amy has 513 balloons. -/
theorem amy_balloon_count :
  ∀ (james_balloons amy_balloons : ℕ),
    james_balloons = 1222 →
    james_balloons = amy_balloons + 709 →
    amy_balloons = 513 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_balloon_count_l3816_381657


namespace NUMINAMATH_CALUDE_total_animals_equals_total_humps_l3816_381633

/-- Represents the composition of a herd of animals -/
structure Herd where
  horses : ℕ
  twoHumpedCamels : ℕ
  oneHumpedCamels : ℕ

/-- Calculates the total number of humps in the herd -/
def totalHumps (h : Herd) : ℕ :=
  2 * h.twoHumpedCamels + h.oneHumpedCamels

/-- Calculates the total number of animals in the herd -/
def totalAnimals (h : Herd) : ℕ :=
  h.horses + h.twoHumpedCamels + h.oneHumpedCamels

/-- Theorem stating that the total number of animals equals the total number of humps
    under specific conditions -/
theorem total_animals_equals_total_humps (h : Herd) :
  h.horses = h.twoHumpedCamels →
  totalHumps h = 200 →
  totalAnimals h = 200 := by
  sorry

#check total_animals_equals_total_humps

end NUMINAMATH_CALUDE_total_animals_equals_total_humps_l3816_381633


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3816_381635

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
  (x + c)^2 / 4 + y^2 / 4 = b^2) → 
  c^2 / a^2 = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3816_381635


namespace NUMINAMATH_CALUDE_triangle_problem_l3816_381690

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a = 3 * Real.sqrt 2 →
  c = Real.sqrt 3 →
  Real.cos C = 2 * Real.sqrt 2 / 3 →
  b < a →
  -- Conclusion
  Real.sin A = Real.sqrt 6 / 3 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3816_381690


namespace NUMINAMATH_CALUDE_smaller_circles_radius_l3816_381696

/-- Given a central circle of radius 2 and 4 identical smaller circles
    touching the central circle and each other, the radius of each smaller circle is 6. -/
theorem smaller_circles_radius (r : ℝ) : r = 6 :=
  by
  -- Define the relationship between the radii
  have h1 : (2 + r)^2 + (2 + r)^2 = (2*r)^2 :=
    sorry
  -- Solve the resulting equation
  have h2 : r^2 - 4*r - 4 = 0 :=
    sorry
  -- Apply the quadratic formula and choose the positive solution
  sorry

end NUMINAMATH_CALUDE_smaller_circles_radius_l3816_381696


namespace NUMINAMATH_CALUDE_solution_set_linear_inequalities_l3816_381665

theorem solution_set_linear_inequalities :
  let S := {x : ℝ | x - 2 > 1 ∧ x < 4}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_linear_inequalities_l3816_381665


namespace NUMINAMATH_CALUDE_percent_relation_l3816_381603

theorem percent_relation (x y z : ℝ) 
  (hxy : x = 1.20 * y) 
  (hyz : y = 0.30 * z) : 
  x = 0.36 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3816_381603


namespace NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l3816_381623

theorem not_prime_two_pow_plus_one (n : ℕ) (d : ℕ) (h_odd : Odd d) (h_div : d ∣ n) :
  ¬ Nat.Prime (2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l3816_381623


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l3816_381602

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define the condition that y decreases as x increases
def decreasing_y (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

-- Define what it means for a point to be in Quadrant III
def in_quadrant_III (x : ℝ) (y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- Theorem statement
theorem linear_function_not_in_quadrant_III (k : ℝ) :
  decreasing_y (linear_function k) →
  ¬∃ x, in_quadrant_III x (linear_function k x) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l3816_381602


namespace NUMINAMATH_CALUDE_gilled_mushroom_count_l3816_381641

/-- Represents the types of mushrooms --/
inductive MushroomType
  | Spotted
  | Gilled

/-- Represents a mushroom --/
structure Mushroom where
  type : MushroomType

/-- Represents a collection of mushrooms on a log --/
structure MushroomLog where
  mushrooms : Finset Mushroom
  total_count : Nat
  spotted_count : Nat
  gilled_count : Nat
  h_total : total_count = mushrooms.card
  h_partition : total_count = spotted_count + gilled_count
  h_types : ∀ m ∈ mushrooms, m.type = MushroomType.Spotted ∨ m.type = MushroomType.Gilled
  h_ratio : spotted_count = 9 * gilled_count

theorem gilled_mushroom_count (log : MushroomLog) (h : log.total_count = 30) :
  log.gilled_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_gilled_mushroom_count_l3816_381641


namespace NUMINAMATH_CALUDE_train_speed_theorem_l3816_381628

-- Define the length of the train in meters
def train_length : ℝ := 300

-- Define the time taken to cross the platform in seconds
def crossing_time : ℝ := 60

-- Define the total distance covered (train length + platform length)
def total_distance : ℝ := 2 * train_length

-- Define the speed conversion factor from m/s to km/h
def speed_conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_theorem :
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * speed_conversion_factor
  speed_kmh = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l3816_381628


namespace NUMINAMATH_CALUDE_parallelogram_base_l3816_381639

/-- Given a parallelogram with area 462 square centimeters and height 21 cm, its base is 22 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 462 → height = 21 → area = base * height → base = 22 := by sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3816_381639


namespace NUMINAMATH_CALUDE_book_page_digits_l3816_381653

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) +
  2 * (min n 99 - min n 9) +
  3 * (n - min n 99)

/-- Theorem: The total number of digits used in numbering the pages of a book with 356 pages is 960 -/
theorem book_page_digits :
  totalDigits 356 = 960 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l3816_381653


namespace NUMINAMATH_CALUDE_loan_period_duration_l3816_381617

/-- The amount of money lent (in Rs.) -/
def loanAmount : ℝ := 3150

/-- The interest rate A charges B (as a decimal) -/
def rateAtoB : ℝ := 0.08

/-- The interest rate B charges C (as a decimal) -/
def rateBtoC : ℝ := 0.125

/-- B's total gain over the period (in Rs.) -/
def totalGain : ℝ := 283.5

/-- The duration of the period in years -/
def periodYears : ℝ := 2

theorem loan_period_duration :
  periodYears * (rateBtoC * loanAmount - rateAtoB * loanAmount) = totalGain :=
by sorry

end NUMINAMATH_CALUDE_loan_period_duration_l3816_381617


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l3816_381605

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The total volume of a mixture -/
def Mixture.volume (m : Mixture) : ℚ := m.milk + m.water

/-- The ratio of milk to water in a mixture -/
def Mixture.ratio (m : Mixture) : ℚ := m.milk / m.water

theorem milk_water_ratio_problem (m1 m2 : Mixture) :
  m1.volume = m2.volume →
  m1.ratio = 7/2 →
  (Mixture.mk (m1.milk + m2.milk) (m1.water + m2.water)).ratio = 5 →
  m2.ratio = 8 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l3816_381605


namespace NUMINAMATH_CALUDE_outfits_count_l3816_381684

theorem outfits_count (shirts : ℕ) (hats : ℕ) : shirts = 5 → hats = 3 → shirts * hats = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3816_381684


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l3816_381613

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∨ t.angle1 = t.angle3 ∨ t.angle2 = t.angle3

-- Define the sum of angles in a triangle
def angleSum (t : Triangle) : ℝ :=
  t.angle1 + t.angle2 + t.angle3

-- Theorem statement
theorem isosceles_triangle_proof (t : Triangle) 
  (h1 : t.angle1 = 40)
  (h2 : t.angle2 = 70)
  (h3 : angleSum t = 180) :
  isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l3816_381613


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l3816_381655

/-- The line y = x + b is tangent to the curve x = √(1 - y²) if and only if b = -√2 -/
theorem line_tangent_to_curve (b : ℝ) : 
  (∀ x y : ℝ, y = x + b ∧ x = Real.sqrt (1 - y^2) → 
    (∃! p : ℝ × ℝ, p.1 = Real.sqrt (1 - p.2^2) ∧ p.2 = p.1 + b)) ↔ 
  b = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_curve_l3816_381655


namespace NUMINAMATH_CALUDE_final_number_is_odd_l3816_381648

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The process of replacing two numbers with their absolute difference cubed -/
def replace_process (a b : ℤ) : ℤ := |a - b|^3

/-- The theorem stating that the final number after the replace process is odd -/
theorem final_number_is_odd (n : ℕ) (h : n = 2017) :
  ∃ (k : ℕ), Odd (sum_to_n n) ∧
  (∀ (a b : ℤ), Odd (a + b) ↔ Odd (replace_process a b)) →
  Odd k := by sorry

end NUMINAMATH_CALUDE_final_number_is_odd_l3816_381648


namespace NUMINAMATH_CALUDE_property_one_property_two_property_three_f_satisfies_all_properties_l3816_381668

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- Property 1: f(x₁x₂) = f(x₁)f(x₂)
theorem property_one : ∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂ := by sorry

-- Property 2: For x ∈ (0, +∞), f'(x) > 0
theorem property_two : ∀ x : ℝ, x > 0 → (deriv f) x > 0 := by sorry

-- Property 3: f'(x) is an odd function
theorem property_three : ∀ x : ℝ, (deriv f) (-x) = -(deriv f) x := by sorry

-- Main theorem: f(x) = x² satisfies all three properties
theorem f_satisfies_all_properties : 
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (∀ x : ℝ, x > 0 → (deriv f) x > 0) ∧ 
  (∀ x : ℝ, (deriv f) (-x) = -(deriv f) x) := by sorry

end NUMINAMATH_CALUDE_property_one_property_two_property_three_f_satisfies_all_properties_l3816_381668


namespace NUMINAMATH_CALUDE_max_x_coordinate_P_max_x_coordinate_P_achieved_l3816_381625

/-- The maximum x-coordinate of point P on line OA, where A is on the ellipse x²/16 + y²/4 = 1 and OA · OP = 6 -/
theorem max_x_coordinate_P (A : ℝ × ℝ) (P : ℝ × ℝ) : 
  (A.1^2 / 16 + A.2^2 / 4 = 1) →  -- A is on the ellipse
  (∃ t : ℝ, P = (t * A.1, t * A.2)) →  -- P is on the line OA
  (A.1 * P.1 + A.2 * P.2 = 6) →  -- OA · OP = 6
  P.1 ≤ Real.sqrt 3 := by
sorry

/-- The maximum x-coordinate of point P is achieved -/
theorem max_x_coordinate_P_achieved (A : ℝ × ℝ) : 
  (A.1^2 / 16 + A.2^2 / 4 = 1) →  -- A is on the ellipse
  ∃ P : ℝ × ℝ, 
    (∃ t : ℝ, P = (t * A.1, t * A.2)) ∧  -- P is on the line OA
    (A.1 * P.1 + A.2 * P.2 = 6) ∧  -- OA · OP = 6
    P.1 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_x_coordinate_P_max_x_coordinate_P_achieved_l3816_381625


namespace NUMINAMATH_CALUDE_walts_investment_l3816_381612

/-- Proves that given the conditions of Walt's investment, the unknown interest rate is 8% -/
theorem walts_investment (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (unknown_investment : ℝ) :
  total_investment = 9000 →
  known_rate = 0.09 →
  total_interest = 770 →
  unknown_investment = 4000 →
  ∃ (unknown_rate : ℝ),
    unknown_rate * unknown_investment + known_rate * (total_investment - unknown_investment) = total_interest ∧
    unknown_rate = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_walts_investment_l3816_381612


namespace NUMINAMATH_CALUDE_cookie_division_l3816_381666

def cookie_area : ℝ := 81.12
def num_friends : ℕ := 6

theorem cookie_division (area_per_person : ℝ) :
  area_per_person = cookie_area / num_friends →
  area_per_person = 13.52 := by
  sorry

end NUMINAMATH_CALUDE_cookie_division_l3816_381666


namespace NUMINAMATH_CALUDE_system_solution_l3816_381619

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k + 1 →
  2*x + y = 1 →
  x + y = 3 →
  k = 7 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3816_381619


namespace NUMINAMATH_CALUDE_instrumental_measurements_insufficient_l3816_381646

-- Define the concept of instrumental measurements
def InstrumentalMeasurement : Type := Unit

-- Define the concept of general geometric statements
def GeneralGeometricStatement : Type := Unit

-- Define the property of being approximate
def is_approximate (m : InstrumentalMeasurement) : Prop := sorry

-- Define the property of applying to infinite configurations
def applies_to_infinite_configurations (s : GeneralGeometricStatement) : Prop := sorry

-- Define the property of being performed on a finite number of instances
def performed_on_finite_instances (m : InstrumentalMeasurement) : Prop := sorry

-- Theorem stating that instrumental measurements are insufficient to justify general geometric statements
theorem instrumental_measurements_insufficient 
  (m : InstrumentalMeasurement) 
  (s : GeneralGeometricStatement) : 
  is_approximate m → 
  applies_to_infinite_configurations s → 
  performed_on_finite_instances m → 
  ¬(∃ (justification : Unit), True) := by sorry

end NUMINAMATH_CALUDE_instrumental_measurements_insufficient_l3816_381646


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3816_381642

theorem complete_square_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -6 ∧ c = -16) :
  ∃ (k m : ℝ), ∀ x, (a * x^2 + b * x + c = 0) ↔ ((x + k)^2 = m) ∧ m = 25 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3816_381642


namespace NUMINAMATH_CALUDE_fred_balloons_l3816_381685

theorem fred_balloons (initial_balloons given_balloons : ℕ) 
  (h1 : initial_balloons = 709)
  (h2 : given_balloons = 221) :
  initial_balloons - given_balloons = 488 :=
by sorry

end NUMINAMATH_CALUDE_fred_balloons_l3816_381685


namespace NUMINAMATH_CALUDE_angle_supplement_complement_relation_l3816_381604

theorem angle_supplement_complement_relation (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 40) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_complement_relation_l3816_381604


namespace NUMINAMATH_CALUDE_initial_girls_count_l3816_381607

theorem initial_girls_count (initial_boys : ℕ) (boys_left : ℕ) (girls_entered : ℕ) (final_total : ℕ) :
  initial_boys = 5 →
  boys_left = 3 →
  girls_entered = 2 →
  final_total = 8 →
  ∃ initial_girls : ℕ, 
    initial_girls = 4 ∧
    final_total = (initial_boys - boys_left) + (initial_girls + girls_entered) :=
by sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3816_381607


namespace NUMINAMATH_CALUDE_divisible_by_three_l3816_381620

theorem divisible_by_three (n : ℕ) : ∃ k : ℤ, 2 * 7^n + 1 = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l3816_381620


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_l3816_381622

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x - 3 * y = 12

-- Define the equidistant condition
def equidistant (x y : ℝ) : Prop := abs x = abs y

-- Define quadrants
def quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem equidistant_points_on_line :
  (∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_I x y) ∧
  (∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_IV x y) ∧
  (¬∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_II x y) ∧
  (¬∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_III x y) :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_l3816_381622


namespace NUMINAMATH_CALUDE_product_equals_root_fraction_l3816_381669

theorem product_equals_root_fraction (a b c : ℝ) :
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1) →
  6 * 15 * 7 = (3 / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_root_fraction_l3816_381669


namespace NUMINAMATH_CALUDE_first_stack_height_is_5_l3816_381693

/-- The height of the first stack of blocks -/
def first_stack_height : ℕ := sorry

/-- The height of the second stack of blocks -/
def second_stack_height : ℕ := first_stack_height + 2

/-- The height of the third stack of blocks -/
def third_stack_height : ℕ := second_stack_height - 5

/-- The height of the fourth stack of blocks -/
def fourth_stack_height : ℕ := third_stack_height + 5

/-- The total number of blocks used -/
def total_blocks : ℕ := 21

theorem first_stack_height_is_5 : 
  first_stack_height = 5 ∧ 
  first_stack_height + second_stack_height + third_stack_height + fourth_stack_height = total_blocks :=
sorry

end NUMINAMATH_CALUDE_first_stack_height_is_5_l3816_381693


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l3816_381659

theorem inverse_of_proposition (p q : Prop) :
  (¬p → ¬q) → (¬q → ¬p) := by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l3816_381659


namespace NUMINAMATH_CALUDE_evaluate_expression_l3816_381679

theorem evaluate_expression : (528 : ℤ) * 528 - (527 * 529) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3816_381679


namespace NUMINAMATH_CALUDE_reflected_quad_area_l3816_381662

/-- A convex quadrilateral in the plane -/
structure ConvexQuadrilateral where
  -- We don't need to define the specifics of the quadrilateral,
  -- just that it exists and has an area
  area : ℝ
  area_pos : area > 0

/-- The quadrilateral formed by reflecting a point inside a convex quadrilateral 
    with respect to the midpoints of its sides -/
def reflectedQuadrilateral (Q : ConvexQuadrilateral) : ConvexQuadrilateral where
  -- We don't need to define how this quadrilateral is constructed,
  -- just that it exists and is related to the original quadrilateral
  area := 2 * Q.area
  area_pos := by
    -- The proof that the area is positive
    sorry

/-- Theorem stating that the area of the reflected quadrilateral 
    is twice the area of the original quadrilateral -/
theorem reflected_quad_area (Q : ConvexQuadrilateral) :
  (reflectedQuadrilateral Q).area = 2 * Q.area := by
  -- The proof of the theorem
  sorry

end NUMINAMATH_CALUDE_reflected_quad_area_l3816_381662


namespace NUMINAMATH_CALUDE_no_prime_sum_for_10003_l3816_381687

/-- A function that returns the number of ways to write a natural number as the sum of two primes -/
def count_prime_sum_representations (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_for_10003 : count_prime_sum_representations 10003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_for_10003_l3816_381687


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l3816_381678

theorem square_rectangle_area_relation :
  let square_side : ℝ → ℝ := λ x => x - 5
  let rect_length : ℝ → ℝ := λ x => x - 4
  let rect_width : ℝ → ℝ := λ x => x + 5
  let square_area : ℝ → ℝ := λ x => (square_side x) ^ 2
  let rect_area : ℝ → ℝ := λ x => (rect_length x) * (rect_width x)
  ∃ x₁ x₂ : ℝ, x₁ > 5 ∧ x₂ > 5 ∧
    2 * x₁^2 - 31 * x₁ + 95 = 0 ∧
    2 * x₂^2 - 31 * x₂ + 95 = 0 ∧
    3 * (square_area x₁) = rect_area x₁ ∧
    3 * (square_area x₂) = rect_area x₂ ∧
    x₁ + x₂ = 31/2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l3816_381678


namespace NUMINAMATH_CALUDE_circles_radius_order_l3816_381616

noncomputable def circle_A_radius : ℝ := 3

noncomputable def circle_B_area : ℝ := 12 * Real.pi

noncomputable def circle_C_area : ℝ := 28 * Real.pi

noncomputable def circle_B_radius : ℝ := Real.sqrt (circle_B_area / Real.pi)

noncomputable def circle_C_radius : ℝ := Real.sqrt (circle_C_area / Real.pi)

theorem circles_radius_order :
  circle_A_radius < circle_B_radius ∧ circle_B_radius < circle_C_radius := by
  sorry

end NUMINAMATH_CALUDE_circles_radius_order_l3816_381616


namespace NUMINAMATH_CALUDE_balance_weights_l3816_381611

def weights : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def target_weight : ℕ := 1998

theorem balance_weights :
  (∀ w ∈ weights, is_power_of_two w) →
  (∃ subset : List ℕ, subset.Subset weights ∧ subset.sum = target_weight ∧ subset.length = 8) ∧
  (∀ subset : List ℕ, subset.Subset weights → subset.sum = target_weight → subset.length ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_balance_weights_l3816_381611


namespace NUMINAMATH_CALUDE_calculation_proof_l3816_381664

theorem calculation_proof : (-1)^2 + |-Real.sqrt 2| + (Real.pi - 3)^0 - Real.sqrt 4 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3816_381664


namespace NUMINAMATH_CALUDE_first_three_decimal_digits_l3816_381649

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2005 → x = (10^n + 1)^(11/8) → 
  ∃ (k : ℕ), x = k + 0.375 + r ∧ 0 ≤ r ∧ r < 0.001 :=
sorry

end NUMINAMATH_CALUDE_first_three_decimal_digits_l3816_381649


namespace NUMINAMATH_CALUDE_ordering_abc_l3816_381698

theorem ordering_abc (a b c : ℝ) : 
  a = 7/9 → b = 0.7 * Real.exp 0.1 → c = Real.cos (2/3) → c > a ∧ a > b :=
sorry

end NUMINAMATH_CALUDE_ordering_abc_l3816_381698


namespace NUMINAMATH_CALUDE_group_size_proof_l3816_381672

theorem group_size_proof (total_rupees : ℚ) (paise_per_rupee : ℕ) : 
  total_rupees = 20.25 →
  paise_per_rupee = 100 →
  ∃ n : ℕ, n * n = total_rupees * paise_per_rupee ∧ n = 45 :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l3816_381672


namespace NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l3816_381688

theorem students_neither_football_nor_cricket 
  (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ) 
  (h1 : total = 410)
  (h2 : football = 325)
  (h3 : cricket = 175)
  (h4 : both = 140) :
  total - (football + cricket - both) = 50 := by
sorry

end NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l3816_381688


namespace NUMINAMATH_CALUDE_ratio_of_system_l3816_381682

theorem ratio_of_system (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_system_l3816_381682


namespace NUMINAMATH_CALUDE_kolya_speed_increase_l3816_381640

theorem kolya_speed_increase (N : ℕ) (x : ℕ) : 
  -- Total problems for each student
  N = (3 * x) / 2 →
  -- Kolya has solved 1/3 of what Seryozha has left
  x / 6 = (x / 2) / 3 →
  -- Seryozha has solved half of his problems
  x = N / 2 →
  -- The factor by which Kolya needs to increase his speed
  (((3 * x) / 2 - x / 6) / (x / 2)) / (x / 6 / x) = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_kolya_speed_increase_l3816_381640


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3816_381618

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 2 * x + y → a + b ≤ x + y ∧ a + b = 2 * Real.sqrt 2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3816_381618


namespace NUMINAMATH_CALUDE_remaining_balance_calculation_l3816_381680

def initial_balance : ℝ := 50
def coffee_expense : ℝ := 10
def tumbler_expense : ℝ := 30

theorem remaining_balance_calculation :
  initial_balance - (coffee_expense + tumbler_expense) = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_calculation_l3816_381680


namespace NUMINAMATH_CALUDE_digit_125_of_4_div_7_l3816_381626

/-- The decimal representation of 4/7 has a 6-digit repeating sequence -/
def repeating_sequence_length : ℕ := 6

/-- The 125th digit after the decimal point in 4/7 -/
def target_digit : ℕ := 125

/-- The function that returns the nth digit in the decimal expansion of 4/7 -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_125_of_4_div_7 : nth_digit target_digit = 2 := by sorry

end NUMINAMATH_CALUDE_digit_125_of_4_div_7_l3816_381626


namespace NUMINAMATH_CALUDE_rectangle_area_l3816_381674

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  side_pos : side > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side ^ 2

/-- Represents the rectangle ABCD containing three squares -/
structure Rectangle where
  small_square1 : Square
  small_square2 : Square
  large_square : Square
  non_overlapping : True  -- Assume squares are non-overlapping
  small_square1_area : small_square1.area = 4
  small_squares_equal : small_square1.side = small_square2.side
  large_square_side : large_square.side = 3 * small_square1.side

/-- The total area of the rectangle -/
def Rectangle.total_area (r : Rectangle) : ℝ :=
  r.small_square1.area + r.small_square2.area + r.large_square.area

/-- The main theorem: The area of rectangle ABCD is 44 square inches -/
theorem rectangle_area (r : Rectangle) : r.total_area = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3816_381674


namespace NUMINAMATH_CALUDE_orange_put_back_l3816_381632

theorem orange_put_back (apple_price orange_price : ℚ)
  (total_fruit : ℕ) (initial_avg_price final_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruit = 10)
  (h4 : initial_avg_price = 54/100)
  (h5 : final_avg_price = 45/100) :
  ∃ (oranges_to_put_back : ℕ),
    oranges_to_put_back = 6 ∧
    ∃ (initial_apples initial_oranges : ℕ),
      initial_apples + initial_oranges = total_fruit ∧
      (initial_apples * apple_price + initial_oranges * orange_price) / total_fruit = initial_avg_price ∧
      ∃ (final_oranges : ℕ),
        final_oranges = initial_oranges - oranges_to_put_back ∧
        (initial_apples * apple_price + final_oranges * orange_price) / (initial_apples + final_oranges) = final_avg_price :=
by
  sorry

end NUMINAMATH_CALUDE_orange_put_back_l3816_381632


namespace NUMINAMATH_CALUDE_square_perimeter_difference_l3816_381630

theorem square_perimeter_difference (a b : ℝ) 
  (h1 : a^2 + b^2 = 85)
  (h2 : a^2 - b^2 = 45) :
  4*a - 4*b = 4*(Real.sqrt 65 - 2*Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_difference_l3816_381630


namespace NUMINAMATH_CALUDE_bandage_overlap_l3816_381658

theorem bandage_overlap (n : ℕ) (l : ℝ) (total : ℝ) (h1 : n = 20) (h2 : l = 15.25) (h3 : total = 248) :
  (n * l - total) / (n - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_bandage_overlap_l3816_381658


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3816_381624

/-- The radius of a circle tangent to four semicircles in a square -/
theorem tangent_circle_radius (s : ℝ) (h : s = 4) : 
  let r := 2 * (Real.sqrt 2 - 1)
  let semicircle_radius := s / 2
  let square_diagonal := s * Real.sqrt 2
  r = square_diagonal / 2 - semicircle_radius :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3816_381624


namespace NUMINAMATH_CALUDE_mountain_climbs_l3816_381600

/-- Proves that Boris needs to climb his mountain 4 times to match Hugo's total climb -/
theorem mountain_climbs (hugo_elevation : ℕ) (boris_difference : ℕ) (hugo_climbs : ℕ) : 
  hugo_elevation = 10000 →
  boris_difference = 2500 →
  hugo_climbs = 3 →
  (hugo_elevation * hugo_climbs) / (hugo_elevation - boris_difference) = 4 := by
sorry

end NUMINAMATH_CALUDE_mountain_climbs_l3816_381600


namespace NUMINAMATH_CALUDE_min_tan_product_l3816_381694

theorem min_tan_product (α β γ : Real) (h_acute : α ∈ Set.Ioo 0 (π/2) ∧ β ∈ Set.Ioo 0 (π/2) ∧ γ ∈ Set.Ioo 0 (π/2)) 
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  ∃ (min : Real), 
    (∀ (α' β' γ' : Real), 
      α' ∈ Set.Ioo 0 (π/2) → β' ∈ Set.Ioo 0 (π/2) → γ' ∈ Set.Ioo 0 (π/2) →
      Real.cos α' ^ 2 + Real.cos β' ^ 2 + Real.cos γ' ^ 2 = 1 →
      Real.tan α' * Real.tan β' * Real.tan γ' ≥ min) ∧
    Real.tan α * Real.tan β * Real.tan γ = min ∧
    min = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_tan_product_l3816_381694


namespace NUMINAMATH_CALUDE_exactly_four_intersections_l3816_381663

/-- The number of intersection points between y=Ax^2 and x^2 + 2y^2 = A+3 -/
def intersection_count (A : ℝ) : ℕ :=
  sorry

/-- Theorem stating that the graphs intersect at exactly 4 points -/
theorem exactly_four_intersections (A : ℝ) (h : A > 0) : intersection_count A = 4 :=
  sorry

end NUMINAMATH_CALUDE_exactly_four_intersections_l3816_381663


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_l3816_381670

/-- The minimum distance sum for a point on the parabola x = (1/4)y^2 -/
theorem min_distance_sum_parabola :
  let parabola := {P : ℝ × ℝ | P.1 = (1/4) * P.2^2}
  let dist_to_A (P : ℝ × ℝ) := Real.sqrt ((P.1 - 0)^2 + (P.2 - 1)^2)
  let dist_to_y_axis (P : ℝ × ℝ) := |P.1|
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 - 1 ∧
    ∀ P ∈ parabola, dist_to_A P + dist_to_y_axis P ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_parabola_l3816_381670


namespace NUMINAMATH_CALUDE_triangle_side_value_l3816_381673

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  f A = 2 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  a = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3816_381673


namespace NUMINAMATH_CALUDE_unique_prime_factorization_and_sum_l3816_381691

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_factorization_and_sum (q r s p1 p2 p3 : ℕ) : 
  (q * r * s = 2206 ∧ 
   is_prime q ∧ is_prime r ∧ is_prime s ∧ 
   q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (p1 + p2 + p3 = q + r + s + 1 ∧ 
   is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ 
   p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) →
  ((q = 2 ∧ r = 3 ∧ s = 367) ∨ (q = 2 ∧ r = 367 ∧ s = 3) ∨ (q = 3 ∧ r = 2 ∧ s = 367) ∨ 
   (q = 3 ∧ r = 367 ∧ s = 2) ∧ (q = 367 ∧ r = 2 ∧ s = 3) ∨ (q = 367 ∧ r = 3 ∧ s = 2)) ∧
  (p1 = 2 ∧ p2 = 3 ∧ p3 = 367) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_factorization_and_sum_l3816_381691


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l3816_381615

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := (List.range (bounces + 1)).map (λ i => initialHeight * bounceRatio^i)
  let ascendDistances := (List.range bounces).map (λ i => initialHeight * bounceRatio^(i+1))
  (descendDistances.sum + ascendDistances.sum)

/-- Theorem: A ball dropped from 25 meters, bouncing 2/3 of its previous height each time,
    and caught after the 4th bounce, travels approximately 88 meters. -/
theorem bouncing_ball_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |totalDistance 25 (2/3) 4 - 88| < ε :=
sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l3816_381615


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3816_381643

theorem solution_satisfies_system :
  let x₁ : ℚ := 1
  let x₂ : ℚ := -1
  let x₃ : ℚ := 1
  let x₄ : ℚ := -1
  let x₅ : ℚ := 1
  (x₁ + 2*x₂ + 2*x₃ + 2*x₄ + 2*x₅ = 1) ∧
  (x₁ + 3*x₂ + 4*x₃ + 4*x₄ + 4*x₅ = 2) ∧
  (x₁ + 3*x₂ + 5*x₃ + 6*x₄ + 6*x₅ = 3) ∧
  (x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 8*x₅ = 4) ∧
  (x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3816_381643


namespace NUMINAMATH_CALUDE_heather_biking_speed_l3816_381610

def total_distance : ℝ := 320
def num_days : ℝ := 8.0

theorem heather_biking_speed : total_distance / num_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_heather_biking_speed_l3816_381610


namespace NUMINAMATH_CALUDE_correct_line_representation_incorrect_representation_A_incorrect_representation_B_incorrect_representation_C_l3816_381636

-- Define a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Theorem for the correct representation (Option D)
theorem correct_line_representation (n : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∃ (l : Line), l.slope = k ∧ pointOnLine l ⟨n, 0⟩ ∧
    ∀ (x y : ℝ), pointOnLine l ⟨x, y⟩ ↔ x = k * y + n :=
sorry

-- Theorem for the incorrectness of Option A
theorem incorrect_representation_A (x₀ y₀ : ℝ) :
  ¬ (∀ (l : Line), ∃ (k : ℝ), ∀ (x y : ℝ),
    pointOnLine l ⟨x, y⟩ ↔ y - y₀ = k * (x - x₀)) :=
sorry

-- Theorem for the incorrectness of Option B
theorem incorrect_representation_B (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂ ∨ y₁ ≠ y₂) :
  ¬ (∀ (l : Line), ∀ (x y : ℝ),
    pointOnLine l ⟨x, y⟩ ↔ (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁)) :=
sorry

-- Theorem for the incorrectness of Option C
theorem incorrect_representation_C :
  ¬ (∀ (l : Line) (a b : ℝ), (¬ pointOnLine l ⟨0, 0⟩) →
    (∀ (x y : ℝ), pointOnLine l ⟨x, y⟩ ↔ x / a + y / b = 1)) :=
sorry

end NUMINAMATH_CALUDE_correct_line_representation_incorrect_representation_A_incorrect_representation_B_incorrect_representation_C_l3816_381636


namespace NUMINAMATH_CALUDE_sum_a_d_equals_two_l3816_381676

theorem sum_a_d_equals_two 
  (a b c d : ℤ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_two_l3816_381676


namespace NUMINAMATH_CALUDE_tank_fill_time_l3816_381614

/-- Represents the time (in hours) it takes for a pipe to empty or fill the tank when working alone -/
structure PipeTime where
  A : ℝ  -- Time for pipe A to empty the tank
  B : ℝ  -- Time for pipe B to empty the tank
  C : ℝ  -- Time for pipe C to fill the tank

/-- Conditions for the tank filling problem -/
def TankConditions (t : PipeTime) : Prop :=
  (1 / t.C - 1 / t.A) * 2 = 1 ∧
  (1 / t.C - 1 / t.B) * 4 = 1 ∧
  1 / t.C * 5 - (1 / t.A + 1 / t.B) * 8 = 0

/-- The main theorem stating the time to fill the tank using only pipe C -/
theorem tank_fill_time (t : PipeTime) (h : TankConditions t) : t.C = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3816_381614


namespace NUMINAMATH_CALUDE_meaningful_expression_l3816_381627

/-- The expression sqrt(a+1)/(a-2) is meaningful iff a ≥ -1 and a ≠ 2 -/
theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 1) ∧ (a ≠ 2) ↔ a ≥ -1 ∧ a ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3816_381627


namespace NUMINAMATH_CALUDE_difference_divisible_by_19_l3816_381645

theorem difference_divisible_by_19 (n : ℕ) : 26^n ≡ 7^n [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_19_l3816_381645


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3816_381675

theorem tangent_line_to_parabola (x y : ℝ) :
  (y = x^2) →  -- The curve equation
  (∃ (m b : ℝ), (y = m*x + b) ∧  -- The tangent line equation
                (-1 = m*1 + b) ∧  -- The line passes through (1, -1)
                (∃ (a : ℝ), y = (2*a)*x - a^2 - a)) →  -- Tangent line touches the curve
  ((y = (2 + 2*Real.sqrt 2)*x - (3 + 2*Real.sqrt 2)) ∨
   (y = (2 - 2*Real.sqrt 2)*x - (3 - 2*Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3816_381675


namespace NUMINAMATH_CALUDE_even_function_inequality_l3816_381601

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_even : is_even_function f)
  (h_increasing : increasing_on_negative f)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : 0 < x₂)
  (h_sum_pos : 0 < x₁ + x₂) :
  f (-x₁) > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3816_381601


namespace NUMINAMATH_CALUDE_mortgage_payment_l3816_381671

theorem mortgage_payment (total : ℝ) (months : ℕ) (ratio : ℝ) (first_payment : ℝ) :
  total = 109300 ∧ 
  months = 7 ∧ 
  ratio = 3 ∧ 
  total = first_payment * (1 - ratio^months) / (1 - ratio) →
  first_payment = 100 := by
sorry

end NUMINAMATH_CALUDE_mortgage_payment_l3816_381671


namespace NUMINAMATH_CALUDE_hyeji_total_water_intake_l3816_381656

-- Define the conversion rate from liters to milliliters
def liters_to_ml (liters : ℝ) : ℝ := liters * 1000

-- Define Hyeji's daily water intake in liters
def daily_intake : ℝ := 2

-- Define the additional amount Hyeji drank in milliliters
def additional_intake : ℝ := 460

-- Theorem to prove
theorem hyeji_total_water_intake :
  liters_to_ml daily_intake + additional_intake = 2460 := by
  sorry

end NUMINAMATH_CALUDE_hyeji_total_water_intake_l3816_381656


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3816_381677

/-- Given a line L1: 2x + y - 3 = 0 and a point P(0, 1), 
    prove that the line L2: 2x + y - 1 = 0 passes through P and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  let L1 := {(x, y) | 2 * x + y - 3 = 0}
  let P := (0, 1)
  let L2 := {(x, y) | 2 * x + y - 1 = 0}
  (P ∈ L2) ∧ (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 → (x2, y2) ∈ L1 → 
    (x2 - x1) * 1 = (y2 - y1) * 2 ↔ 
    (x2 - x1) * 1 = (y2 - y1) * 2) := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l3816_381677


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l3816_381650

theorem tan_alpha_minus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α - Real.pi/4) = -1/7 ∨ Real.tan (α - Real.pi/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l3816_381650
