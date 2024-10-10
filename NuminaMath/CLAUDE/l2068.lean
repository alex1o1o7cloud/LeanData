import Mathlib

namespace flash_interval_l2068_206840

/-- Proves that the time between each flash is 6 seconds, given that a light flashes 450 times in ¾ of an hour. -/
theorem flash_interval (flashes : ℕ) (time : ℚ) (h1 : flashes = 450) (h2 : time = 3/4) :
  (time * 3600) / flashes = 6 := by
  sorry

end flash_interval_l2068_206840


namespace friend_lunch_cost_l2068_206812

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 11 → difference = 3 → friend_cost = total / 2 + difference / 2 → friend_cost = 7 := by
  sorry

end friend_lunch_cost_l2068_206812


namespace min_value_theorem_l2068_206891

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  1/a + 2/b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 1 ∧ 1/a₀ + 2/b₀ = 9 :=
sorry

end min_value_theorem_l2068_206891


namespace initial_shirts_l2068_206854

theorem initial_shirts (S : ℕ) : S + 4 = 16 → S = 12 := by
  sorry

end initial_shirts_l2068_206854


namespace log_expression_equality_l2068_206829

theorem log_expression_equality : 2 * (Real.log 256 / Real.log 4) - (Real.log (1/16) / Real.log 4) = 10 := by
  sorry

end log_expression_equality_l2068_206829


namespace dormitory_students_l2068_206873

theorem dormitory_students (F S : ℝ) (h1 : F + S = 1) 
  (h2 : 4/5 * F = F - F/5) 
  (h3 : S - 4 * (F/5) = 4/5 * F) 
  (h4 : S - (S - 4 * (F/5)) = 0.2) : 
  S = 2/3 := by sorry

end dormitory_students_l2068_206873


namespace finley_age_l2068_206825

/-- Represents the ages of the individuals in the problem -/
structure Ages where
  jill : ℕ
  roger : ℕ
  alex : ℕ
  finley : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.roger = 2 * ages.jill + 5 ∧
  ages.roger + 15 - (ages.jill + 15) = ages.finley - 30 ∧
  ages.jill = 20 ∧
  ages.roger = (ages.jill + ages.alex) / 2 ∧
  ages.alex = 3 * (ages.finley + 10) - 5

/-- The theorem stating Finley's age -/
theorem finley_age (ages : Ages) (h : problem_conditions ages) : ages.finley = 15 := by
  sorry

#check finley_age

end finley_age_l2068_206825


namespace x_squared_equals_three_l2068_206875

theorem x_squared_equals_three (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = x / 2) : x^2 = 3 := by
  sorry

end x_squared_equals_three_l2068_206875


namespace right_handed_players_count_l2068_206879

/-- Calculates the number of right-handed players on a cricket team given specific conditions -/
theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) (left_handed_thrower_percent : ℚ)
  (right_handed_thrower_avg : ℕ) (left_handed_thrower_avg : ℕ) (total_runs : ℕ)
  (left_handed_non_thrower_runs : ℕ) :
  total_players = 120 →
  throwers = 55 →
  left_handed_thrower_percent = 1/5 →
  right_handed_thrower_avg = 25 →
  left_handed_thrower_avg = 30 →
  total_runs = 3620 →
  left_handed_non_thrower_runs = 720 →
  ∃ (right_handed_players : ℕ), right_handed_players = 164 :=
by sorry

end right_handed_players_count_l2068_206879


namespace seashell_collection_l2068_206831

/-- The number of seashells collected by Stefan, Vail, and Aiguo -/
theorem seashell_collection (stefan vail aiguo : ℕ) 
  (h1 : stefan = vail + 16)
  (h2 : vail = aiguo - 5)
  (h3 : aiguo = 20) :
  stefan + vail + aiguo = 66 := by
  sorry

end seashell_collection_l2068_206831


namespace min_value_quadratic_l2068_206821

theorem min_value_quadratic (x y : ℝ) : 2 * x^2 + 3 * y^2 - 8 * x + 6 * y + 25 ≥ 14 := by
  sorry

end min_value_quadratic_l2068_206821


namespace volumes_equal_l2068_206890

/-- The region bounded by x² = 4y, x² = -4y, x = 4, x = -4 -/
def Region1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ (x ≤ 4 ∧ x ≥ -4)

/-- The region defined by x²y² ≤ 16, x² + (y-2)² ≥ 4, x² + (y+2)² ≥ 4 -/
def Region2 (x y : ℝ) : Prop :=
  x^2 * y^2 ≤ 16 ∧ x^2 + (y-2)^2 ≥ 4 ∧ x^2 + (y+2)^2 ≥ 4

/-- The volume of the solid obtained by rotating Region1 around the y-axis -/
noncomputable def V1 : ℝ := sorry

/-- The volume of the solid obtained by rotating Region2 around the y-axis -/
noncomputable def V2 : ℝ := sorry

/-- The volumes of the two solids are equal -/
theorem volumes_equal : V1 = V2 := by sorry

end volumes_equal_l2068_206890


namespace sum_of_120_mod_980_l2068_206843

-- Define the sum of first n natural numbers
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- State the theorem
theorem sum_of_120_mod_980 : sum_of_first_n 120 % 980 = 320 := by
  sorry

end sum_of_120_mod_980_l2068_206843


namespace purification_cost_is_one_l2068_206896

/-- The cost to purify a gallon of fresh water -/
def purification_cost (water_per_person : ℚ) (family_size : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (water_per_person * family_size)

/-- Theorem: The cost to purify a gallon of fresh water is $1 -/
theorem purification_cost_is_one :
  purification_cost (1/2) 6 3 = 1 := by
  sorry

end purification_cost_is_one_l2068_206896


namespace final_brownie_count_l2068_206862

def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def additional_brownies : ℕ := 24

theorem final_brownie_count :
  initial_brownies - father_ate - mooney_ate + additional_brownies = 36 := by
  sorry

end final_brownie_count_l2068_206862


namespace max_overlap_theorem_l2068_206861

/-- The area of the equilateral triangle -/
def triangle_area : ℝ := 2019

/-- The maximum overlap area when folding the triangle -/
def max_overlap_area : ℝ := 673

/-- The fold line is parallel to one of the triangle's sides -/
axiom fold_parallel : True

theorem max_overlap_theorem :
  ∀ (overlap_area : ℝ),
  overlap_area ≤ max_overlap_area :=
sorry

end max_overlap_theorem_l2068_206861


namespace no_fourfold_digit_move_l2068_206846

theorem no_fourfold_digit_move :
  ∀ (N : ℕ), ∀ (a : ℕ), ∀ (n : ℕ), ∀ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) →
    (x < 10^n) →
    (N = a * 10^n + x) →
    (10 * x + a ≠ 4 * N) :=
by sorry

end no_fourfold_digit_move_l2068_206846


namespace total_surfers_l2068_206828

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 50

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 30

/-- The number of surfers on Venice beach -/
def venice_surfers : ℕ := 20

/-- The ratio of surfers on Malibu beach -/
def malibu_ratio : ℕ := 5

/-- The ratio of surfers on Santa Monica beach -/
def santa_monica_ratio : ℕ := 3

/-- The ratio of surfers on Venice beach -/
def venice_ratio : ℕ := 2

theorem total_surfers : 
  malibu_surfers + santa_monica_surfers + venice_surfers = 100 ∧
  malibu_surfers * santa_monica_ratio = santa_monica_surfers * malibu_ratio ∧
  venice_surfers * santa_monica_ratio = santa_monica_surfers * venice_ratio :=
by sorry

end total_surfers_l2068_206828


namespace urn_problem_l2068_206835

/-- Given two urns with different compositions of colored balls, 
    prove that the number of blue balls in the second urn is 15 --/
theorem urn_problem (N : ℕ) : 
  (5 : ℚ) / 10 * (10 : ℚ) / (10 + N) +  -- Probability of both balls being green
  (5 : ℚ) / 10 * (N : ℚ) / (10 + N) =   -- Probability of both balls being blue
  (52 : ℚ) / 100 →                      -- Total probability of same color
  N = 15 := by
sorry

end urn_problem_l2068_206835


namespace exists_marked_points_with_distance_l2068_206839

/-- Represents a marked point on the segment -/
structure MarkedPoint where
  position : ℚ
  deriving Repr

/-- The process of marking points on a segment of length 3^n -/
def markPoints (n : ℕ) : List MarkedPoint :=
  sorry

/-- Theorem stating the existence of two marked points with distance k -/
theorem exists_marked_points_with_distance (n : ℕ) (k : ℕ) 
  (h : 1 ≤ k ∧ k ≤ 3^n) : 
  ∃ (p q : MarkedPoint), p ∈ markPoints n ∧ q ∈ markPoints n ∧ 
    |p.position - q.position| = k :=
  sorry

end exists_marked_points_with_distance_l2068_206839


namespace polynomial_roots_k_values_l2068_206885

/-- The set of all distinct possible values of k for the polynomial x^2 - kx + 36 
    with only positive integer roots -/
def possible_k_values : Set ℤ := {12, 13, 15, 20, 37}

/-- A polynomial of the form x^2 - kx + 36 -/
def polynomial (k : ℤ) (x : ℝ) : ℝ := x^2 - k*x + 36

theorem polynomial_roots_k_values :
  ∀ k : ℤ, (∃ r₁ r₂ : ℤ, r₁ > 0 ∧ r₂ > 0 ∧ 
    ∀ x : ℝ, polynomial k x = 0 ↔ x = r₁ ∨ x = r₂) ↔ 
  k ∈ possible_k_values :=
sorry

end polynomial_roots_k_values_l2068_206885


namespace largest_divisor_of_polynomial_l2068_206811

theorem largest_divisor_of_polynomial (n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (m : ℤ), (m^4 - 5*m^2 + 6) % k = 0) ∧ 
  (∀ (l : ℕ), l > k → ∃ (m : ℤ), (m^4 - 5*m^2 + 6) % l ≠ 0) → k = 1 := by
  sorry

end largest_divisor_of_polynomial_l2068_206811


namespace find_x_l2068_206817

-- Define the # operation
def sharp (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

-- Theorem statement
theorem find_x : 
  ∃ (x : ℤ), 
    (∀ (p : ℤ), sharp (sharp (sharp p x) x) x = -4) ∧ 
    (sharp (sharp (sharp 18 x) x) x = -4) → 
    x = -21 := by
  sorry

end find_x_l2068_206817


namespace negation_of_existence_negation_of_quadratic_inequality_l2068_206880

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ ∀ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - x ≤ 0) ↔ (∀ x > 0, x^2 - x > 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l2068_206880


namespace max_dot_product_OM_OC_l2068_206882

/-- Given points in a 2D Cartesian coordinate system -/
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

/-- M is a moving point with x-coordinate between -2 and 2 -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | -2 ≤ p.1 ∧ p.1 ≤ 2}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: The maximum value of OM · OC is 4 -/
theorem max_dot_product_OM_OC :
  ∃ (m : ℝ × ℝ), m ∈ M ∧ 
    ∀ (n : ℝ × ℝ), n ∈ M → 
      dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) ≥ 
      dot_product (n.1 - O.1, n.2 - O.2) (C.1 - O.1, C.2 - O.2) ∧
    dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) = 4 :=
sorry

end max_dot_product_OM_OC_l2068_206882


namespace total_rats_l2068_206870

/-- The number of rats each person has -/
structure RatCounts where
  kenia : ℕ
  hunter : ℕ
  elodie : ℕ

/-- The conditions of the rat problem -/
def rat_problem (r : RatCounts) : Prop :=
  r.kenia = 3 * (r.hunter + r.elodie) ∧
  r.elodie = 30 ∧
  r.elodie = r.hunter + 10

theorem total_rats (r : RatCounts) (h : rat_problem r) : 
  r.kenia + r.hunter + r.elodie = 200 := by
  sorry

end total_rats_l2068_206870


namespace triangle_property_triangle_area_l2068_206849

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_property (t : Triangle) 
  (h : t.c - t.a * Real.cos t.B = (Real.sqrt 2 / 2) * t.b) : 
  t.A = π / 4 := by sorry

theorem triangle_area (t : Triangle) 
  (h1 : t.c - t.a * Real.cos t.B = (Real.sqrt 2 / 2) * t.b)
  (h2 : t.c = 4 * Real.sqrt 2)
  (h3 : Real.cos t.B = 7 * Real.sqrt 2 / 10) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = 2 := by sorry

end triangle_property_triangle_area_l2068_206849


namespace cube_surface_covering_l2068_206837

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  side_length : ℝ
  height : ℝ
  area : ℝ

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  side_length : ℝ
  surface_area : ℝ

/-- A covering is a collection of shapes that cover a surface. -/
structure Covering where
  shapes : List Rhombus
  total_area : ℝ

/-- Theorem: It is possible to cover the surface of a cube with fewer than six rhombuses. -/
theorem cube_surface_covering (c : Cube) : 
  ∃ (cov : Covering), cov.shapes.length < 6 ∧ cov.total_area = c.surface_area := by
  sorry


end cube_surface_covering_l2068_206837


namespace binomial_1300_2_l2068_206845

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by sorry

end binomial_1300_2_l2068_206845


namespace complex_equality_l2068_206889

theorem complex_equality (z₁ z₂ : ℂ) (h : Complex.abs (z₁ + 2 * z₂) = Complex.abs (2 * z₁ + z₂)) :
  ∀ a : ℝ, Complex.abs (z₁ + a * z₂) = Complex.abs (a * z₁ + z₂) := by
  sorry

end complex_equality_l2068_206889


namespace negative_sqrt_two_squared_equals_two_l2068_206869

theorem negative_sqrt_two_squared_equals_two :
  (-Real.sqrt 2)^2 = 2 := by
  sorry

end negative_sqrt_two_squared_equals_two_l2068_206869


namespace tan_alpha_plus_pi_fourth_l2068_206899

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 2) : 
  Real.tan (α + π / 4) = -2 / 3 := by
  sorry

end tan_alpha_plus_pi_fourth_l2068_206899


namespace unique_solution_3n_plus_1_equals_a_squared_l2068_206857

theorem unique_solution_3n_plus_1_equals_a_squared :
  ∀ a n : ℕ+, 3^(n : ℕ) + 1 = (a : ℕ)^2 → a = 2 ∧ n = 1 := by
  sorry

end unique_solution_3n_plus_1_equals_a_squared_l2068_206857


namespace cosA_sinB_value_l2068_206809

theorem cosA_sinB_value (A B : Real) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : (4 + Real.tan A ^ 2) * (5 + Real.tan B ^ 2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = 1 / Real.sqrt 6 := by
sorry

end cosA_sinB_value_l2068_206809


namespace candy_mixture_theorem_l2068_206872

def candy_mixture (initial_blue initial_red added_blue added_red final_blue : ℚ) : Prop :=
  ∃ (x y : ℚ),
    x > 0 ∧ y > 0 ∧
    initial_blue = 1/10 ∧
    initial_red = 1/4 ∧
    added_blue = 1/4 ∧
    added_red = 3/4 ∧
    (initial_blue * x + added_blue * y) / (x + y) = final_blue

theorem candy_mixture_theorem :
  ∀ initial_blue initial_red added_blue added_red final_blue,
    candy_mixture initial_blue initial_red added_blue added_red final_blue →
    final_blue = 4/25 →
    ∃ final_red : ℚ, final_red = 9/20 :=
by sorry

end candy_mixture_theorem_l2068_206872


namespace triangle_perimeter_l2068_206860

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  c^2 = a * Real.cos B + b * Real.cos A →
  a = 3 →
  b = 3 →
  a + b + c = 7 :=
by
  sorry

end triangle_perimeter_l2068_206860


namespace basketball_team_selection_l2068_206852

def team_size : ℕ := 16
def lineup_size : ℕ := 5
def num_twins : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose (team_size - num_twins) lineup_size) +
  (num_twins * Nat.choose (team_size - num_twins) (lineup_size - 1)) +
  (Nat.choose (team_size - num_twins) (lineup_size - num_twins)) = 4368 := by
  sorry

end basketball_team_selection_l2068_206852


namespace interest_first_year_l2068_206804

def initial_deposit : ℝ := 5000
def balance_after_first_year : ℝ := 5500
def second_year_increase : ℝ := 0.1
def total_increase : ℝ := 0.21

theorem interest_first_year :
  balance_after_first_year - initial_deposit = 500 :=
sorry

end interest_first_year_l2068_206804


namespace cube_volume_from_space_diagonal_l2068_206895

/-- The volume of a cube with space diagonal 3√3 is 27 -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 3 * Real.sqrt 3 → s^3 = 27 := by
  sorry

end cube_volume_from_space_diagonal_l2068_206895


namespace unique_four_digit_cube_divisible_by_16_and_9_l2068_206868

theorem unique_four_digit_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∃ m : ℕ, n = m^3) ∧ 
  n % 16 = 0 ∧ n % 9 = 0 :=
by sorry

end unique_four_digit_cube_divisible_by_16_and_9_l2068_206868


namespace bottles_per_case_l2068_206800

/-- Given a company that produces bottles of water and uses cases to hold them,
    this theorem proves the number of bottles per case. -/
theorem bottles_per_case
  (total_bottles : ℕ)
  (total_cases : ℕ)
  (h1 : total_bottles = 60000)
  (h2 : total_cases = 12000)
  : total_bottles / total_cases = 5 := by
  sorry

end bottles_per_case_l2068_206800


namespace reciprocal_and_opposite_of_negative_four_l2068_206894

theorem reciprocal_and_opposite_of_negative_four :
  (1 / (-4 : ℝ) = -1/4) ∧ (-((-4) : ℝ) = 4) := by sorry

end reciprocal_and_opposite_of_negative_four_l2068_206894


namespace missing_score_proof_l2068_206805

theorem missing_score_proof (known_scores : List ℝ) (mean : ℝ) : 
  known_scores = [81, 73, 86, 73] →
  mean = 79.2 →
  ∃ (missing_score : ℝ), 
    (List.sum known_scores + missing_score) / 5 = mean ∧
    missing_score = 83 := by
  sorry

end missing_score_proof_l2068_206805


namespace andrea_pony_pasture_cost_l2068_206892

/-- Calculates the monthly pasture cost for Andrea's pony --/
theorem andrea_pony_pasture_cost :
  let daily_food_cost : ℕ := 10
  let lesson_cost : ℕ := 60
  let lessons_per_week : ℕ := 2
  let total_annual_expense : ℕ := 15890
  let days_per_year : ℕ := 365
  let weeks_per_year : ℕ := 52

  let annual_food_cost := daily_food_cost * days_per_year
  let annual_lesson_cost := lesson_cost * lessons_per_week * weeks_per_year
  let annual_pasture_cost := total_annual_expense - (annual_food_cost + annual_lesson_cost)
  let monthly_pasture_cost := annual_pasture_cost / 12

  monthly_pasture_cost = 500 := by sorry

end andrea_pony_pasture_cost_l2068_206892


namespace coal_burn_duration_l2068_206832

/-- Given a factory with 300 tons of coal, this theorem establishes the relationship
    between the number of days the coal can burn and the average daily consumption. -/
theorem coal_burn_duration (x : ℝ) (y : ℝ) (h : x > 0) :
  y = 300 / x ↔ y * x = 300 :=
by sorry

end coal_burn_duration_l2068_206832


namespace retail_price_maximizes_profit_l2068_206883

/-- The profit function for a shopping mall selling items -/
def profit_function (p : ℝ) : ℝ := (8300 - 170*p - p^2)*(p - 20)

/-- The derivative of the profit function -/
def profit_derivative (p : ℝ) : ℝ := -3*p^2 - 300*p + 11700

theorem retail_price_maximizes_profit :
  ∃ (p : ℝ), p > 20 ∧ 
  ∀ (q : ℝ), q > 20 → profit_function p ≥ profit_function q ∧
  p = 30 := by
  sorry

#check retail_price_maximizes_profit

end retail_price_maximizes_profit_l2068_206883


namespace canoe_upstream_speed_l2068_206886

/-- The speed of a canoe rowing upstream, given its downstream speed and the stream speed -/
theorem canoe_upstream_speed (downstream_speed stream_speed : ℝ) : 
  downstream_speed = 10 → stream_speed = 2 → 
  downstream_speed - 2 * stream_speed = 6 := by sorry

end canoe_upstream_speed_l2068_206886


namespace map_scale_l2068_206827

/-- Given a map where 12 cm represents 90 km, prove that 20 cm represents 150 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm / 12 = real_km / 90) :
  (20 * real_km) / map_cm = 150 := by
  sorry

end map_scale_l2068_206827


namespace geometric_sequence_problem_l2068_206877

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  (a 20) ^ 2 - 10 * (a 20) + 16 = 0 →                   -- a_20 is a root
  (a 60) ^ 2 - 10 * (a 60) + 16 = 0 →                   -- a_60 is a root
  (a 30 * a 40 * a 50) / 2 = 32 := by
  sorry

end geometric_sequence_problem_l2068_206877


namespace simplify_expression_l2068_206819

theorem simplify_expression (x : ℝ) (h : x = Real.tan (60 * π / 180)) :
  (x + 1 - 8 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - x)) * (3 - x) = -3 - 3 * Real.sqrt 3 := by
  sorry

end simplify_expression_l2068_206819


namespace true_compound_propositions_l2068_206893

-- Define propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom p₁_true : p₁
axiom p₂_false : ¬p₂
axiom p₃_false : ¬p₃
axiom p₄_true : p₄

-- Theorem to prove
theorem true_compound_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
by sorry

end true_compound_propositions_l2068_206893


namespace special_dog_food_weight_l2068_206876

/-- The weight of each bag of special dog food for a puppy -/
theorem special_dog_food_weight :
  let first_period_days : ℕ := 60
  let total_days : ℕ := 365
  let first_period_consumption : ℕ := 2  -- ounces per day
  let second_period_consumption : ℕ := 4  -- ounces per day
  let ounces_per_pound : ℕ := 16
  let number_of_bags : ℕ := 17
  
  let total_consumption : ℕ := 
    first_period_days * first_period_consumption + 
    (total_days - first_period_days) * second_period_consumption
  
  let total_pounds : ℚ := total_consumption / ounces_per_pound
  let bag_weight : ℚ := total_pounds / number_of_bags
  
  ∃ (weight : ℚ), abs (weight - bag_weight) < 0.005 ∧ weight = 4.93 :=
by sorry

end special_dog_food_weight_l2068_206876


namespace function_properties_l2068_206888

def StrictlyDecreasing (f : ℝ → ℝ) :=
  ∀ x y, x < y → f x > f y

def StrictlyConvex (f : ℝ → ℝ) :=
  ∀ x y t, 0 < t → t < 1 → f (t * x + (1 - t) * y) < t * f x + (1 - t) * f y

theorem function_properties (f : ℝ → ℝ) (h1 : StrictlyDecreasing f) (h2 : StrictlyConvex f) :
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 1 →
    (x₂ * f x₁ > x₁ * f x₂) ∧
    ((f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by sorry

end function_properties_l2068_206888


namespace maze_max_candies_l2068_206820

/-- Represents a station in the maze --/
structure Station where
  candies : ℕ  -- Number of candies given at this station
  entries : ℕ  -- Number of times Jirka can enter this station

/-- The maze configuration --/
def Maze : List Station :=
  [⟨5, 3⟩, ⟨3, 2⟩, ⟨3, 2⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩]

/-- The maximum number of candies Jirka can collect --/
def maxCandies : ℕ := 30

theorem maze_max_candies :
  (Maze.map (fun s => s.candies * s.entries)).sum = maxCandies := by
  sorry


end maze_max_candies_l2068_206820


namespace grandma_age_l2068_206864

-- Define the number of grandchildren
def num_grandchildren : ℕ := 5

-- Define the average age of the entire group
def group_average_age : ℚ := 26

-- Define the average age of the grandchildren
def grandchildren_average_age : ℚ := 7

-- Define the age difference between grandpa and grandma
def age_difference : ℕ := 1

-- Theorem statement
theorem grandma_age :
  ∃ (grandpa_age grandma_age : ℕ),
    -- The average age of the group is 26
    (grandpa_age + grandma_age + num_grandchildren * grandchildren_average_age) / (2 + num_grandchildren : ℚ) = group_average_age ∧
    -- Grandma is one year younger than grandpa
    grandpa_age = grandma_age + age_difference ∧
    -- Grandma's age is 73
    grandma_age = 73 := by
  sorry

end grandma_age_l2068_206864


namespace max_distance_l2068_206855

theorem max_distance (x y z w v : ℝ) 
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  ∃ (x' y' z' w' v' : ℝ), 
    |x' - y'| = 1 ∧ 
    |y' - z'| = 2 ∧ 
    |z' - w'| = 3 ∧ 
    |w' - v'| = 5 ∧ 
    |x' - v'| = 11 ∧
    ∀ (a b c d e : ℝ), 
      |a - b| = 1 → 
      |b - c| = 2 → 
      |c - d| = 3 → 
      |d - e| = 5 → 
      |a - e| ≤ 11 :=
by
  sorry

end max_distance_l2068_206855


namespace ellipse_focal_length_implies_m_8_l2068_206847

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

-- Define the condition for major axis along y-axis
def major_axis_y (m : ℝ) : Prop :=
  m - 2 > 10 - m

-- Define the focal length
def focal_length (m : ℝ) : ℝ :=
  4

-- Theorem statement
theorem ellipse_focal_length_implies_m_8 :
  ∀ m : ℝ,
  (∀ x y : ℝ, ellipse_equation x y m) →
  major_axis_y m →
  focal_length m = 4 →
  m = 8 := by
  sorry

end ellipse_focal_length_implies_m_8_l2068_206847


namespace expression_value_l2068_206813

theorem expression_value : -2^4 + 3 * (-1)^6 - (-2)^3 = -5 := by
  sorry

end expression_value_l2068_206813


namespace percentage_of_240_l2068_206871

theorem percentage_of_240 : (3 / 8 : ℚ) / 100 * 240 = 0.9 := by
  sorry

end percentage_of_240_l2068_206871


namespace base4_21202_equals_610_l2068_206851

def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem base4_21202_equals_610 :
  base4ToBase10 [2, 0, 2, 1, 2] = 610 := by
  sorry

end base4_21202_equals_610_l2068_206851


namespace triangle_properties_l2068_206826

/-- Triangle ABC with given properties -/
structure Triangle where
  b : ℝ
  c : ℝ
  cosC : ℝ
  h_b : b = 2
  h_c : c = 3
  h_cosC : cosC = 1/3

/-- Theorems about the triangle -/
theorem triangle_properties (t : Triangle) :
  ∃ (a : ℝ) (area : ℝ) (cosBminusC : ℝ),
    -- Side length a
    a = 3 ∧
    -- Area of the triangle
    area = 2 * Real.sqrt 2 ∧
    -- Cosine of B minus C
    cosBminusC = 23/27 := by
  sorry

end triangle_properties_l2068_206826


namespace f_equals_g_l2068_206807

def N : Set ℕ := {n : ℕ | n > 0}

theorem f_equals_g
  (f g : ℕ → ℕ)
  (f_onto : ∀ y : ℕ, ∃ x : ℕ, f x = y)
  (g_one_one : ∀ x y : ℕ, g x = g y → x = y)
  (f_ge_g : ∀ n : ℕ, f n ≥ g n)
  : ∀ n : ℕ, f n = g n :=
sorry

end f_equals_g_l2068_206807


namespace cats_in_meow_and_paw_l2068_206887

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := meow_cats + paw_cats

/-- Theorem stating that the total number of cats in Cat Cafe Meow and Cat Cafe Paw is 40 -/
theorem cats_in_meow_and_paw : total_cats = 40 := by
  sorry

end cats_in_meow_and_paw_l2068_206887


namespace trig_identity_proof_l2068_206834

/-- Proves that sin 42° * cos 18° - cos 138° * cos 72° = √3/2 -/
theorem trig_identity_proof : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end trig_identity_proof_l2068_206834


namespace total_balloon_cost_l2068_206801

def fred_balloons : ℕ := 10
def fred_cost : ℚ := 1

def sam_balloons : ℕ := 46
def sam_cost : ℚ := 3/2

def dan_balloons : ℕ := 16
def dan_cost : ℚ := 3/4

theorem total_balloon_cost :
  (fred_balloons : ℚ) * fred_cost +
  (sam_balloons : ℚ) * sam_cost +
  (dan_balloons : ℚ) * dan_cost = 91 := by
  sorry

end total_balloon_cost_l2068_206801


namespace cone_base_radius_l2068_206830

theorem cone_base_radius (α : Real) (n : Nat) (r : Real) (h₁ : α = 30 * π / 180) (h₂ : n = 11) (h₃ : r = 3) :
  let R := r * (1 / Real.sin (π / n) - 1 / Real.tan (π / 4 + α / 2))
  R = r / Real.sin (π / n) - Real.sqrt 3 := by
  sorry

end cone_base_radius_l2068_206830


namespace not_prime_n4_plus_n2_plus_1_l2068_206818

theorem not_prime_n4_plus_n2_plus_1 (n : ℕ) (h : n ≥ 2) :
  ¬ Nat.Prime (n^4 + n^2 + 1) := by
  sorry

end not_prime_n4_plus_n2_plus_1_l2068_206818


namespace function_with_two_symmetries_is_periodic_l2068_206898

/-- A function with two lines of symmetry is periodic -/
theorem function_with_two_symmetries_is_periodic
  (f : ℝ → ℝ) (m n : ℝ) (hm : m ≠ n)
  (sym_m : ∀ x, f x = f (2 * m - x))
  (sym_n : ∀ x, f x = f (2 * n - x)) :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

end function_with_two_symmetries_is_periodic_l2068_206898


namespace fifth_term_of_arithmetic_sequence_l2068_206842

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem fifth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) :
  a 5 = 10 := by
sorry

end fifth_term_of_arithmetic_sequence_l2068_206842


namespace family_gathering_handshakes_l2068_206866

/-- The number of unique handshakes in a family gathering with twins and triplets -/
theorem family_gathering_handshakes :
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 3
  let twins_per_set : ℕ := 2
  let triplets_per_set : ℕ := 3
  let total_twins : ℕ := twin_sets * twins_per_set
  let total_triplets : ℕ := triplet_sets * triplets_per_set
  let twin_handshakes : ℕ := total_twins * (total_twins - twins_per_set)
  let triplet_handshakes : ℕ := total_triplets * (total_triplets - triplets_per_set)
  let twin_triplet_handshakes : ℕ := total_twins * total_triplets
  let total_handshakes : ℕ := twin_handshakes + triplet_handshakes + twin_triplet_handshakes
  327 = total_handshakes / 2 := by
  sorry

end family_gathering_handshakes_l2068_206866


namespace scientific_notation_of_passenger_trips_l2068_206865

theorem scientific_notation_of_passenger_trips :
  let trips : ℝ := 56.99 * 1000000
  trips = 5.699 * (10 ^ 7) := by sorry

end scientific_notation_of_passenger_trips_l2068_206865


namespace remainder_theorem_a_value_l2068_206841

/-- The polynomial function f(x) = x^6 - 8x^3 + 6 -/
def f (x : ℝ) : ℝ := x^6 - 8*x^3 + 6

/-- The remainder function R(x) = 7x - 8 -/
def R (x : ℝ) : ℝ := 7*x - 8

/-- Theorem stating that R(x) is the remainder when f(x) is divided by (x-1)(x-2) -/
theorem remainder_theorem :
  ∀ x : ℝ, ∃ q : ℝ → ℝ, f x = ((x - 1) * (x - 2)) * (q x) + R x :=
by
  sorry

/-- Corollary: The value of a in the remainder 7x - a is 8 -/
theorem a_value : ∃ a : ℝ, a = 8 ∧ 
  (∀ x : ℝ, ∃ q : ℝ → ℝ, f x = ((x - 1) * (x - 2)) * (q x) + (7*x - a)) :=
by
  sorry

end remainder_theorem_a_value_l2068_206841


namespace sandy_shopping_money_l2068_206848

theorem sandy_shopping_money (initial_amount : ℝ) : 
  (initial_amount * 0.7 = 210) → initial_amount = 300 := by
  sorry

end sandy_shopping_money_l2068_206848


namespace division_problem_l2068_206838

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 5/2) : 
  c / a = 2/15 := by sorry

end division_problem_l2068_206838


namespace quadratic_inequality_range_l2068_206844

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + 1 < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 1 :=
sorry

end quadratic_inequality_range_l2068_206844


namespace original_number_of_girls_l2068_206858

theorem original_number_of_girls (b g : ℚ) : 
  b > 0 ∧ g > 0 →  -- Initial numbers are positive
  3 * (g - 20) = b →  -- After 20 girls leave, ratio is 3 boys to 1 girl
  4 * (b - 60) = g - 20 →  -- After 60 boys leave, ratio is 1 boy to 4 girls
  g = 460 / 11 := by
sorry

end original_number_of_girls_l2068_206858


namespace intersection_of_M_and_N_l2068_206867

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end intersection_of_M_and_N_l2068_206867


namespace absolute_value_inequality_l2068_206808

theorem absolute_value_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end absolute_value_inequality_l2068_206808


namespace tiffany_lives_gained_l2068_206810

theorem tiffany_lives_gained (initial_lives lost_lives final_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : lost_lives = 14)
  (h3 : final_lives = 56) : 
  final_lives - (initial_lives - lost_lives) = 27 := by
  sorry

end tiffany_lives_gained_l2068_206810


namespace rational_sqrt_equation_l2068_206859

theorem rational_sqrt_equation (a b : ℚ) : 
  a - b * Real.sqrt 2 = (1 + Real.sqrt 2)^2 → a = 3 ∧ b = -2 := by
  sorry

end rational_sqrt_equation_l2068_206859


namespace solve_quadratic_equation_l2068_206816

theorem solve_quadratic_equation (x : ℝ) :
  (1/3 - x)^2 = 4 ↔ x = -5/3 ∨ x = 7/3 := by
  sorry

end solve_quadratic_equation_l2068_206816


namespace algorithm_properties_l2068_206823

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the properties of algorithms
def yields_definite_result (a : Algorithm) : Prop := sorry
def multiple_algorithms_exist : Prop := sorry
def terminates_in_finite_steps (a : Algorithm) : Prop := sorry

-- Theorem stating the correct properties of algorithms
theorem algorithm_properties :
  (∀ a : Algorithm, yields_definite_result a) ∧
  multiple_algorithms_exist ∧
  (∀ a : Algorithm, terminates_in_finite_steps a) := by
  sorry

end algorithm_properties_l2068_206823


namespace two_possible_w_values_l2068_206884

theorem two_possible_w_values 
  (w : ℂ) 
  (h_exists : ∃ (u v : ℂ), u ≠ v ∧ ∀ (z : ℂ), (z - u) * (z - v) = (z - w * u) * (z - w * v)) : 
  w = 1 ∨ w = -1 :=
sorry

end two_possible_w_values_l2068_206884


namespace consecutive_even_numbers_sum_l2068_206833

/-- Given three consecutive even numbers whose sum is 246, the first number is 80 -/
theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ a + b + c = 246 ∧ Even a ∧ Even b ∧ Even c) → 
  n = 80 := by
sorry

end consecutive_even_numbers_sum_l2068_206833


namespace sum_of_specific_triangles_l2068_206803

/-- Triangle operation that takes three integers and returns their sum minus the last -/
def triangle_op (a b c : ℤ) : ℤ := a + b - c

/-- The sum of two triangle operations -/
def sum_of_triangles (a₁ b₁ c₁ a₂ b₂ c₂ : ℤ) : ℤ :=
  triangle_op a₁ b₁ c₁ + triangle_op a₂ b₂ c₂

/-- Theorem stating that the sum of triangle operations (1,3,4) and (2,5,6) is 1 -/
theorem sum_of_specific_triangles :
  sum_of_triangles 1 3 4 2 5 6 = 1 := by sorry

end sum_of_specific_triangles_l2068_206803


namespace arithmetic_mean_function_constant_l2068_206897

/-- A function from ℤ² to ℤ⁺ satisfying the arithmetic mean property -/
def ArithmeticMeanFunction (f : ℤ × ℤ → ℕ+) : Prop :=
  ∀ x y : ℤ, (f (x, y) : ℚ) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

/-- If a function satisfies the arithmetic mean property, then it is constant -/
theorem arithmetic_mean_function_constant (f : ℤ × ℤ → ℕ+) 
  (h : ArithmeticMeanFunction f) : 
  ∀ p q : ℤ × ℤ, f p = f q :=
sorry

end arithmetic_mean_function_constant_l2068_206897


namespace gold_coins_distribution_l2068_206856

theorem gold_coins_distribution (x y : ℕ) (h : x * x - y * y = 81 * (x - y)) : x + y = 81 := by
  sorry

end gold_coins_distribution_l2068_206856


namespace latus_rectum_of_parabola_l2068_206881

/-- Given a parabola y = ax^2 where a < 0, its latus rectum has the equation y = -1/(4a) -/
theorem latus_rectum_of_parabola (a : ℝ) (h : a < 0) :
  let parabola := λ x : ℝ => a * x^2
  let latus_rectum := λ y : ℝ => y = -1 / (4 * a)
  ∀ x : ℝ, latus_rectum (parabola x) := by sorry

end latus_rectum_of_parabola_l2068_206881


namespace no_integer_solution_l2068_206850

theorem no_integer_solution (a b c d : ℤ) : 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧
   a * 62^3 + b * 62^2 + c * 62 + d = 2) → False :=
by sorry

end no_integer_solution_l2068_206850


namespace inequality_solution_l2068_206802

/-- The solution set of the inequality x^2 - (a + a^2)x + a^3 > 0 for any real number a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 ∨ a > 1 then {x | x > a^2 ∨ x < a}
  else if a = 0 then {x | x ≠ 0}
  else if 0 < a ∧ a < 1 then {x | x > a ∨ x < a^2}
  else {x | x ≠ 1}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + a^2) * x + a^3 > 0} = solution_set a :=
by sorry

end inequality_solution_l2068_206802


namespace joshua_bottle_caps_l2068_206824

theorem joshua_bottle_caps (initial_caps : ℕ) (bought_caps : ℕ) : 
  initial_caps = 40 → bought_caps = 7 → initial_caps + bought_caps = 47 := by
  sorry

end joshua_bottle_caps_l2068_206824


namespace positive_roots_quadratic_l2068_206815

/-- For a quadratic equation (n-2)x^2 - 2nx + n + 3 = 0, both roots are positive
    if and only if n ∈ (-∞, -3) ∪ (2, 6] -/
theorem positive_roots_quadratic (n : ℝ) : 
  (∀ x : ℝ, (n - 2) * x^2 - 2 * n * x + n + 3 = 0 → x > 0) ↔ 
  (n < -3 ∨ (2 < n ∧ n ≤ 6)) := by
  sorry

end positive_roots_quadratic_l2068_206815


namespace real_roots_iff_k_leq_one_l2068_206878

theorem real_roots_iff_k_leq_one (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 := by
  sorry

end real_roots_iff_k_leq_one_l2068_206878


namespace vector_product_l2068_206806

theorem vector_product (m n : ℝ) : 
  let a : Fin 2 → ℝ := ![m, n]
  let b : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), a = k • b) → 
  (‖a‖ = 2 * ‖b‖) →
  m * n = -8 := by sorry

end vector_product_l2068_206806


namespace angle_sine_relation_l2068_206874

theorem angle_sine_relation (A B : Real) (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2) :
  A > B ↔ Real.sin A > Real.sin B :=
sorry

end angle_sine_relation_l2068_206874


namespace freshman_percentage_l2068_206814

theorem freshman_percentage (total_students : ℝ) (freshman : ℝ) 
  (h1 : freshman > 0) 
  (h2 : total_students > 0) 
  (h3 : freshman * 0.4 * 0.2 = total_students * 0.048) : 
  freshman / total_students = 0.6 := by
sorry

end freshman_percentage_l2068_206814


namespace digital_earth_definition_l2068_206822

/-- Definition of Digital Earth -/
def DigitalEarth : Type := Unit

/-- Property of Digital Earth being a digitized, informational virtual Earth -/
def is_digitized_informational_virtual_earth (de : DigitalEarth) : Prop :=
  -- This is left abstract as the problem doesn't provide specific criteria
  True

/-- Theorem stating that Digital Earth refers to a digitized, informational virtual Earth -/
theorem digital_earth_definition :
  ∀ (de : DigitalEarth), is_digitized_informational_virtual_earth de :=
sorry

end digital_earth_definition_l2068_206822


namespace circle_ratio_after_increase_l2068_206836

theorem circle_ratio_after_increase (r : ℝ) (h : r > 0) :
  let new_radius := r + 2
  let new_circumference := 2 * Real.pi * new_radius
  let new_diameter := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end circle_ratio_after_increase_l2068_206836


namespace milk_powder_cost_july_l2068_206853

/-- The cost of milk powder and coffee in July -/
def july_cost (june_cost : ℝ) : ℝ × ℝ :=
  (0.4 * june_cost, 3 * june_cost)

/-- The total cost of the mixture in July -/
def mixture_cost (june_cost : ℝ) : ℝ :=
  1.5 * (july_cost june_cost).1 + 1.5 * (july_cost june_cost).2

theorem milk_powder_cost_july :
  ∃ (june_cost : ℝ),
    june_cost > 0 ∧
    mixture_cost june_cost = 5.1 ∧
    (july_cost june_cost).1 = 0.4 :=
by sorry

end milk_powder_cost_july_l2068_206853


namespace stevens_skittles_l2068_206863

theorem stevens_skittles (erasers : ℕ) (groups : ℕ) (items_per_group : ℕ) (skittles : ℕ) :
  erasers = 4276 →
  groups = 154 →
  items_per_group = 57 →
  skittles + erasers = groups * items_per_group →
  skittles = 4502 := by
  sorry

end stevens_skittles_l2068_206863
