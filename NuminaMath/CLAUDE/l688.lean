import Mathlib

namespace NUMINAMATH_CALUDE_stone_density_l688_68884

/-- Given a cylindrical container with water and a stone, this theorem relates
    the density of the stone to the water level changes under different conditions. -/
theorem stone_density (S : ℝ) (ρ h₁ h₂ : ℝ) (hS : S > 0) (hρ : ρ > 0) (hh₁ : h₁ > 0) (hh₂ : h₂ > 0) :
  let ρ_s := (ρ * h₁) / h₂
  ρ_s = (ρ * S * h₁) / (S * h₂) :=
by sorry

end NUMINAMATH_CALUDE_stone_density_l688_68884


namespace NUMINAMATH_CALUDE_unique_x_satisfying_three_inequalities_l688_68848

theorem unique_x_satisfying_three_inequalities :
  ∃! (x : ℕ), (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37) ∧
              ¬(2 * x ≥ 21) ∧ ¬(x > 7) :=
by sorry

end NUMINAMATH_CALUDE_unique_x_satisfying_three_inequalities_l688_68848


namespace NUMINAMATH_CALUDE_logarithm_inequality_l688_68883

theorem logarithm_inequality (m n p : ℝ) 
  (hm : 0 < m ∧ m < 1) 
  (hn : 0 < n ∧ n < 1) 
  (hp : 0 < p ∧ p < 1) 
  (h_log : Real.log m / Real.log 3 = Real.log n / Real.log 5 ∧ 
           Real.log n / Real.log 5 = Real.log p / Real.log 10) : 
  m^(1/3) < n^(1/5) ∧ n^(1/5) < p^(1/10) := by
sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l688_68883


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l688_68881

theorem cube_root_125_times_fourth_root_256_times_sqrt_16 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l688_68881


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l688_68866

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3*x^2 + 1}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = {x | -1 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l688_68866


namespace NUMINAMATH_CALUDE_robert_ate_seven_chocolates_l688_68892

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference between Robert's and Nickel's chocolate consumption -/
def robert_nickel_difference : ℕ := 2

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := nickel_chocolates + robert_nickel_difference

theorem robert_ate_seven_chocolates : robert_chocolates = 7 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_seven_chocolates_l688_68892


namespace NUMINAMATH_CALUDE_rectangle_properties_l688_68894

/-- Properties of a rectangle with specific dimensions --/
theorem rectangle_properties (w : ℝ) (h : w > 0) :
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 200 →
  (l * w = 1600 ∧ perimeter - (perimeter - 5) = 5) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_properties_l688_68894


namespace NUMINAMATH_CALUDE_bucky_fish_count_l688_68899

/-- The number of fish Bucky caught on Sunday -/
def F : ℕ := 5

/-- The price of the video game -/
def game_price : ℕ := 60

/-- The amount Bucky earned last weekend -/
def last_weekend_earnings : ℕ := 35

/-- The price of a trout -/
def trout_price : ℕ := 5

/-- The price of a blue-gill -/
def blue_gill_price : ℕ := 4

/-- The percentage of trout caught -/
def trout_percentage : ℚ := 3/5

/-- The percentage of blue-gill caught -/
def blue_gill_percentage : ℚ := 2/5

/-- The additional amount Bucky needs to save -/
def additional_savings : ℕ := 2

theorem bucky_fish_count :
  F * (trout_percentage * trout_price + blue_gill_percentage * blue_gill_price) =
  game_price - last_weekend_earnings - additional_savings :=
sorry

end NUMINAMATH_CALUDE_bucky_fish_count_l688_68899


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_l688_68839

theorem least_positive_integer_multiple (x : ℕ) : x = 47 ↔ 
  (x > 0 ∧ ∀ y : ℕ, y > 0 → y < x → ¬((2*y)^2 + 2*47*2*y + 47^2) % 47 = 0) ∧
  ((2*x)^2 + 2*47*2*x + 47^2) % 47 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_l688_68839


namespace NUMINAMATH_CALUDE_tammy_caught_30_times_l688_68880

/-- Calculates the number of times Tammy caught the ball given the conditions of the problem. -/
def tammys_catches (joe_catches : ℕ) : ℕ :=
  let derek_catches := 2 * joe_catches - 4
  let tammy_catches := derek_catches / 3 + 16
  tammy_catches

/-- Theorem stating that Tammy caught the ball 30 times given the problem conditions. -/
theorem tammy_caught_30_times : tammys_catches 23 = 30 := by
  sorry

#eval tammys_catches 23

end NUMINAMATH_CALUDE_tammy_caught_30_times_l688_68880


namespace NUMINAMATH_CALUDE_pony_jeans_discount_rate_l688_68885

theorem pony_jeans_discount_rate :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let total_savings : ℚ := 9
  let total_discount_rate : ℚ := 25

  ∀ (fox_discount pony_discount : ℚ),
    fox_discount + pony_discount = total_discount_rate →
    fox_quantity * fox_price * (fox_discount / 100) + 
    pony_quantity * pony_price * (pony_discount / 100) = total_savings →
    pony_discount = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_pony_jeans_discount_rate_l688_68885


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l688_68801

theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_radius := 2.5 * r
  let new_height := 3 * h
  (π * new_radius^2 * new_height) / (π * r^2 * h) = 18.75 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l688_68801


namespace NUMINAMATH_CALUDE_vector_addition_l688_68888

theorem vector_addition : 
  let v1 : Fin 3 → ℝ := ![(-5 : ℝ), 1, -4]
  let v2 : Fin 3 → ℝ := ![0, 8, -4]
  v1 + v2 = ![(-5 : ℝ), 9, -8] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l688_68888


namespace NUMINAMATH_CALUDE_max_value_quadratic_l688_68852

theorem max_value_quadratic (x : ℝ) : 
  (∃ (z : ℝ), z = x^2 - 14*x + 10) → 
  (∃ (max_z : ℝ), max_z = -39 ∧ ∀ (y : ℝ), y = x^2 - 14*x + 10 → y ≤ max_z) :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l688_68852


namespace NUMINAMATH_CALUDE_fraction_addition_l688_68826

theorem fraction_addition : (4 / 7 : ℚ) / 5 + 1 / 3 = 47 / 105 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l688_68826


namespace NUMINAMATH_CALUDE_points_per_treasure_l688_68868

/-- Calculates the points per treasure in Tiffany's video game. -/
theorem points_per_treasure (treasures_level1 treasures_level2 total_score : ℕ) : 
  treasures_level1 = 3 → treasures_level2 = 5 → total_score = 48 →
  total_score / (treasures_level1 + treasures_level2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_points_per_treasure_l688_68868


namespace NUMINAMATH_CALUDE_point_inside_circle_parameter_range_l688_68879

theorem point_inside_circle_parameter_range :
  ∀ a : ℝ, 
  (∃ x y : ℝ, (x - a)^2 + (y + a)^2 = 4 ∧ (1 - a)^2 + (1 + a)^2 < 4) →
  -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_parameter_range_l688_68879


namespace NUMINAMATH_CALUDE_square_difference_equality_l688_68864

theorem square_difference_equality : 1005^2 - 995^2 - 1002^2 + 996^2 = 8012 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l688_68864


namespace NUMINAMATH_CALUDE_inequality_preservation_l688_68838

theorem inequality_preservation (a b c : ℝ) (h : a > b) : c - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l688_68838


namespace NUMINAMATH_CALUDE_cosine_arctangent_equation_solution_l688_68867

theorem cosine_arctangent_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (2 * x)) = x / 2 →
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (2 * x)) = x / 2 ∧ x^2 = (Real.sqrt 17 - 1) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cosine_arctangent_equation_solution_l688_68867


namespace NUMINAMATH_CALUDE_inequality_implies_x_greater_than_one_l688_68871

theorem inequality_implies_x_greater_than_one (x : ℝ) :
  x * (x^2 + 1) > (x + 1) * (x^2 - x + 1) → x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_x_greater_than_one_l688_68871


namespace NUMINAMATH_CALUDE_fraction_of_total_l688_68845

theorem fraction_of_total (total : ℝ) (r_amount : ℝ) (h1 : total = 5000) (h2 : r_amount = 2000.0000000000002) :
  r_amount / total = 0.40000000000000004 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_total_l688_68845


namespace NUMINAMATH_CALUDE_f_positive_iff_f_inequality_iff_l688_68834

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the first part of the problem
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x < -1/3 ∨ x > 3 := by sorry

-- Theorem for the second part of the problem
theorem f_inequality_iff (a : ℝ) : 
  (∀ x : ℝ, f x + 3 * |x + 2| ≥ |a - 1|) ↔ -4 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_f_inequality_iff_l688_68834


namespace NUMINAMATH_CALUDE_bandage_overlap_l688_68861

theorem bandage_overlap (n : ℕ) (l : ℝ) (total : ℝ) (h1 : n = 20) (h2 : l = 15.25) (h3 : total = 248) :
  (n * l - total) / (n - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_bandage_overlap_l688_68861


namespace NUMINAMATH_CALUDE_apples_to_oranges_ratio_l688_68805

/-- Represents the number of fruits of each type on the display -/
structure FruitDisplay where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Defines the conditions of the fruit display -/
def validFruitDisplay (d : FruitDisplay) : Prop :=
  d.oranges = 2 * d.bananas ∧
  d.bananas = 5 ∧
  d.apples + d.oranges + d.bananas = 35

/-- Theorem stating that for a valid fruit display, the ratio of apples to oranges is 2:1 -/
theorem apples_to_oranges_ratio (d : FruitDisplay) (h : validFruitDisplay d) :
  d.apples * 1 = d.oranges * 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_to_oranges_ratio_l688_68805


namespace NUMINAMATH_CALUDE_problem_statement_l688_68869

theorem problem_statement : (481 + 426)^2 - 4 * 481 * 426 = 3025 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l688_68869


namespace NUMINAMATH_CALUDE_value_of_a_l688_68872

theorem value_of_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l688_68872


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l688_68814

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : 
  (x * y ≥ 4) ∧ (x + y ≥ 9/2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l688_68814


namespace NUMINAMATH_CALUDE_f_minimum_value_g_range_condition_l688_68818

noncomputable section

def f (x : ℝ) := 2 * x * Real.log x

def g (a x : ℝ) := -x^2 + a*x - 3

theorem f_minimum_value :
  ∃ (m : ℝ), m = 2 / Real.exp 1 ∧ ∀ x > 0, f x ≥ m :=
sorry

theorem g_range_condition (a : ℝ) :
  (∃ x > 0, f x ≤ g a x) → a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_g_range_condition_l688_68818


namespace NUMINAMATH_CALUDE_min_value_of_s_min_value_of_s_achieved_l688_68849

theorem min_value_of_s (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (y*z/x + z*x/y + x*y/z) ≥ Real.sqrt 3 := by
  sorry

theorem min_value_of_s_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧
    b*c/a + c*a/b + a*b/c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_s_min_value_of_s_achieved_l688_68849


namespace NUMINAMATH_CALUDE_det_equals_polynomial_l688_68824

/-- The determinant of a 3x3 matrix with polynomial entries -/
def matrix_det (y : ℝ) : ℝ :=
  let a11 := 2*y + 3
  let a12 := y - 1
  let a13 := y + 2
  let a21 := y + 1
  let a22 := 2*y
  let a23 := y
  let a31 := y
  let a32 := y
  let a33 := 2*y - 1
  a11 * (a22 * a33 - a23 * a32) - 
  a12 * (a21 * a33 - a23 * a31) + 
  a13 * (a21 * a32 - a22 * a31)

theorem det_equals_polynomial (y : ℝ) : 
  matrix_det y = 4*y^3 + 8*y^2 - 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_det_equals_polynomial_l688_68824


namespace NUMINAMATH_CALUDE_chess_tournament_games_l688_68850

/-- The number of games played in a chess tournament with n participants,
    where each participant plays exactly one game with each other participant. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 22 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 231. -/
theorem chess_tournament_games :
  tournament_games 22 = 231 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l688_68850


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l688_68810

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The distance from a point to the left focus -/
def dist_to_left_focus (x y : ℝ) : ℝ := sorry

/-- The distance from a point to the right focus -/
def dist_to_right_focus (x y : ℝ) : ℝ := sorry

/-- Theorem: If P(x,y) is on the right branch of the hyperbola and
    its distance to the left focus is 12, then its distance to the right focus is 4 -/
theorem hyperbola_focus_distance (x y : ℝ) :
  is_on_hyperbola x y ∧ x > 0 ∧ dist_to_left_focus x y = 12 →
  dist_to_right_focus x y = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l688_68810


namespace NUMINAMATH_CALUDE_polynomial_expansion_l688_68817

theorem polynomial_expansion :
  ∀ t : ℝ, (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 
    6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l688_68817


namespace NUMINAMATH_CALUDE_winnie_lollipops_l688_68829

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (cherry wintergreen grape shrimp friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp) % friends

theorem winnie_lollipops :
  lollipops_kept 36 125 8 241 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l688_68829


namespace NUMINAMATH_CALUDE_flower_purchase_cost_katie_flower_purchase_cost_l688_68841

/-- The cost of buying roses and daisies at a fixed price per flower -/
theorem flower_purchase_cost 
  (price_per_flower : ℕ) 
  (num_roses : ℕ) 
  (num_daisies : ℕ) : 
  price_per_flower * (num_roses + num_daisies) = 
    price_per_flower * num_roses + price_per_flower * num_daisies :=
by sorry

/-- The total cost of Katie's flower purchase -/
theorem katie_flower_purchase_cost : 
  (5 : ℕ) + 5 = 10 ∧ 6 * 10 = 60 :=
by sorry

end NUMINAMATH_CALUDE_flower_purchase_cost_katie_flower_purchase_cost_l688_68841


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l688_68815

theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  2 * c^2 - 2 * a^2 = b^2 →
  2 * c * Real.cos A - 2 * a * Real.cos C = b :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_angle_relation_l688_68815


namespace NUMINAMATH_CALUDE_soda_consumption_per_person_l688_68828

/- Define the problem parameters -/
def people_attending : ℕ := 5 * 12  -- five dozens
def cans_per_box : ℕ := 10
def cost_per_box : ℕ := 2
def family_members : ℕ := 6
def payment_per_member : ℕ := 4

/- Define the theorem -/
theorem soda_consumption_per_person :
  let total_payment := family_members * payment_per_member
  let boxes_bought := total_payment / cost_per_box
  let total_cans := boxes_bought * cans_per_box
  total_cans / people_attending = 2 := by sorry

end NUMINAMATH_CALUDE_soda_consumption_per_person_l688_68828


namespace NUMINAMATH_CALUDE_interview_score_calculation_l688_68886

/-- Calculate the interview score based on individual scores and their proportions -/
theorem interview_score_calculation 
  (basic_knowledge : ℝ) 
  (communication_skills : ℝ) 
  (work_attitude : ℝ) 
  (basic_knowledge_proportion : ℝ) 
  (communication_skills_proportion : ℝ) 
  (work_attitude_proportion : ℝ) 
  (h1 : basic_knowledge = 92) 
  (h2 : communication_skills = 87) 
  (h3 : work_attitude = 94) 
  (h4 : basic_knowledge_proportion = 0.2) 
  (h5 : communication_skills_proportion = 0.3) 
  (h6 : work_attitude_proportion = 0.5) :
  basic_knowledge * basic_knowledge_proportion + 
  communication_skills * communication_skills_proportion + 
  work_attitude * work_attitude_proportion = 91.5 := by
sorry

end NUMINAMATH_CALUDE_interview_score_calculation_l688_68886


namespace NUMINAMATH_CALUDE_problem_statement_l688_68832

theorem problem_statement : 2 * ((40 / 8) + (34 / 12)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l688_68832


namespace NUMINAMATH_CALUDE_equation_solution_l688_68812

theorem equation_solution (x : ℝ) :
  x^2 + x + 1 = 1 / (x^2 - x + 1) ∧ x^2 - x + 1 ≠ 0 → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l688_68812


namespace NUMINAMATH_CALUDE_divisibility_by_three_l688_68831

theorem divisibility_by_three (x y : ℤ) : 
  (3 ∣ x^2 + y^2) → (3 ∣ x) ∧ (3 ∣ y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l688_68831


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l688_68870

theorem angle_terminal_side_value (a : ℝ) (h : a > 0) :
  let x := 5 * a
  let y := -12 * a
  let r := Real.sqrt (x^2 + y^2)
  let sinα := y / r
  let cosα := x / r
  2 * sinα + cosα = -19 / 13 := by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l688_68870


namespace NUMINAMATH_CALUDE_tuesday_lost_revenue_l688_68889

/-- Represents a movie theater with its capacity, ticket price, and tickets sold. -/
structure MovieTheater where
  capacity : ℕ
  ticketPrice : ℚ
  ticketsSold : ℕ

/-- Calculates the lost revenue for a movie theater. -/
def lostRevenue (theater : MovieTheater) : ℚ :=
  (theater.capacity - theater.ticketsSold) * theater.ticketPrice

/-- Theorem stating that the lost revenue for the given theater scenario is $208.00. -/
theorem tuesday_lost_revenue :
  let theater : MovieTheater := ⟨50, 8, 24⟩
  lostRevenue theater = 208 := by sorry

end NUMINAMATH_CALUDE_tuesday_lost_revenue_l688_68889


namespace NUMINAMATH_CALUDE_universally_energetic_characterization_no_specific_energetic_triplets_l688_68835

/-- A triplet (a, b, c) is n-energetic if it satisfies the given conditions --/
def isNEnergetic (a b c n : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Nat.gcd a (Nat.gcd b c) = 1 ∧ (a^n + b^n + c^n) % (a + b + c) = 0

/-- A triplet (a, b, c) is universally energetic if it is n-energetic for all n ≥ 1 --/
def isUniversallyEnergetic (a b c : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → isNEnergetic a b c n

/-- The set of all universally energetic triplets --/
def universallyEnergeticTriplets : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ isUniversallyEnergetic t.1 t.2.1 t.2.2}

theorem universally_energetic_characterization :
    universallyEnergeticTriplets = {(1, 1, 1), (1, 1, 4)} := by sorry

theorem no_specific_energetic_triplets :
    ∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 →
      (isNEnergetic a b c 2004 ∧ isNEnergetic a b c 2005 ∧ ¬isNEnergetic a b c 2007) → False := by sorry

end NUMINAMATH_CALUDE_universally_energetic_characterization_no_specific_energetic_triplets_l688_68835


namespace NUMINAMATH_CALUDE_winter_fest_attendance_l688_68897

theorem winter_fest_attendance (total_students : ℕ) (attending_students : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1400) 
  (h2 : attending_students = 800) (h3 : girls + boys = total_students) 
  (h4 : 3 * girls / 4 + 3 * boys / 5 = attending_students) : 
  3 * girls / 4 = 600 := by
sorry

end NUMINAMATH_CALUDE_winter_fest_attendance_l688_68897


namespace NUMINAMATH_CALUDE_house_rent_calculation_l688_68830

def salary : ℚ := 170000

def food_fraction : ℚ := 1/5
def clothes_fraction : ℚ := 3/5
def remaining_amount : ℚ := 17000

def house_rent_fraction : ℚ := 1/10

theorem house_rent_calculation :
  house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining_amount = salary :=
by sorry

end NUMINAMATH_CALUDE_house_rent_calculation_l688_68830


namespace NUMINAMATH_CALUDE_percentage_problem_l688_68819

/-- Given that P% of 820 is 20 less than 15% of 1500, prove that P = 25 -/
theorem percentage_problem (P : ℝ) (h : P / 100 * 820 = 15 / 100 * 1500 - 20) : P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l688_68819


namespace NUMINAMATH_CALUDE_correct_number_of_pitchers_l688_68840

/-- The number of glasses each pitcher can serve -/
def glasses_per_pitcher : ℕ := 5

/-- The total number of glasses served -/
def total_glasses_served : ℕ := 30

/-- The number of pitchers prepared -/
def pitchers_prepared : ℕ := total_glasses_served / glasses_per_pitcher

theorem correct_number_of_pitchers : pitchers_prepared = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_pitchers_l688_68840


namespace NUMINAMATH_CALUDE_x_range_l688_68865

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x : ℝ) : Prop := x^2 + 3*x ≥ 0

-- Define the theorem
theorem x_range :
  ∀ x : ℝ, (¬(p x ∧ q x) ∧ ¬(¬(p x))) → (-2 ≤ x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l688_68865


namespace NUMINAMATH_CALUDE_distance_between_foci_l688_68859

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l688_68859


namespace NUMINAMATH_CALUDE_hotel_reunion_attendees_l688_68878

theorem hotel_reunion_attendees (total_guests oates_attendees hall_attendees : ℕ) 
  (h1 : total_guests = 100)
  (h2 : oates_attendees = 40)
  (h3 : hall_attendees = 70)
  (h4 : total_guests ≤ oates_attendees + hall_attendees) :
  oates_attendees + hall_attendees - total_guests = 10 := by
  sorry

end NUMINAMATH_CALUDE_hotel_reunion_attendees_l688_68878


namespace NUMINAMATH_CALUDE_rectangle_fold_area_l688_68827

theorem rectangle_fold_area (a b : ℝ) (h1 : a = 4) (h2 : b = 8) : 
  let diagonal := Real.sqrt (a^2 + b^2)
  let height := diagonal / 2
  (1/2) * diagonal * height = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_fold_area_l688_68827


namespace NUMINAMATH_CALUDE_function_properties_l688_68816

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p ≠ 0 ∧ ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h_even : IsEven f)
    (h_shift : ∀ x, f (x + 1) = -f x)
    (h_incr : IsIncreasingOn f (-1) 0) :
    (IsPeriodic f 2) ∧ 
    (∀ x, f (2 - x) = f x) ∧
    (f 2 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l688_68816


namespace NUMINAMATH_CALUDE_trigonometric_identities_l688_68823

theorem trigonometric_identities (x : Real) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.tan x = -2) : 
  (Real.sin x - Real.cos x = -3 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (2 * π - x) * Real.cos (π - x) - Real.sin x ^ 2) / 
   (Real.cos (π + x) * Real.cos (π/2 - x) + Real.cos x ^ 2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l688_68823


namespace NUMINAMATH_CALUDE_equation_solution_l688_68837

theorem equation_solution (x : ℝ) (h : x > 0) :
  (x - 3) / 8 = 5 / (x - 8) ↔ x = 16 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l688_68837


namespace NUMINAMATH_CALUDE_amy_balloon_count_l688_68860

/-- Given that James has 1222 balloons and 709 more balloons than Amy,
    prove that Amy has 513 balloons. -/
theorem amy_balloon_count :
  ∀ (james_balloons amy_balloons : ℕ),
    james_balloons = 1222 →
    james_balloons = amy_balloons + 709 →
    amy_balloons = 513 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_balloon_count_l688_68860


namespace NUMINAMATH_CALUDE_sum_of_cubes_l688_68821

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l688_68821


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l688_68895

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (A B C : ℤ) : ℝ → ℝ := fun x ↦ A * x^2 + B * x + C

theorem quadratic_function_m_value
  (A B C : ℤ)
  (h1 : QuadraticFunction A B C 2 = 0)
  (h2 : 100 < QuadraticFunction A B C 9 ∧ QuadraticFunction A B C 9 < 110)
  (h3 : 150 < QuadraticFunction A B C 10 ∧ QuadraticFunction A B C 10 < 160)
  (h4 : ∃ m : ℤ, 10000 * m < QuadraticFunction A B C 200 ∧ QuadraticFunction A B C 200 < 10000 * (m + 1)) :
  ∃ m : ℤ, m = 16 ∧ 10000 * m < QuadraticFunction A B C 200 ∧ QuadraticFunction A B C 200 < 10000 * (m + 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l688_68895


namespace NUMINAMATH_CALUDE_octal_subtraction_l688_68858

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Subtraction operation in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Conversion from octal to decimal --/
def from_octal (n : OctalNumber) : ℕ :=
  sorry

theorem octal_subtraction :
  octal_sub (to_octal 43) (to_octal 22) = to_octal 21 :=
by sorry

end NUMINAMATH_CALUDE_octal_subtraction_l688_68858


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l688_68825

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 40 →
  football = 26 →
  tennis = 20 →
  neither = 11 →
  ∃ both : ℕ, both = 17 ∧ total = football + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l688_68825


namespace NUMINAMATH_CALUDE_unique_function_is_zero_l688_68804

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f (x - y) + f (f (1 - x * y))

-- Theorem statement
theorem unique_function_is_zero :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_function_is_zero_l688_68804


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l688_68876

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 48 →
  triangle_height = 48 →
  (square_perimeter / 4)^2 = (1/2) * x * triangle_height →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l688_68876


namespace NUMINAMATH_CALUDE_max_ratio_squared_l688_68820

theorem max_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) : 
  (∃ ρ : ℝ, ρ > 0 ∧ ρ^2 = 2 ∧ 
    (∀ r : ℝ, r > ρ → 
      ¬∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ 
        a^2 + y^2 = (a - x)^2 + (b - y)^2 ∧ 
        a^2 + y^2 = b^2 - x^2 + y^2 ∧ 
        r = a / b)) := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l688_68820


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l688_68862

theorem inverse_of_proposition (p q : Prop) :
  (¬p → ¬q) → (¬q → ¬p) := by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l688_68862


namespace NUMINAMATH_CALUDE_house_selling_price_l688_68851

theorem house_selling_price 
  (original_price : ℝ)
  (profit_percentage : ℝ)
  (commission_percentage : ℝ)
  (h1 : original_price = 80000)
  (h2 : profit_percentage = 20)
  (h3 : commission_percentage = 5)
  : original_price + (profit_percentage / 100) * original_price + (commission_percentage / 100) * original_price = 100000 := by
  sorry

end NUMINAMATH_CALUDE_house_selling_price_l688_68851


namespace NUMINAMATH_CALUDE_probability_two_black_two_white_l688_68800

def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4
def drawn_black : ℕ := 2
def drawn_white : ℕ := 2

theorem probability_two_black_two_white :
  (Nat.choose black_balls drawn_black * Nat.choose white_balls drawn_white) /
  Nat.choose total_balls drawn_balls = 7 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_two_white_l688_68800


namespace NUMINAMATH_CALUDE_range_of_a_l688_68813

-- Define the system of linear equations
def system (x y a : ℝ) : Prop :=
  (3 * x + 5 * y = 6 * a) ∧ (2 * x + 6 * y = 3 * a + 3)

-- Define the constraint
def constraint (x y : ℝ) : Prop :=
  x - y > 0

-- Theorem statement
theorem range_of_a (x y a : ℝ) :
  system x y a → constraint x y → a > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l688_68813


namespace NUMINAMATH_CALUDE_valid_marking_exists_l688_68877

/-- Represents a marking of cells in a 9x9 table -/
def Marking := Fin 9 → Fin 9 → Bool

/-- Checks if two adjacent rows have at least 6 marked cells -/
def validRows (m : Marking) : Prop :=
  ∀ i : Fin 8, (Finset.sum (Finset.univ.filter (λ j => m i j || m (i + 1) j)) (λ _ => 1) : ℕ) ≥ 6

/-- Checks if two adjacent columns have at most 5 marked cells -/
def validColumns (m : Marking) : Prop :=
  ∀ j : Fin 8, (Finset.sum (Finset.univ.filter (λ i => m i j || m i (j + 1))) (λ _ => 1) : ℕ) ≤ 5

/-- Theorem stating that a valid marking exists -/
theorem valid_marking_exists : ∃ m : Marking, validRows m ∧ validColumns m := by
  sorry

end NUMINAMATH_CALUDE_valid_marking_exists_l688_68877


namespace NUMINAMATH_CALUDE_set_operations_l688_68844

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

def B : Set ℝ := {x : ℝ | x < -3 ∨ x > 1}

theorem set_operations :
  (A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B = {x : ℝ | x < -3 ∨ x > 0}) ∧
  ((Aᶜ) ∩ (Bᶜ) = {x : ℝ | -3 ≤ x ∧ x ≤ 0}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l688_68844


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l688_68887

theorem system_of_equations_solution
  (a b c x y z : ℝ)
  (h1 : x - a * y + a^2 * z = a^3)
  (h2 : x - b * y + b^2 * z = b^3)
  (h3 : x - c * y + c^2 * z = c^3)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hca : c ≠ a) :
  x = a * b * c ∧ y = a * b + b * c + c * a ∧ z = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l688_68887


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l688_68874

theorem tan_alpha_minus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α - Real.pi/4) = -1/7 ∨ Real.tan (α - Real.pi/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l688_68874


namespace NUMINAMATH_CALUDE_smallest_k_for_p_cubed_minus_k_div_24_l688_68863

-- Define p as the largest prime number with 1007 digits
def p : Nat := sorry

-- Define the property that p is prime
axiom p_is_prime : Nat.Prime p

-- Define the property that p has 1007 digits
axiom p_has_1007_digits : 10^1006 ≤ p ∧ p < 10^1007

-- Define the property that p is the largest such prime
axiom p_is_largest : ∀ q : Nat, Nat.Prime q → 10^1006 ≤ q ∧ q < 10^1007 → q ≤ p

-- Theorem statement
theorem smallest_k_for_p_cubed_minus_k_div_24 :
  (∃ k : Nat, k > 0 ∧ (p^3 - k) % 24 = 0) ∧
  (∀ k : Nat, k > 0 ∧ (p^3 - k) % 24 = 0 → k ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_p_cubed_minus_k_div_24_l688_68863


namespace NUMINAMATH_CALUDE_max_perimeter_after_cut_l688_68853

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Maximum perimeter after cutting out a smaller rectangle -/
theorem max_perimeter_after_cut (original : Rectangle) (cutout : Rectangle) :
  original.length = 20 ∧ 
  original.width = 16 ∧ 
  cutout.length = 10 ∧ 
  cutout.width = 5 →
  ∃ (remaining : Rectangle), 
    perimeter remaining = 92 ∧ 
    ∀ (other : Rectangle), perimeter other ≤ perimeter remaining :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_after_cut_l688_68853


namespace NUMINAMATH_CALUDE_matrix_sum_of_squares_l688_68893

open Matrix

theorem matrix_sum_of_squares (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (transpose B = (2 : ℝ) • (B⁻¹)) →
  x^2 + y^2 + z^2 + w^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_of_squares_l688_68893


namespace NUMINAMATH_CALUDE_painted_cubes_l688_68803

theorem painted_cubes (n : ℕ) (h : n = 10) : 
  n^3 - (n - 2)^3 = 488 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l688_68803


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l688_68846

/-- The area of a square with one side on y = 8 and endpoints on y = x^2 + 4x + 3 is 36 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (8 = x₁^2 + 4*x₁ + 3) ∧
  (8 = x₂^2 + 4*x₂ + 3) ∧
  ((x₂ - x₁)^2 = 36) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l688_68846


namespace NUMINAMATH_CALUDE_perfect_number_examples_sum_xy_is_one_k_equals_36_when_S_is_perfect_number_l688_68822

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Theorem 1: 11 is not a perfect number and 53 is a perfect number -/
theorem perfect_number_examples :
  (¬ is_perfect_number 11) ∧ (is_perfect_number 53) := by sorry

/-- Theorem 2: Given x^2 + y^2 - 4x + 2y + 5 = 0, prove x + y = 1 -/
theorem sum_xy_is_one (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 5 = 0) :
  x + y = 1 := by sorry

/-- Definition of S -/
def S (x y k : ℝ) : ℝ := 2*x^2 + y^2 + 2*x*y + 12*x + k

/-- Theorem 3: Given S = 2x^2 + y^2 + 2xy + 12x + k, 
    prove that k = 36 when S is a perfect number -/
theorem k_equals_36_when_S_is_perfect_number (x y : ℝ) :
  (∃ a b : ℝ, S x y 36 = a^2 + b^2) → 
  (∀ k : ℝ, (∃ a b : ℝ, S x y k = a^2 + b^2) → k = 36) := by sorry

end NUMINAMATH_CALUDE_perfect_number_examples_sum_xy_is_one_k_equals_36_when_S_is_perfect_number_l688_68822


namespace NUMINAMATH_CALUDE_sons_age_l688_68890

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 34 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 32 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l688_68890


namespace NUMINAMATH_CALUDE_tracy_candies_l688_68873

theorem tracy_candies (x : ℕ) : x = 68 :=
  -- Initial number of candies
  have h1 : x > 0 := by sorry

  -- After eating 1/4, the remaining candies are divisible by 3 (for giving 1/3 to Rachel)
  have h2 : ∃ k : ℕ, 3 * k = 3 * x / 4 := by sorry

  -- After giving 1/3 to Rachel, the remaining candies are even (for Tracy and mom to eat 12 each)
  have h3 : ∃ m : ℕ, 2 * m = x / 2 := by sorry

  -- After Tracy and mom eat 12 each, the remaining candies are between 7 and 11
  have h4 : 7 ≤ x / 2 - 24 ∧ x / 2 - 24 ≤ 11 := by sorry

  -- Final number of candies is 5
  have h5 : ∃ b : ℕ, 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 24 - b = 5 := by sorry

  sorry

end NUMINAMATH_CALUDE_tracy_candies_l688_68873


namespace NUMINAMATH_CALUDE_homogeneous_de_solution_l688_68811

/-- The homogeneous differential equation -/
def homogeneous_de (x y : ℝ) (dx dy : ℝ) : Prop :=
  (x^2 - y^2) * dy - 2 * y * x * dx = 0

/-- The general solution to the homogeneous differential equation -/
def general_solution (x y C : ℝ) : Prop :=
  x^2 + y^2 = C * y

/-- Theorem stating that the general solution satisfies the homogeneous differential equation -/
theorem homogeneous_de_solution (x y C : ℝ) :
  general_solution x y C →
  ∃ (dx dy : ℝ), homogeneous_de x y dx dy :=
sorry

end NUMINAMATH_CALUDE_homogeneous_de_solution_l688_68811


namespace NUMINAMATH_CALUDE_cherry_sales_analysis_l688_68842

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

end NUMINAMATH_CALUDE_cherry_sales_analysis_l688_68842


namespace NUMINAMATH_CALUDE_special_function_value_l688_68807

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem special_function_value :
  ∀ f : ℝ → ℝ, special_function f → f 1 = 2 → f (-3) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l688_68807


namespace NUMINAMATH_CALUDE_minimum_value_a_l688_68891

theorem minimum_value_a (a : ℝ) : (∀ x₁ x₂ x₃ x₄ : ℝ, ∃ k₁ k₂ k₃ k₄ : ℤ,
  (x₁ - k₁ - (x₂ - k₂))^2 + (x₁ - k₁ - (x₃ - k₃))^2 + (x₁ - k₁ - (x₄ - k₄))^2 +
  (x₂ - k₂ - (x₃ - k₃))^2 + (x₂ - k₂ - (x₄ - k₄))^2 + (x₃ - k₃ - (x₄ - k₄))^2 ≤ a) →
  a ≥ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_a_l688_68891


namespace NUMINAMATH_CALUDE_power_values_l688_68847

theorem power_values (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(4*m + 3*n) = 432 ∧ a^(5*m - 2*n) = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_power_values_l688_68847


namespace NUMINAMATH_CALUDE_min_value_theorem_l688_68854

theorem min_value_theorem (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + y₀ + z₀ = 1 ∧ 
    1/x₀ + 4/y₀ + 9/z₀ = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l688_68854


namespace NUMINAMATH_CALUDE_pauls_toys_l688_68855

theorem pauls_toys (toys_per_box : ℕ) (number_of_boxes : ℕ) (h1 : toys_per_box = 8) (h2 : number_of_boxes = 4) :
  toys_per_box * number_of_boxes = 32 := by
  sorry

end NUMINAMATH_CALUDE_pauls_toys_l688_68855


namespace NUMINAMATH_CALUDE_sector_area_l688_68802

/-- The area of a sector of a circle with radius 4 cm and arc length 3.5 cm is 7 cm² -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h1 : r = 4) (h2 : arc_length = 3.5) :
  (arc_length / (2 * π * r)) * (π * r^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l688_68802


namespace NUMINAMATH_CALUDE_hyperbola_center_l688_68857

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  center = (7, 2) ↔ f1 = (3, -2) ∧ f2 = (11, 6) := by
  sorry

#check hyperbola_center

end NUMINAMATH_CALUDE_hyperbola_center_l688_68857


namespace NUMINAMATH_CALUDE_worker_count_l688_68875

theorem worker_count (total : ℕ) (increased_total : ℕ) (extra_contribution : ℕ) : 
  (total = 300000) → 
  (increased_total = 325000) → 
  (extra_contribution = 50) → 
  (∃ (n : ℕ), (n * (total / n) = total) ∧ 
              (n * (total / n + extra_contribution) = increased_total) ∧ 
              (n = 500)) := by
  sorry

end NUMINAMATH_CALUDE_worker_count_l688_68875


namespace NUMINAMATH_CALUDE_line_equation_through_ellipse_points_l688_68808

/-- The equation of a line passing through two points on an ellipse -/
theorem line_equation_through_ellipse_points 
  (A B : ℝ × ℝ) -- Two points on the ellipse
  (h_ellipse_A : (A.1^2 / 16) + (A.2^2 / 12) = 1) -- A is on the ellipse
  (h_ellipse_B : (B.1^2 / 16) + (B.2^2 / 12) = 1) -- B is on the ellipse
  (h_midpoint : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1)) -- Midpoint of AB is (2, 1)
  : ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ 
                    a * B.1 + b * B.2 + c = 0 ∧ 
                    (a, b, c) = (3, 2, -8) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_ellipse_points_l688_68808


namespace NUMINAMATH_CALUDE_sum_lent_is_400_l688_68896

/-- Prove that the sum lent is 400, given the conditions of the problem -/
theorem sum_lent_is_400 
  (interest_rate : ℚ) 
  (time_period : ℕ) 
  (interest_difference : ℚ) 
  (h1 : interest_rate = 4 / 100)
  (h2 : time_period = 8)
  (h3 : interest_difference = 272) :
  ∃ (sum_lent : ℚ), 
    sum_lent * interest_rate * time_period = sum_lent - interest_difference ∧ 
    sum_lent = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_is_400_l688_68896


namespace NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l688_68856

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (1 - 2*m) + y^2 / (m + 3) = 1 ∧ (1 - 2*m) * (m + 3) < 0

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 3 - 2*m = 0

-- Theorem for the range of m where p is true
theorem p_range (m : ℝ) : p m ↔ m < -3 ∨ m > 1/2 := by sorry

-- Theorem for the range of m where q is true
theorem q_range (m : ℝ) : q m ↔ m ≤ -3 ∨ m ≥ 1 := by sorry

-- Theorem for the range of m where "p ∨ q" is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -3 < m ∧ m ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l688_68856


namespace NUMINAMATH_CALUDE_mechanic_worked_six_hours_l688_68836

def mechanic_hours (total_cost parts_cost labor_rate : ℚ) : ℚ :=
  let parts_total := 2 * parts_cost
  let labor_cost := total_cost - parts_total
  let minutes_worked := labor_cost / labor_rate
  minutes_worked / 60

theorem mechanic_worked_six_hours :
  mechanic_hours 220 20 0.5 = 6 := by sorry

end NUMINAMATH_CALUDE_mechanic_worked_six_hours_l688_68836


namespace NUMINAMATH_CALUDE_cracker_problem_l688_68843

/-- The number of crackers Darren and Calvin bought together -/
def total_crackers (darren_boxes calvin_boxes crackers_per_box : ℕ) : ℕ :=
  (darren_boxes + calvin_boxes) * crackers_per_box

theorem cracker_problem :
  ∀ (darren_boxes calvin_boxes crackers_per_box : ℕ),
    darren_boxes = 4 →
    crackers_per_box = 24 →
    calvin_boxes = 2 * darren_boxes - 1 →
    total_crackers darren_boxes calvin_boxes crackers_per_box = 264 := by
  sorry

end NUMINAMATH_CALUDE_cracker_problem_l688_68843


namespace NUMINAMATH_CALUDE_prime_representation_l688_68833

theorem prime_representation (p : ℕ) (hp : p.Prime) (hp_gt_2 : p > 2) :
  (p % 8 = 1 → ∃ x y : ℤ, ↑p = x^2 + 16 * y^2) ∧
  (p % 8 = 5 → ∃ x y : ℤ, ↑p = 4 * x^2 + 4 * x * y + 5 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_representation_l688_68833


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l688_68806

/-- Given a cone with a 45° angle between the generatrix and base, and height 1,
    the length of the generatrix is √2. -/
theorem cone_generatrix_length 
  (angle : ℝ) 
  (height : ℝ) 
  (h_angle : angle = Real.pi / 4) 
  (h_height : height = 1) : 
  Real.sqrt 2 = 
    Real.sqrt (height ^ 2 + height ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l688_68806


namespace NUMINAMATH_CALUDE_parallel_line_condition_perpendicular_line_condition_l688_68882

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (3, 0)

-- Define the given lines
def line1 (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 4 * x + y - 14 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Theorem for the first condition
theorem parallel_line_condition :
  parallel_line A.1 A.2 ∧
  ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), line1 (x + k) (y - 4 * k) :=
sorry

-- Theorem for the second condition
theorem perpendicular_line_condition :
  perpendicular_line B.1 B.2 ∧
  ∀ (x y : ℝ), perpendicular_line x y ↔ ∃ (k : ℝ), line2 (x + 2 * k) (y + k) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_condition_perpendicular_line_condition_l688_68882


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l688_68809

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≤ 5) ∨
               (5 = y - 2 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≤ x + 3)}

-- Define what it means for a set to be three rays with a common point
def is_three_rays_with_common_point (S : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, ∃ r₁ r₂ r₃ : Set (ℝ × ℝ),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    r₁ ∩ r₂ = {p} ∧ r₁ ∩ r₃ = {p} ∧ r₂ ∩ r₃ = {p} ∧
    (∀ q ∈ r₁, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (0, -1)) ∧
    (∀ q ∈ r₂, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (-1, 0)) ∧
    (∀ q ∈ r₃, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (1, 1))

-- State the theorem
theorem T_is_three_rays_with_common_point : is_three_rays_with_common_point T := by
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l688_68809


namespace NUMINAMATH_CALUDE_x_cube_minus_3x_eq_6_l688_68898

theorem x_cube_minus_3x_eq_6 (x : ℝ) (h : x^3 - 3*x = 6) :
  x^6 + 27*x^2 = 36*x^2 + 36*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_minus_3x_eq_6_l688_68898
