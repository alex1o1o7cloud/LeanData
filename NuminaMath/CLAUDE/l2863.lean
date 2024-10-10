import Mathlib

namespace rice_mixture_cost_l2863_286302

/-- Given two varieties of rice mixed in a specific ratio to obtain a mixture with a known cost,
    this theorem proves that the cost of the first variety can be determined. -/
theorem rice_mixture_cost
  (cost_second : ℝ)  -- Cost per kg of the second variety of rice
  (ratio : ℝ)        -- Ratio of the first variety to the second in the mixture
  (cost_mixture : ℝ) -- Cost per kg of the resulting mixture
  (h1 : cost_second = 8.75)
  (h2 : ratio = 0.8333333333333334)
  (h3 : cost_mixture = 7.50)
  : ∃ (cost_first : ℝ), 
    cost_first * (ratio / (1 + ratio)) + cost_second * (1 / (1 + ratio)) = cost_mixture ∧ 
    cost_first = 7.25 := by
  sorry


end rice_mixture_cost_l2863_286302


namespace dog_tail_length_l2863_286399

/-- Represents the length of a dog's body parts and total length --/
structure DogMeasurements where
  body : ℝ
  head : ℝ
  tail : ℝ
  total : ℝ

/-- Theorem stating the tail length of a dog given specific proportions --/
theorem dog_tail_length (d : DogMeasurements) 
  (h1 : d.tail = d.body / 2)
  (h2 : d.head = d.body / 6)
  (h3 : d.total = d.body + d.head + d.tail)
  (h4 : d.total = 30) : 
  d.tail = 9 := by
  sorry

end dog_tail_length_l2863_286399


namespace abc_inequality_l2863_286398

theorem abc_inequality (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a^x = b*c) (eq2 : b^y = c*a) (eq3 : c^z = a*b) :
  1/(2+x) + 1/(2+y) + 1/(2+z) ≤ 3/4 := by sorry

end abc_inequality_l2863_286398


namespace sin_90_degrees_l2863_286394

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l2863_286394


namespace complex_product_real_implies_m_equals_negative_one_l2863_286309

theorem complex_product_real_implies_m_equals_negative_one (m : ℂ) : 
  (∃ (r : ℝ), (m^2 + Complex.I) * (1 + m * Complex.I) = r) → m = -1 := by
  sorry

end complex_product_real_implies_m_equals_negative_one_l2863_286309


namespace clea_escalator_ride_time_l2863_286329

/-- Represents the escalator scenario for Clea -/
structure EscalatorScenario where
  /-- Time (in seconds) for Clea to walk down a stationary escalator -/
  stationary_time : ℝ
  /-- Time (in seconds) for Clea to walk down a moving escalator -/
  moving_time : ℝ
  /-- Slowdown factor for the escalator during off-peak hours -/
  slowdown_factor : ℝ

/-- Calculates the time for Clea to ride the slower escalator without walking -/
def ride_time (scenario : EscalatorScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that given the specific scenario, the ride time is 60 seconds -/
theorem clea_escalator_ride_time :
  let scenario : EscalatorScenario :=
    { stationary_time := 80
      moving_time := 30
      slowdown_factor := 0.8 }
  ride_time scenario = 60 := by
  sorry

end clea_escalator_ride_time_l2863_286329


namespace function_and_tangent_line_properties_l2863_286366

noncomputable section

-- Define the constant e
def e : ℝ := Real.exp 1

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2

-- Define the tangent line function
def tangentLine (b : ℝ) (x : ℝ) : ℝ := (e - 2) * x + b

theorem function_and_tangent_line_properties :
  ∃ (a b : ℝ),
    (∀ x : ℝ, tangentLine b x = (Real.exp 1 - f a 1) + (Real.exp 1 - 2 * a) * (x - 1)) ∧
    a = 1 ∧
    b = 1 ∧
    (∀ x : ℝ, x ≥ 0 → f a x > x^2 + 4*x - 14) :=
sorry

end function_and_tangent_line_properties_l2863_286366


namespace fourth_group_frequency_l2863_286356

theorem fourth_group_frequency 
  (groups : Fin 6 → ℝ) 
  (first_three_sum : (groups 0) + (groups 1) + (groups 2) = 0.65)
  (last_two_sum : (groups 4) + (groups 5) = 0.32)
  (all_sum_to_one : (groups 0) + (groups 1) + (groups 2) + (groups 3) + (groups 4) + (groups 5) = 1) :
  groups 3 = 0.03 := by
  sorry

end fourth_group_frequency_l2863_286356


namespace max_vertex_sum_l2863_286360

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h_T_pos : T > 0
  h_passes_through : ∀ x y, y = a * x * (x - T) → 
    ((x = 0 ∧ y = 0) ∨ (x = T ∧ y = 0) ∨ (x = T + 1 ∧ y = 50))

/-- The sum of coordinates of the vertex of the parabola -/
def vertexSum (p : Parabola) : ℚ :=
  p.T / 2 - (p.a * p.T^2) / 4

/-- The theorem stating the maximum value of the vertex sum -/
theorem max_vertex_sum (p : Parabola) : 
  vertexSum p ≤ -23/2 :=
sorry

end max_vertex_sum_l2863_286360


namespace pythagorean_squares_area_l2863_286391

theorem pythagorean_squares_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_sum_areas : a^2 + b^2 + 2*c^2 = 500) : c^2 = 500/3 :=
by sorry

end pythagorean_squares_area_l2863_286391


namespace work_completion_time_l2863_286385

theorem work_completion_time (x : ℝ) 
  (h1 : x > 0) 
  (h2 : 1/x + 1/18 = 1/6) : x = 9 := by
  sorry

end work_completion_time_l2863_286385


namespace caravan_feet_head_difference_l2863_286387

theorem caravan_feet_head_difference : 
  let num_hens : ℕ := 50
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let feet_per_hen : ℕ := 2
  let feet_per_goat : ℕ := 4
  let feet_per_camel : ℕ := 4
  let feet_per_keeper : ℕ := 2
  let total_heads : ℕ := num_hens + num_goats + num_camels + num_keepers
  let total_feet : ℕ := num_hens * feet_per_hen + num_goats * feet_per_goat + 
                        num_camels * feet_per_camel + num_keepers * feet_per_keeper
  total_feet - total_heads = 224 := by
sorry

end caravan_feet_head_difference_l2863_286387


namespace friends_at_reception_l2863_286370

def wedding_reception (total_guests : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : ℕ :=
  total_guests - 2 * (bride_couples + groom_couples)

theorem friends_at_reception :
  wedding_reception 300 30 30 = 180 := by
  sorry

end friends_at_reception_l2863_286370


namespace angle_terminal_side_formula_l2863_286363

/-- Given a point P(-4,3) on the terminal side of angle α, prove that 2sin α + cos α = 2/5 -/
theorem angle_terminal_side_formula (α : Real) (P : ℝ × ℝ) : 
  P = (-4, 3) → 2 * Real.sin α + Real.cos α = 2/5 := by
  sorry

end angle_terminal_side_formula_l2863_286363


namespace complex_purely_imaginary_l2863_286337

theorem complex_purely_imaginary (a : ℝ) :
  (a = 1 → ∃ (z : ℂ), z = (a - 1) * (a + 2) + (a + 3) * I ∧ z.re = 0) ∧
  (∃ (b : ℝ), b ≠ 1 ∧ ∃ (z : ℂ), z = (b - 1) * (b + 2) + (b + 3) * I ∧ z.re = 0) :=
by sorry

end complex_purely_imaginary_l2863_286337


namespace two_digit_number_property_l2863_286335

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →  -- N is a two-digit number
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →  -- Property condition
  (N = 32 ∨ N = 64 ∨ N = 96) := by
sorry

end two_digit_number_property_l2863_286335


namespace quadratic_polynomial_discriminant_l2863_286311

theorem quadratic_polynomial_discriminant 
  (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : ∃! x, a * x^2 + b * x + c = x - 2) 
  (h2 : ∃! x, a * x^2 + b * x + c = 1 - x / 2) : 
  b^2 - 4*a*c = -1/2 := by
sorry

end quadratic_polynomial_discriminant_l2863_286311


namespace max_complex_norm_squared_l2863_286390

theorem max_complex_norm_squared (θ : ℝ) : 
  let z : ℂ := 2 * Complex.cos θ + Complex.I * Complex.sin θ
  ∃ (M : ℝ), M = 4 ∧ ∀ θ' : ℝ, Complex.normSq z ≤ M :=
sorry

end max_complex_norm_squared_l2863_286390


namespace inequality_representation_l2863_286321

theorem inequality_representation (x : ℝ) : 
  (x + 4 < 10) ↔ (∃ y, y = x + 4 ∧ y < 10) :=
sorry

end inequality_representation_l2863_286321


namespace sick_days_per_year_l2863_286339

/-- Represents the number of hours in a workday -/
def hoursPerDay : ℕ := 8

/-- Represents the number of hours remaining after using half of the allotment -/
def remainingHours : ℕ := 80

/-- Theorem stating that the number of sick days per year is 20 -/
theorem sick_days_per_year :
  ∀ (sickDays vacationDays : ℕ),
  sickDays = vacationDays →
  sickDays + vacationDays = 2 * (remainingHours / hoursPerDay) →
  sickDays = 20 := by sorry

end sick_days_per_year_l2863_286339


namespace consecutive_missing_factors_l2863_286364

theorem consecutive_missing_factors (n : ℕ) (h1 : n > 30) : 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 → (k ≠ 16 ∧ k ≠ 17 → n % k = 0)) →
  (∃ (m : ℕ), m ≥ 1 ∧ m < 30 ∧ n % m ≠ 0 ∧ n % (m + 1) ≠ 0) →
  (∀ (j : ℕ), j ≥ 1 ∧ j < 30 ∧ n % j ≠ 0 ∧ n % (j + 1) ≠ 0 → j = 16) :=
by sorry

end consecutive_missing_factors_l2863_286364


namespace certain_number_proof_l2863_286357

theorem certain_number_proof (x : ℝ) : 
  (3 - (1/5) * x) - (4 - (1/7) * 210) = 114 → x = -425 :=
by
  sorry

end certain_number_proof_l2863_286357


namespace correct_sacks_per_day_l2863_286312

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def days_of_harvest : ℕ := 49

/-- The total number of sacks after the harvest period -/
def total_sacks : ℕ := 1862

/-- The number of oranges in each sack -/
def oranges_per_sack : ℕ := 42

/-- Theorem stating that the number of sacks harvested per day is correct -/
theorem correct_sacks_per_day : 
  sacks_per_day * days_of_harvest = total_sacks :=
sorry

end correct_sacks_per_day_l2863_286312


namespace max_value_of_a_l2863_286361

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a*x + 1) / x

def g (x : ℝ) : ℝ := Real.exp x - Real.log x + 2*x^2 + 1

theorem max_value_of_a (h : ∀ x > 0, x * f x a ≤ g x) :
  a ≤ Real.exp 1 + 1 :=
sorry

end

end max_value_of_a_l2863_286361


namespace total_shells_is_61_l2863_286377

def bucket_a_initial : ℕ := 5
def bucket_a_additional : ℕ := 12

def bucket_b_initial : ℕ := 8
def bucket_b_additional : ℕ := 15

def bucket_c_initial : ℕ := 3
def bucket_c_additional : ℕ := 18

def total_shells : ℕ := 
  (bucket_a_initial + bucket_a_additional) + 
  (bucket_b_initial + bucket_b_additional) + 
  (bucket_c_initial + bucket_c_additional)

theorem total_shells_is_61 : total_shells = 61 := by
  sorry

end total_shells_is_61_l2863_286377


namespace system_of_equations_solution_transformed_system_solution_l2863_286332

theorem system_of_equations_solution (x y : ℝ) :
  x + 2*y = 9 ∧ 2*x + y = 6 → x - y = -3 ∧ x + y = 5 := by sorry

theorem transformed_system_solution (m n x y : ℝ) :
  (m = 5 ∧ n = 4 ∧ 2*m - 3*n = -2 ∧ 3*m + 5*n = 35) →
  (2*(x+2) - 3*(y-1) = -2 ∧ 3*(x+2) + 5*(y-1) = 35 → x = 3 ∧ y = 5) := by sorry

end system_of_equations_solution_transformed_system_solution_l2863_286332


namespace saline_drip_duration_l2863_286324

/-- Calculates the duration of a saline drip treatment -/
theorem saline_drip_duration 
  (drop_rate : ℕ) 
  (drops_per_ml : ℚ) 
  (total_volume : ℚ) : 
  drop_rate = 20 →
  drops_per_ml = 100 / 5 →
  total_volume = 120 →
  (total_volume * drops_per_ml / drop_rate) / 60 = 2 := by
  sorry

end saline_drip_duration_l2863_286324


namespace cookie_problem_l2863_286376

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The number of bags -/
def num_bags : ℕ := 9

/-- The difference in cookies between boxes and bags -/
def cookie_difference : ℕ := 33

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

theorem cookie_problem :
  cookies_per_box * num_boxes = cookies_per_bag * num_bags + cookie_difference :=
by sorry

end cookie_problem_l2863_286376


namespace sum_c_plus_d_l2863_286340

theorem sum_c_plus_d (a b c d : ℤ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : a + d = 8) : 
  c + d = 3 := by
sorry

end sum_c_plus_d_l2863_286340


namespace initial_trees_count_l2863_286313

/-- The number of oak trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of oak trees cut down -/
def cut_trees : ℕ := 2

/-- The number of oak trees remaining after cutting -/
def remaining_trees : ℕ := 7

/-- Theorem stating that the initial number of trees is 9 -/
theorem initial_trees_count : initial_trees = 9 := by sorry

end initial_trees_count_l2863_286313


namespace circle_tangent_lines_l2863_286330

/-- Given a circle with equation (x-1)^2 + (y+3)^2 = 4 and a point (-1, -1),
    the tangent lines from this point to the circle have equations x = -1 or y = -1 -/
theorem circle_tangent_lines (x y : ℝ) :
  let circle := (x - 1)^2 + (y + 3)^2 = 4
  let point := ((-1 : ℝ), (-1 : ℝ))
  let tangent1 := x = -1
  let tangent2 := y = -1
  (∃ (t : ℝ), circle ∧ (tangent1 ∨ tangent2) ∧
    (point.1 = t ∧ point.2 = -1) ∨ (point.1 = -1 ∧ point.2 = t)) :=
by sorry

end circle_tangent_lines_l2863_286330


namespace sin_4phi_value_l2863_286320

theorem sin_4phi_value (φ : ℝ) : 
  Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (4 * φ) = 12 * Real.sqrt 8 / 625 := by
sorry

end sin_4phi_value_l2863_286320


namespace multiply_subtract_equal_computation_result_l2863_286351

theorem multiply_subtract_equal (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem computation_result : 65 * 1515 - 25 * 1515 = 60600 := by sorry

end multiply_subtract_equal_computation_result_l2863_286351


namespace number_of_happy_arrangements_l2863_286318

/-- Represents the types of chains --/
inductive Chain : Type
| Silver : Chain
| Gold : Chain
| Iron : Chain

/-- Represents the types of stones --/
inductive Stone : Type
| CubicZirconia : Stone
| Emerald : Stone
| Quartz : Stone

/-- Represents the types of pendants --/
inductive Pendant : Type
| Star : Pendant
| Sun : Pendant
| Moon : Pendant

/-- Represents a piece of jewelry --/
structure Jewelry :=
  (chain : Chain)
  (stone : Stone)
  (pendant : Pendant)

/-- Represents an arrangement of three pieces of jewelry --/
structure Arrangement :=
  (left : Jewelry)
  (middle : Jewelry)
  (right : Jewelry)

/-- Predicate to check if an arrangement satisfies Polina's conditions --/
def satisfiesConditions (arr : Arrangement) : Prop :=
  (arr.middle.chain = Chain.Iron ∧ arr.middle.pendant = Pendant.Sun) ∧
  ((arr.left.chain = Chain.Gold ∧ arr.right.chain = Chain.Silver) ∨
   (arr.left.chain = Chain.Silver ∧ arr.right.chain = Chain.Gold)) ∧
  (arr.left.stone ≠ arr.middle.stone ∧ arr.left.stone ≠ arr.right.stone ∧ arr.middle.stone ≠ arr.right.stone) ∧
  (arr.left.pendant ≠ arr.middle.pendant ∧ arr.left.pendant ≠ arr.right.pendant ∧ arr.middle.pendant ≠ arr.right.pendant) ∧
  (arr.left.chain ≠ arr.middle.chain ∧ arr.left.chain ≠ arr.right.chain ∧ arr.middle.chain ≠ arr.right.chain)

/-- The theorem to be proved --/
theorem number_of_happy_arrangements :
  ∃! (n : ℕ), ∃ (arrangements : Finset Arrangement),
    arrangements.card = n ∧
    (∀ arr ∈ arrangements, satisfiesConditions arr) ∧
    (∀ arr : Arrangement, satisfiesConditions arr → arr ∈ arrangements) :=
sorry

end number_of_happy_arrangements_l2863_286318


namespace hulk_jump_exceeds_2km_l2863_286367

def hulk_jump (n : ℕ) : ℝ :=
  if n = 0 then 0.5 else 2^(n - 1)

theorem hulk_jump_exceeds_2km : 
  (∀ k < 13, hulk_jump k ≤ 2000) ∧ 
  hulk_jump 13 > 2000 := by sorry

end hulk_jump_exceeds_2km_l2863_286367


namespace distance_between_points_l2863_286334

/-- The distance between two points A(-1, 2) and B(-4, 6) is 5. -/
theorem distance_between_points : 
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (-4, 6)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by
  sorry

end distance_between_points_l2863_286334


namespace f_monotone_increasing_l2863_286306

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(2/3) + m * x + 1

-- State the theorem
theorem f_monotone_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →  -- f is an even function
  ∀ x ≥ 0, ∀ y ≥ x, f m x ≤ f m y :=  -- f is monotonically increasing on [0, +∞)
by
  sorry

end f_monotone_increasing_l2863_286306


namespace solve_family_income_problem_l2863_286386

def family_income_problem (initial_members : ℕ) (initial_average : ℝ) 
  (final_members : ℕ) (final_average : ℝ) : Prop :=
  let initial_total := initial_members * initial_average
  let final_total := final_members * final_average
  let deceased_income := initial_total - final_total
  initial_members = 4 ∧ 
  final_members = 3 ∧ 
  initial_average = 782 ∧ 
  final_average = 650 ∧ 
  deceased_income = 1178

theorem solve_family_income_problem : 
  ∃ (initial_members final_members : ℕ) (initial_average final_average : ℝ),
    family_income_problem initial_members initial_average final_members final_average :=
by
  sorry

end solve_family_income_problem_l2863_286386


namespace thirteenth_row_sum_l2863_286354

def row_sum (n : ℕ) : ℕ :=
  3 * 2^(n-1)

theorem thirteenth_row_sum :
  row_sum 13 = 12288 :=
by sorry

end thirteenth_row_sum_l2863_286354


namespace largest_inscribed_triangle_area_for_radius_6_l2863_286307

/-- The area of the largest possible triangle inscribed in a circle,
    where one side of the triangle is a diameter of the circle. -/
def largest_inscribed_triangle_area (r : ℝ) : ℝ :=
  2 * r * r

theorem largest_inscribed_triangle_area_for_radius_6 :
  largest_inscribed_triangle_area 6 = 36 := by
  sorry

#eval largest_inscribed_triangle_area 6

end largest_inscribed_triangle_area_for_radius_6_l2863_286307


namespace quadrilaterals_on_circle_l2863_286380

/-- The number of convex quadrilaterals formed by 15 points on a circle -/
theorem quadrilaterals_on_circle (n : ℕ) (h : n = 15) : 
  Nat.choose n 4 = 1365 :=
by sorry

end quadrilaterals_on_circle_l2863_286380


namespace units_digit_of_quotient_l2863_286379

theorem units_digit_of_quotient (n : ℕ) : (4^1993 + 5^1993) % 9 = 0 :=
by sorry

end units_digit_of_quotient_l2863_286379


namespace x_squared_coefficient_of_product_l2863_286348

/-- The coefficient of x^2 in the expansion of (3x^2 + 4x + 5)(6x^2 + 7x + 8) is 82 -/
theorem x_squared_coefficient_of_product : 
  let p₁ : Polynomial ℝ := 3 * X^2 + 4 * X + 5
  let p₂ : Polynomial ℝ := 6 * X^2 + 7 * X + 8
  (p₁ * p₂).coeff 2 = 82 := by
sorry

end x_squared_coefficient_of_product_l2863_286348


namespace smallest_n_congruence_l2863_286308

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ k : ℕ+, k < n → ¬(1023 * k.val ≡ 2147 * k.val [ZMOD 30])) ∧ 
  (1023 * n.val ≡ 2147 * n.val [ZMOD 30]) := by
  sorry

end smallest_n_congruence_l2863_286308


namespace pyramid_height_formula_l2863_286350

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base -/
  base_perimeter : ℝ
  /-- The distance from the apex to each vertex of the square base -/
  apex_to_vertex : ℝ

/-- The height of the pyramid from its peak to the center of the square base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

theorem pyramid_height_formula (p : RightPyramid) 
  (h1 : p.base_perimeter = 40)
  (h2 : p.apex_to_vertex = 15) : 
  pyramid_height p = 5 * Real.sqrt 7 := by
  sorry

end pyramid_height_formula_l2863_286350


namespace no_harmonic_point_reciprocal_unique_harmonic_point_range_of_m_l2863_286315

-- Definition of a harmonic point
def is_harmonic_point (x y : ℝ) : Prop := x = y

-- Part 1: No harmonic point for y = -4/x
theorem no_harmonic_point_reciprocal : ¬∃ x : ℝ, x ≠ 0 ∧ is_harmonic_point x (-4/x) := by sorry

-- Part 2: Quadratic function with one harmonic point
theorem unique_harmonic_point (a c : ℝ) :
  a ≠ 0 ∧ 
  (∃! x : ℝ, is_harmonic_point x (a * x^2 + 6 * x + c)) ∧
  is_harmonic_point (5/2) (a * (5/2)^2 + 6 * (5/2) + c) →
  a = -1 ∧ c = -25/4 := by sorry

-- Part 3: Range of m for quadratic function with given min and max
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 ≥ -1) ∧
  (∀ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 ≤ 3) ∧
  (∃ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 = -1) ∧
  (∃ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 = 3) →
  3 ≤ m ∧ m ≤ 5 := by sorry

end no_harmonic_point_reciprocal_unique_harmonic_point_range_of_m_l2863_286315


namespace no_numbers_seven_times_digit_sum_l2863_286355

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem no_numbers_seven_times_digit_sum : 
  ∀ n : ℕ, n > 0 ∧ n < 10000 → n ≠ 7 * (sum_of_digits n) :=
by
  sorry

end no_numbers_seven_times_digit_sum_l2863_286355


namespace fraction_comparison_l2863_286389

def first_numerator : ℕ := 100^99
def first_denominator : ℕ := 9777777  -- 97...7 with 7 digits

def second_numerator : ℕ := 55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555  -- 100 digits of 5
def second_denominator : ℕ := 55555  -- 5 digits of 5

theorem fraction_comparison :
  (first_numerator : ℚ) / first_denominator < (second_numerator : ℚ) / second_denominator :=
sorry

end fraction_comparison_l2863_286389


namespace p_sufficient_not_necessary_for_q_l2863_286378

-- Define propositions p and q
def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x y : ℝ) : Prop := ¬(x = -1 ∧ y = -1)

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end p_sufficient_not_necessary_for_q_l2863_286378


namespace g_value_at_4_l2863_286347

/-- The cubic polynomial f(x) = x^3 - 2x + 5 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 5

/-- g is a cubic polynomial such that g(0) = 1 and its roots are the squares of the roots of f -/
def g : ℝ → ℝ :=
  sorry

theorem g_value_at_4 : g 4 = -9/25 := by
  sorry

end g_value_at_4_l2863_286347


namespace sum_of_angles_l2863_286341

/-- The number of 90-degree angles in a rectangle -/
def rectangle_angles : ℕ := 4

/-- The number of 90-degree angles in a square -/
def square_angles : ℕ := 4

/-- The sum of 90-degree angles in a rectangle and a square -/
def total_angles : ℕ := rectangle_angles + square_angles

theorem sum_of_angles : total_angles = 8 := by
  sorry

end sum_of_angles_l2863_286341


namespace all_gp_lines_pass_through_origin_l2863_286316

/-- A line in 2D space defined by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The set of all lines where a, b, c form a geometric progression -/
def GPLines : Set Line :=
  {l : Line | isGeometricProgression l.a l.b l.c}

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

theorem all_gp_lines_pass_through_origin :
  ∀ l ∈ GPLines, pointOnLine ⟨0, 0⟩ l :=
sorry

end all_gp_lines_pass_through_origin_l2863_286316


namespace common_external_tangent_y_intercept_l2863_286358

def circle1_center : ℝ × ℝ := (3, 3)
def circle2_center : ℝ × ℝ := (15, 10)
def circle1_radius : ℝ := 5
def circle2_radius : ℝ := 10

theorem common_external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), y = m * x + b → 
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 = circle1_radius^2 ∨
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2) →
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 > circle1_radius^2 ∧
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 > circle2_radius^2) ∨
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 < circle1_radius^2 ∧
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 < circle2_radius^2)) ∧
  b = 446 / 95 := by
sorry

end common_external_tangent_y_intercept_l2863_286358


namespace tens_digit_of_4032_pow_4033_minus_4036_l2863_286323

theorem tens_digit_of_4032_pow_4033_minus_4036 :
  (4032^4033 - 4036) % 100 / 10 = 9 := by sorry

end tens_digit_of_4032_pow_4033_minus_4036_l2863_286323


namespace donation_start_age_l2863_286381

def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_donation : ℕ := 440000

theorem donation_start_age :
  ∃ (start_age : ℕ),
    start_age = current_age - (total_donation / annual_donation) ∧
    start_age = 16 := by
  sorry

end donation_start_age_l2863_286381


namespace pascal_triangle_sum_l2863_286327

/-- The number of elements in a row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def sumOfElements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_sum :
  sumOfElements 29 = 465 := by sorry

end pascal_triangle_sum_l2863_286327


namespace strip_covering_theorem_l2863_286382

/-- A strip of width w -/
def Strip (w : ℝ) := Set (ℝ × ℝ)

/-- A set of points can be covered by a strip -/
def Coverable (S : Set (ℝ × ℝ)) (w : ℝ) :=
  ∃ (strip : Strip w), S ⊆ strip

/-- Main theorem -/
theorem strip_covering_theorem (S : Set (ℝ × ℝ)) (n : ℕ) 
  (h1 : Fintype S)
  (h2 : Fintype.card S = n)
  (h3 : n ≥ 3)
  (h4 : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → 
    Coverable {A, B, C} 1) :
  Coverable S 2 := by
  sorry

end strip_covering_theorem_l2863_286382


namespace fractional_inequality_solution_set_l2863_286310

theorem fractional_inequality_solution_set (x : ℝ) :
  (x + 1) / (x + 2) ≥ 0 ↔ (x ≥ -1 ∨ x < -2) ∧ x ≠ -2 := by
  sorry

end fractional_inequality_solution_set_l2863_286310


namespace floor_sqrt_26_squared_l2863_286325

theorem floor_sqrt_26_squared : ⌊Real.sqrt 26⌋^2 = 25 := by
  sorry

end floor_sqrt_26_squared_l2863_286325


namespace fake_coin_determinable_l2863_286326

/-- Represents the result of a weighing on a two-pan balance scale -/
inductive WeighingResult
  | Left : WeighingResult  -- Left pan is heavier
  | Right : WeighingResult -- Right pan is heavier
  | Equal : WeighingResult -- Pans are balanced

/-- Represents the state of a coin -/
inductive CoinState
  | Normal : CoinState
  | Heavier : CoinState
  | Lighter : CoinState

/-- Represents a weighing on a two-pan balance scale -/
def Weighing := (Fin 25 → Bool) → WeighingResult

/-- Represents the strategy for determining the state of the fake coin -/
def Strategy := Weighing → Weighing → CoinState

/-- Theorem stating that it's possible to determine whether the fake coin
    is lighter or heavier using only two weighings -/
theorem fake_coin_determinable :
  ∃ (s : Strategy),
    ∀ (fake : Fin 25) (state : CoinState),
      state ≠ CoinState.Normal →
        ∀ (w₁ w₂ : Weighing),
          (∀ (f : Fin 25 → Bool),
            w₁ f = WeighingResult.Left ↔ (state = CoinState.Heavier ∧ f fake) ∨
                                         (state = CoinState.Lighter ∧ ¬f fake)) →
          (∀ (f : Fin 25 → Bool),
            w₂ f = WeighingResult.Left ↔ (state = CoinState.Heavier ∧ f fake) ∨
                                         (state = CoinState.Lighter ∧ ¬f fake)) →
          s w₁ w₂ = state :=
sorry

end fake_coin_determinable_l2863_286326


namespace min_value_theorem_min_value_achievable_l2863_286319

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3*b = 1) :
  1/a + 3/b ≥ 16 :=
sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3*b = 1 ∧ 1/a + 3/b = 16 :=
sorry

end min_value_theorem_min_value_achievable_l2863_286319


namespace imaginary_part_of_complex_fraction_l2863_286362

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + 3*Complex.I) / (3 - Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2863_286362


namespace triangle_side_sharing_l2863_286393

/-- A point on a circle -/
structure Point

/-- A triangle formed by three points -/
structure Triangle (Point : Type) where
  p1 : Point
  p2 : Point
  p3 : Point

/-- A side of a triangle -/
structure Side (Point : Type) where
  p1 : Point
  p2 : Point

/-- Definition of 8 points on a circle -/
def circle_points : Finset Point := sorry

/-- Definition of all possible triangles formed by the 8 points -/
def all_triangles : Finset (Triangle Point) := sorry

/-- Definition of all possible sides formed by the 8 points -/
def all_sides : Finset (Side Point) := sorry

/-- Function to get the sides of a triangle -/
def triangle_sides (t : Triangle Point) : Finset (Side Point) := sorry

theorem triangle_side_sharing :
  ∀ (triangles : Finset (Triangle Point)),
    triangles ⊆ all_triangles →
    triangles.card = 9 →
    ∃ (t1 t2 : Triangle Point) (s : Side Point),
      t1 ∈ triangles ∧ t2 ∈ triangles ∧ t1 ≠ t2 ∧
      s ∈ triangle_sides t1 ∧ s ∈ triangle_sides t2 :=
sorry

end triangle_side_sharing_l2863_286393


namespace rental_car_distance_l2863_286388

theorem rental_car_distance (fixed_fee : ℝ) (per_km_charge : ℝ) (total_bill : ℝ) (km_travelled : ℝ) : 
  fixed_fee = 45 →
  per_km_charge = 0.12 →
  total_bill = 74.16 →
  total_bill = fixed_fee + per_km_charge * km_travelled →
  km_travelled = 243 := by
sorry

end rental_car_distance_l2863_286388


namespace cubic_factorization_l2863_286368

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end cubic_factorization_l2863_286368


namespace specialist_time_calculation_l2863_286395

theorem specialist_time_calculation (days_in_hospital : ℕ) (bed_charge_per_day : ℕ) 
  (specialist_charge_per_hour : ℕ) (ambulance_charge : ℕ) (total_bill : ℕ) : 
  days_in_hospital = 3 →
  bed_charge_per_day = 900 →
  specialist_charge_per_hour = 250 →
  ambulance_charge = 1800 →
  total_bill = 4625 →
  (total_bill - (days_in_hospital * bed_charge_per_day + ambulance_charge)) / 
    (2 * (specialist_charge_per_hour / 60)) = 15 := by
  sorry

end specialist_time_calculation_l2863_286395


namespace nested_square_root_18_l2863_286304

theorem nested_square_root_18 :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (18 + x) → x = 6 := by sorry

end nested_square_root_18_l2863_286304


namespace trajectory_equation_l2863_286322

theorem trajectory_equation (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -2 → x^2 + y^2 / 2 = 1 := by
  sorry

end trajectory_equation_l2863_286322


namespace jungkook_has_bigger_number_l2863_286359

theorem jungkook_has_bigger_number :
  let yoongi_collected : ℕ := 4
  let jungkook_collected : ℕ := 6 + 3
  jungkook_collected > yoongi_collected :=
by
  sorry

end jungkook_has_bigger_number_l2863_286359


namespace complex_magnitude_l2863_286336

theorem complex_magnitude (z : ℂ) : z * (1 - Complex.I) = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l2863_286336


namespace ken_share_l2863_286300

def total_amount : ℕ := 5250

theorem ken_share (ken : ℕ) (tony : ℕ) 
  (h1 : ken + tony = total_amount) 
  (h2 : tony = 2 * ken) : 
  ken = 1750 := by
  sorry

end ken_share_l2863_286300


namespace perpendicular_lines_l2863_286375

/-- The slope of the line 2x - 3y + 5 = 0 -/
def m₁ : ℚ := 2 / 3

/-- The slope of the line bx - 3y + 1 = 0 -/
def m₂ (b : ℚ) : ℚ := b / 3

/-- The condition for perpendicular lines -/
def perpendicular (m₁ m₂ : ℚ) : Prop := m₁ * m₂ = -1

theorem perpendicular_lines (b : ℚ) : 
  perpendicular m₁ (m₂ b) → b = -9/2 := by
  sorry

end perpendicular_lines_l2863_286375


namespace intersection_of_A_and_B_l2863_286396

def A : Set Int := {-1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end intersection_of_A_and_B_l2863_286396


namespace fraction_zero_implies_x_equals_one_l2863_286349

theorem fraction_zero_implies_x_equals_one :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_equals_one_l2863_286349


namespace walkers_speed_l2863_286303

/-- Proves that a walker's speed is 5 mph given specific conditions involving a cyclist --/
theorem walkers_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (walker_catchup_time : ℝ) : ℝ :=
  let walker_speed : ℝ :=
    (cyclist_speed * cyclist_travel_time) / walker_catchup_time
  by
    sorry

#check walkers_speed 20 (5/60) (20/60) = 5

end walkers_speed_l2863_286303


namespace degree_of_g_l2863_286353

/-- Given polynomials f and g, where h(x) = f(g(x)) + g(x), 
    the degree of h(x) is 6, and the degree of f(x) is 3, 
    then the degree of g(x) is 2. -/
theorem degree_of_g (f g h : Polynomial ℝ) :
  (∀ x, h.eval x = (f.comp g).eval x + g.eval x) →
  h.degree = 6 →
  f.degree = 3 →
  g.degree = 2 := by
sorry

end degree_of_g_l2863_286353


namespace sum_inequality_l2863_286342

theorem sum_inequality (t1 t2 t3 t4 t5 : ℝ) :
  (1 - t1) * Real.exp t1 +
  (1 - t2) * Real.exp (t1 + t2) +
  (1 - t3) * Real.exp (t1 + t2 + t3) +
  (1 - t4) * Real.exp (t1 + t2 + t3 + t4) +
  (1 - t5) * Real.exp (t1 + t2 + t3 + t4 + t5) ≤ Real.exp (Real.exp (Real.exp (Real.exp 1))) := by
  sorry

end sum_inequality_l2863_286342


namespace equation_solution_l2863_286338

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 4 ∧ x₂ = -6) ∧ 
  (∀ x : ℝ, 2 * (x + 1)^2 - 49 = 1 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l2863_286338


namespace equal_share_of_candles_total_divisible_by_four_l2863_286343

/- Define the number of candles for each person -/
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

/- Define the total number of candles -/
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles

/- The theorem to prove -/
theorem equal_share_of_candles : total_candles / 4 = 37 := by
  sorry

/- Additional helper theorem to show the total is divisible by 4 -/
theorem total_divisible_by_four : total_candles % 4 = 0 := by
  sorry

end equal_share_of_candles_total_divisible_by_four_l2863_286343


namespace number_calculation_l2863_286345

theorem number_calculation (x : ℝ) : (0.8 * 90 = 0.7 * x + 30) → x = 60 := by
  sorry

end number_calculation_l2863_286345


namespace count_400000_to_500000_by_50_l2863_286373

def count_sequence (start : ℕ) (increment : ℕ) (end_value : ℕ) : ℕ :=
  (end_value - start) / increment + 1

theorem count_400000_to_500000_by_50 :
  count_sequence 400000 50 500000 = 2000 := by
  sorry

end count_400000_to_500000_by_50_l2863_286373


namespace football_club_balance_l2863_286305

/-- Calculates the final balance of a football club after player transactions -/
def final_balance (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) 
  (players_bought : ℕ) (buying_price : ℝ) : ℝ :=
  initial_balance + players_sold * selling_price - players_bought * buying_price

/-- Theorem: The final balance of the football club is $60 million -/
theorem football_club_balance : 
  final_balance 100 2 10 4 15 = 60 := by
  sorry

end football_club_balance_l2863_286305


namespace sqrt_21000_l2863_286317

theorem sqrt_21000 (h : Real.sqrt 2.1 = 1.449) : Real.sqrt 21000 = 144.9 := by
  sorry

end sqrt_21000_l2863_286317


namespace girls_average_score_l2863_286384

-- Define the variables
def num_girls : ℝ := 1
def num_boys : ℝ := 1.8 * num_girls
def class_average : ℝ := 75
def girls_score_ratio : ℝ := 1.2

-- Theorem statement
theorem girls_average_score :
  ∃ (girls_score : ℝ),
    girls_score * num_girls + (girls_score / girls_score_ratio) * num_boys = 
    class_average * (num_girls + num_boys) ∧
    girls_score = 84 := by
  sorry

end girls_average_score_l2863_286384


namespace exponent_division_l2863_286374

theorem exponent_division (x : ℝ) : x^10 / x^2 = x^8 := by
  sorry

end exponent_division_l2863_286374


namespace apple_stack_count_l2863_286346

def pyramid_stack (base_length : ℕ) (base_width : ℕ) : ℕ :=
  let layers := List.range (base_length - 1)
  let regular_layers := layers.map (λ i => (base_length - i) * (base_width - i))
  let top_layer := 2
  regular_layers.sum + top_layer

theorem apple_stack_count : pyramid_stack 6 9 = 156 := by
  sorry

end apple_stack_count_l2863_286346


namespace rachel_rona_age_ratio_l2863_286331

/-- Given the ages of Rachel, Rona, and Collete, prove that the ratio of Rachel's age to Rona's age is 2:1 -/
theorem rachel_rona_age_ratio (rachel_age rona_age collete_age : ℕ) : 
  rachel_age > rona_age →
  collete_age = rona_age / 2 →
  rona_age = 8 →
  rachel_age - collete_age = 12 →
  rachel_age / rona_age = 2 := by
  sorry


end rachel_rona_age_ratio_l2863_286331


namespace inequality_solution_set_l2863_286397

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) → a > 1 := by
  sorry

end inequality_solution_set_l2863_286397


namespace female_students_like_pe_l2863_286301

def total_students : ℕ := 1500
def male_percentage : ℚ := 2/5
def female_dislike_pe_percentage : ℚ := 13/20

theorem female_students_like_pe : 
  (total_students : ℚ) * (1 - male_percentage) * (1 - female_dislike_pe_percentage) = 315 := by
  sorry

end female_students_like_pe_l2863_286301


namespace tinplate_allocation_l2863_286344

theorem tinplate_allocation (total_tinplates : ℕ) 
  (bodies_per_tinplate : ℕ) (bottoms_per_tinplate : ℕ) 
  (bodies_to_bottoms_ratio : ℚ) :
  total_tinplates = 36 →
  bodies_per_tinplate = 25 →
  bottoms_per_tinplate = 40 →
  bodies_to_bottoms_ratio = 1/2 →
  ∃ (bodies_tinplates bottoms_tinplates : ℕ),
    bodies_tinplates + bottoms_tinplates = total_tinplates ∧
    bodies_tinplates * bodies_per_tinplate * 2 = bottoms_tinplates * bottoms_per_tinplate ∧
    bodies_tinplates = 16 ∧
    bottoms_tinplates = 20 :=
by sorry

end tinplate_allocation_l2863_286344


namespace plan_b_more_economical_l2863_286352

/-- Proves that Plan B (fixed money spent) is more economical than Plan A (fixed amount of gasoline) for two refuelings with different prices. -/
theorem plan_b_more_economical (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (2 * x * y) / (x + y) < (x + y) / 2 := by
  sorry

#check plan_b_more_economical

end plan_b_more_economical_l2863_286352


namespace square_sum_from_diff_and_product_l2863_286333

theorem square_sum_from_diff_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end square_sum_from_diff_and_product_l2863_286333


namespace divisibility_property_l2863_286365

theorem divisibility_property (n : ℕ) : (n - 1) ∣ (n^2 + n - 2) := by
  sorry

end divisibility_property_l2863_286365


namespace school_size_calculation_l2863_286372

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  grade11_students : ℕ
  sample_size : ℕ
  grade10_sample : ℕ
  grade12_sample : ℕ

/-- The theorem stating the conditions and the conclusion to be proved -/
theorem school_size_calculation (s : School)
  (h1 : s.grade11_students = 600)
  (h2 : s.sample_size = 50)
  (h3 : s.grade10_sample = 15)
  (h4 : s.grade12_sample = 20) :
  s.total_students = 2000 := by
  sorry


end school_size_calculation_l2863_286372


namespace inequality_proof_l2863_286371

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 := by
  sorry

end inequality_proof_l2863_286371


namespace room_height_is_12_l2863_286328

def room_length : ℝ := 25
def room_width : ℝ := 15
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def whitewash_cost_per_sqft : ℝ := 2
def total_cost : ℝ := 1812

theorem room_height_is_12 (h : ℝ) :
  (2 * (room_length + room_width) * h - (door_area + num_windows * window_area)) * whitewash_cost_per_sqft = total_cost →
  h = 12 := by
  sorry

end room_height_is_12_l2863_286328


namespace hyperbola_standard_equation_l2863_286314

/-- The standard equation of a hyperbola with given asymptotes and passing through a specific point -/
theorem hyperbola_standard_equation (x y : ℝ) :
  (∀ (t : ℝ), y = (2/3) * t ∨ y = -(2/3) * t) →  -- Asymptotes condition
  (x = Real.sqrt 6 ∧ y = 2) →                    -- Point condition
  (3 * y^2 / 4) - (x^2 / 3) = 1 :=               -- Standard equation
by sorry

end hyperbola_standard_equation_l2863_286314


namespace triangle_inequality_for_specific_triangle_l2863_286392

/-- A triangle with sides of length 3, 4, and x is valid if and only if 1 < x < 7 -/
theorem triangle_inequality_for_specific_triangle (x : ℝ) :
  (3 + 4 > x ∧ 3 + x > 4 ∧ 4 + x > 3) ↔ (1 < x ∧ x < 7) := by
  sorry

end triangle_inequality_for_specific_triangle_l2863_286392


namespace nell_baseball_cards_count_l2863_286369

/-- Represents the number of cards Nell has --/
structure NellCards where
  initialBaseball : Nat
  initialAce : Nat
  currentAce : Nat
  baseballDifference : Nat

/-- Calculates the current number of baseball cards Nell has --/
def currentBaseballCards (cards : NellCards) : Nat :=
  cards.currentAce + cards.baseballDifference

/-- Theorem stating that Nell's current baseball cards equal 178 --/
theorem nell_baseball_cards_count (cards : NellCards) 
  (h1 : cards.initialBaseball = 438)
  (h2 : cards.initialAce = 18)
  (h3 : cards.currentAce = 55)
  (h4 : cards.baseballDifference = 123) :
  currentBaseballCards cards = 178 := by
  sorry


end nell_baseball_cards_count_l2863_286369


namespace proposition_condition_l2863_286383

theorem proposition_condition (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end proposition_condition_l2863_286383
