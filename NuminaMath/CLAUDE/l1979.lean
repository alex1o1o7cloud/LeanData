import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1979_197930

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 576) 
  (h2 : height = 18) : 
  area / height = 32 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1979_197930


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l1979_197957

/-- Calculates the total revenue from concert ticket sales given specific discount conditions -/
theorem concert_ticket_revenue : 
  let ticket_price : ℝ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_discount : ℝ := 0.4
  let second_discount : ℝ := 0.15
  let total_attendees : ℕ := 48

  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_attendees := total_attendees - first_group_size - second_group_size
  let full_price_revenue := remaining_attendees * ticket_price

  first_group_revenue + second_group_revenue + full_price_revenue = 820 :=
by
  sorry


end NUMINAMATH_CALUDE_concert_ticket_revenue_l1979_197957


namespace NUMINAMATH_CALUDE_magnitude_of_w_l1979_197958

theorem magnitude_of_w (w : ℂ) (h : w^2 = -7 + 24*I) : Complex.abs w = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_w_l1979_197958


namespace NUMINAMATH_CALUDE_health_drink_sales_correct_l1979_197966

/-- Represents the health drink inventory and sales data -/
structure HealthDrinkSales where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  unit_price_increase : ℝ
  selling_price : ℝ
  discounted_quantity : ℕ
  discount_rate : ℝ

/-- Calculates the quantity of the first batch and the total profit -/
def calculate_quantity_and_profit (sales : HealthDrinkSales) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem health_drink_sales_correct (sales : HealthDrinkSales) 
  (h1 : sales.first_batch_cost = 40000)
  (h2 : sales.second_batch_cost = 88000)
  (h3 : sales.unit_price_increase = 2)
  (h4 : sales.selling_price = 28)
  (h5 : sales.discounted_quantity = 100)
  (h6 : sales.discount_rate = 0.2) :
  let (quantity, profit) := calculate_quantity_and_profit sales
  quantity = 2000 ∧ profit = 39440 :=
sorry

end NUMINAMATH_CALUDE_health_drink_sales_correct_l1979_197966


namespace NUMINAMATH_CALUDE_rosie_pies_l1979_197945

/-- Calculates the number of pies Rosie can make given the available apples and pears. -/
def calculate_pies (apples_per_3_pies : ℕ) (pears_per_3_pies : ℕ) (available_apples : ℕ) (available_pears : ℕ) : ℕ :=
  min (available_apples * 3 / apples_per_3_pies) (available_pears * 3 / pears_per_3_pies)

/-- Proves that Rosie can make 9 pies with 36 apples and 18 pears, given that she can make 3 pies out of 12 apples and 6 pears. -/
theorem rosie_pies : calculate_pies 12 6 36 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l1979_197945


namespace NUMINAMATH_CALUDE_marching_band_composition_l1979_197927

theorem marching_band_composition (total : ℕ) (brass : ℕ) (woodwind : ℕ) (percussion : ℕ)
  (h1 : total = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind)
  (h4 : total = brass + woodwind + percussion) :
  brass = 10 := by
sorry

end NUMINAMATH_CALUDE_marching_band_composition_l1979_197927


namespace NUMINAMATH_CALUDE_circle_center_l1979_197991

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 5

/-- Theorem: The center of the circle defined by x^2 + y^2 - 2x - 4y = 0 is at (1, 2) -/
theorem circle_center : CircleCenter 1 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1979_197991


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l1979_197901

/-- Given two polynomials over ℝ, prove their sum equals a specific polynomial -/
theorem polynomial_sum_simplification (x : ℝ) :
  (3 * x^4 - 2 * x^3 + 5 * x^2 - 8 * x + 10) + 
  (7 * x^5 - 3 * x^4 + x^3 - 7 * x^2 + 2 * x - 2) = 
  7 * x^5 - x^3 - 2 * x^2 - 6 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l1979_197901


namespace NUMINAMATH_CALUDE_tangent_segment_region_area_l1979_197972

theorem tangent_segment_region_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 6) : 
  let outer_radius := r * Real.sqrt 2
  let area := π * (outer_radius^2 - r^2)
  area = 9 * π := by sorry

end NUMINAMATH_CALUDE_tangent_segment_region_area_l1979_197972


namespace NUMINAMATH_CALUDE_lemon_ratio_l1979_197913

def lemon_problem (levi jayden eli ian : ℕ) : Prop :=
  levi = 5 ∧
  jayden = levi + 6 ∧
  jayden = eli / 3 ∧
  levi + jayden + eli + ian = 115 ∧
  eli * 2 = ian

theorem lemon_ratio :
  ∀ levi jayden eli ian : ℕ,
    lemon_problem levi jayden eli ian →
    eli * 2 = ian :=
by
  sorry

end NUMINAMATH_CALUDE_lemon_ratio_l1979_197913


namespace NUMINAMATH_CALUDE_emily_beads_count_l1979_197978

theorem emily_beads_count (beads_per_necklace : ℕ) (necklaces_made : ℕ) (total_beads : ℕ) : 
  beads_per_necklace = 8 → necklaces_made = 2 → total_beads = beads_per_necklace * necklaces_made → total_beads = 16 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l1979_197978


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1979_197928

theorem geometric_progression_fourth_term 
  (a : ℝ → ℝ) -- Sequence of real numbers
  (h1 : a 1 = 2^(1/2)) -- First term
  (h2 : a 2 = 2^(1/3)) -- Second term
  (h3 : a 3 = 2^(1/6)) -- Third term
  (h_geom : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1) -- Geometric progression condition
  : a 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1979_197928


namespace NUMINAMATH_CALUDE_circle_path_in_right_triangle_l1979_197941

theorem circle_path_in_right_triangle : 
  ∀ (a b c : ℝ) (r : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 →  -- Triangle side lengths
    r = 1 →                   -- Circle radius
    a^2 + b^2 = c^2 →         -- Right triangle condition
    (a + b + c) - 6*r = 12 := by  -- Path length
  sorry

end NUMINAMATH_CALUDE_circle_path_in_right_triangle_l1979_197941


namespace NUMINAMATH_CALUDE_muffins_baked_by_macadams_class_l1979_197998

theorem muffins_baked_by_macadams_class (brier_muffins flannery_muffins total_muffins : ℕ) 
  (h1 : brier_muffins = 18)
  (h2 : flannery_muffins = 17)
  (h3 : total_muffins = 55) :
  total_muffins - (brier_muffins + flannery_muffins) = 20 := by
  sorry

end NUMINAMATH_CALUDE_muffins_baked_by_macadams_class_l1979_197998


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1979_197920

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1979_197920


namespace NUMINAMATH_CALUDE_f_less_than_neg_two_f_two_zeros_iff_l1979_197946

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - exp (x - a) + a

theorem f_less_than_neg_two (x : ℝ) (h : x > 0) : f 0 x < -2 := by
  sorry

theorem f_two_zeros_iff (a : ℝ) :
  (∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_less_than_neg_two_f_two_zeros_iff_l1979_197946


namespace NUMINAMATH_CALUDE_articles_with_equal_price_l1979_197919

/-- Represents the cost price of a single article -/
def cost_price : ℝ := sorry

/-- Represents the selling price of a single article -/
def selling_price : ℝ := sorry

/-- The number of articles whose selling price equals the cost price of 50 articles -/
def N : ℝ := sorry

/-- The gain percentage -/
def gain_percent : ℝ := 100

theorem articles_with_equal_price :
  (50 * cost_price = N * selling_price) →
  (selling_price = 2 * cost_price) →
  (N = 25) :=
by sorry

end NUMINAMATH_CALUDE_articles_with_equal_price_l1979_197919


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1979_197932

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem inequality_equivalence (x : ℝ) (hx : x > 0) :
  (lg x ^ 2 - 3 * lg x + 3) / (lg x - 1) < 1 ↔ x < 10 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1979_197932


namespace NUMINAMATH_CALUDE_factorization_equality_l1979_197989

theorem factorization_equality (p : ℝ) : (p - 4) * (p + 1) + 3 * p = (p + 2) * (p - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1979_197989


namespace NUMINAMATH_CALUDE_quadratic_one_zero_properties_l1979_197951

/-- A quadratic function with exactly one zero -/
structure QuadraticWithOneZero where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : ∃! x, x^2 + a*x + b = 0

theorem quadratic_one_zero_properties (f : QuadraticWithOneZero) :
  (f.a^2 - f.b^2 ≤ 4) ∧
  (f.a^2 + 1/f.b ≥ 4) ∧
  (∀ c x₁ x₂, (∀ x, x^2 + f.a*x + f.b < c ↔ x₁ < x ∧ x < x₂) → |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_properties_l1979_197951


namespace NUMINAMATH_CALUDE_range_of_x_l1979_197976

def P (x : ℝ) : Prop := (x + 1) / (x - 3) ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem range_of_x (x : ℝ) : 
  P x ∧ ¬Q x ↔ x ≤ -1 ∨ x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l1979_197976


namespace NUMINAMATH_CALUDE_paper_airplane_class_composition_l1979_197911

theorem paper_airplane_class_composition 
  (total_students : ℕ) 
  (total_airplanes : ℕ) 
  (girls_airplanes : ℕ) 
  (boys_airplanes : ℕ) 
  (h1 : total_students = 21)
  (h2 : total_airplanes = 69)
  (h3 : girls_airplanes = 2)
  (h4 : boys_airplanes = 5) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boys * boys_airplanes + girls * girls_airplanes = total_airplanes ∧
    boys = 9 ∧ 
    girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_paper_airplane_class_composition_l1979_197911


namespace NUMINAMATH_CALUDE_election_winning_percentage_bound_l1979_197939

def total_votes_sept30 : ℕ := 15000
def total_votes_oct10 : ℕ := 22000
def geoff_votes_sept30 : ℕ := 150
def additional_votes_needed_sept30 : ℕ := 5000
def additional_votes_needed_oct10 : ℕ := 2000

def winning_percentage : ℚ :=
  (geoff_votes_sept30 + additional_votes_needed_sept30 + additional_votes_needed_oct10) / total_votes_oct10

theorem election_winning_percentage_bound :
  winning_percentage < 325/1000 := by sorry

end NUMINAMATH_CALUDE_election_winning_percentage_bound_l1979_197939


namespace NUMINAMATH_CALUDE_function_properties_l1979_197925

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + a*x^2 - 1

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 1

-- Theorem statement
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ∧
    (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) ∧
    a = 4 ∧
    ∃ b : ℝ, (b = 0 ∨ b = 4) ∧
      (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = g b x₁ ∧ f a x₂ = g b x₂) ∧
      (∀ x₃ : ℝ, x₃ ≠ x₁ ∧ x₃ ≠ x₂ → f a x₃ ≠ g b x₃) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1979_197925


namespace NUMINAMATH_CALUDE_max_cart_length_l1979_197909

/-- The maximum length of a rectangular cart that can navigate through a right-angled corridor -/
theorem max_cart_length (corridor_width : ℝ) (cart_width : ℝ) :
  corridor_width = 1.5 →
  cart_width = 1 →
  ∃ (max_length : ℝ), max_length = 3 * Real.sqrt 2 - 2 ∧
    ∀ (cart_length : ℝ), cart_length ≤ max_length →
      ∃ (θ : ℝ), 0 < θ ∧ θ < Real.pi / 2 ∧
        cart_length ≤ (3 * (Real.sin θ + Real.cos θ) - 2) / (2 * Real.sin θ * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_max_cart_length_l1979_197909


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1979_197992

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (7 + 18 * i) / (3 - 4 * i) = -51/25 + 82/25 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1979_197992


namespace NUMINAMATH_CALUDE_three_digit_cube_units_digit_l1979_197953

theorem three_digit_cube_units_digit :
  ∀ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧ 
    (n = (n % 10)^3) →
    (n = 125 ∨ n = 216 ∨ n = 729) :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_units_digit_l1979_197953


namespace NUMINAMATH_CALUDE_solve_for_y_l1979_197910

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1979_197910


namespace NUMINAMATH_CALUDE_conditioner_shampoo_ratio_l1979_197999

/-- Proves the ratio of daily conditioner use to daily shampoo use -/
theorem conditioner_shampoo_ratio 
  (daily_shampoo : ℝ) 
  (total_volume : ℝ) 
  (days : ℕ) 
  (h1 : daily_shampoo = 1)
  (h2 : total_volume = 21)
  (h3 : days = 14) :
  (total_volume - daily_shampoo * days) / days / daily_shampoo = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_conditioner_shampoo_ratio_l1979_197999


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1979_197902

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_composition_of_three : g (g (g (g 3))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1979_197902


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1979_197907

def sale_month1 : ℕ := 5420
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470
def average_sale : ℕ := 6100
def num_months : ℕ := 6

theorem fourth_month_sale :
  sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6 + 6350 = average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l1979_197907


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l1979_197947

theorem log_inequality_solution_set :
  ∀ x : ℝ, (Real.log (x - 1) < 1) ↔ (1 < x ∧ x < 11) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l1979_197947


namespace NUMINAMATH_CALUDE_g_inverse_sum_l1979_197977

/-- The function g(x) defined piecewise -/
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 4 * x

/-- Theorem stating that c + d = 7.25 given the conditions -/
theorem g_inverse_sum (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) →
  c + d = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_sum_l1979_197977


namespace NUMINAMATH_CALUDE_fruit_water_content_l1979_197926

theorem fruit_water_content (m : ℝ) : 
  m > 0 ∧ m ≤ 100 →  -- m is a percentage, so it's between 0 and 100
  (100 - m + m * (1 - (m - 5) / 100) = 50) →  -- equation from step 6 in the solution
  m = 80 := by sorry

end NUMINAMATH_CALUDE_fruit_water_content_l1979_197926


namespace NUMINAMATH_CALUDE_bank_layoff_optimization_l1979_197903

/-- Represents the problem of maximizing bank profit through layoffs --/
theorem bank_layoff_optimization :
  let initial_employees : ℕ := 320
  let initial_profit_per_employee : ℝ := 200000
  let profit_increase_per_layoff : ℝ := 20000
  let layoff_expense : ℝ := 60000
  let min_employees : ℕ := (3 * initial_employees) / 4
  let profit (x : ℕ) : ℝ := 
    (initial_employees - x) * (initial_profit_per_employee + profit_increase_per_layoff * x) - layoff_expense * x
  ∃ (optimal_layoffs : ℕ), 
    optimal_layoffs = 80 ∧ 
    optimal_layoffs ≤ initial_employees - min_employees ∧
    ∀ (x : ℕ), x ≤ initial_employees - min_employees → profit x ≤ profit optimal_layoffs :=
by sorry

end NUMINAMATH_CALUDE_bank_layoff_optimization_l1979_197903


namespace NUMINAMATH_CALUDE_point_in_planar_region_l1979_197940

/-- A point (m, 1) is within the planar region represented by 2x + 3y - 5 > 0 if and only if m > 1 -/
theorem point_in_planar_region (m : ℝ) : 2*m + 3*1 - 5 > 0 ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_planar_region_l1979_197940


namespace NUMINAMATH_CALUDE_smallest_music_class_size_l1979_197922

theorem smallest_music_class_size :
  ∀ (x : ℕ),
  (∃ (total : ℕ), total = 5 * x + 2 ∧ total > 40) →
  (∀ (y : ℕ), y < x → ¬(∃ (total : ℕ), total = 5 * y + 2 ∧ total > 40)) →
  5 * x + 2 = 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_music_class_size_l1979_197922


namespace NUMINAMATH_CALUDE_marcy_cat_time_l1979_197997

/-- Given that Marcy spends 12 minutes petting her cat and 1/3 of that time combing it,
    prove that she spends 16 minutes in total with her cat. -/
theorem marcy_cat_time (petting_time : ℝ) (combing_ratio : ℝ) : 
  petting_time = 12 → combing_ratio = 1/3 → petting_time + combing_ratio * petting_time = 16 := by
sorry

end NUMINAMATH_CALUDE_marcy_cat_time_l1979_197997


namespace NUMINAMATH_CALUDE_f_f_3_equals_13_9_l1979_197974

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

theorem f_f_3_equals_13_9 : f (f 3) = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_f_f_3_equals_13_9_l1979_197974


namespace NUMINAMATH_CALUDE_novelists_count_l1979_197944

theorem novelists_count (total : ℕ) (ratio_novelists : ℕ) (ratio_poets : ℕ) (novelists : ℕ) : 
  total = 24 →
  ratio_novelists = 5 →
  ratio_poets = 3 →
  ratio_novelists + ratio_poets = novelists + (total - novelists) →
  novelists * (ratio_novelists + ratio_poets) = total * ratio_novelists →
  novelists = 15 := by
sorry

end NUMINAMATH_CALUDE_novelists_count_l1979_197944


namespace NUMINAMATH_CALUDE_negative_a_sixth_div_a_cube_l1979_197967

theorem negative_a_sixth_div_a_cube (a : ℝ) : (-a)^6 / a^3 = a^3 := by sorry

end NUMINAMATH_CALUDE_negative_a_sixth_div_a_cube_l1979_197967


namespace NUMINAMATH_CALUDE_laptop_price_l1979_197965

theorem laptop_price : ∃ (x : ℝ), x = 400 ∧ 
  (∃ (price_C price_D : ℝ), 
    price_C = 0.8 * x - 60 ∧ 
    price_D = 0.7 * x ∧ 
    price_D - price_C = 20) := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l1979_197965


namespace NUMINAMATH_CALUDE_tangent_line_at_e_l1979_197963

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_e :
  let p : ℝ × ℝ := (Real.exp 1, f (Real.exp 1))
  let m : ℝ := deriv f (Real.exp 1)
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  tangent_line = λ x => 2 * x - Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_e_l1979_197963


namespace NUMINAMATH_CALUDE_vacation_cost_l1979_197921

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 5 = 50) → C = 375 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l1979_197921


namespace NUMINAMATH_CALUDE_book_purchases_l1979_197959

theorem book_purchases (people_A : ℕ) (people_B : ℕ) (people_both : ℕ) (people_only_B : ℕ) (people_only_A : ℕ) : 
  people_A = 2 * people_B →
  people_both = 500 →
  people_both = 2 * people_only_B →
  people_A = people_only_A + people_both →
  people_B = people_only_B + people_both →
  people_only_A = 1000 := by
sorry

end NUMINAMATH_CALUDE_book_purchases_l1979_197959


namespace NUMINAMATH_CALUDE_estimate_total_students_l1979_197908

/-- Represents the survey data and estimated total students -/
structure SurveyData where
  total_students : ℕ  -- Estimated total number of first-year students
  first_survey : ℕ    -- Number of students in the first survey
  second_survey : ℕ   -- Number of students in the second survey
  overlap : ℕ         -- Number of students in both surveys

/-- The theorem states that given the survey conditions, 
    the estimated total number of first-year students is 400 -/
theorem estimate_total_students (data : SurveyData) :
  data.first_survey = 80 →
  data.second_survey = 100 →
  data.overlap = 20 →
  data.total_students = 400 :=
by sorry

end NUMINAMATH_CALUDE_estimate_total_students_l1979_197908


namespace NUMINAMATH_CALUDE_inventory_problem_l1979_197916

theorem inventory_problem (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  speedsters = (3 * total) / 4 →
  convertibles = (3 * speedsters) / 5 →
  convertibles = 54 →
  total - speedsters = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_inventory_problem_l1979_197916


namespace NUMINAMATH_CALUDE_intersection_M_N_l1979_197990

-- Define set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Define set N (domain of log|x|)
def N : Set ℝ := {x | x ≠ 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1979_197990


namespace NUMINAMATH_CALUDE_unique_solution_l1979_197971

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x + y - 1 = 0 ∧ x - 2*y + 2 = 0

/-- The solution to the system of equations -/
def solution : ℝ × ℝ := (0, 1)

/-- Theorem stating that the solution is unique and satisfies the system -/
theorem unique_solution :
  system solution.1 solution.2 ∧
  ∀ x y : ℝ, system x y → (x, y) = solution := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1979_197971


namespace NUMINAMATH_CALUDE_equation_solutions_l1979_197962

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 5) * (x - 3) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 5) * (x - 3)
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → 
    (f x / g x = 1 ↔ x = 4 ∨ x = 4 + 2 * Real.sqrt 10 ∨ x = 4 - 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1979_197962


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1979_197917

/-- The intersection point of the diagonals of a parallelogram with opposite vertices (2, -3) and (10, 9) is (6, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (10, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (6, 3) := by
sorry


end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1979_197917


namespace NUMINAMATH_CALUDE_leg_length_in_45_45_90_triangle_l1979_197938

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = side * Real.sqrt 2

/-- The length of a leg in a 45-45-90 triangle with hypotenuse 9 is 9 -/
theorem leg_length_in_45_45_90_triangle (t : RightIsoscelesTriangle) 
  (h : t.hypotenuse = 9) : t.side = 9 := by
  sorry

end NUMINAMATH_CALUDE_leg_length_in_45_45_90_triangle_l1979_197938


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1979_197964

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = (10^n * 6 - 6) / (10^n - 1)) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1979_197964


namespace NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l1979_197983

theorem sum_positive_implies_at_least_one_positive (x y : ℝ) : x + y > 0 → x > 0 ∨ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l1979_197983


namespace NUMINAMATH_CALUDE_problem_statement_l1979_197929

theorem problem_statement (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1979_197929


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1979_197973

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 3/(y+2) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 3/(y₀+2) = 1 ∧ x₀ + y₀ = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1979_197973


namespace NUMINAMATH_CALUDE_min_value_of_f_l1979_197912

/-- The function f(x) = 2x³ - 6x² + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f m x ≥ f m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f m x = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f m x ≤ f m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f m x = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1979_197912


namespace NUMINAMATH_CALUDE_element_in_set_l1979_197923

theorem element_in_set : ∀ (a b : ℕ), 1 ∈ ({a, b, 1} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1979_197923


namespace NUMINAMATH_CALUDE_fraction_simplification_specific_case_l1979_197960

theorem fraction_simplification (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

theorem specific_case : 
  let a : ℚ := 12
  let b : ℚ := 16
  let c : ℚ := 9
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_specific_case_l1979_197960


namespace NUMINAMATH_CALUDE_wheel_distance_theorem_l1979_197936

/-- Represents the properties and movement of a wheel -/
structure Wheel where
  rotations_per_minute : ℕ
  cm_per_rotation : ℕ

/-- Calculates the distance in meters that a wheel moves in one hour -/
def distance_in_one_hour (w : Wheel) : ℚ :=
  (w.rotations_per_minute * 60 * w.cm_per_rotation) / 100

/-- Theorem stating that a wheel with given properties moves 420 meters in one hour -/
theorem wheel_distance_theorem (w : Wheel) 
  (h1 : w.rotations_per_minute = 20) 
  (h2 : w.cm_per_rotation = 35) : 
  distance_in_one_hour w = 420 := by
  sorry

#eval distance_in_one_hour ⟨20, 35⟩

end NUMINAMATH_CALUDE_wheel_distance_theorem_l1979_197936


namespace NUMINAMATH_CALUDE_complement_of_M_l1979_197986

-- Define the set M
def M : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1979_197986


namespace NUMINAMATH_CALUDE_franks_daily_cookie_consumption_l1979_197934

/-- Proves that Frank eats 1 cookie each day given the conditions of the problem -/
theorem franks_daily_cookie_consumption :
  let days : ℕ := 6
  let trays_per_day : ℕ := 2
  let cookies_per_tray : ℕ := 12
  let ted_cookies : ℕ := 4
  let cookies_left : ℕ := 134
  let total_baked : ℕ := days * trays_per_day * cookies_per_tray
  let franks_total_consumption : ℕ := total_baked - ted_cookies - cookies_left
  franks_total_consumption / days = 1 := by
  sorry

end NUMINAMATH_CALUDE_franks_daily_cookie_consumption_l1979_197934


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l1979_197975

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeros :
  trailingZeros 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l1979_197975


namespace NUMINAMATH_CALUDE_inequality_proof_l1979_197954

theorem inequality_proof (n : ℕ) : (2*n + 1)^n ≥ (2*n)^n + (2*n - 1)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1979_197954


namespace NUMINAMATH_CALUDE_multiplicative_inverse_of_3_mod_47_l1979_197985

theorem multiplicative_inverse_of_3_mod_47 : ∃ x : ℕ, x < 47 ∧ (3 * x) % 47 = 1 :=
by
  use 16
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_of_3_mod_47_l1979_197985


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l1979_197949

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure any additional person must sit next to someone -/
def minOccupiedSeats (totalSeats : ℕ) : ℕ :=
  totalSeats / 3

theorem min_occupied_seats_for_150 :
  minOccupiedSeats 150 = 50 := by
  sorry

#eval minOccupiedSeats 150

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l1979_197949


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1979_197996

/-- Given two 2D vectors a and b, where a = (2,1), a + b = (1,k), and a ⟂ b, prove that k = 3 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  a + b = (1, k) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1979_197996


namespace NUMINAMATH_CALUDE_equilateral_triangle_coverage_l1979_197918

theorem equilateral_triangle_coverage (small_side : ℝ) (large_side : ℝ) : 
  small_side = 1 →
  large_side = 15 →
  (large_side / small_side) ^ 2 = 225 :=
by
  sorry

#check equilateral_triangle_coverage

end NUMINAMATH_CALUDE_equilateral_triangle_coverage_l1979_197918


namespace NUMINAMATH_CALUDE_initial_work_plan_l1979_197970

/-- Proves that the initial plan was to complete the work in 28 days given the conditions of the problem. -/
theorem initial_work_plan (total_men : Nat) (absent_men : Nat) (days_with_reduced_men : Nat) 
  (h1 : total_men = 42)
  (h2 : absent_men = 6)
  (h3 : days_with_reduced_men = 14) : 
  (total_men * ((total_men - absent_men) * days_with_reduced_men)) / (total_men - absent_men) = 28 := by
  sorry

#eval (42 * ((42 - 6) * 14)) / (42 - 6)

end NUMINAMATH_CALUDE_initial_work_plan_l1979_197970


namespace NUMINAMATH_CALUDE_solve_for_a_l1979_197994

theorem solve_for_a (x a : ℝ) (h1 : 2 * x - 5 * a = 3 * a + 22) (h2 : x = 3) : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1979_197994


namespace NUMINAMATH_CALUDE_tangent_line_m_range_l1979_197948

/-- The range of m for a line mx - y - 5m + 4 = 0 tangent to a circle (x+1)^2 + y^2 = 4 -/
theorem tangent_line_m_range :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), (x + 1)^2 + y^2 = 4 ∧ m*x - y - 5*m + 4 = 0) →
  (∃ (Q : ℝ × ℝ), (Q.1 + 1)^2 + Q.2^2 = 4 ∧ 
    ∃ (P : ℝ × ℝ), m*P.1 - P.2 - 5*m + 4 = 0 ∧
    Real.cos (30 * π / 180) = (Q.1 - P.1) / (4 * ((Q.1 - P.1)^2 + (Q.2 - P.2)^2).sqrt)) →
  0 ≤ m ∧ m ≤ 12/5 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_m_range_l1979_197948


namespace NUMINAMATH_CALUDE_machining_defect_probability_l1979_197906

theorem machining_defect_probability (defect_rate1 defect_rate2 : ℝ) 
  (h1 : defect_rate1 = 0.03) 
  (h2 : defect_rate2 = 0.05) 
  (h3 : 0 ≤ defect_rate1 ∧ defect_rate1 ≤ 1) 
  (h4 : 0 ≤ defect_rate2 ∧ defect_rate2 ≤ 1) :
  1 - (1 - defect_rate1) * (1 - defect_rate2) = 0.0785 := by
  sorry

#check machining_defect_probability

end NUMINAMATH_CALUDE_machining_defect_probability_l1979_197906


namespace NUMINAMATH_CALUDE_center_square_side_length_l1979_197915

theorem center_square_side_length :
  ∀ (total_side : ℝ) (l_region_count : ℕ) (l_region_fraction : ℝ),
    total_side = 20 →
    l_region_count = 4 →
    l_region_fraction = 1/5 →
    let total_area := total_side^2
    let l_regions_area := l_region_count * l_region_fraction * total_area
    let center_area := total_area - l_regions_area
    center_area.sqrt = 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_center_square_side_length_l1979_197915


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1979_197942

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≤ Real.sqrt 5}

-- Define the set N
def N : Set ℝ := {1, 2, 3, 4}

-- Theorem statement
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1979_197942


namespace NUMINAMATH_CALUDE_antonio_meatballs_l1979_197995

/-- Given a recipe for meatballs and family size, calculate how many meatballs Antonio will eat -/
theorem antonio_meatballs (hamburger_per_meatball : ℚ) (family_size : ℕ) (total_hamburger : ℕ) :
  hamburger_per_meatball = 1/8 →
  family_size = 8 →
  total_hamburger = 4 →
  (total_hamburger / hamburger_per_meatball) / family_size = 4 :=
by sorry

end NUMINAMATH_CALUDE_antonio_meatballs_l1979_197995


namespace NUMINAMATH_CALUDE_first_day_is_saturday_l1979_197993

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  saturdays : Nat
  sundays : Nat

/-- Theorem: In a 30-day month with 5 Saturdays and 5 Sundays, the first day is Saturday -/
theorem first_day_is_saturday (m : Month) (h1 : m.days = 30) (h2 : m.saturdays = 5) (h3 : m.sundays = 5) :
  ∃ (first_day : DayOfWeek), first_day = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_first_day_is_saturday_l1979_197993


namespace NUMINAMATH_CALUDE_abs_value_equality_l1979_197952

theorem abs_value_equality (m : ℝ) : |m| = |-3| → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_equality_l1979_197952


namespace NUMINAMATH_CALUDE_original_group_size_l1979_197935

/-- Proves that the original number of men in a group is 12, given the conditions of the problem -/
theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 8 →
  absent_men = 3 →
  final_days = 10 →
  ∃ (original_men : ℕ),
    original_men > 0 ∧
    (original_men : ℚ) / initial_days = (original_men - absent_men : ℚ) / final_days ∧
    original_men = 12 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l1979_197935


namespace NUMINAMATH_CALUDE_solution_verification_l1979_197914

theorem solution_verification (x : ℝ) : 
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧ 
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧ 
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) := by
sorry

end NUMINAMATH_CALUDE_solution_verification_l1979_197914


namespace NUMINAMATH_CALUDE_jerry_cans_count_l1979_197961

/-- The number of cans Jerry can carry at once -/
def cans_per_trip : ℕ := 4

/-- The time in seconds it takes to drain 4 cans -/
def drain_time : ℕ := 30

/-- The time in seconds for a round trip to the sink/recycling bin -/
def round_trip_time : ℕ := 20

/-- The total time in seconds to throw all cans away -/
def total_time : ℕ := 350

/-- The time in seconds for one complete cycle (draining and round trip) -/
def cycle_time : ℕ := drain_time + round_trip_time

theorem jerry_cans_count : 
  (total_time / cycle_time) * cans_per_trip = 28 := by sorry

end NUMINAMATH_CALUDE_jerry_cans_count_l1979_197961


namespace NUMINAMATH_CALUDE_parabola_values_l1979_197981

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_values (p : Parabola) : 
  (p.y_coord 4 = 5) ∧ 
  (p.y_coord 2 = -3) ∧ 
  (p.y_coord 6 = 3) ∧
  (∀ x : ℝ, p.y_coord x = p.y_coord (8 - x)) →
  p.a = -2 ∧ p.b = 16 ∧ p.c = -27 := by
  sorry

end NUMINAMATH_CALUDE_parabola_values_l1979_197981


namespace NUMINAMATH_CALUDE_sum_inequality_l1979_197968

theorem sum_inequality (a b c : ℝ) (k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1) (hk : k ≥ 3) : 
  (1 / (a^k * (b + c)) + 1 / (b^k * (a + c)) + 1 / (c^k * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1979_197968


namespace NUMINAMATH_CALUDE_horse_purchase_problem_l1979_197987

/-- The problem of three people buying a horse -/
theorem horse_purchase_problem (x y z : ℚ) : 
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  (x = 60/17 ∧ y = 136/17 ∧ z = 156/17) := by
  sorry

end NUMINAMATH_CALUDE_horse_purchase_problem_l1979_197987


namespace NUMINAMATH_CALUDE_fish_catch_calculation_l1979_197955

/-- Prove that given the conditions, Erica caught 80 kg of fish in the past four months --/
theorem fish_catch_calculation (price : ℝ) (total_earnings : ℝ) (past_catch : ℝ) :
  price = 20 →
  total_earnings = 4800 →
  total_earnings = price * (past_catch + 2 * past_catch) →
  past_catch = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_calculation_l1979_197955


namespace NUMINAMATH_CALUDE_remainder_1949_1995_mod_7_l1979_197956

theorem remainder_1949_1995_mod_7 : 1949^1995 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1949_1995_mod_7_l1979_197956


namespace NUMINAMATH_CALUDE_factorization_equality_l1979_197905

theorem factorization_equality (a b : ℝ) : a^2 - 4*a*b^2 = a*(a - 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1979_197905


namespace NUMINAMATH_CALUDE_expression_evaluation_l1979_197943

theorem expression_evaluation : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1979_197943


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1979_197979

theorem least_positive_integer_with_given_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 6 = 3 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1979_197979


namespace NUMINAMATH_CALUDE_double_acute_angle_l1979_197950

theorem double_acute_angle (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < 2 * θ ∧ 2 * θ < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_l1979_197950


namespace NUMINAMATH_CALUDE_probability_largest_is_six_correct_l1979_197904

def probability_largest_is_six (n m k : ℕ) : ℚ :=
  (Nat.choose m k : ℚ) / (Nat.choose n k : ℚ)

theorem probability_largest_is_six_correct : 
  probability_largest_is_six 10 6 4 = (Nat.choose 6 4 : ℚ) / (Nat.choose 10 4 : ℚ) :=
by
  sorry

#eval probability_largest_is_six 10 6 4

end NUMINAMATH_CALUDE_probability_largest_is_six_correct_l1979_197904


namespace NUMINAMATH_CALUDE_distance_sum_theorem_l1979_197931

/-- The curve C in the xy-plane -/
def C (x y : ℝ) : Prop := x^2/9 + y^2 = 1

/-- The line l in the xy-plane -/
def l (x y : ℝ) : Prop := y - x = Real.sqrt 2

/-- The point P -/
def P : ℝ × ℝ := (0, 2)

/-- Points A and B are the intersection points of C and l -/
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ l A.1 A.2 ∧ C B.1 B.2 ∧ l B.1 B.2 ∧ A ≠ B

theorem distance_sum_theorem (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
  Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) =
  18 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_theorem_l1979_197931


namespace NUMINAMATH_CALUDE_system_no_solution_l1979_197982

def has_no_solution (a b c : ℤ) : Prop :=
  2 / a = -b / 5 ∧ -b / 5 = 1 / -c ∧ 2 / a ≠ 2 * b / a

theorem system_no_solution : 
  {(a, b, c) : ℤ × ℤ × ℤ | has_no_solution a b c} = 
  {(-2, 5, 1), (2, -5, -1), (10, -1, -5)} := by sorry

end NUMINAMATH_CALUDE_system_no_solution_l1979_197982


namespace NUMINAMATH_CALUDE_face_D_opposite_Y_l1979_197933

-- Define the faces of the cube
inductive Face
| A | B | C | D | E | Y

-- Define the structure of the net
structure Net :=
  (faces : List Face)
  (adjacent : Face → Face → Bool)

-- Define the structure of the cube
structure Cube :=
  (faces : List Face)
  (opposite : Face → Face)

-- Define the folding operation
def fold (net : Net) : Cube :=
  sorry

-- The theorem to prove
theorem face_D_opposite_Y (net : Net) (cube : Cube) :
  net.faces = [Face.A, Face.B, Face.C, Face.D, Face.E, Face.Y] →
  cube = fold net →
  cube.opposite Face.Y = Face.D :=
sorry

end NUMINAMATH_CALUDE_face_D_opposite_Y_l1979_197933


namespace NUMINAMATH_CALUDE_train_length_l1979_197900

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 12 → ∃ length : ℝ, 
  (length ≥ 399) ∧ (length ≤ 401) ∧ (length = speed * time * 1000 / 3600) := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1979_197900


namespace NUMINAMATH_CALUDE_puppies_adopted_per_day_l1979_197969

theorem puppies_adopted_per_day 
  (initial_puppies : ℕ) 
  (additional_puppies : ℕ) 
  (adoption_days : ℕ) 
  (h1 : initial_puppies = 2)
  (h2 : additional_puppies = 34)
  (h3 : adoption_days = 9)
  (h4 : (initial_puppies + additional_puppies) % adoption_days = 0) :
  (initial_puppies + additional_puppies) / adoption_days = 4 := by
sorry

end NUMINAMATH_CALUDE_puppies_adopted_per_day_l1979_197969


namespace NUMINAMATH_CALUDE_statements_are_false_l1979_197988

theorem statements_are_false : 
  (¬ ∀ (x : ℚ), ∃ (y : ℚ), (x < y ∧ y < -x) ∨ (-x < y ∧ y < x)) ∧ 
  (¬ ∀ (x : ℚ), x ≠ 0 → ∃ (y : ℚ), (x < y ∧ y < x⁻¹) ∨ (x⁻¹ < y ∧ y < x)) :=
by sorry

end NUMINAMATH_CALUDE_statements_are_false_l1979_197988


namespace NUMINAMATH_CALUDE_no_zeroes_g_l1979_197937

/-- A function f satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  continuous : Continuous f
  differentiable : Differentiable ℝ f
  condition : ∀ x, x * (deriv f x) + f x > 0

/-- The function g(x) = xf(x) + 1 -/
def g (sf : SpecialFunction) (x : ℝ) : ℝ := x * sf.f x + 1

/-- Theorem stating that g has no zeroes for x > 0 -/
theorem no_zeroes_g (sf : SpecialFunction) : ∀ x > 0, g sf x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeroes_g_l1979_197937


namespace NUMINAMATH_CALUDE_same_height_time_l1979_197924

/-- Represents the height of a ball as a function of time -/
def ball_height (a h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2)^2 + h

theorem same_height_time : 
  ∀ (a h : ℝ), a ≠ 0 →
  ∃ (t : ℝ), t > 0 ∧ 
  ball_height a h t = ball_height a h (t - 2) ∧
  t = 2.2 :=
sorry

end NUMINAMATH_CALUDE_same_height_time_l1979_197924


namespace NUMINAMATH_CALUDE_line_proof_l1979_197980

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := 3*x + y - 2 = 0

-- Define the line to be proved
def prove_line (x y : ℝ) : Prop := x - 3*y + 1 = 0

-- Function to get the center of a circle
def circle_center (circle : (ℝ → ℝ → Prop)) : ℝ × ℝ := sorry

-- Function to check if two lines are perpendicular
def perpendicular (line1 line2 : ℝ → ℝ → Prop) : Prop := sorry

theorem line_proof :
  let center := circle_center circle_C
  prove_line center.1 center.2 ∧ 
  perpendicular prove_line given_line := by sorry

end NUMINAMATH_CALUDE_line_proof_l1979_197980


namespace NUMINAMATH_CALUDE_zero_of_f_floor_l1979_197984

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem zero_of_f_floor (x : ℝ) (hx : f x = 0) : Int.floor x = 2 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_floor_l1979_197984
