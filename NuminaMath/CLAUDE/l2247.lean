import Mathlib

namespace simplify_expression_l2247_224745

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = 35/2 - 3 * Real.sqrt 34 := by
  sorry

end simplify_expression_l2247_224745


namespace yard_area_l2247_224731

def yard_length : ℝ := 20
def yard_width : ℝ := 18
def square_cutout_side : ℝ := 4
def rect_cutout_length : ℝ := 2
def rect_cutout_width : ℝ := 5

theorem yard_area : 
  yard_length * yard_width - 
  square_cutout_side * square_cutout_side - 
  rect_cutout_length * rect_cutout_width = 334 := by
sorry

end yard_area_l2247_224731


namespace total_legs_l2247_224753

/-- The number of bees -/
def num_bees : ℕ := 50

/-- The number of ants -/
def num_ants : ℕ := 35

/-- The number of spiders -/
def num_spiders : ℕ := 20

/-- The number of legs a bee has -/
def bee_legs : ℕ := 6

/-- The number of legs an ant has -/
def ant_legs : ℕ := 6

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- Theorem stating the total number of legs -/
theorem total_legs : 
  num_bees * bee_legs + num_ants * ant_legs + num_spiders * spider_legs = 670 := by
  sorry

end total_legs_l2247_224753


namespace stratified_sampling_proportion_l2247_224768

/-- Represents the number of athletes selected in a stratified sampling -/
structure StratifiedSample where
  male : ℕ
  female : ℕ

/-- Represents the composition of the track and field team -/
def team : StratifiedSample :=
  { male := 56, female := 42 }

/-- Calculates the ratio of male to female athletes -/
def ratio (s : StratifiedSample) : ℚ :=
  s.male / s.female

/-- Theorem: In a stratified sampling, if 8 male athletes are selected,
    then 6 female athletes should be selected to maintain the same proportion -/
theorem stratified_sampling_proportion :
  ∀ (sample : StratifiedSample),
    sample.male = 8 →
    ratio sample = ratio team →
    sample.female = 6 := by
  sorry

end stratified_sampling_proportion_l2247_224768


namespace kevin_kangaroo_hops_l2247_224782

theorem kevin_kangaroo_hops (n : ℕ) (a : ℚ) (r : ℚ) : 
  n = 7 ∧ a = 1 ∧ r = 3/4 → 
  4 * (a * (1 - r^n) / (1 - r)) = 7086/2048 := by
sorry

end kevin_kangaroo_hops_l2247_224782


namespace carpet_cost_per_meter_l2247_224790

/-- Proves that the cost per meter of carpet is 30 paise given the room dimensions, carpet width, and total cost. -/
theorem carpet_cost_per_meter (room_length : ℝ) (room_width : ℝ) (carpet_width : ℝ) (total_cost : ℝ) :
  room_length = 15 →
  room_width = 6 →
  carpet_width = 0.75 →
  total_cost = 36 →
  (total_cost * 100) / (room_length * room_width / carpet_width) = 30 := by
  sorry

#check carpet_cost_per_meter

end carpet_cost_per_meter_l2247_224790


namespace bank_savings_exceed_50_dollars_l2247_224779

/-- The sum of a geometric sequence with first term 5 and ratio 2, after n terms -/
def geometric_sum (n : ℕ) : ℚ := 5 * (2^n - 1)

/-- The smallest number of days needed for the sum to exceed 5000 cents -/
def smallest_day : ℕ := 10

theorem bank_savings_exceed_50_dollars :
  (∀ k < smallest_day, geometric_sum k ≤ 5000) ∧
  geometric_sum smallest_day > 5000 := by sorry

end bank_savings_exceed_50_dollars_l2247_224779


namespace square_sum_product_equality_l2247_224704

theorem square_sum_product_equality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end square_sum_product_equality_l2247_224704


namespace brendas_age_l2247_224730

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda is 3 years old -/
theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B)     -- Addison's age is four times Brenda's age
  (h2 : J = B + 9)     -- Janet is nine years older than Brenda
  (h3 : A = J)         -- Addison and Janet are twins (same age)
  : B = 3 := by        -- Prove that Brenda's age (B) is 3
sorry


end brendas_age_l2247_224730


namespace equal_roots_coefficients_l2247_224755

def polynomial (x p q : ℝ) : ℝ := x^4 - 10*x^3 + 37*x^2 + p*x + q

theorem equal_roots_coefficients :
  ∀ (p q : ℝ),
  (∃ (x₁ x₃ : ℝ), 
    (∀ x : ℝ, polynomial x p q = 0 ↔ x = x₁ ∨ x = x₃) ∧
    (x₁ + x₃ = 5) ∧
    (x₁ * x₃ = 6)) →
  p = -60 ∧ q = 36 := by
sorry

end equal_roots_coefficients_l2247_224755


namespace parabola_hyperbola_configuration_l2247_224747

/-- Theorem: Value of 'a' for a specific parabola and hyperbola configuration -/
theorem parabola_hyperbola_configuration (p t a : ℝ) : 
  p > 0 → 
  t > 0 → 
  a > 0 → 
  t^2 = 2*p*1 → 
  (1 + p/2)^2 + t^2 = 5^2 → 
  (∃ k : ℝ, k = 4/(1+a) ∧ k = 3/a) → 
  a = 3 :=
by sorry

end parabola_hyperbola_configuration_l2247_224747


namespace inverse_composition_problem_l2247_224705

/-- Given functions h and k where k⁻¹ ∘ h = λ z, 7 * z - 4, prove that h⁻¹(k(12)) = 16/7 -/
theorem inverse_composition_problem (h k : ℝ → ℝ) 
  (hk : Function.LeftInverse k⁻¹ h ∧ Function.RightInverse k⁻¹ h) 
  (h_def : ∀ z, k⁻¹ (h z) = 7 * z - 4) : 
  h⁻¹ (k 12) = 16/7 := by
  sorry

end inverse_composition_problem_l2247_224705


namespace common_number_in_overlapping_lists_l2247_224770

theorem common_number_in_overlapping_lists (list : List ℝ) : 
  list.length = 8 →
  (list.take 5).sum / 5 = 6 →
  (list.drop 3).sum / 5 = 9 →
  list.sum / 8 = 7.5 →
  ∃ x ∈ list.take 5 ∩ list.drop 3, x = 7.5 :=
by sorry

end common_number_in_overlapping_lists_l2247_224770


namespace double_room_cost_is_60_l2247_224702

/-- Represents the hotel booking scenario -/
structure HotelBooking where
  total_rooms : ℕ
  single_room_cost : ℕ
  total_revenue : ℕ
  single_rooms_booked : ℕ

/-- Calculates the cost of a double room given the hotel booking information -/
def double_room_cost (booking : HotelBooking) : ℕ :=
  let double_rooms := booking.total_rooms - booking.single_rooms_booked
  let single_room_revenue := booking.single_rooms_booked * booking.single_room_cost
  let double_room_revenue := booking.total_revenue - single_room_revenue
  double_room_revenue / double_rooms

/-- Theorem stating that the double room cost is 60 for the given scenario -/
theorem double_room_cost_is_60 (booking : HotelBooking) 
  (h1 : booking.total_rooms = 260)
  (h2 : booking.single_room_cost = 35)
  (h3 : booking.total_revenue = 14000)
  (h4 : booking.single_rooms_booked = 64) :
  double_room_cost booking = 60 := by
  sorry

end double_room_cost_is_60_l2247_224702


namespace midpoint_quadrilateral_area_in_regular_hexagon_l2247_224772

/-- Represents a regular hexagon -/
structure RegularHexagon :=
  (side_length : ℝ)

/-- Represents the quadrilateral formed by joining midpoints of non-adjacent sides -/
structure MidpointQuadrilateral :=
  (hexagon : RegularHexagon)

/-- The area of the midpoint quadrilateral in a regular hexagon -/
def midpoint_quadrilateral_area (q : MidpointQuadrilateral) : ℝ :=
  q.hexagon.side_length * q.hexagon.side_length

theorem midpoint_quadrilateral_area_in_regular_hexagon 
  (h : RegularHexagon) 
  (hside : h.side_length = 12) :
  midpoint_quadrilateral_area ⟨h⟩ = 144 := by
  sorry

#check midpoint_quadrilateral_area_in_regular_hexagon

end midpoint_quadrilateral_area_in_regular_hexagon_l2247_224772


namespace existence_of_a_l2247_224732

theorem existence_of_a (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end existence_of_a_l2247_224732


namespace percent_relation_l2247_224781

theorem percent_relation (x y : ℝ) (h : 0.2 * (x - y) = 0.15 * (x + y)) : y = x / 7 := by
  sorry

end percent_relation_l2247_224781


namespace fifth_flower_is_e_l2247_224778

def flowers := ['a', 'b', 'c', 'd', 'e', 'f', 'g']

theorem fifth_flower_is_e : flowers[4] = 'e' := by
  sorry

end fifth_flower_is_e_l2247_224778


namespace rectangle_ratio_is_two_l2247_224737

/-- Configuration of squares and rectangles -/
structure SquareRectConfig where
  inner_square_side : ℝ
  rect_short_side : ℝ
  rect_long_side : ℝ

/-- The configuration satisfies the problem conditions -/
def valid_config (c : SquareRectConfig) : Prop :=
  c.inner_square_side > 0 ∧
  c.rect_short_side > 0 ∧
  c.rect_long_side > 0 ∧
  c.inner_square_side + 2 * c.rect_short_side = 3 * c.inner_square_side ∧
  c.inner_square_side + c.rect_long_side = 3 * c.inner_square_side

theorem rectangle_ratio_is_two (c : SquareRectConfig) (h : valid_config c) :
  c.rect_long_side / c.rect_short_side = 2 := by
  sorry

end rectangle_ratio_is_two_l2247_224737


namespace fans_with_all_items_l2247_224767

def stadium_capacity : ℕ := 5000
def hotdog_interval : ℕ := 75
def soda_interval : ℕ := 45
def popcorn_interval : ℕ := 50
def max_all_items : ℕ := 100

theorem fans_with_all_items :
  let lcm := Nat.lcm (Nat.lcm hotdog_interval soda_interval) popcorn_interval
  min (stadium_capacity / lcm) max_all_items = 11 := by
  sorry

end fans_with_all_items_l2247_224767


namespace difference_of_squares_l2247_224788

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end difference_of_squares_l2247_224788


namespace community_service_arrangements_l2247_224777

def number_of_arrangements (n m k : ℕ) (a b : Fin n) : ℕ :=
  let without_ab := Nat.choose m k
  let with_one := 2 * Nat.choose (m - 1) (k - 1)
  without_ab + 2 * with_one

theorem community_service_arrangements :
  number_of_arrangements 8 6 3 0 1 = 80 := by
  sorry

end community_service_arrangements_l2247_224777


namespace total_gold_stars_l2247_224760

def shelby_stars : List Nat := [4, 6, 3, 5, 2, 3, 7]
def alex_stars : List Nat := [5, 3, 6, 4, 7, 2, 5]

theorem total_gold_stars :
  (shelby_stars.sum + alex_stars.sum) = 62 := by
  sorry

end total_gold_stars_l2247_224760


namespace missing_number_proof_l2247_224791

theorem missing_number_proof : ∃ x : ℤ, (4 + 3) + (8 - 3 - x) = 11 ∧ x = 1 := by
  sorry

end missing_number_proof_l2247_224791


namespace apple_count_difference_l2247_224715

theorem apple_count_difference (initial_green : ℕ) (red_green_difference : ℕ) (delivered_green : ℕ) : 
  initial_green = 546 →
  red_green_difference = 1850 →
  delivered_green = 2725 →
  (initial_green + delivered_green) - (initial_green + red_green_difference) = 875 :=
by
  sorry

end apple_count_difference_l2247_224715


namespace mixture_ratio_l2247_224797

/-- Proves that combining 5 liters of Mixture A (2/3 alcohol, 1/3 water) with 14 liters of Mixture B (4/7 alcohol, 3/7 water) results in a mixture with an alcohol to water volume ratio of 34:23 -/
theorem mixture_ratio (mixture_a_volume : ℚ) (mixture_b_volume : ℚ)
  (mixture_a_alcohol_ratio : ℚ) (mixture_a_water_ratio : ℚ)
  (mixture_b_alcohol_ratio : ℚ) (mixture_b_water_ratio : ℚ)
  (h1 : mixture_a_volume = 5)
  (h2 : mixture_b_volume = 14)
  (h3 : mixture_a_alcohol_ratio = 2/3)
  (h4 : mixture_a_water_ratio = 1/3)
  (h5 : mixture_b_alcohol_ratio = 4/7)
  (h6 : mixture_b_water_ratio = 3/7) :
  (mixture_a_volume * mixture_a_alcohol_ratio + mixture_b_volume * mixture_b_alcohol_ratio) /
  (mixture_a_volume * mixture_a_water_ratio + mixture_b_volume * mixture_b_water_ratio) = 34/23 :=
by sorry

end mixture_ratio_l2247_224797


namespace unique_pair_sum_and_quotient_l2247_224734

theorem unique_pair_sum_and_quotient :
  ∃! (x y : ℕ), x + y = 2015 ∧ ∃ (s : ℕ), x = 25 * y + s ∧ s < y := by
  sorry

end unique_pair_sum_and_quotient_l2247_224734


namespace greatest_value_quadratic_inequality_eight_satisfies_inequality_exists_no_greater_value_l2247_224735

theorem greatest_value_quadratic_inequality :
  ∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → x ≤ 8 :=
by
  sorry

theorem eight_satisfies_inequality :
  8^2 - 12*8 + 32 = 0 :=
by
  sorry

theorem exists_no_greater_value :
  ¬∃ y : ℝ, y > 8 ∧ y^2 - 12*y + 32 ≤ 0 :=
by
  sorry

end greatest_value_quadratic_inequality_eight_satisfies_inequality_exists_no_greater_value_l2247_224735


namespace intersection_A_B_range_of_a_l2247_224794

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x+2)*(4-x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a+1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∪ C a = B → a ∈ Set.Icc (-2) 3 := by sorry

end intersection_A_B_range_of_a_l2247_224794


namespace simplify_sqrt_product_l2247_224786

theorem simplify_sqrt_product : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 225 * Real.sqrt 15 := by
  sorry

end simplify_sqrt_product_l2247_224786


namespace rectangle_area_change_l2247_224796

theorem rectangle_area_change (L W : ℝ) (h : L * W = 540) :
  (0.8 * L) * (1.2 * W) = 518.4 := by sorry

end rectangle_area_change_l2247_224796


namespace intersection_area_is_pi_l2247_224718

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- Define set M
def M : Set (ℝ × ℝ) := {p | f p.1 + f p.2 ≤ 0}

-- Define set N
def N : Set (ℝ × ℝ) := {p | f p.1 - f p.2 ≥ 0}

-- Theorem statement
theorem intersection_area_is_pi : MeasureTheory.volume (M ∩ N) = π := by sorry

end intersection_area_is_pi_l2247_224718


namespace expression_evaluation_l2247_224739

theorem expression_evaluation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 := by
  sorry

end expression_evaluation_l2247_224739


namespace pure_imaginary_value_l2247_224706

theorem pure_imaginary_value (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 3*a + 2 : ℝ) + (a - 2 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → 
  a = 1 := by
sorry

end pure_imaginary_value_l2247_224706


namespace other_factor_l2247_224750

def f (k : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - 18*x^2 + 52*x + k

theorem other_factor (k : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, f k x = (x - 2) * c) → 
  (∃ d : ℝ, ∀ x : ℝ, f k x = (x + 5) * d) :=
sorry

end other_factor_l2247_224750


namespace triangle_side_length_l2247_224719

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → c = 4 → Real.cos C = -(1/4 : ℝ) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  b = 7/2 := by sorry

end triangle_side_length_l2247_224719


namespace sum_equals_3004_5_l2247_224775

/-- Define the recursive function for the sum -/
def S (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 3 + (1/3) * 2
  else (2003 - n + 1 : ℚ) + (1/3) * S (n-1)

/-- The main theorem stating that S(2001) equals 3004.5 -/
theorem sum_equals_3004_5 : S 2001 = 3004.5 := by
  sorry

end sum_equals_3004_5_l2247_224775


namespace point_outside_circle_l2247_224733

theorem point_outside_circle (a b : ℝ) :
  (∃ x y, x^2 + y^2 = 1 ∧ a*x + b*y = 1) →
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l2247_224733


namespace count_valid_numbers_l2247_224774

def is_valid_number (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  let a := tens + units
  10 ≤ n ∧ n < 100 ∧
  (3*n % 10 + 5*n % 10 + 7*n % 10 + 9*n % 10 = a)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 3 :=
sorry

end count_valid_numbers_l2247_224774


namespace solution_difference_l2247_224792

theorem solution_difference (p q : ℝ) : 
  p ≠ q → 
  ((6 * p - 18) / (p^2 + 4*p - 21) = p + 3) →
  ((6 * q - 18) / (q^2 + 4*q - 21) = q + 3) →
  p > q →
  p - q = 2 := by
sorry

end solution_difference_l2247_224792


namespace estimated_weight_not_exact_weight_estimated_weight_is_approximation_l2247_224793

/-- Represents the linear regression model for height and weight --/
structure HeightWeightModel where
  slope : ℝ
  intercept : ℝ

/-- The estimated weight based on the linear regression model --/
def estimated_weight (model : HeightWeightModel) (height : ℝ) : ℝ :=
  model.slope * height + model.intercept

/-- The given linear regression model for the problem --/
def given_model : HeightWeightModel :=
  { slope := 0.85, intercept := -85.71 }

/-- Theorem stating that the estimated weight for a 160cm tall girl is not necessarily her exact weight --/
theorem estimated_weight_not_exact_weight :
  ∃ (actual_weight : ℝ), 
    estimated_weight given_model 160 ≠ actual_weight ∧ 
    actual_weight > 0 := by
  sorry

/-- Theorem stating that the estimated weight is just an approximation --/
theorem estimated_weight_is_approximation (height : ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ 
    ∀ (actual_weight : ℝ), 
      actual_weight > 0 →
      |estimated_weight given_model height - actual_weight| < ε := by
  sorry

end estimated_weight_not_exact_weight_estimated_weight_is_approximation_l2247_224793


namespace fraction_is_one_fifth_l2247_224700

/-- The total number of states in the collection -/
def total_states : ℕ := 50

/-- The number of states that joined the union between 1790 and 1809 -/
def states_1790_1809 : ℕ := 10

/-- The fraction of states that joined between 1790 and 1809 -/
def fraction_1790_1809 : ℚ := states_1790_1809 / total_states

theorem fraction_is_one_fifth : fraction_1790_1809 = 1 / 5 := by
  sorry

end fraction_is_one_fifth_l2247_224700


namespace cube_root_simplification_l2247_224710

theorem cube_root_simplification :
  ∀ (x : ℝ), x > 0 → (x^(1/3) : ℝ) = (154^(1/3) / 9^(1/3) : ℝ) ↔ x = 17 + 1/9 :=
by sorry

end cube_root_simplification_l2247_224710


namespace intersection_M_N_l2247_224756

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def N : Set ℕ := {x | Real.sqrt (2^x - 1) < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by sorry

end intersection_M_N_l2247_224756


namespace discriminant_positive_increasing_when_m_le_8_l2247_224764

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m - 2) * x - m

-- Theorem 1: The discriminant is always positive
theorem discriminant_positive (m : ℝ) : m^2 + 4 > 0 := by sorry

-- Theorem 2: The function is increasing for x ≥ 3 when m ≤ 8
theorem increasing_when_m_le_8 (m : ℝ) (h : m ≤ 8) :
  ∀ x ≥ 3, ∀ y > x, f m y > f m x := by sorry

end discriminant_positive_increasing_when_m_le_8_l2247_224764


namespace intersection_empty_intersection_equals_A_l2247_224709

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x : ℝ | x^2 - 4*x - 12 > 0}

-- Theorem 1: A ∩ B = ∅ iff -2 ≤ a ≤ 3
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

-- Theorem 2: A ∩ B = A iff a < -5 or a > 6
theorem intersection_equals_A (a : ℝ) : A a ∩ B = A a ↔ a < -5 ∨ a > 6 := by
  sorry

end intersection_empty_intersection_equals_A_l2247_224709


namespace hyperbola_properties_l2247_224717

/-- Definition of the hyperbola C -/
def C (x y : ℝ) : Prop := y = Real.sqrt 3 * (1 / (2 * x) + x / 3)

/-- C is a hyperbola -/
axiom C_is_hyperbola : ∃ (a b : ℝ), ∀ (x y : ℝ), C x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1

/-- Statement about the asymptote, focus, and intersection properties of C -/
theorem hyperbola_properties :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → ∃ y, C x y ∧ |y| > 1/ε) ∧ 
  C 1 (Real.sqrt 3) ∧
  (∀ t : ℝ, ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ y₁ = x₁ + t ∧ y₂ = x₂ + t) :=
sorry

end hyperbola_properties_l2247_224717


namespace water_tank_capacity_l2247_224795

theorem water_tank_capacity (x : ℝ) : 
  (2/3 : ℝ) * x - (1/3 : ℝ) * x = 15 → x = 45 := by
  sorry

end water_tank_capacity_l2247_224795


namespace geometric_sequence_product_l2247_224708

/-- Given a geometric sequence {a_n} where a₄ = 4, prove that a₂ * a₆ = 16 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 4 = 4 →                            -- given condition
  a 2 * a 6 = 16 :=                    -- theorem to prove
by
  sorry

end geometric_sequence_product_l2247_224708


namespace solution_implies_m_range_l2247_224723

/-- A function representing the quadratic equation x^2 - mx + 2 = 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- The theorem stating that if the equation x^2 - mx + 2 = 0 has a solution 
    in the interval [1, 2], then m is in the range [2√2, 3] -/
theorem solution_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f m x = 0) → 
  m ∈ Set.Icc (2 * Real.sqrt 2) 3 := by
  sorry


end solution_implies_m_range_l2247_224723


namespace repeating_decimal_as_fraction_sum_of_numerator_and_denominator_l2247_224761

def repeating_decimal : ℚ := 0.363636

theorem repeating_decimal_as_fraction :
  repeating_decimal = 4 / 11 :=
sorry

theorem sum_of_numerator_and_denominator :
  (4 : ℕ) + 11 = 15 :=
sorry

end repeating_decimal_as_fraction_sum_of_numerator_and_denominator_l2247_224761


namespace area_of_special_triangle_l2247_224726

/-- A scalene triangle with given properties -/
structure ScaleneTriangle where
  -- A, B, C are the angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- rA, rB, rC are the exradii
  rA : ℝ
  rB : ℝ
  rC : ℝ
  -- Conditions
  angle_sum : A + B + C = π
  exradii_condition : 20 * (rB^2 * rC^2 + rC^2 * rA^2 + rA^2 * rB^2) = 19 * (rA * rB * rC)^2
  tan_sum : Real.tan (A/2) + Real.tan (B/2) + Real.tan (C/2) = 2.019
  inradius : ℝ := 1

/-- The area of a scalene triangle with the given properties is 2019/25 -/
theorem area_of_special_triangle (t : ScaleneTriangle) : 
  (2 * t.inradius * (Real.tan (t.A/2) + Real.tan (t.B/2) + Real.tan (t.C/2))) = 2019/25 := by
  sorry

end area_of_special_triangle_l2247_224726


namespace angle_measure_proof_l2247_224707

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = (180 - x) / 2 - 25) → x = 50 := by
  sorry

end angle_measure_proof_l2247_224707


namespace equation_solution_l2247_224771

theorem equation_solution :
  ∃! x : ℚ, (x + 2 ≠ 0) ∧ ((x^2 + 2*x + 3) / (x + 2) = x + 4) :=
by
  -- The unique solution is x = -5/4
  use -5/4
  sorry

end equation_solution_l2247_224771


namespace x_value_l2247_224783

theorem x_value : ∃ x : ℝ, x ≠ 0 ∧ x = 3 * (1 / x * (-x)) + 3 → x = 0 := by
  sorry

end x_value_l2247_224783


namespace pyramid_face_area_l2247_224799

/-- The total area of triangular faces of a right square-based pyramid -/
theorem pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) : 
  base_edge = 8 → lateral_edge = 9 → 
  (4 * (1/2 * base_edge * Real.sqrt ((lateral_edge^2) - (base_edge/2)^2))) = 16 * Real.sqrt 65 := by
  sorry

end pyramid_face_area_l2247_224799


namespace all_points_on_line_l2247_224751

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def isOnLine (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem all_points_on_line :
  let p1 : Point := ⟨8, 2⟩
  let p2 : Point := ⟨2, -10⟩
  let points : List Point := [⟨5, -4⟩, ⟨4, -6⟩, ⟨10, 6⟩, ⟨0, -14⟩, ⟨1, -12⟩]
  ∀ p ∈ points, isOnLine p p1 p2 := by
  sorry

end all_points_on_line_l2247_224751


namespace computers_produced_per_month_l2247_224724

/-- The number of days in a month -/
def days_per_month : ℕ := 28

/-- The number of computers produced in 30 minutes -/
def computers_per_interval : ℕ := 3

/-- The number of 30-minute intervals in a day -/
def intervals_per_day : ℕ := 24 * 2

/-- Calculates the number of computers produced in a month -/
def computers_per_month : ℕ :=
  days_per_month * intervals_per_day * computers_per_interval

/-- Theorem stating that the number of computers produced per month is 4032 -/
theorem computers_produced_per_month :
  computers_per_month = 4032 := by
  sorry


end computers_produced_per_month_l2247_224724


namespace inequality_solution_set_l2247_224773

theorem inequality_solution_set (x : ℝ) : 
  |5 - 2*x| - 1 > 0 ↔ x < 2 ∨ x > 3 := by sorry

end inequality_solution_set_l2247_224773


namespace arithmetic_sequence_sum_l2247_224725

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : 
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  m ≥ 2 →
  S (m - 1) = 16 →
  S m = 25 →
  S (m + 2) = 49 →
  m = 5 := by
sorry

end arithmetic_sequence_sum_l2247_224725


namespace f_is_quadratic_l2247_224713

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

#check f_is_quadratic

end f_is_quadratic_l2247_224713


namespace initial_mean_calculation_l2247_224758

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (new_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 48 ∧ 
  new_mean = 41.5 →
  ∃ (initial_mean : ℝ),
    initial_mean * n + (correct_value - wrong_value) = new_mean * n ∧
    initial_mean = 41 :=
by sorry

end initial_mean_calculation_l2247_224758


namespace square_perimeter_l2247_224703

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ (r : ℝ × ℝ), r.1 = s/2 ∧ r.2 = s ∧ 2*(r.1 + r.2) = 24) → 4*s = 32 := by
  sorry

end square_perimeter_l2247_224703


namespace sine_cosine_values_l2247_224748

def angle_on_line (α : Real) : Prop :=
  ∃ (x y : Real), y = Real.sqrt 3 * x ∧ 
  (Real.cos α = x / Real.sqrt (x^2 + y^2)) ∧
  (Real.sin α = y / Real.sqrt (x^2 + y^2))

theorem sine_cosine_values (α : Real) (h : angle_on_line α) :
  (Real.sin α = Real.sqrt 3 / 2 ∧ Real.cos α = 1 / 2) ∨
  (Real.sin α = -Real.sqrt 3 / 2 ∧ Real.cos α = -1 / 2) := by
  sorry

end sine_cosine_values_l2247_224748


namespace partial_fraction_decomposition_l2247_224742

theorem partial_fraction_decomposition (x : ℝ) 
  (h1 : x ≠ 7/8) (h2 : x ≠ 4/5) (h3 : x ≠ 1/2) :
  (306 * x^2 - 450 * x + 162) / ((8*x-7)*(5*x-4)*(2*x-1)) = 
  9 / (8*x-7) + 6 / (5*x-4) + 3 / (2*x-1) := by
  sorry

end partial_fraction_decomposition_l2247_224742


namespace min_tangent_length_l2247_224746

/-- The minimum length of a tangent from a point on the line x - y + 1 = 0 to the circle (x - 2)² + (y + 1)² = 1 is √7 -/
theorem min_tangent_length (x y : ℝ) : 
  let line := {(x, y) | x - y + 1 = 0}
  let circle := {(x, y) | (x - 2)^2 + (y + 1)^2 = 1}
  let tangent_length (p : ℝ × ℝ) := 
    Real.sqrt ((p.1 - 2)^2 + (p.2 + 1)^2 - 1)
  ∃ (p : ℝ × ℝ), p ∈ line ∧ 
    ∀ (q : ℝ × ℝ), q ∈ line → tangent_length p ≤ tangent_length q ∧
    tangent_length p = Real.sqrt 7 :=
by sorry


end min_tangent_length_l2247_224746


namespace largest_divisors_ratio_l2247_224784

theorem largest_divisors_ratio (N : ℕ) (h1 : N > 1) 
  (h2 : ∃ (a : ℕ), a ∣ N ∧ 6 * a ∣ N ∧ a ≠ 1 ∧ 6 * a ≠ N) :
  (N / 2) / (N / 3) = 3 / 2 := by
sorry

end largest_divisors_ratio_l2247_224784


namespace minimum_value_theorem_l2247_224741

theorem minimum_value_theorem (x : ℝ) (h : x > -2) :
  x + 1 / (x + 2) ≥ 0 ∧ ∃ y > -2, y + 1 / (y + 2) = 0 := by
  sorry

end minimum_value_theorem_l2247_224741


namespace nights_with_new_habit_l2247_224736

/-- Represents the number of nights a candle lasts when burned for 1 hour per night -/
def initial_nights_per_candle : ℕ := 8

/-- Represents the number of hours Carmen burns a candle each night after changing her habit -/
def hours_per_night : ℕ := 2

/-- Represents the total number of candles Carmen uses -/
def total_candles : ℕ := 6

/-- Theorem stating the total number of nights Carmen can burn candles with the new habit -/
theorem nights_with_new_habit : 
  (total_candles * initial_nights_per_candle) / hours_per_night = 24 := by
  sorry

end nights_with_new_habit_l2247_224736


namespace function_characterization_l2247_224759

/-- The set of positive rational numbers -/
def PositiveRationals : Set ℚ := {x : ℚ | 0 < x}

/-- The condition on x, y, z -/
def Condition (x y z : ℚ) : Prop := (x + y + z + 1 = 4 * x * y * z) ∧ (x ∈ PositiveRationals) ∧ (y ∈ PositiveRationals) ∧ (z ∈ PositiveRationals)

/-- The property that f must satisfy -/
def SatisfiesProperty (f : ℚ → ℝ) : Prop :=
  ∀ x y z, Condition x y z → f x + f y + f z = 1

/-- The theorem statement -/
theorem function_characterization :
  ∀ f : ℚ → ℝ, (∀ x ∈ PositiveRationals, f x = f x) →
    SatisfiesProperty f →
    ∃ a : ℝ, ∀ x ∈ PositiveRationals, f x = a * (1 / (2 * x + 1)) + (1 - a) * (1 / 3) :=
sorry

end function_characterization_l2247_224759


namespace inequality_iff_solution_set_l2247_224738

def inequality (x : ℝ) : Prop :=
  (3 / (x + 2)) + (4 / (x + 6)) > 1

def solution_set (x : ℝ) : Prop :=
  x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
by sorry

end inequality_iff_solution_set_l2247_224738


namespace set_equality_l2247_224754

def U := Set ℝ

def M : Set ℝ := {x | x > -1}

def N : Set ℝ := {x | -2 < x ∧ x < 3}

theorem set_equality : {x : ℝ | x ≤ -2} = (M ∪ N)ᶜ := by sorry

end set_equality_l2247_224754


namespace cyclic_quadrilateral_max_area_l2247_224757

/-- A quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral (a b c d : ℝ) where
  angle_sum : ℝ -- Sum of all interior angles
  area : ℝ -- Area of the quadrilateral

/-- Definition of a cyclic quadrilateral -/
def is_cyclic (q : Quadrilateral a b c d) : Prop :=
  q.angle_sum = 2 * Real.pi

/-- Theorem: Among all quadrilaterals with given side lengths, 
    the cyclic quadrilateral has the largest area -/
theorem cyclic_quadrilateral_max_area 
  {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∀ q : Quadrilateral a b c d, 
    ∃ q_cyclic : Quadrilateral a b c d, 
      is_cyclic q_cyclic ∧ q.area ≤ q_cyclic.area :=
sorry

end cyclic_quadrilateral_max_area_l2247_224757


namespace express_delivery_growth_rate_l2247_224714

/-- Proves that the equation 5000(1+x)^2 = 7500 correctly represents the average annual growth rate
    for an initial value of 5000, a final value of 7500, over a 2-year period. -/
theorem express_delivery_growth_rate (x : ℝ) : 
  (5000 : ℝ) * (1 + x)^2 = 7500 ↔ 
  (∃ (initial final : ℝ) (years : ℕ), 
    initial = 5000 ∧ 
    final = 7500 ∧ 
    years = 2 ∧ 
    final = initial * (1 + x)^years) :=
by sorry

end express_delivery_growth_rate_l2247_224714


namespace paul_chickens_left_l2247_224769

/-- The number of chickens Paul has left after selling some -/
def chickens_left (initial : ℕ) (sold_neighbor : ℕ) (sold_gate : ℕ) : ℕ :=
  initial - sold_neighbor - sold_gate

/-- Theorem stating that Paul is left with 43 chickens -/
theorem paul_chickens_left : chickens_left 80 12 25 = 43 := by
  sorry

end paul_chickens_left_l2247_224769


namespace longest_to_shortest_l2247_224798

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hyp_short : shorterLeg = hypotenuse / 2
  hyp_long : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
structure FourTriangles where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  hyp_relation1 : t1.longerLeg = t2.hypotenuse
  hyp_relation2 : t2.longerLeg = t3.hypotenuse
  hyp_relation3 : t3.longerLeg = t4.hypotenuse

theorem longest_to_shortest (triangles : FourTriangles) 
    (h : triangles.t1.hypotenuse = 16) : 
    triangles.t4.longerLeg = 9 := by
  sorry

end longest_to_shortest_l2247_224798


namespace smallest_land_fraction_150_members_l2247_224752

/-- Represents a noble family with land division rules -/
structure NobleFamily :=
  (total_members : ℕ)
  (founder_land : ℝ)
  (divide_land : ℝ → ℕ → ℝ)
  (transfer_to_state : ℝ → ℝ)

/-- The smallest possible fraction of land a family member could receive -/
def smallest_land_fraction (family : NobleFamily) : ℚ :=
  1 / (2 * 3^49)

/-- Theorem stating the smallest possible fraction of land for a family of 150 members -/
theorem smallest_land_fraction_150_members 
  (family : NobleFamily) 
  (h_members : family.total_members = 150) :
  smallest_land_fraction family = 1 / (2 * 3^49) :=
sorry

end smallest_land_fraction_150_members_l2247_224752


namespace cube_surface_area_l2247_224744

/-- Given a cube with volume 729 cubic centimeters, its surface area is 486 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) : 
  volume = 729 → 
  volume = side ^ 3 → 
  6 * side ^ 2 = 486 := by
sorry

end cube_surface_area_l2247_224744


namespace notebooks_ordered_l2247_224716

theorem notebooks_ordered (initial final lost : ℕ) (h1 : initial = 4) (h2 : lost = 2) (h3 : final = 8) :
  ∃ ordered : ℕ, initial + ordered - lost = final ∧ ordered = 6 := by
  sorry

end notebooks_ordered_l2247_224716


namespace chocolate_bar_squares_l2247_224762

theorem chocolate_bar_squares (gerald_bars : ℕ) (students : ℕ) (squares_per_student : ℕ) :
  gerald_bars = 7 →
  students = 24 →
  squares_per_student = 7 →
  (gerald_bars + 2 * gerald_bars) * (squares_in_each_bar : ℕ) = students * squares_per_student →
  squares_in_each_bar = 8 :=
by sorry

end chocolate_bar_squares_l2247_224762


namespace rectangle_diagonal_l2247_224729

theorem rectangle_diagonal (a b d : ℝ) : 
  a = 6 → a * b = 48 → d^2 = a^2 + b^2 → d = 10 := by sorry

end rectangle_diagonal_l2247_224729


namespace sqrt_five_irrational_l2247_224785

theorem sqrt_five_irrational : Irrational (Real.sqrt 5) := by
  sorry

end sqrt_five_irrational_l2247_224785


namespace speed_A_calculation_l2247_224789

-- Define the speeds of A and B
def speed_A : ℝ := 5.913043478260869
def speed_B : ℝ := 7.555555555555555

-- Define the time when B overtakes A (in hours)
def overtake_time : ℝ := 1.8

-- Define the head start time of A (in hours)
def head_start : ℝ := 0.5

-- Theorem statement
theorem speed_A_calculation :
  speed_A * (overtake_time + head_start) = speed_B * overtake_time :=
by sorry

end speed_A_calculation_l2247_224789


namespace rental_fee_calculation_l2247_224787

/-- The rental fee for a truck, given the total cost, per-mile charge, and miles driven. -/
def rental_fee (total_cost per_mile_charge miles_driven : ℚ) : ℚ :=
  total_cost - per_mile_charge * miles_driven

/-- Theorem stating that the rental fee is $20.99 under the given conditions. -/
theorem rental_fee_calculation :
  rental_fee 95.74 0.25 299 = 20.99 := by
  sorry

end rental_fee_calculation_l2247_224787


namespace unique_function_satisfying_equation_l2247_224727

-- Define the property that a function must satisfy
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f (k * y)) = x + y

-- State the theorem
theorem unique_function_satisfying_equation (k : ℝ) (hk : k ≠ 0) :
  ∃! f : ℝ → ℝ, SatisfiesEquation f k ∧ f = id := by sorry

end unique_function_satisfying_equation_l2247_224727


namespace existence_of_special_number_l2247_224712

/-- Given a positive integer, returns the sum of its digits. -/
def sum_of_digits (m : ℕ+) : ℕ := sorry

/-- Given a positive integer, returns the number of its digits. -/
def num_digits (m : ℕ+) : ℕ := sorry

/-- Checks if all digits of a positive integer are non-zero. -/
def all_digits_nonzero (m : ℕ+) : Prop := sorry

theorem existence_of_special_number :
  ∀ n : ℕ+, ∃ m : ℕ+,
    (num_digits m = n) ∧
    (all_digits_nonzero m) ∧
    (m.val % sum_of_digits m = 0) :=
sorry

end existence_of_special_number_l2247_224712


namespace company_workers_problem_l2247_224763

theorem company_workers_problem (total_workers : ℕ) 
  (h1 : total_workers % 3 = 0)  -- Ensures total_workers is divisible by 3
  (h2 : total_workers ≠ 0)      -- Ensures total_workers is not zero
  (h3 : (total_workers / 3) % 5 = 0)  -- Ensures workers without plan is divisible by 5
  (h4 : (2 * total_workers / 3) % 5 = 0)  -- Ensures workers with plan is divisible by 5
  (h5 : 40 * (2 * total_workers / 3) / 100 = 128)  -- 128 male workers
  : (7 * total_workers / 15 : ℕ) = 224 := by
  sorry

end company_workers_problem_l2247_224763


namespace quadrilateral_area_l2247_224766

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Theorem: Area of quadrilateral ABCD is 62.5√3 -/
theorem quadrilateral_area (ABCD : Quadrilateral) :
  angle ABCD.A ABCD.B ABCD.C = π / 2 →
  angle ABCD.A ABCD.C ABCD.D = π / 3 →
  distance ABCD.A ABCD.C = 25 →
  distance ABCD.C ABCD.D = 10 →
  ∃ E : Point, distance ABCD.A E = 15 →
  area ABCD = 62.5 * Real.sqrt 3 := by
  sorry

end quadrilateral_area_l2247_224766


namespace polar_bears_research_l2247_224780

theorem polar_bears_research (time_per_round : ℕ) (sunday_rounds : ℕ) (total_time : ℕ) :
  time_per_round = 30 →
  sunday_rounds = 15 →
  total_time = 780 →
  ∃ (saturday_additional_rounds : ℕ),
    saturday_additional_rounds = 10 ∧
    total_time = time_per_round * (1 + saturday_additional_rounds + sunday_rounds) :=
by sorry

end polar_bears_research_l2247_224780


namespace arrangements_count_l2247_224743

/-- Represents the number of male students -/
def num_male_students : ℕ := 3

/-- Represents the number of female students -/
def num_female_students : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := num_male_students + num_female_students

/-- Represents whether female students can stand at the ends of the row -/
def female_at_ends : Prop := False

/-- Represents whether female students A and B can be adjacent to female student C -/
def female_AB_adjacent_C : Prop := False

/-- Calculates the number of different arrangements given the conditions -/
def num_arrangements : ℕ := 144

/-- Theorem stating that the number of arrangements is 144 given the conditions -/
theorem arrangements_count :
  num_male_students = 3 ∧
  num_female_students = 3 ∧
  total_students = num_male_students + num_female_students ∧
  ¬female_at_ends ∧
  ¬female_AB_adjacent_C →
  num_arrangements = 144 :=
by sorry

end arrangements_count_l2247_224743


namespace van_capacity_l2247_224749

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 33 →
  adults = 9 →
  vans = 6 →
  (students + adults) / vans = 7 :=
by
  sorry

end van_capacity_l2247_224749


namespace first_year_after_2010_with_digit_sum_3_l2247_224722

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year < 3000 ∧ sum_of_digits year = 3

theorem first_year_after_2010_with_digit_sum_3 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2100 :=
sorry

end first_year_after_2010_with_digit_sum_3_l2247_224722


namespace books_sold_l2247_224711

theorem books_sold (initial_books final_books : ℕ) 
  (h1 : initial_books = 255)
  (h2 : final_books = 145) :
  initial_books - final_books = 110 := by
  sorry

end books_sold_l2247_224711


namespace complex_purely_imaginary_l2247_224728

theorem complex_purely_imaginary (a : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  ((a : ℂ) + Complex.I) / ((1 : ℂ) + 2 * Complex.I) = Complex.I * ((1 - 2 * a : ℝ) / 5) →
  a = -2 := by
sorry

end complex_purely_imaginary_l2247_224728


namespace fruit_basket_problem_l2247_224721

theorem fruit_basket_problem (total_fruit : ℕ) 
  (jacques_apples jacques_pears gillian_apples gillian_pears : ℕ) : 
  total_fruit = 25 →
  jacques_apples = 1 →
  jacques_pears = 3 →
  gillian_apples = 3 →
  gillian_pears = 2 →
  ∃ (initial_apples initial_pears : ℕ),
    initial_apples + initial_pears = total_fruit ∧
    initial_apples - jacques_apples - gillian_apples = 
      initial_pears - jacques_pears - gillian_pears →
  initial_pears = 13 := by
sorry

end fruit_basket_problem_l2247_224721


namespace B_roster_l2247_224776

def A : Set ℤ := {-2, 2, 3, 4}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_roster : B = {4, 9, 16} := by
  sorry

end B_roster_l2247_224776


namespace geometric_progression_sum_180_l2247_224720

theorem geometric_progression_sum_180 :
  ∃ (a b c d : ℝ) (e f g h : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧
    a + b + c + d = 180 ∧
    b / a = c / b ∧ c / b = d / c ∧
    c = a + 36 ∧
    e + f + g + h = 180 ∧
    f / e = g / f ∧ g / f = h / g ∧
    g = e + 36 ∧
    ((a = 9/2 ∧ b = 27/2 ∧ c = 81/2 ∧ d = 243/2) ∨
     (a = 12 ∧ b = 24 ∧ c = 48 ∧ d = 96)) ∧
    ((e = 9/2 ∧ f = 27/2 ∧ g = 81/2 ∧ h = 243/2) ∨
     (e = 12 ∧ f = 24 ∧ g = 48 ∧ h = 96)) ∧
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) :=
by sorry

end geometric_progression_sum_180_l2247_224720


namespace count_pairs_eq_738_l2247_224701

/-- The number of pairs (a, b) with 1 ≤ a < b ≤ 57 such that a^2 mod 57 < b^2 mod 57 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let (a, b) := p
    1 ≤ a ∧ a < b ∧ b ≤ 57 ∧ (a^2 % 57 < b^2 % 57))
    (Finset.product (Finset.range 58) (Finset.range 58))).card

theorem count_pairs_eq_738 : count_pairs = 738 := by
  sorry

end count_pairs_eq_738_l2247_224701


namespace clothing_distribution_l2247_224740

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 59)
  (h2 : first_load = 32)
  (h3 : num_small_loads = 9)
  : (total - first_load) / num_small_loads = 3 := by
  sorry

end clothing_distribution_l2247_224740


namespace prime_condition_characterization_l2247_224765

/-- The set of polynomials with coefficients from {0,1,...,p-1} and degree less than p -/
def K_p (p : ℕ) : Set (Polynomial ℤ) :=
  {f | ∀ i, (f.coeff i < p ∧ f.coeff i ≥ 0) ∧ f.degree < p}

/-- The condition that for all pairs of polynomials P,Q in K_p, 
    if P(Q(n)) ≡ n (mod p) for all integers n, then deg(P) = deg(Q) -/
def condition (p : ℕ) : Prop :=
  ∀ P Q : Polynomial ℤ, P ∈ K_p p → Q ∈ K_p p →
    (∀ n : ℤ, (P.comp Q).eval n ≡ n [ZMOD p]) →
    P.degree = Q.degree

theorem prime_condition_characterization :
  ∀ p : ℕ, p.Prime → (condition p ↔ p ∈ ({2, 3, 5, 7} : Set ℕ)) := by
  sorry

#check prime_condition_characterization

end prime_condition_characterization_l2247_224765
