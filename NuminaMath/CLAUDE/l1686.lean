import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_expression_l1686_168682

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) : 
  y / (2 * x - y) ≠ 6 / 1 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1686_168682


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1686_168640

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_coefficients :
  (∀ x, f (x + 5) = 4 * x^2 + 9 * x + 6) →
  (∃ a b c, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c, (∀ x, f x = a * x^2 + b * x + c) ∧ a + b + c = 34) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1686_168640


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1686_168603

theorem product_of_three_numbers (x y z n : ℝ) : 
  x + y + z = 150 ∧ 
  x ≤ y ∧ x ≤ z ∧ 
  z ≤ y ∧
  7 * x = n ∧ 
  y - 10 = n ∧ 
  z + 10 = n → 
  x * y * z = 48000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1686_168603


namespace NUMINAMATH_CALUDE_revenue_decrease_65_percent_l1686_168607

/-- Represents the change in revenue when tax is reduced and consumption is increased -/
def revenue_change (tax_reduction : ℝ) (consumption_increase : ℝ) : ℝ :=
  (1 - tax_reduction) * (1 + consumption_increase) - 1

/-- Theorem stating that a 15% tax reduction and 10% consumption increase results in a 6.5% revenue decrease -/
theorem revenue_decrease_65_percent :
  revenue_change 0.15 0.10 = -0.065 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_65_percent_l1686_168607


namespace NUMINAMATH_CALUDE_abc_value_l1686_168688

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a * (b + c) = 156)
  (eq2 : b * (c + a) = 168)
  (eq3 : c * (a + b) = 180) :
  a * b * c = 288 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1686_168688


namespace NUMINAMATH_CALUDE_perimeter_is_96_l1686_168684

/-- A figure composed of perpendicular line segments -/
structure PerpendicularFigure where
  x : ℝ
  y : ℝ
  area : ℝ
  x_eq_2y : x = 2 * y
  area_eq_252 : area = 252

/-- The perimeter of the perpendicular figure -/
def perimeter (f : PerpendicularFigure) : ℝ :=
  16 * f.y

theorem perimeter_is_96 (f : PerpendicularFigure) : perimeter f = 96 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_96_l1686_168684


namespace NUMINAMATH_CALUDE_san_diego_zoo_tickets_l1686_168651

/-- Given a family of 7 members visiting the San Diego Zoo, prove that 3 adult tickets were purchased. --/
theorem san_diego_zoo_tickets (total_cost : ℕ) (adult_price child_price : ℕ) 
  (h1 : total_cost = 119)
  (h2 : adult_price = 21)
  (h3 : child_price = 14) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = 7 ∧
    adult_tickets * adult_price + child_tickets * child_price = total_cost ∧
    adult_tickets = 3 := by
  sorry

end NUMINAMATH_CALUDE_san_diego_zoo_tickets_l1686_168651


namespace NUMINAMATH_CALUDE_f_value_at_half_l1686_168621

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is [a-1, 2a] -/
def HasDomain (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x ≠ 0 → a - 1 ≤ x ∧ x ≤ 2 * a

/-- The function f(x) = ax² + bx + 3a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + 3 * a + b

theorem f_value_at_half (a b : ℝ) :
  IsEven (f a b) → HasDomain (f a b) a → f a b (1/2) = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_half_l1686_168621


namespace NUMINAMATH_CALUDE_sophomore_count_l1686_168653

theorem sophomore_count (total : ℕ) (soph_percent : ℚ) (junior_percent : ℚ) :
  total = 36 →
  soph_percent = 1/5 →
  junior_percent = 3/20 →
  ∃ (soph junior : ℕ),
    soph + junior = total ∧
    soph_percent * soph = junior_percent * junior ∧
    soph = 16 :=
by sorry

end NUMINAMATH_CALUDE_sophomore_count_l1686_168653


namespace NUMINAMATH_CALUDE_sum_abc_equals_eight_l1686_168606

theorem sum_abc_equals_eight (a b c : ℝ) 
  (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 - 2*(a - 5)*(b - 6) = 0) : 
  a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_eight_l1686_168606


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_from_max_ratio_l1686_168652

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity_from_max_ratio (e : Ellipse) :
  (∀ p : ℝ × ℝ, ∃ q : ℝ × ℝ, distance p e.F₁ / distance p e.F₂ ≤ distance q e.F₁ / distance q e.F₂) →
  (∃ p : ℝ × ℝ, distance p e.F₁ / distance p e.F₂ = 3) →
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_from_max_ratio_l1686_168652


namespace NUMINAMATH_CALUDE_train_travel_time_l1686_168611

/-- Given a train that travels 360 miles in 3 hours, prove that it takes 2 hours to travel an additional 240 miles at the same rate. -/
theorem train_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) :
  initial_distance = 360 →
  initial_time = 3 →
  additional_distance = 240 →
  (additional_distance / (initial_distance / initial_time)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l1686_168611


namespace NUMINAMATH_CALUDE_special_function_is_x_plus_one_l1686_168627

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

/-- Theorem stating that the special function is x + 1 -/
theorem special_function_is_x_plus_one (f : ℝ → ℝ) (hf : special_function f) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_x_plus_one_l1686_168627


namespace NUMINAMATH_CALUDE_light_travel_distance_l1686_168679

/-- The distance light travels in one year in kilometers -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℝ := 70

/-- The expected distance light travels in the given number of years -/
def expected_distance : ℝ := 6.62256 * (10 ^ 14)

/-- Theorem stating that the distance light travels in the given number of years
    is equal to the expected distance -/
theorem light_travel_distance : light_year_distance * years = expected_distance := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1686_168679


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cone_l1686_168639

/-- The volume of a pyramid inscribed in a cone, where the pyramid's base is an isosceles triangle -/
theorem pyramid_volume_in_cone (V : ℝ) (α : ℝ) :
  let cone_volume := V
  let base_angle := α
  let pyramid_volume := (2 * V / Real.pi) * Real.sin α * (Real.cos (α / 2))^2
  0 < V → 0 < α → α < π →
  pyramid_volume = (2 * cone_volume / Real.pi) * Real.sin base_angle * (Real.cos (base_angle / 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cone_l1686_168639


namespace NUMINAMATH_CALUDE_sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1_l1686_168637

theorem sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1 :
  Real.sqrt 12 - ((-1 : ℝ) ^ (0 : ℕ)) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1_l1686_168637


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1686_168609

/-- For a regular polygon with exterior angles of 45 degrees, the sum of interior angles is 1080 degrees. -/
theorem regular_polygon_interior_angle_sum : 
  ∀ (n : ℕ), n > 2 → (360 / n = 45) → (n - 2) * 180 = 1080 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1686_168609


namespace NUMINAMATH_CALUDE_sign_determination_l1686_168691

theorem sign_determination (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l1686_168691


namespace NUMINAMATH_CALUDE_division_problem_l1686_168690

theorem division_problem :
  ∃ (quotient : ℕ), 136 = 15 * quotient + 1 ∧ quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1686_168690


namespace NUMINAMATH_CALUDE_union_equals_real_l1686_168677

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - 5| < a}

-- State the theorem
theorem union_equals_real (a : ℝ) (h : 11 ∈ B a) : A ∪ B a = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_real_l1686_168677


namespace NUMINAMATH_CALUDE_soccer_committee_count_l1686_168694

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The total number of possible organizing committees -/
def total_committees : ℕ := 34134175

theorem soccer_committee_count :
  (num_teams * (Nat.choose team_size host_selection) *
   (Nat.choose team_size non_host_selection ^ (num_teams - 1))) = total_committees := by
  sorry

end NUMINAMATH_CALUDE_soccer_committee_count_l1686_168694


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_53_l1686_168626

theorem least_positive_integer_multiple_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), 0 < y ∧ y < x → ¬(53 ∣ (2*y)^2 + 2*47*(2*y) + 47^2)) ∧
  (53 ∣ (2*x)^2 + 2*47*(2*x) + 47^2) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_53_l1686_168626


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l1686_168614

theorem least_k_for_inequality : 
  ∃ k : ℕ+, (∀ a : ℝ, a ∈ Set.Icc 0 1 → ∀ n : ℕ+, (a^(k:ℝ) * (1 - a)^(n:ℝ) < 1 / ((n:ℝ) + 1)^3)) ∧ 
  (∀ k' : ℕ+, k' < k → ∃ a : ℝ, a ∈ Set.Icc 0 1 ∧ ∃ n : ℕ+, a^(k':ℝ) * (1 - a)^(n:ℝ) ≥ 1 / ((n:ℝ) + 1)^3) ∧
  k = 4 :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l1686_168614


namespace NUMINAMATH_CALUDE_f_s_not_multiplicative_for_other_s_l1686_168638

/-- The count of integer solutions to x_1^2 + x_2^2 + ... + x_s^2 = n -/
def r_s (s n : ℕ) : ℕ := sorry

/-- f_s(n) = r_s(n) / (2s) -/
def f_s (s n : ℕ) : ℚ := (r_s s n : ℚ) / (2 * s : ℚ)

/-- The multiplication rule for f_s -/
def multiplication_rule (s : ℕ) : Prop :=
  ∀ m n : ℕ, Nat.Coprime m n → f_s s (m * n) = f_s s m * f_s s n

theorem f_s_not_multiplicative_for_other_s :
  ∀ s : ℕ, s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8 →
    ∃ m n : ℕ, f_s s (m * n) ≠ f_s s m * f_s s n :=
by sorry

end NUMINAMATH_CALUDE_f_s_not_multiplicative_for_other_s_l1686_168638


namespace NUMINAMATH_CALUDE_sampling_properties_l1686_168696

/-- Represents a club with male and female members -/
structure Club where
  male_members : ℕ
  female_members : ℕ

/-- Represents a sample drawn from the club -/
structure Sample where
  size : ℕ
  males_selected : ℕ
  females_selected : ℕ

/-- The probability of selecting a male from the club -/
def prob_select_male (c : Club) (s : Sample) : ℚ :=
  s.males_selected / c.male_members

/-- The probability of selecting a female from the club -/
def prob_select_female (c : Club) (s : Sample) : ℚ :=
  s.females_selected / c.female_members

/-- Theorem about the sampling properties of a specific club and sample -/
theorem sampling_properties (c : Club) (s : Sample) 
    (h_male : c.male_members = 30)
    (h_female : c.female_members = 20)
    (h_sample_size : s.size = 5)
    (h_males_selected : s.males_selected = 2)
    (h_females_selected : s.females_selected = 3) :
  (∃ (sampling_method : String), sampling_method = "random") ∧
  (¬ ∃ (sampling_method : String), sampling_method = "stratified") ∧
  prob_select_male c s < prob_select_female c s :=
by sorry

end NUMINAMATH_CALUDE_sampling_properties_l1686_168696


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l1686_168663

theorem walking_rate_ratio (usual_time new_time distance : ℝ) 
  (h1 : usual_time = 36)
  (h2 : new_time = usual_time - 4)
  (h3 : distance > 0)
  (h4 : usual_time > 0)
  (h5 : new_time > 0) :
  (distance / new_time) / (distance / usual_time) = 9 / 8 := by
sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l1686_168663


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1686_168644

theorem cube_volume_from_surface_area :
  ∀ s : ℝ, 
    s > 0 →
    6 * s^2 = 150 →
    s^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1686_168644


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l1686_168671

theorem set_equality_implies_a_equals_one (a : ℝ) :
  let A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
  let B : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}
  (A ∪ B) ⊆ (A ∩ B) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l1686_168671


namespace NUMINAMATH_CALUDE_alley_width_l1686_168612

theorem alley_width (l : ℝ) (h₁ h₂ : ℝ) (θ₁ θ₂ : ℝ) (w : ℝ) 
  (hl : l = 10)
  (hh₁ : h₁ = 4)
  (hh₂ : h₂ = 3)
  (hθ₁ : θ₁ = 30 * π / 180)
  (hθ₂ : θ₂ = 120 * π / 180) :
  w = 5 * (Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_alley_width_l1686_168612


namespace NUMINAMATH_CALUDE_picnic_cost_l1686_168692

def sandwich_price : ℚ := 6
def fruit_salad_price : ℚ := 4
def cheese_platter_price : ℚ := 8
def soda_price : ℚ := 2.5
def snack_bag_price : ℚ := 4.5

def num_people : ℕ := 6
def num_sandwiches : ℕ := 6
def num_fruit_salads : ℕ := 4
def num_cheese_platters : ℕ := 3
def num_sodas : ℕ := 12
def num_snack_bags : ℕ := 5

def sandwich_discount (n : ℕ) : ℕ := n / 6
def cheese_platter_discount (n : ℕ) : ℚ := if n ≥ 2 then 0.1 else 0
def soda_discount (n : ℕ) : ℕ := (n / 10) * 2
def snack_bag_discount (n : ℕ) : ℕ := n / 2

def total_cost : ℚ :=
  (num_sandwiches - sandwich_discount num_sandwiches) * sandwich_price +
  num_fruit_salads * fruit_salad_price +
  (num_cheese_platters * cheese_platter_price) * (1 - cheese_platter_discount num_cheese_platters) +
  (num_sodas - soda_discount num_sodas) * soda_price +
  num_snack_bags * snack_bag_price - snack_bag_discount num_snack_bags

theorem picnic_cost : total_cost = 113.1 := by
  sorry

end NUMINAMATH_CALUDE_picnic_cost_l1686_168692


namespace NUMINAMATH_CALUDE_coin_arrangements_l1686_168633

/-- Represents the number of gold coins -/
def gold_coins : ℕ := 6

/-- Represents the number of silver coins -/
def silver_coins : ℕ := 4

/-- Represents the total number of coins -/
def total_coins : ℕ := gold_coins + silver_coins

/-- Represents the number of possible color arrangements -/
def color_arrangements : ℕ := Nat.choose total_coins silver_coins

/-- Represents the number of possible face orientations -/
def face_orientations : ℕ := total_coins + 1

/-- The main theorem stating the number of distinguishable arrangements -/
theorem coin_arrangements :
  color_arrangements * face_orientations = 2310 := by sorry

end NUMINAMATH_CALUDE_coin_arrangements_l1686_168633


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_l1686_168646

/-- Given three points X, Y, and Z in a plane satisfying certain conditions, 
    prove that the sum of X's coordinates is 34. -/
theorem sum_of_x_coordinates (X Y Z : ℝ × ℝ) : 
  (dist X Z) / (dist X Y) = 2/3 →
  (dist Z Y) / (dist X Y) = 1/3 →
  Y = (1, 9) →
  Z = (-1, 3) →
  X.1 + X.2 = 34 := by sorry


end NUMINAMATH_CALUDE_sum_of_x_coordinates_l1686_168646


namespace NUMINAMATH_CALUDE_jellybean_distribution_l1686_168617

/-- Proves that given 70 jellybeans divided equally among 3 nephews and 2 nieces, each child receives 14 jellybeans. -/
theorem jellybean_distribution (total_jellybeans : ℕ) (num_nephews : ℕ) (num_nieces : ℕ)
  (h1 : total_jellybeans = 70)
  (h2 : num_nephews = 3)
  (h3 : num_nieces = 2) :
  total_jellybeans / (num_nephews + num_nieces) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_distribution_l1686_168617


namespace NUMINAMATH_CALUDE_total_get_well_cards_l1686_168634

/-- Represents the number of cards Mariela received in different categories -/
structure CardCounts where
  handwritten : ℕ
  multilingual : ℕ
  multiplePages : ℕ

/-- Calculates the total number of cards given the counts for each category -/
def totalCards (counts : CardCounts) : ℕ :=
  counts.handwritten + counts.multilingual + counts.multiplePages

/-- Theorem stating the total number of get well cards Mariela received -/
theorem total_get_well_cards 
  (hospital : CardCounts) 
  (home : CardCounts) 
  (h1 : hospital.handwritten = 152)
  (h2 : hospital.multilingual = 98)
  (h3 : hospital.multiplePages = 153)
  (h4 : totalCards hospital = 403)
  (h5 : home.handwritten = 121)
  (h6 : home.multilingual = 66)
  (h7 : home.multiplePages = 100)
  (h8 : totalCards home = 287) :
  totalCards hospital + totalCards home = 690 := by
  sorry

#check total_get_well_cards

end NUMINAMATH_CALUDE_total_get_well_cards_l1686_168634


namespace NUMINAMATH_CALUDE_coinciding_rest_days_l1686_168641

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 7

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 5

/-- Total number of days -/
def total_days : ℕ := 1200

/-- Number of rest days Al has in one cycle -/
def al_rest_days : ℕ := 2

/-- Number of rest days Barb has in one cycle -/
def barb_rest_days : ℕ := 1

/-- The theorem to prove -/
theorem coinciding_rest_days : 
  ∃ (n : ℕ), n = (total_days / (Nat.lcm al_cycle barb_cycle)) * 
    (al_rest_days * barb_rest_days) ∧ n = 34 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_l1686_168641


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1686_168673

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 15 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1686_168673


namespace NUMINAMATH_CALUDE_quadratic_function_comparison_l1686_168670

theorem quadratic_function_comparison (y₁ y₂ : ℝ) : 
  ((-1 : ℝ)^2 - 2*(-1) = y₁) → 
  ((2 : ℝ)^2 - 2*2 = y₂) → 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_function_comparison_l1686_168670


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1686_168660

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Structure representing a community's income distribution -/
structure CommunityIncome where
  totalFamilies : ℕ
  highIncome : ℕ
  middleIncome : ℕ
  lowIncome : ℕ
  high_income_valid : highIncome ≤ totalFamilies
  middle_income_valid : middleIncome ≤ totalFamilies
  low_income_valid : lowIncome ≤ totalFamilies
  total_sum_valid : highIncome + middleIncome + lowIncome = totalFamilies

/-- Function to determine the most appropriate sampling method -/
def mostAppropriateSamplingMethod (community : CommunityIncome) (sampleSize : ℕ) : SamplingMethod :=
  SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is most appropriate for the given community -/
theorem stratified_sampling_most_appropriate 
  (community : CommunityIncome) 
  (sampleSize : ℕ) 
  (sample_size_valid : sampleSize ≤ community.totalFamilies) :
  mostAppropriateSamplingMethod community sampleSize = SamplingMethod.Stratified :=
by
  sorry

#check stratified_sampling_most_appropriate

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1686_168660


namespace NUMINAMATH_CALUDE_sally_peaches_l1686_168657

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked : ℕ := 55

/-- The total number of peaches after picking -/
def total_peaches : ℕ := 68

/-- The initial number of peaches Sally had -/
def initial_peaches : ℕ := total_peaches - peaches_picked

theorem sally_peaches : initial_peaches + peaches_picked = total_peaches := by
  sorry

end NUMINAMATH_CALUDE_sally_peaches_l1686_168657


namespace NUMINAMATH_CALUDE_classroom_students_count_l1686_168604

theorem classroom_students_count : ∃! n : ℕ, n < 60 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ n = 22 := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_count_l1686_168604


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l1686_168687

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem no_roots_of_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l1686_168687


namespace NUMINAMATH_CALUDE_existence_of_cube_triplet_l1686_168658

theorem existence_of_cube_triplet :
  ∃ n₀ : ℕ, ∀ m : ℕ, m ≥ n₀ →
    ∃ a b c : ℕ+,
      (m ^ 3 : ℝ) < (a : ℝ) ∧
      (a : ℝ) < (b : ℝ) ∧
      (b : ℝ) < (c : ℝ) ∧
      (c : ℝ) < ((m + 1) ^ 3 : ℝ) ∧
      ∃ k : ℕ, (a * b * c : ℕ) = k ^ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_cube_triplet_l1686_168658


namespace NUMINAMATH_CALUDE_square_diff_plus_six_b_l1686_168648

theorem square_diff_plus_six_b (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6*b = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_plus_six_b_l1686_168648


namespace NUMINAMATH_CALUDE_sugar_per_cup_l1686_168630

def total_sugar : ℝ := 84.6
def num_cups : ℕ := 12

theorem sugar_per_cup : 
  (total_sugar / num_cups : ℝ) = 7.05 := by sorry

end NUMINAMATH_CALUDE_sugar_per_cup_l1686_168630


namespace NUMINAMATH_CALUDE_max_white_rooks_8x8_l1686_168605

/-- Represents a chessboard configuration with black and white rooks -/
structure ChessboardConfig where
  size : Nat
  blackRooks : Nat
  whiteRooks : Nat
  differentCells : Bool
  onlyAttackOpposite : Bool

/-- Defines the maximum number of white rooks for a given configuration -/
def maxWhiteRooks (config : ChessboardConfig) : Nat :=
  sorry

/-- Theorem stating the maximum number of white rooks for the given configuration -/
theorem max_white_rooks_8x8 :
  let config : ChessboardConfig := {
    size := 8,
    blackRooks := 6,
    whiteRooks := 14,
    differentCells := true,
    onlyAttackOpposite := true
  }
  maxWhiteRooks config = 14 := by sorry

end NUMINAMATH_CALUDE_max_white_rooks_8x8_l1686_168605


namespace NUMINAMATH_CALUDE_money_division_l1686_168680

theorem money_division (A B C : ℚ) (h1 : A = (1/3) * (B + C))
                                   (h2 : ∃ x, B = x * (A + C))
                                   (h3 : A = B + 15)
                                   (h4 : A + B + C = 540) :
  ∃ x, B = x * (A + C) ∧ x = 2/9 := by
sorry

end NUMINAMATH_CALUDE_money_division_l1686_168680


namespace NUMINAMATH_CALUDE_simplify_expression_l1686_168686

theorem simplify_expression (z y : ℝ) : (4 - 5*z + 2*y) - (6 + 7*z - 3*y) = -2 - 12*z + 5*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1686_168686


namespace NUMINAMATH_CALUDE_compare_powers_l1686_168661

theorem compare_powers : 
  let a : ℝ := 2^(4/3)
  let b : ℝ := 4^(2/5)
  let c : ℝ := 25^(1/3)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_compare_powers_l1686_168661


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1686_168649

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 15 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1686_168649


namespace NUMINAMATH_CALUDE_long_sleeve_shirts_count_l1686_168629

theorem long_sleeve_shirts_count (total_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9)
  (h2 : short_sleeve_shirts = 4) :
  total_shirts - short_sleeve_shirts = 5 := by
sorry

end NUMINAMATH_CALUDE_long_sleeve_shirts_count_l1686_168629


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1686_168659

/-- Given a function y = x(1-ax) where 0 < x < 1/a, if the maximum value of y is 1/12, then a = 3 -/
theorem max_value_implies_a (a : ℝ) : 
  (∃ (y : ℝ → ℝ), (∀ x : ℝ, 0 < x → x < 1/a → y x = x*(1-a*x)) ∧ 
   (∃ M : ℝ, M = 1/12 ∧ ∀ x : ℝ, 0 < x → x < 1/a → y x ≤ M)) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1686_168659


namespace NUMINAMATH_CALUDE_anya_original_position_l1686_168656

def Friend := Fin 5

structure Seating :=
  (positions : Friend → Fin 5)
  (bijective : Function.Bijective positions)

def sum_positions (s : Seating) : Nat :=
  (List.range 5).sum

-- Define the movements
def move_right (s : Seating) (f : Friend) (n : Nat) : Seating := sorry
def move_left (s : Seating) (f : Friend) (n : Nat) : Seating := sorry
def swap (s : Seating) (f1 f2 : Friend) : Seating := sorry
def move_to_end (s : Seating) (f : Friend) : Seating := sorry

theorem anya_original_position 
  (initial : Seating) 
  (anya varya galya diana ella : Friend) 
  (h_distinct : anya ≠ varya ∧ anya ≠ galya ∧ anya ≠ diana ∧ anya ≠ ella ∧ 
                varya ≠ galya ∧ varya ≠ diana ∧ varya ≠ ella ∧ 
                galya ≠ diana ∧ galya ≠ ella ∧ 
                diana ≠ ella) 
  (final : Seating) 
  (h_movements : final = move_to_end (swap (move_left (move_right initial varya 3) galya 1) diana ella) anya) 
  (h_sum_equal : sum_positions initial = sum_positions final) :
  initial.positions anya = 3 := by sorry

end NUMINAMATH_CALUDE_anya_original_position_l1686_168656


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l1686_168698

/-- Represents the profit function for helmet sales -/
def profit_function (x : ℝ) : ℝ :=
  -20 * x^2 + 1400 * x - 60000

/-- The optimal selling price for helmets -/
def optimal_price : ℝ := 70

theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function optimal_price ≥ profit_function x :=
sorry

#check optimal_price_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l1686_168698


namespace NUMINAMATH_CALUDE_sophomore_sample_size_l1686_168620

/-- Represents the number of students to be selected from a stratum in stratified sampling. -/
def stratified_sample (total_population : ℕ) (stratum_size : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_size * sample_size) / total_population

/-- Theorem stating that in the given stratified sampling scenario, 
    32 sophomores should be selected. -/
theorem sophomore_sample_size : 
  stratified_sample 2000 640 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_size_l1686_168620


namespace NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l1686_168672

/-- Calculates the total cost of James' shopping trip -/
def shopping_trip_cost : ℝ :=
  let milk_price : ℝ := 4.50
  let milk_tax_rate : ℝ := 0.20
  let bananas_price : ℝ := 3.00
  let bananas_tax_rate : ℝ := 0.15
  let baguette_price : ℝ := 2.50
  let cereal_price : ℝ := 6.00
  let cereal_discount : ℝ := 0.20
  let cereal_tax_rate : ℝ := 0.12
  let eggs_price : ℝ := 3.50
  let eggs_coupon : ℝ := 1.00
  let eggs_tax_rate : ℝ := 0.18

  let milk_total := milk_price * (1 + milk_tax_rate)
  let bananas_total := bananas_price * (1 + bananas_tax_rate)
  let baguette_total := baguette_price
  let cereal_discounted := cereal_price * (1 - cereal_discount)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_price - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)

  milk_total + bananas_total + baguette_total + cereal_total + eggs_total

theorem shopping_trip_cost_theorem : shopping_trip_cost = 19.68 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l1686_168672


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1686_168674

/-- A quadratic trinomial in x and y -/
def QuadraticTrinomial (a b c : ℝ) := fun (x y : ℝ) ↦ a*x^2 + b*x*y + c*y^2

/-- Predicate for a quadratic trinomial being a perfect square -/
def IsPerfectSquare (q : (ℝ → ℝ → ℝ)) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), q x y = (a*x + b*y)^2

theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquare (QuadraticTrinomial 4 m 9) → (m = 12 ∨ m = -12) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1686_168674


namespace NUMINAMATH_CALUDE_barbara_savings_l1686_168689

/-- The number of weeks needed to save for a wristwatch -/
def weeks_to_save (watch_cost : ℕ) (weekly_allowance : ℕ) (current_savings : ℕ) : ℕ :=
  ((watch_cost - current_savings) + weekly_allowance - 1) / weekly_allowance

/-- Theorem: Given the conditions, Barbara needs 16 more weeks to save for the wristwatch -/
theorem barbara_savings : weeks_to_save 100 5 20 = 16 := by
  sorry

end NUMINAMATH_CALUDE_barbara_savings_l1686_168689


namespace NUMINAMATH_CALUDE_speed_of_k_l1686_168645

-- Define the speeds and time delay
def speed_a : ℝ := 30
def speed_b : ℝ := 40
def delay : ℝ := 5

-- Define the theorem
theorem speed_of_k (speed_k : ℝ) : 
  -- a, b, k start from the same place and travel in the same direction
  -- a travels at speed_a km/hr
  -- b travels at speed_b km/hr
  -- b starts delay hours after a
  -- b and k overtake a at the same instant
  -- k starts at the same time as a
  (∃ (t : ℝ), t > 0 ∧ 
    speed_b * t = speed_a * (t + delay) ∧
    speed_k * (t + delay) = speed_a * (t + delay)) →
  -- Then the speed of k is 35 km/hr
  speed_k = 35 := by
sorry

end NUMINAMATH_CALUDE_speed_of_k_l1686_168645


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1686_168602

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 60) (h2 : x = 37) : x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1686_168602


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1686_168625

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^2 - 4*y^2 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1686_168625


namespace NUMINAMATH_CALUDE_min_distance_parabola_point_l1686_168600

/-- The minimum distance from a point on the parabola to Q plus its y-coordinate -/
theorem min_distance_parabola_point (x y : ℝ) (h : x^2 = -4*y) :
  ∃ (min : ℝ), min = 2 ∧ 
  ∀ (x' y' : ℝ), x'^2 = -4*y' → 
    abs y + Real.sqrt ((x' + 2*Real.sqrt 2)^2 + y'^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_parabola_point_l1686_168600


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1686_168616

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (Ca I C H : ℕ) (wCa wI wC wH : ℝ) : ℝ :=
  Ca * wCa + I * wI + C * wC + H * wH

/-- The molecular weight of the given compound is 602.794 amu -/
theorem compound_molecular_weight :
  let Ca : ℕ := 2
  let I : ℕ := 4
  let C : ℕ := 1
  let H : ℕ := 3
  let wCa : ℝ := 40.08
  let wI : ℝ := 126.90
  let wC : ℝ := 12.01
  let wH : ℝ := 1.008
  molecularWeight Ca I C H wCa wI wC wH = 602.794 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1686_168616


namespace NUMINAMATH_CALUDE_crocodile_earnings_exceed_peter_l1686_168693

theorem crocodile_earnings_exceed_peter (n : ℕ) : (∀ k < n, 2^k ≤ 64*k + 1) ∧ 2^n > 64*n + 1 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_crocodile_earnings_exceed_peter_l1686_168693


namespace NUMINAMATH_CALUDE_jacks_mopping_rate_l1686_168676

theorem jacks_mopping_rate (bathroom_area kitchen_area total_time : ℝ) 
  (h1 : bathroom_area = 24)
  (h2 : kitchen_area = 80)
  (h3 : total_time = 13) :
  (bathroom_area + kitchen_area) / total_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_jacks_mopping_rate_l1686_168676


namespace NUMINAMATH_CALUDE_area_triangle_DBC_l1686_168635

/-- Given a triangle ABC with vertices A(0,10), B(0,0), and C(12,0),
    and midpoints D of AB, E of BC, and F of AC,
    prove that the area of triangle DBC is 30. -/
theorem area_triangle_DBC (A B C D E F : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (12, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (1/2 : ℝ) * (C.1 - B.1) * D.2 = 30 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_DBC_l1686_168635


namespace NUMINAMATH_CALUDE_xiao_ming_math_score_l1686_168615

theorem xiao_ming_math_score :
  let average_three := 94
  let subjects := 3
  let average_two := average_three - 1
  let total_score := average_three * subjects
  let chinese_english_score := average_two * (subjects - 1)
  total_score - chinese_english_score = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_math_score_l1686_168615


namespace NUMINAMATH_CALUDE_clark_number_is_23_l1686_168610

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digits_form_unique_prime (n : ℕ) : Prop :=
  is_prime n ∧
  n < 100 ∧
  ∀ m : ℕ, m ≠ n → (m = n % 10 * 10 + n / 10 ∨ m = n) → ¬(is_prime m)

def digits_are_ambiguous (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_prime m ∧ 
    ((m % 10 = n % 10 ∧ m / 10 = n / 10) ∨ 
     (m % 10 = n / 10 ∧ m / 10 = n % 10))

theorem clark_number_is_23 :
  ∃! n : ℕ, digits_form_unique_prime n ∧ digits_are_ambiguous n ∧ n = 23 :=
sorry

end NUMINAMATH_CALUDE_clark_number_is_23_l1686_168610


namespace NUMINAMATH_CALUDE_addition_puzzle_l1686_168666

theorem addition_puzzle (A B C D : Nat) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  700 + 10 * A + 5 + 100 * B + 70 + C = 900 + 30 + 8 →
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_addition_puzzle_l1686_168666


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l1686_168622

/-- Calculates the value of a machine after a given number of years, 
    given its initial value and yearly depreciation rate. -/
def machine_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

/-- Theorem stating that a machine purchased for $8,000 with a 10% yearly depreciation rate
    will have a value of $6,480 after two years. -/
theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 := by
  sorry

#eval machine_value 8000 0.1 2

end NUMINAMATH_CALUDE_machine_value_after_two_years_l1686_168622


namespace NUMINAMATH_CALUDE_angle_is_135_degrees_l1686_168695

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_135_degrees (a b : ℝ × ℝ) 
  (sum_condition : a.1 + b.1 = 2 ∧ a.2 + b.2 = -1)
  (a_condition : a = (1, 2)) :
  angle_between_vectors a b = 135 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_angle_is_135_degrees_l1686_168695


namespace NUMINAMATH_CALUDE_unique_n_exists_l1686_168642

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem unique_n_exists : ∃! n : ℕ, n > 0 ∧ n + S n + S (S n) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_exists_l1686_168642


namespace NUMINAMATH_CALUDE_range_of_m_l1686_168697

def is_circle (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 2*m*x + 2*m^2 - 2*m = 0

def hyperbola_eccentricity_in_range (m : ℝ) : Prop :=
  ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ e^2 = 1 + m/5

def p (m : ℝ) : Prop := is_circle m
def q (m : ℝ) : Prop := hyperbola_eccentricity_in_range m

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (2 ≤ m ∧ m < 15) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1686_168697


namespace NUMINAMATH_CALUDE_problem_equivalence_l1686_168643

theorem problem_equivalence (y x : ℝ) (h : x ≠ -1) :
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 ∧
  (1 + 2 / (x + 1)) / ((x^2 + 6*x + 9) / (x + 1)) = 1 / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_equivalence_l1686_168643


namespace NUMINAMATH_CALUDE_jacket_price_proof_l1686_168628

theorem jacket_price_proof (S P : ℝ) (h1 : S = P + 0.4 * S) 
  (h2 : 0.8 * S - P = 18) : P = 54 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_proof_l1686_168628


namespace NUMINAMATH_CALUDE_expression_evaluation_l1686_168608

theorem expression_evaluation :
  let a : ℚ := -3/2
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1686_168608


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l1686_168623

/-- Given a circle and a parabola, if the circle is tangent to the directrix of the parabola,
    then the parameter p of the parabola is 2. -/
theorem circle_tangent_to_parabola_directrix (x y : ℝ) (p : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) →  -- Circle equation
  (p > 0) →                   -- p is positive
  (∃ (y : ℝ), y^2 = 2*p*x) →  -- Parabola equation
  (∃ (x₀ : ℝ), ∀ (x y : ℝ), x^2 + y^2 - 6*x - 7 = 0 → |x - x₀| ≥ p/2) →  -- Circle is tangent to directrix
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l1686_168623


namespace NUMINAMATH_CALUDE_evaluate_expression_l1686_168699

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3*x^3 - 5*x^2 + 9*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1686_168699


namespace NUMINAMATH_CALUDE_vector_magnitude_condition_l1686_168678

theorem vector_magnitude_condition (n : Type*) [NormedAddCommGroup n] :
  ∃ (a b : n),
    (‖a‖ = ‖b‖ ∧ ‖a + b‖ ≠ ‖a - b‖) ∧
    (‖a‖ ≠ ‖b‖ ∧ ‖a + b‖ = ‖a - b‖) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_condition_l1686_168678


namespace NUMINAMATH_CALUDE_grocery_store_buyers_difference_l1686_168655

/-- Given information about buyers in a grocery store over three days, 
    prove the difference in buyers between today and yesterday --/
theorem grocery_store_buyers_difference 
  (buyers_day_before_yesterday : ℕ) 
  (buyers_yesterday : ℕ) 
  (buyers_today : ℕ) 
  (total_buyers : ℕ) 
  (h1 : buyers_day_before_yesterday = 50)
  (h2 : buyers_yesterday = buyers_day_before_yesterday / 2)
  (h3 : total_buyers = buyers_day_before_yesterday + buyers_yesterday + buyers_today)
  (h4 : total_buyers = 140) :
  buyers_today - buyers_yesterday = 40 := by
sorry


end NUMINAMATH_CALUDE_grocery_store_buyers_difference_l1686_168655


namespace NUMINAMATH_CALUDE_total_seedlings_transferred_l1686_168667

def seedlings_day1 : ℕ := 200

def seedlings_day2 (day1 : ℕ) : ℕ := 2 * day1

theorem total_seedlings_transferred : 
  seedlings_day1 + seedlings_day2 seedlings_day1 = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_seedlings_transferred_l1686_168667


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1686_168669

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (b^2 - 4*a*c < 0) = False := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1686_168669


namespace NUMINAMATH_CALUDE_range_of_positives_in_K_l1686_168632

/-- Definition of the list K -/
def list_K : List ℤ := List.range 40 |>.map (fun i => -25 + 3 * i)

/-- The range of positive integers in list K -/
def positive_range (L : List ℤ) : ℤ :=
  let positives := L.filter (· > 0)
  positives.maximum.getD 0 - positives.minimum.getD 0

/-- Theorem: The range of positive integers in list K is 90 -/
theorem range_of_positives_in_K : positive_range list_K = 90 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positives_in_K_l1686_168632


namespace NUMINAMATH_CALUDE_three_digit_power_of_2_and_5_l1686_168664

theorem three_digit_power_of_2_and_5 : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ m : ℕ, n = 2^m) ∧ 
  (∃ k : ℕ, n = 5^k) :=
sorry

end NUMINAMATH_CALUDE_three_digit_power_of_2_and_5_l1686_168664


namespace NUMINAMATH_CALUDE_proposition_B_is_false_l1686_168631

-- Define propositions as boolean variables
variable (p q : Prop)

-- Define the proposition B
def proposition_B (p q : Prop) : Prop :=
  (¬p ∧ ¬q) → (¬p ∧ ¬q)

-- Theorem stating that proposition B is false
theorem proposition_B_is_false :
  ∃ p q : Prop, ¬(proposition_B p q) :=
sorry

end NUMINAMATH_CALUDE_proposition_B_is_false_l1686_168631


namespace NUMINAMATH_CALUDE_min_value_theorem_l1686_168647

theorem min_value_theorem (x y : ℝ) :
  (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 ≥ 1/6 ∧
  ((y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 = 1/6 ↔ x = 5/2 ∧ y = 5/6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1686_168647


namespace NUMINAMATH_CALUDE_vegetables_in_soup_serving_l1686_168685

/-- Proves that the number of cups of vegetables in one serving of soup is 1 -/
theorem vegetables_in_soup_serving (V : ℝ) : V = 1 :=
  by
  -- One serving contains V cups of vegetables and 2.5 cups of broth
  have h1 : V + 2.5 = (14 * 2) / 8 := by sorry
  -- 8 servings require 14 pints of vegetables and broth combined
  -- 1 pint = 2 cups
  -- So, 14 pints = 14 * 2 cups = 28 cups
  -- Solve the equation: 8 * (V + 2.5) = 28
  sorry

end NUMINAMATH_CALUDE_vegetables_in_soup_serving_l1686_168685


namespace NUMINAMATH_CALUDE_exam_passing_marks_l1686_168624

theorem exam_passing_marks (T : ℝ) (P : ℝ) : 
  (0.3 * T = P - 60) →
  (0.4 * T + 10 = P) →
  (0.5 * T - 5 = P + 40) →
  P = 210 := by
  sorry

end NUMINAMATH_CALUDE_exam_passing_marks_l1686_168624


namespace NUMINAMATH_CALUDE_power_inequalities_l1686_168619

theorem power_inequalities :
  (∀ (x : ℝ), x > 1 → ∀ (a b : ℝ), 0 < a → a < b → x^a < x^b) ∧
  (∀ (x y z : ℝ), 1 < x → x < y → 0 < z → z < 1 → x^z > y^z) :=
sorry

end NUMINAMATH_CALUDE_power_inequalities_l1686_168619


namespace NUMINAMATH_CALUDE_equation_solution_l1686_168681

theorem equation_solution (a : ℝ) (x : ℝ) : 
  a = 3 → (5 * a - x = 13 ↔ x = 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1686_168681


namespace NUMINAMATH_CALUDE_fraction_sum_problem_l1686_168675

theorem fraction_sum_problem (x y : ℚ) (h : x / y = 2 / 7) : (x + y) / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_problem_l1686_168675


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l1686_168613

theorem magnitude_of_complex_number : Complex.abs (5/6 + 2*Complex.I) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l1686_168613


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l1686_168665

/-- The axis of symmetry of a shifted sine function -/
theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.sin (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 + π / 6
  ∀ x : ℝ, f (axis + x) = f (axis - x) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l1686_168665


namespace NUMINAMATH_CALUDE_xy_bounds_l1686_168683

theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_bounds_l1686_168683


namespace NUMINAMATH_CALUDE_min_value_expression_l1686_168650

theorem min_value_expression (a b c : ℕ) (h1 : b > a) (h2 : a > c) (h3 : c > 0) (h4 : b ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c - a)^2 + (a - c)^2 : ℚ) / (b^2 : ℚ) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1686_168650


namespace NUMINAMATH_CALUDE_c_7_equals_448_l1686_168618

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem c_7_equals_448 : c 7 = 448 := by
  sorry

end NUMINAMATH_CALUDE_c_7_equals_448_l1686_168618


namespace NUMINAMATH_CALUDE_book_has_180_pages_l1686_168668

/-- Calculates the number of pages in a book given reading habits and time to finish --/
def book_pages (weekday_pages : ℕ) (weekend_pages : ℕ) (weeks : ℕ) : ℕ :=
  let weekdays := 5 * weeks
  let weekends := 2 * weeks
  weekday_pages * weekdays + weekend_pages * weekends

/-- Theorem stating that a book has 180 pages given specific reading habits and time --/
theorem book_has_180_pages :
  book_pages 10 20 2 = 180 := by
  sorry

#eval book_pages 10 20 2

end NUMINAMATH_CALUDE_book_has_180_pages_l1686_168668


namespace NUMINAMATH_CALUDE_primeDivisorsOf50FactorialIs15_l1686_168654

/-- The number of prime divisors of 50! -/
def primeDivisorsOf50Factorial : ℕ :=
  (List.range 51).filter (fun n => n.Prime && n > 1) |>.length

/-- Theorem: The number of prime divisors of 50! is 15 -/
theorem primeDivisorsOf50FactorialIs15 : primeDivisorsOf50Factorial = 15 := by
  sorry

end NUMINAMATH_CALUDE_primeDivisorsOf50FactorialIs15_l1686_168654


namespace NUMINAMATH_CALUDE_library_books_count_l1686_168662

/-- The number of books in a library after two years of purchases -/
def library_books (initial_books : ℕ) (books_last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial_books + books_last_year + multiplier * books_last_year

/-- Theorem stating that the library now has 300 books -/
theorem library_books_count : library_books 100 50 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1686_168662


namespace NUMINAMATH_CALUDE_omega_range_l1686_168601

theorem omega_range (ω : ℝ) (f : ℝ → ℝ) : 
  ω > 0 → 
  (∀ x, f x = Real.sin (ω * x + π / 4)) →
  (∀ x y, π / 2 < x → x < y → y < π → f y < f x) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_omega_range_l1686_168601


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l1686_168636

theorem sum_of_distinct_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l1686_168636
