import Mathlib

namespace xyz_sum_l779_77991

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 24) (hxz : x * z = 48) (hyz : y * z = 72) :
  x + y + z = 22 := by
  sorry

end xyz_sum_l779_77991


namespace triangle_side_length_l779_77981

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (area : Real) :
  area = Real.sqrt 3 →
  B = 60 * π / 180 →
  a^2 + c^2 = 3 * a * c →
  b = 2 * Real.sqrt 2 :=
by sorry

end triangle_side_length_l779_77981


namespace part1_part2_l779_77947

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection point of l₁ and l₂
def intersection : ℝ × ℝ := (-2, 2)

-- Define the line parallel to 3x + y - 1 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x + y - 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Part 1: Prove that if l passes through the intersection and is parallel to 3x + y - 1 = 0,
-- then its equation is 3x + y + 4 = 0
theorem part1 (l : ℝ → ℝ → Prop) :
  (∀ x y, l x y ↔ ∃ k, 3 * x + y + k = 0) →
  l (intersection.1) (intersection.2) →
  (∀ x y, l x y → parallel_line x y) →
  (∀ x y, l x y ↔ 3 * x + y + 4 = 0) :=
sorry

-- Part 2: Prove that if l passes through the intersection and the distance from A to l is 5,
-- then its equation is either x = -2 or 12x - 5y + 34 = 0
theorem part2 (l : ℝ → ℝ → Prop) :
  l (intersection.1) (intersection.2) →
  (∀ x y, l x y → (((x - point_A.1) ^ 2 + (y - point_A.2) ^ 2) : ℝ).sqrt = 5) →
  (∀ x y, l x y ↔ x = -2 ∨ 12 * x - 5 * y + 34 = 0) :=
sorry

end part1_part2_l779_77947


namespace selina_shorts_sold_l779_77958

/-- Represents the number of pairs of shorts Selina sold -/
def shorts_sold : ℕ := sorry

/-- The price of a pair of pants in dollars -/
def pants_price : ℕ := 5

/-- The price of a pair of shorts in dollars -/
def shorts_price : ℕ := 3

/-- The price of a shirt in dollars -/
def shirt_price : ℕ := 4

/-- The number of pairs of pants Selina sold -/
def pants_sold : ℕ := 3

/-- The number of shirts Selina sold -/
def shirts_sold : ℕ := 5

/-- The price of each new shirt Selina bought -/
def new_shirt_price : ℕ := 10

/-- The number of new shirts Selina bought -/
def new_shirts_bought : ℕ := 2

/-- The amount of money Selina left the store with -/
def money_left : ℕ := 30

theorem selina_shorts_sold :
  shorts_sold = 5 ∧
  pants_sold * pants_price + shirts_sold * shirt_price + shorts_sold * shorts_price =
    money_left + new_shirts_bought * new_shirt_price :=
by sorry

end selina_shorts_sold_l779_77958


namespace no_integral_points_on_tangent_line_l779_77900

theorem no_integral_points_on_tangent_line (k m n : ℤ) : 
  ∀ x y : ℤ, (m^3 - m) * x + (n^3 - n) * y ≠ (3*k + 1)^2 := by
  sorry

end no_integral_points_on_tangent_line_l779_77900


namespace nuts_in_third_box_l779_77979

theorem nuts_in_third_box (x y z : ℕ) : 
  x + 6 = y + z → y + 10 = x + z → z = 8 := by sorry

end nuts_in_third_box_l779_77979


namespace luke_lawn_mowing_earnings_l779_77928

theorem luke_lawn_mowing_earnings :
  ∀ (L : ℝ),
  (∃ (total_earnings : ℝ),
    total_earnings = L + 18 ∧
    total_earnings = 3 * 9) →
  L = 9 := by
sorry

end luke_lawn_mowing_earnings_l779_77928


namespace janet_action_figures_l779_77952

/-- Calculates the final number of action figures Janet has after selling, buying, and receiving a gift. -/
theorem janet_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) (gift_multiplier : ℕ) : 
  initial = 10 → sold = 6 → bought = 4 → gift_multiplier = 2 →
  (initial - sold + bought) * (gift_multiplier + 1) = 24 :=
by
  sorry

end janet_action_figures_l779_77952


namespace sneakers_final_price_l779_77926

/-- Calculates the final price of sneakers after applying a coupon and membership discount -/
theorem sneakers_final_price
  (original_price : ℝ)
  (coupon_discount : ℝ)
  (membership_discount_rate : ℝ)
  (h1 : original_price = 120)
  (h2 : coupon_discount = 10)
  (h3 : membership_discount_rate = 0.1) :
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  let final_price := price_after_coupon - membership_discount
  final_price = 99 := by
sorry

end sneakers_final_price_l779_77926


namespace geometry_class_size_l779_77929

theorem geometry_class_size :
  ∀ (total_students : ℕ),
  (total_students / 2 : ℚ) = (total_students : ℚ) / 2 →
  ((total_students / 2) / 5 : ℚ) = (total_students : ℚ) / 10 →
  (total_students : ℚ) / 10 = 10 →
  total_students = 100 :=
by
  sorry

end geometry_class_size_l779_77929


namespace longest_segment_through_interior_point_l779_77953

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- Define the properties of a convex polygon
  -- (This is a simplified representation)
  vertices : Set (ℝ × ℝ)
  is_convex : Bool

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A direction in 2D space -/
def Direction := ℝ × ℝ

/-- Checks if a point is inside a convex polygon -/
def is_inside (K : ConvexPolygon) (P : Point) : Prop := sorry

/-- The length of the intersection of a line with a polygon -/
def intersection_length (K : ConvexPolygon) (P : Point) (d : Direction) : ℝ := sorry

/-- The theorem statement -/
theorem longest_segment_through_interior_point 
  (K : ConvexPolygon) (P : Point) (h : is_inside K P) :
  ∃ (d : Direction), 
    ∀ (Q : Point), is_inside K Q → 
      intersection_length K P d ≥ intersection_length K Q d := by sorry

end longest_segment_through_interior_point_l779_77953


namespace quadratic_equation_condition_l779_77951

/-- For the equation (m-1)x^2 + mx - 1 = 0 to be a quadratic equation in x,
    m must not equal 1. -/
theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, (m - 1) * x^2 + m * x - 1 = 0 → (m - 1) ≠ 0) ↔ m ≠ 1 :=
by sorry

end quadratic_equation_condition_l779_77951


namespace tony_fish_count_l779_77954

/-- The number of fish Tony has after a given number of years -/
def fish_count (initial_fish : ℕ) (years : ℕ) : ℕ :=
  initial_fish + years * (3 - 2)

/-- Theorem: Tony will have 15 fish after 10 years -/
theorem tony_fish_count : fish_count 5 10 = 15 := by
  sorry

end tony_fish_count_l779_77954


namespace bird_families_to_asia_l779_77912

theorem bird_families_to_asia (total_to_africa : ℕ) (difference : ℕ) : total_to_africa = 42 → difference = 11 → total_to_africa - difference = 31 := by
  sorry

end bird_families_to_asia_l779_77912


namespace stratified_sampling_seniors_l779_77960

theorem stratified_sampling_seniors (total_students : ℕ) (seniors : ℕ) (sample_size : ℕ) 
  (h_total : total_students = 900)
  (h_seniors : seniors = 400)
  (h_sample : sample_size = 45) :
  (seniors * sample_size) / total_students = 20 := by
sorry

end stratified_sampling_seniors_l779_77960


namespace triangle_circle_relation_l779_77909

theorem triangle_circle_relation 
  (AO' AO₁ AB AC t s s₁ s₂ s₃ r r₁ α : ℝ) 
  (h1 : AO' * Real.sin (α/2) = r ∧ r = t/s)
  (h2 : AO₁ * Real.sin (α/2) = r₁ ∧ r₁ = t/s₁)
  (h3 : AO' * AO₁ = t^2 / (s * s₁ * Real.sin (α/2)^2))
  (h4 : Real.sin (α/2)^2 = (s₂ * s₃) / (AB * AC)) :
  AO' * AO₁ = AB * AC := by
  sorry

end triangle_circle_relation_l779_77909


namespace expression_value_l779_77915

theorem expression_value (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end expression_value_l779_77915


namespace min_value_2a_plus_b_l779_77927

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 1 / a + 2 / b = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 2 / b' = 1 → 2 * a + b ≤ 2 * a' + b') ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 1 / a₀ + 2 / b₀ = 1 ∧ 2 * a₀ + b₀ = 8) := by
  sorry

end min_value_2a_plus_b_l779_77927


namespace chris_age_l779_77986

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Five years ago, Chris was the same age as Amy is now
  ages.chris - 5 = ages.amy ∧
  -- In 4 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 4 = (3/4) * (ages.amy + 4)

/-- The theorem to be proved -/
theorem chris_age (ages : Ages) :
  problem_conditions ages → ages.chris = 15.55 := by
  sorry

end chris_age_l779_77986


namespace peach_difference_l779_77902

def red_peaches : ℕ := 19
def yellow_peaches : ℕ := 11

theorem peach_difference : red_peaches - yellow_peaches = 8 := by
  sorry

end peach_difference_l779_77902


namespace min_value_of_expression_l779_77934

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 :=
by sorry

end min_value_of_expression_l779_77934


namespace no_integer_roots_l779_77924

theorem no_integer_roots (a b : ℤ) : 
  ¬ ∃ (x : ℤ), (x^2 + 10*a*x + 5*b + 3 = 0) ∨ (x^2 + 10*a*x + 5*b - 3 = 0) :=
by sorry

end no_integer_roots_l779_77924


namespace inequality_solution_set_l779_77970

theorem inequality_solution_set (x : ℝ) :
  -6 * x^2 - x + 2 ≤ 0 ↔ x ≥ (1/2 : ℝ) ∨ x ≤ -(2/3 : ℝ) := by sorry

end inequality_solution_set_l779_77970


namespace limit_S_over_a_squared_ln_a_nonzero_l779_77992

/-- The area S(a) bounded by the curve y = (a-x)ln x and the x-axis for a > 1 -/
noncomputable def S (a : ℝ) : ℝ := ∫ x in (1)..(a), (a - x) * Real.log x

/-- The limit of S(a)/(a^2 ln a) as a approaches infinity is a non-zero real number -/
theorem limit_S_over_a_squared_ln_a_nonzero :
  ∃ (L : ℝ), L ≠ 0 ∧ Filter.Tendsto (fun a => S a / (a^2 * Real.log a)) Filter.atTop (nhds L) := by
  sorry

end limit_S_over_a_squared_ln_a_nonzero_l779_77992


namespace cube_volume_from_surface_area_l779_77911

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 486 → volume = (surface_area / 6) ^ (3/2) → volume = 729 := by
  sorry

end cube_volume_from_surface_area_l779_77911


namespace total_cost_is_21_93_l779_77966

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa paid for cherries in dollars -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

/-- Theorem stating that the total cost of fruits is $21.93 -/
theorem total_cost_is_21_93 : total_cost = 21.93 := by
  sorry

end total_cost_is_21_93_l779_77966


namespace lattice_points_on_segment_l779_77945

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating the number of lattice points on the given line segment --/
theorem lattice_points_on_segment : latticePointCount 5 13 35 97 = 7 := by
  sorry

end lattice_points_on_segment_l779_77945


namespace total_values_count_l779_77978

theorem total_values_count (initial_mean correct_mean : ℚ) 
  (incorrect_value correct_value : ℚ) (n : ℕ) : 
  initial_mean = 150 → 
  correct_mean = 151 → 
  incorrect_value = 135 → 
  correct_value = 165 → 
  (n : ℚ) * initial_mean + incorrect_value = (n : ℚ) * correct_mean + correct_value → 
  n = 30 := by
  sorry

#check total_values_count

end total_values_count_l779_77978


namespace rod_cutting_l779_77985

theorem rod_cutting (rod_length : ℝ) (total_pieces : ℝ) (piece_length : ℝ) : 
  rod_length = 47.5 →
  total_pieces = 118.75 →
  piece_length = rod_length / total_pieces →
  piece_length = 0.4 := by
sorry

end rod_cutting_l779_77985


namespace unique_solution_l779_77959

def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def condition1 (a b : ℕ) : Prop := is_divisible (a^2 + 4*a + 3) b

def condition2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 2*a - 16*b - 8 = 0

def condition3 (a b : ℕ) : Prop := is_divisible (a + 2*b + 1) 4

def condition4 (a b : ℕ) : Prop := is_prime (a + 6*b + 1)

def exactly_three_true (a b : ℕ) : Prop :=
  (condition1 a b ∧ condition2 a b ∧ condition3 a b ∧ ¬condition4 a b) ∨
  (condition1 a b ∧ condition2 a b ∧ ¬condition3 a b ∧ condition4 a b) ∨
  (condition1 a b ∧ ¬condition2 a b ∧ condition3 a b ∧ condition4 a b) ∨
  (¬condition1 a b ∧ condition2 a b ∧ condition3 a b ∧ condition4 a b)

theorem unique_solution :
  ∀ a b : ℕ, exactly_three_true a b ↔ (a = 6 ∧ b = 1) ∨ (a = 18 ∧ b = 7) :=
sorry

end unique_solution_l779_77959


namespace arithmetic_progression_common_difference_l779_77982

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (eq : 26 * x^2 + 23 * x * y - 3 * y^2 - 19 = 0) 
  (progression : ∃ (a d : ℤ), x = a + 5 * d ∧ y = a + 10 * d ∧ d < 0) :
  ∃ (a : ℤ), x = a + 5 * (-3) ∧ y = a + 10 * (-3) := by
sorry

end arithmetic_progression_common_difference_l779_77982


namespace synthetic_method_deduces_result_from_cause_l779_77938

/-- The synthetic method in mathematics -/
def synthetic_method : Type := Unit

/-- Property of deducing result from cause -/
def deduces_result_from_cause (m : Type) : Prop := sorry

/-- The synthetic method is a way of thinking in mathematics -/
axiom synthetic_method_is_way_of_thinking : synthetic_method = Unit

/-- Theorem: The synthetic method deduces the result from the cause -/
theorem synthetic_method_deduces_result_from_cause : 
  deduces_result_from_cause synthetic_method :=
sorry

end synthetic_method_deduces_result_from_cause_l779_77938


namespace initial_loss_percentage_l779_77996

-- Define the cost price of a pencil
def cost_price : ℚ := 1 / 13

-- Define the selling price when selling 20 pencils for 1 rupee
def selling_price_20 : ℚ := 1 / 20

-- Define the selling price when selling 10 pencils for 1 rupee (30% gain)
def selling_price_10 : ℚ := 1 / 10

-- Define the percentage loss
def percentage_loss : ℚ := ((cost_price - selling_price_20) / cost_price) * 100

-- Theorem stating the initial loss percentage
theorem initial_loss_percentage : 
  (selling_price_10 = cost_price + 0.3 * cost_price) → 
  (percentage_loss = 35) :=
by
  sorry

end initial_loss_percentage_l779_77996


namespace basketball_score_ratio_l779_77920

/-- Given the scoring information for two basketball teams, prove the ratio of 2-pointers scored by the opponents to Mark's team. -/
theorem basketball_score_ratio :
  let marks_two_pointers : ℕ := 25
  let marks_three_pointers : ℕ := 8
  let marks_free_throws : ℕ := 10
  let opponents_three_pointers : ℕ := marks_three_pointers / 2
  let opponents_free_throws : ℕ := marks_free_throws / 2
  let total_points : ℕ := 201
  ∃ (x : ℚ),
    (2 * marks_two_pointers + 3 * marks_three_pointers + marks_free_throws) +
    (2 * (x * marks_two_pointers) + 3 * opponents_three_pointers + opponents_free_throws) = total_points ∧
    x = 2 := by
  sorry

end basketball_score_ratio_l779_77920


namespace number_difference_l779_77993

theorem number_difference (x y : ℝ) : 
  (35 + x) / 2 = 45 →
  (35 + x + y) / 3 = 40 →
  |y - 35| = 5 := by
sorry

end number_difference_l779_77993


namespace evaluate_expression_l779_77967

theorem evaluate_expression (b : ℕ) (h : b = 4) :
  b^3 * b^6 * 2 = 524288 := by
  sorry

end evaluate_expression_l779_77967


namespace combined_tax_rate_l779_77901

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.30) 
  (h2 : mindy_rate = 0.20) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end combined_tax_rate_l779_77901


namespace stock_recovery_l779_77956

theorem stock_recovery (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  let price_after_drops := initial_price * (1 - 0.1)^4
  ∃ n : ℕ, n ≥ 5 ∧ price_after_drops * (1 + 0.1)^n ≥ initial_price :=
by sorry

end stock_recovery_l779_77956


namespace perfect_square_example_l779_77955

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem perfect_square_example : is_perfect_square (4^10 * 5^5 * 6^10) := by
  sorry

end perfect_square_example_l779_77955


namespace survey_result_l779_77918

/-- Represents the number of questionnaires collected from each unit -/
structure QuestionnaireData where
  total : ℕ
  sample : ℕ
  sample_b : ℕ

/-- Proves that given the conditions from the survey, the number of questionnaires drawn from unit D is 60 -/
theorem survey_result (data : QuestionnaireData) 
  (h_total : data.total = 1000)
  (h_sample : data.sample = 150)
  (h_sample_b : data.sample_b = 30)
  (h_arithmetic : ∃ (a d : ℚ), ∀ i : Fin 4, a + i * d = (data.total : ℚ) / 4)
  (h_prop_arithmetic : ∃ (b e : ℚ), ∀ i : Fin 4, b + i * e = (data.sample : ℚ) / 4 ∧ b + 1 * e = data.sample_b) :
  ∃ (b e : ℚ), b + 3 * e = 60 := by
sorry

end survey_result_l779_77918


namespace arithmetic_sequence_fifth_term_l779_77914

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 12) : 
  a 5 = 6 := by
sorry

end arithmetic_sequence_fifth_term_l779_77914


namespace line_parallel_in_perp_planes_l779_77935

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the relation of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_in_perp_planes
  (α β : Plane) (l m n : Line)
  (h1 : perp_plane α β)
  (h2 : l = intersection α β)
  (h3 : in_plane n β)
  (h4 : perp_line n l)
  (h5 : perp_line_plane m α) :
  parallel m n :=
by sorry

end line_parallel_in_perp_planes_l779_77935


namespace garden_flowers_l779_77999

theorem garden_flowers (white_flowers : ℕ) (additional_red_needed : ℕ) (current_red_flowers : ℕ) : 
  white_flowers = 555 →
  additional_red_needed = 208 →
  white_flowers = current_red_flowers + additional_red_needed →
  current_red_flowers = 347 := by
sorry

end garden_flowers_l779_77999


namespace go_complexity_ratio_l779_77933

/-- The upper limit of the state space complexity of Go -/
def M : ℝ := 3^361

/-- The total number of atoms of ordinary matter in the observable universe -/
def N : ℝ := 10^80

/-- Theorem stating that M/N is approximately equal to 10^93 -/
theorem go_complexity_ratio : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |M / N - 10^93| < ε := by
  sorry

end go_complexity_ratio_l779_77933


namespace cookie_cost_l779_77975

theorem cookie_cost (cheeseburger_cost milkshake_cost coke_cost fries_cost tax : ℚ)
  (toby_initial toby_change : ℚ) (cookie_count : ℕ) :
  cheeseburger_cost = 365/100 ∧ 
  milkshake_cost = 2 ∧ 
  coke_cost = 1 ∧ 
  fries_cost = 4 ∧ 
  tax = 1/5 ∧
  toby_initial = 15 ∧ 
  toby_change = 7 ∧
  cookie_count = 3 →
  let total_before_cookies := 2 * cheeseburger_cost + milkshake_cost + coke_cost + fries_cost + tax
  let total_spent := 2 * (toby_initial - toby_change)
  let cookie_total_cost := total_spent - total_before_cookies
  cookie_total_cost / cookie_count = 1/2 := by
sorry

end cookie_cost_l779_77975


namespace trig_identity_l779_77976

theorem trig_identity (α : Real) (h : Real.sin (π / 4 + α) = 1 / 2) :
  Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α) * Real.cos (7 * π / 4 - α) = -1 / 2 := by
  sorry

end trig_identity_l779_77976


namespace smallest_four_digit_multiple_of_18_l779_77971

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end smallest_four_digit_multiple_of_18_l779_77971


namespace arithmetic_progression_problem_l779_77946

theorem arithmetic_progression_problem (a d : ℚ) : 
  (3 * ((a - d) + a) = 2 * (a + d)) →
  ((a - 2)^2 = (a - d) * (a + d)) →
  ((a = 5 ∧ d = 4) ∨ (a = 5/4 ∧ d = 1)) := by
sorry

end arithmetic_progression_problem_l779_77946


namespace mikes_initial_amount_solve_mikes_initial_amount_l779_77974

/-- Proves that Mike's initial amount is $90 given the conditions of the problem -/
theorem mikes_initial_amount (carol_initial : ℕ) (carol_weekly_savings : ℕ) 
  (mike_weekly_savings : ℕ) (weeks : ℕ) (mike_initial : ℕ) : Prop :=
  carol_initial = 60 →
  carol_weekly_savings = 9 →
  mike_weekly_savings = 3 →
  weeks = 5 →
  carol_initial + carol_weekly_savings * weeks = mike_initial + mike_weekly_savings * weeks →
  mike_initial = 90

/-- The main theorem that proves Mike's initial amount -/
theorem solve_mikes_initial_amount : 
  ∃ (mike_initial : ℕ), mikes_initial_amount 60 9 3 5 mike_initial :=
by
  sorry

end mikes_initial_amount_solve_mikes_initial_amount_l779_77974


namespace jason_has_21_toys_l779_77930

/-- The number of toys Rachel has -/
def rachel_toys : ℕ := 1

/-- The number of toys John has -/
def john_toys : ℕ := rachel_toys + 6

/-- The number of toys Jason has -/
def jason_toys : ℕ := 3 * john_toys

/-- Theorem: Jason has 21 toys -/
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end jason_has_21_toys_l779_77930


namespace parallel_planes_condition_l779_77943

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem parallel_planes_condition 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ α ≠ γ ∧ β ≠ γ)
  (h_parallel : parallel m n)
  (h_perp1 : perpendicular n α)
  (h_perp2 : perpendicular m β) :
  plane_parallel α β :=
sorry

end parallel_planes_condition_l779_77943


namespace battery_current_l779_77963

/-- Given a battery with voltage 48V, prove that when the resistance R is 12Ω, 
    the current I is 4A, where I is related to R by the function I = 48/R. -/
theorem battery_current (R : ℝ) (I : ℝ) : 
  R = 12 → I = 48 / R → I = 4 := by sorry

end battery_current_l779_77963


namespace corn_acres_calculation_l779_77936

def total_land : ℝ := 1634
def beans_ratio : ℝ := 4.5
def wheat_ratio : ℝ := 2.3
def corn_ratio : ℝ := 3.8
def barley_ratio : ℝ := 3.4

theorem corn_acres_calculation :
  let total_ratio := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
  let acres_per_part := total_land / total_ratio
  let corn_acres := corn_ratio * acres_per_part
  ∃ ε > 0, |corn_acres - 443.51| < ε :=
by sorry

end corn_acres_calculation_l779_77936


namespace volume_of_specific_tetrahedron_l779_77994

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the given tetrahedron is approximately 13.416 -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 3,
    RS := 7
  }
  abs (tetrahedronVolume t - 13.416) < 0.001 := by
  sorry

end volume_of_specific_tetrahedron_l779_77994


namespace negation_of_existence_negation_of_proposition_l779_77932

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) :=
by sorry

end negation_of_existence_negation_of_proposition_l779_77932


namespace annual_pension_l779_77961

/-- The annual pension problem -/
theorem annual_pension (c p q : ℝ) (x : ℝ) (k : ℝ) :
  (k * Real.sqrt (x + c) = k * Real.sqrt x + 3 * p) →
  (k * Real.sqrt (x + 2 * c) = k * Real.sqrt x + 4 * q) →
  (k * Real.sqrt x = (16 * q^2 - 18 * p^2) / (12 * p - 8 * q)) :=
by sorry

end annual_pension_l779_77961


namespace equal_roots_sum_inverse_a_and_c_l779_77987

theorem equal_roots_sum_inverse_a_and_c (a c : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, x * x * a + 2 * x + 2 - c = 0 ∧ 
   ∀ y : ℝ, y * y * a + 2 * y + 2 - c = 0 → y = x) →
  1 / a + c = 2 := by
sorry

end equal_roots_sum_inverse_a_and_c_l779_77987


namespace range_of_a_l779_77906

/-- The range of values for a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (a^2 - 2*a - 2)^x < (a^2 - 2*a - 2)^y) ∧ 
  ¬(0 < a ∧ a < 4) →
  a ≥ 4 ∨ a < -1 :=
by sorry

end range_of_a_l779_77906


namespace x_range_l779_77977

theorem x_range (x : ℝ) (h1 : (1 : ℝ) / x < 3) (h2 : (1 : ℝ) / x > -2) :
  x > (1 : ℝ) / 3 ∨ x < -(1 : ℝ) / 2 := by
  sorry

end x_range_l779_77977


namespace halfway_point_between_fractions_l779_77949

theorem halfway_point_between_fractions :
  let a := (1 : ℚ) / 9
  let b := (1 : ℚ) / 11
  let midpoint := (a + b) / 2
  midpoint = 10 / 99 := by sorry

end halfway_point_between_fractions_l779_77949


namespace xiao_ming_school_time_l779_77923

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two Time values in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Converts minutes to Time -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60,
    minutes := m % 60,
    valid := by sorry }

theorem xiao_ming_school_time :
  let morning_arrival : Time := { hours := 7, minutes := 50, valid := by sorry }
  let morning_departure : Time := { hours := 11, minutes := 50, valid := by sorry }
  let afternoon_arrival : Time := { hours := 14, minutes := 10, valid := by sorry }
  let afternoon_departure : Time := { hours := 17, minutes := 0, valid := by sorry }
  let morning_time := timeDifference morning_arrival morning_departure
  let afternoon_time := timeDifference afternoon_arrival afternoon_departure
  let total_time := morning_time + afternoon_time
  minutesToTime total_time = { hours := 6, minutes := 50, valid := by sorry } :=
by sorry

end xiao_ming_school_time_l779_77923


namespace gas_station_sales_l779_77921

/-- The total number of boxes sold at a gas station -/
def total_boxes (chocolate_boxes sugar_boxes gum_boxes : ℕ) : ℕ :=
  chocolate_boxes + sugar_boxes + gum_boxes

/-- Theorem: The gas station sold 9 boxes in total -/
theorem gas_station_sales : total_boxes 2 5 2 = 9 := by
  sorry

end gas_station_sales_l779_77921


namespace money_difference_l779_77942

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Mrs. Hilt's coin counts -/
def mrs_hilt_coins : Fin 3 → ℕ
| 0 => 2  -- pennies
| 1 => 2  -- nickels
| 2 => 2  -- dimes
| _ => 0

/-- Jacob's coin counts -/
def jacob_coins : Fin 3 → ℕ
| 0 => 4  -- pennies
| 1 => 1  -- nickel
| 2 => 1  -- dime
| _ => 0

/-- The value of a coin type in dollars -/
def coin_value : Fin 3 → ℚ
| 0 => penny_value
| 1 => nickel_value
| 2 => dime_value
| _ => 0

/-- Calculate the total value of coins -/
def total_value (coins : Fin 3 → ℕ) : ℚ :=
  (coins 0 : ℚ) * penny_value + (coins 1 : ℚ) * nickel_value + (coins 2 : ℚ) * dime_value

theorem money_difference :
  total_value mrs_hilt_coins - total_value jacob_coins = 13 / 100 := by
  sorry

end money_difference_l779_77942


namespace solve_equation_l779_77908

theorem solve_equation : ∃ x : ℝ, (75 / 100 * 4500 = (1 / 4) * x + 144) ∧ x = 12924 := by
  sorry

end solve_equation_l779_77908


namespace unique_solution_for_x_l779_77984

theorem unique_solution_for_x (x y z : ℤ) 
  (h1 : x > y ∧ y > z ∧ z > 0)
  (h2 : x + y + z + x*y + y*z + z*x = 104) : 
  x = 6 := by sorry

end unique_solution_for_x_l779_77984


namespace fixed_point_of_exponential_function_l779_77903

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2016) + 2016
  f 2016 = 2017 := by
  sorry

end fixed_point_of_exponential_function_l779_77903


namespace geometric_sequence_sum_ratio_l779_77944

/-- Given a geometric sequence {a_n} with S_n being the sum of its first n terms,
    if a_6 = 8a_3, then S_6 / S_3 = 9 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h_sum : ∀ n, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) 
    (h_ratio : a 6 = 8 * a 3) : 
  S 6 / S 3 = 9 := by
sorry

end geometric_sequence_sum_ratio_l779_77944


namespace square_sum_value_l779_77917

theorem square_sum_value (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -9) : x^2 + 9*y^2 = 90 := by
  sorry

end square_sum_value_l779_77917


namespace diagonal_intersection_probability_l779_77925

theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let intersecting_diagonals := vertices.choose 4
  intersecting_diagonals / (total_diagonals.choose 2 : ℚ) = 
    n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end diagonal_intersection_probability_l779_77925


namespace complex_fraction_evaluation_l779_77910

theorem complex_fraction_evaluation : 
  (((10/3 / 10 + 0.175 / 0.35) / (1.75 - (28/17) * (51/56))) - 
   ((11/18 - 1/15) / 1.4) / ((0.5 - 1/9) * 3)) = 1/2 := by
  sorry

end complex_fraction_evaluation_l779_77910


namespace distance_home_to_school_l779_77940

/-- The distance between home and school satisfies the given conditions -/
theorem distance_home_to_school :
  ∃ (d : ℝ), d > 0 ∧
  ∃ (t : ℝ), t > 0 ∧
  (5 * (t + 7/60) = d) ∧
  (10 * (t - 8/60) = d) ∧
  d = 2.5 := by
  sorry

end distance_home_to_school_l779_77940


namespace sqrt_27_minus_3_sqrt_one_third_l779_77905

theorem sqrt_27_minus_3_sqrt_one_third : 
  Real.sqrt 27 - 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := by
  sorry

end sqrt_27_minus_3_sqrt_one_third_l779_77905


namespace magnitude_of_BC_l779_77931

/-- Given vectors BA and AC in R², prove that the magnitude of BC is 5 -/
theorem magnitude_of_BC (BA AC : ℝ × ℝ) : 
  BA = (3, -2) → AC = (0, 6) → ‖BA + AC‖ = 5 := by sorry

end magnitude_of_BC_l779_77931


namespace trapezoid_perimeter_l779_77919

/-- A trapezoid with the given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  angle_between_diagonals : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with the given properties has a perimeter of 22 -/
theorem trapezoid_perimeter (t : Trapezoid) 
  (h1 : t.base1 = 3)
  (h2 : t.base2 = 5)
  (h3 : t.diagonal = 8)
  (h4 : t.angle_between_diagonals = 60 * π / 180) :
  perimeter t = 22 := by sorry

end trapezoid_perimeter_l779_77919


namespace bike_price_calculation_l779_77969

theorem bike_price_calculation (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 200)
  (h2 : upfront_percentage = 0.20) :
  upfront_payment / upfront_percentage = 1000 := by
  sorry

end bike_price_calculation_l779_77969


namespace number_thought_of_l779_77990

theorem number_thought_of (x : ℝ) : (6 * x^2 - 10) / 3 + 15 = 95 → x = 5 * Real.sqrt 15 / 3 := by
  sorry

end number_thought_of_l779_77990


namespace inequality_holds_l779_77997

theorem inequality_holds (r s : ℝ) (hr : 0 ≤ r ∧ r < 2) (hs : s > 0) :
  (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) > 3 * r^2 * s := by
  sorry

end inequality_holds_l779_77997


namespace square_field_area_l779_77983

/-- The area of a square field with a diagonal of 26 meters is 338.0625 square meters. -/
theorem square_field_area (d : ℝ) (h : d = 26) : 
  let s := d / Real.sqrt 2
  s^2 = 338.0625 := by sorry

end square_field_area_l779_77983


namespace fourth_power_sum_equality_l779_77989

theorem fourth_power_sum_equality : 120^4 + 97^4 + 84^4 + 27^4 = 174^4 := by
  sorry

end fourth_power_sum_equality_l779_77989


namespace bob_win_probability_l779_77937

theorem bob_win_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 5/8)
  (h_tie : p_tie = 1/8) : 
  1 - p_lose - p_tie = 1/4 := by
  sorry

end bob_win_probability_l779_77937


namespace quadratic_rewrite_sum_l779_77904

theorem quadratic_rewrite_sum (a b c : ℤ) : 
  (49 : ℤ) * x^2 + 70 * x - 121 = 0 ↔ (a * x + b)^2 = c ∧ 
  a > 0 ∧ 
  a + b + c = -134 := by
sorry

end quadratic_rewrite_sum_l779_77904


namespace number_of_classes_l779_77972

theorem number_of_classes (single_sided_per_class_per_day : ℕ)
                          (double_sided_per_class_per_day : ℕ)
                          (school_days_per_week : ℕ)
                          (total_single_sided_per_week : ℕ)
                          (total_double_sided_per_week : ℕ)
                          (h1 : single_sided_per_class_per_day = 175)
                          (h2 : double_sided_per_class_per_day = 75)
                          (h3 : school_days_per_week = 5)
                          (h4 : total_single_sided_per_week = 16000)
                          (h5 : total_double_sided_per_week = 7000) :
  ⌊(total_single_sided_per_week + total_double_sided_per_week : ℚ) /
   ((single_sided_per_class_per_day + double_sided_per_class_per_day) * school_days_per_week)⌋ = 18 :=
by sorry

end number_of_classes_l779_77972


namespace score_difference_l779_77907

def score_distribution : List (ℝ × ℝ) := [
  (0.15, 60),
  (0.25, 75),
  (0.35, 85),
  (0.20, 95),
  (0.05, 110)
]

def median_score : ℝ := 85

def mean_score : ℝ := (score_distribution.map (λ (p, s) => p * s)).sum

theorem score_difference : median_score - mean_score = 3 := by sorry

end score_difference_l779_77907


namespace trajectory_equation_l779_77973

/-- 
Given a point P(x, y) in the Cartesian coordinate system,
if the product of its distances to the x-axis and y-axis equals 1,
then the equation of its trajectory is xy = ± 1.
-/
theorem trajectory_equation (x y : ℝ) : 
  (|x| * |y| = 1) → (x * y = 1 ∨ x * y = -1) := by
  sorry

end trajectory_equation_l779_77973


namespace divides_condition_l779_77998

theorem divides_condition (a b : ℕ) : 
  (a^b + b) ∣ (a^(2*b) + 2*b) ↔ 
  (a = 0) ∨ (b = 0) ∨ (a = 2 ∧ b = 1) := by
  sorry

-- Define 0^0 = 1
axiom zero_pow_zero : (0 : ℕ)^(0 : ℕ) = 1

end divides_condition_l779_77998


namespace base_conversion_l779_77922

/-- Given that in base x, the decimal number 67 is written as 47, prove that x = 15 -/
theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 67) : x = 15 := by
  sorry

end base_conversion_l779_77922


namespace total_sample_variance_stratified_sampling_l779_77962

/-- Calculates the total sample variance for stratified sampling of student heights -/
theorem total_sample_variance_stratified_sampling 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (male_mean : ℝ) 
  (female_mean : ℝ) 
  (male_variance : ℝ) 
  (female_variance : ℝ) 
  (h_male_count : male_count = 100)
  (h_female_count : female_count = 60)
  (h_male_mean : male_mean = 172)
  (h_female_mean : female_mean = 164)
  (h_male_variance : male_variance = 18)
  (h_female_variance : female_variance = 30) :
  let total_count := male_count + female_count
  let combined_mean := (male_count * male_mean + female_count * female_mean) / total_count
  let total_variance := 
    (male_count : ℝ) / total_count * (male_variance + (male_mean - combined_mean)^2) +
    (female_count : ℝ) / total_count * (female_variance + (female_mean - combined_mean)^2)
  total_variance = 37.5 := by
sorry


end total_sample_variance_stratified_sampling_l779_77962


namespace total_earnings_value_l779_77941

def friday_earnings : ℚ := 147
def saturday_earnings : ℚ := 2 * friday_earnings + 7
def sunday_earnings : ℚ := friday_earnings + 78
def monday_earnings : ℚ := 0.75 * friday_earnings
def tuesday_earnings : ℚ := 1.25 * monday_earnings
def wednesday_earnings : ℚ := 0.8 * tuesday_earnings

def total_earnings : ℚ := friday_earnings + saturday_earnings + sunday_earnings + 
                          monday_earnings + tuesday_earnings + wednesday_earnings

theorem total_earnings_value : total_earnings = 1031.3125 := by
  sorry

end total_earnings_value_l779_77941


namespace min_values_xy_and_x_plus_2y_l779_77980

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x * y ≤ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x + 2*y ≤ a + 2*b) ∧
  x * y = 36 ∧
  x + 2*y = 19 + 6 * Real.sqrt 2 := by
  sorry

end min_values_xy_and_x_plus_2y_l779_77980


namespace permutation_sum_of_digits_l779_77964

def digit_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

def geometric_sum : ℕ := (10^7 - 1) / 9

theorem permutation_sum_of_digits (n : ℕ) (h : n = 7) :
  (n.factorial * digit_sum * geometric_sum : ℕ) = 22399997760 := by
  sorry

end permutation_sum_of_digits_l779_77964


namespace max_value_implies_a_l779_77995

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

theorem max_value_implies_a (a : ℝ) (h_a : a ≠ 0) :
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x = 1) →
  a = 3/4 ∨ a = 1/2 := by
sorry

end max_value_implies_a_l779_77995


namespace refrigerator_theorem_l779_77913

def refrigerator_problem (P : ℝ) : Prop :=
  let discount_rate : ℝ := 0.20
  let profit_rate : ℝ := 0.10
  let additional_costs : ℝ := 375
  let selling_price : ℝ := 18975
  let purchase_price : ℝ := P * (1 - discount_rate)
  let total_price : ℝ := purchase_price + additional_costs
  (P * (1 + profit_rate) = selling_price) → (total_price = 14175)

theorem refrigerator_theorem :
  ∃ P : ℝ, refrigerator_problem P :=
sorry

end refrigerator_theorem_l779_77913


namespace quadratic_one_solution_l779_77939

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x, 4 * x^2 + n * x + 4 = 0) ↔ (n = 8 ∨ n = -8) :=
by sorry

end quadratic_one_solution_l779_77939


namespace platform_length_calculation_l779_77968

-- Define the given parameters
def train_length : ℝ := 250
def train_speed_kmph : ℝ := 72
def train_speed_mps : ℝ := 20
def time_to_cross : ℝ := 25

-- Define the theorem
theorem platform_length_calculation :
  let total_distance := train_speed_mps * time_to_cross
  let platform_length := total_distance - train_length
  platform_length = 250 := by sorry

end platform_length_calculation_l779_77968


namespace repeating_decimal_equals_fraction_l779_77965

/-- Represents the repeating decimal 0.37246̄ as a rational number -/
def repeating_decimal : ℚ := 37245 / 99900

/-- Theorem stating that the repeating decimal 0.37246̄ is equal to 37245/99900 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 37245 / 99900 := by
  sorry

end repeating_decimal_equals_fraction_l779_77965


namespace mean_of_car_counts_l779_77950

theorem mean_of_car_counts : 
  let counts : List ℝ := [30, 14, 14, 21, 25]
  (counts.sum / counts.length : ℝ) = 20.8 := by
sorry

end mean_of_car_counts_l779_77950


namespace small_cube_edge_length_l779_77957

theorem small_cube_edge_length (large_cube_edge : ℕ) (small_cube_edge : ℕ) : 
  large_cube_edge = 12 →
  (large_cube_edge / small_cube_edge) > 0 →
  6 * ((large_cube_edge / small_cube_edge - 2) ^ 2) = 12 * (large_cube_edge / small_cube_edge - 2) →
  small_cube_edge = 3 := by
sorry

end small_cube_edge_length_l779_77957


namespace zanders_stickers_l779_77948

theorem zanders_stickers (S : ℚ) : 
  (1/5 : ℚ) * S + (3/10 : ℚ) * (S - (1/5 : ℚ) * S) = 44 → S = 100 := by
sorry

end zanders_stickers_l779_77948


namespace number_of_students_l779_77988

theorem number_of_students (total_skittles : ℕ) (skittles_per_student : ℕ) (h1 : total_skittles = 27) (h2 : skittles_per_student = 3) :
  total_skittles / skittles_per_student = 9 :=
by sorry

end number_of_students_l779_77988


namespace complex_number_properties_l779_77916

theorem complex_number_properties (z : ℂ) (h : (2 + I) * z = 1 + 3 * I) : 
  Complex.abs z = Real.sqrt 2 ∧ z^2 - 2*z + 2 = 0 := by
  sorry

end complex_number_properties_l779_77916
