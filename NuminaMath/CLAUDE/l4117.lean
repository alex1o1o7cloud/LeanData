import Mathlib

namespace tmall_double_eleven_sales_scientific_notation_l4117_411744

theorem tmall_double_eleven_sales_scientific_notation :
  let billion : ℕ := 10^9
  let sales : ℕ := 2684 * billion
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (a * 10^n : ℝ) = sales ∧ a = 2.684 ∧ n = 11 :=
by sorry

end tmall_double_eleven_sales_scientific_notation_l4117_411744


namespace quadratic_root_difference_l4117_411792

theorem quadratic_root_difference (m : ℝ) : 
  ∃ (x₁ x₂ : ℂ), x₁^2 + m*x₁ + 3 = 0 ∧ 
                 x₂^2 + m*x₂ + 3 = 0 ∧ 
                 x₁ ≠ x₂ ∧
                 Complex.abs (x₁ - x₂) = 2 → 
  m = 5 := by
sorry

end quadratic_root_difference_l4117_411792


namespace shaded_area_of_semicircles_l4117_411773

/-- The shaded area of semicircles in a pattern --/
theorem shaded_area_of_semicircles (d : ℝ) (l : ℝ) : 
  d = 3 → l = 24 → (l / d) * (π * d^2 / 8) = 18 * π := by sorry

end shaded_area_of_semicircles_l4117_411773


namespace pink_highlighters_count_l4117_411723

theorem pink_highlighters_count (total yellow blue : ℕ) (h1 : total = 15) (h2 : yellow = 7) (h3 : blue = 5) :
  ∃ pink : ℕ, pink + yellow + blue = total ∧ pink = 3 := by
  sorry

end pink_highlighters_count_l4117_411723


namespace arithmetic_sequence_sum_divisibility_l4117_411772

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_divisibility :
  (arithmetic_sequence_sum 1 8 313) % 8 = 0 := by
  sorry

end arithmetic_sequence_sum_divisibility_l4117_411772


namespace angle_D_measure_l4117_411752

-- Define the hexagon and its angles
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_valid_hexagon (h : ConvexHexagon) : Prop :=
  h.A > 0 ∧ h.B > 0 ∧ h.C > 0 ∧ h.D > 0 ∧ h.E > 0 ∧ h.F > 0 ∧
  h.A + h.B + h.C + h.D + h.E + h.F = 720

-- Define the conditions of the problem
def satisfies_conditions (h : ConvexHexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧
  h.D = h.E ∧ h.E = h.F ∧
  h.A + 30 = h.D

-- Theorem statement
theorem angle_D_measure (h : ConvexHexagon) 
  (h_valid : is_valid_hexagon h) 
  (h_cond : satisfies_conditions h) : 
  h.D = 135 :=
sorry

end angle_D_measure_l4117_411752


namespace ten_cut_patterns_l4117_411735

/-- Represents a grid with cells that can be cut into rectangles and squares. -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (total_cells : ℕ)
  (removed_cells : ℕ)

/-- Represents a way to cut the grid. -/
structure CutPattern :=
  (rectangles : ℕ)
  (squares : ℕ)

/-- The number of valid cut patterns for a given grid. -/
def valid_cut_patterns (g : Grid) (p : CutPattern) : ℕ := sorry

/-- The main theorem stating that there are exactly 10 ways to cut the specific grid. -/
theorem ten_cut_patterns :
  ∃ (g : Grid) (p : CutPattern),
    g.rows = 3 ∧
    g.cols = 6 ∧
    g.total_cells = 17 ∧
    g.removed_cells = 1 ∧
    p.rectangles = 8 ∧
    p.squares = 1 ∧
    valid_cut_patterns g p = 10 := by sorry

end ten_cut_patterns_l4117_411735


namespace hydrochloric_acid_moles_required_l4117_411771

/-- Represents a chemical substance with its coefficient in a chemical equation -/
structure Substance where
  name : String
  coefficient : ℕ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Substance
  products : List Substance

def sodium_bisulfite : Substance := ⟨"NaHSO3", 1⟩
def hydrochloric_acid : Substance := ⟨"HCl", 1⟩
def sodium_chloride : Substance := ⟨"NaCl", 1⟩
def water : Substance := ⟨"H2O", 1⟩
def sulfur_dioxide : Substance := ⟨"SO2", 1⟩

def reaction : Reaction :=
  ⟨[sodium_bisulfite, hydrochloric_acid], [sodium_chloride, water, sulfur_dioxide]⟩

/-- The number of moles of a substance required or produced in a reaction -/
def moles_required (s : Substance) (n : ℕ) : ℕ := s.coefficient * n

theorem hydrochloric_acid_moles_required :
  moles_required hydrochloric_acid 2 = 2 :=
sorry

end hydrochloric_acid_moles_required_l4117_411771


namespace inversion_number_reverse_l4117_411716

/-- An array of 8 distinct integers -/
def Array8 := Fin 8 → ℤ

/-- The inversion number of an array -/
def inversionNumber (A : Array8) : ℕ :=
  sorry

/-- Theorem: Given an array of 8 distinct integers with inversion number 2,
    the inversion number of its reverse (excluding the last element) is at least 19 -/
theorem inversion_number_reverse (A : Array8) 
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j)
  (h_inv_num : inversionNumber A = 2) :
  inversionNumber (fun i => A (⟨7 - i.val, sorry⟩ : Fin 8)) ≥ 19 :=
sorry

end inversion_number_reverse_l4117_411716


namespace f_has_three_zeros_l4117_411766

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x^2 - 2*x) * Real.log x + (a - 1/2) * x^2 + 2*(1 - a)*x + a

theorem f_has_three_zeros (a : ℝ) (h : a < -2) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by sorry

end f_has_three_zeros_l4117_411766


namespace parabola_directrix_l4117_411703

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 4*x + 4) / 8

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -1/4

/-- Theorem: The directrix of the given parabola is y = -1/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_d : ℝ, directrix_equation y_d :=
sorry

end parabola_directrix_l4117_411703


namespace total_peppers_weight_l4117_411776

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℝ := 0.33

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℝ := 0.33

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℝ := green_peppers + red_peppers

/-- Theorem stating that the total weight of peppers is 0.66 pounds -/
theorem total_peppers_weight : total_peppers = 0.66 := by sorry

end total_peppers_weight_l4117_411776


namespace weight_order_l4117_411763

theorem weight_order (P Q R S T : ℝ) 
  (h1 : P < 1000) (h2 : Q < 1000) (h3 : R < 1000) (h4 : S < 1000) (h5 : T < 1000)
  (h6 : Q + S = 1200) (h7 : R + T = 2100) (h8 : Q + T = 800) (h9 : Q + R = 900) (h10 : P + T = 700) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
by sorry

end weight_order_l4117_411763


namespace window_purchase_savings_l4117_411745

/-- The regular price of a window in dollars -/
def regular_price : ℕ := 100

/-- The number of windows Alice needs -/
def alice_windows : ℕ := 9

/-- The number of windows Bob needs -/
def bob_windows : ℕ := 11

/-- The number of windows purchased that qualify for the special deal -/
def special_deal_threshold : ℕ := 10

/-- The number of free windows given in the special deal -/
def free_windows : ℕ := 2

/-- Calculate the cost of windows with the special deal applied -/
def cost_with_deal (n : ℕ) : ℕ :=
  let sets := n / special_deal_threshold
  let remainder := n % special_deal_threshold
  (n - sets * free_windows) * regular_price

/-- The main theorem stating the savings when purchasing together -/
theorem window_purchase_savings :
  cost_with_deal alice_windows + cost_with_deal bob_windows -
  cost_with_deal (alice_windows + bob_windows) = 200 := by
  sorry

end window_purchase_savings_l4117_411745


namespace inscribed_sphere_volume_l4117_411738

-- Define the cone
def cone_base_diameter : ℝ := 16
def cone_vertex_angle : ℝ := 90

-- Define the sphere
def sphere_touches_lateral_surfaces : Prop := sorry
def sphere_rests_on_table : Prop := sorry

-- Calculate the volume of the sphere
noncomputable def sphere_volume : ℝ := 
  let base_radius := cone_base_diameter / 2
  let cone_height := base_radius * 2
  let sphere_radius := base_radius / Real.sqrt 2
  (4 / 3) * Real.pi * (sphere_radius ^ 3)

-- Theorem statement
theorem inscribed_sphere_volume 
  (h1 : cone_vertex_angle = 90)
  (h2 : sphere_touches_lateral_surfaces)
  (h3 : sphere_rests_on_table) :
  sphere_volume = (512 * Real.sqrt 2 * Real.pi) / 3 := by sorry

end inscribed_sphere_volume_l4117_411738


namespace divisibility_by_2008_l4117_411790

theorem divisibility_by_2008 (k m : ℕ) (h1 : ∃ (u : ℕ), k = 25 * (2 * u + 1)) (h2 : ∃ (v : ℕ), m = 25 * v) :
  2008 ∣ (2^k + 4^m) :=
sorry

end divisibility_by_2008_l4117_411790


namespace kamal_math_marks_l4117_411779

/-- Proves that given Kamal's marks in English, Physics, Chemistry, and Biology,
    with a specific average for all 5 subjects, his marks in Mathematics can be determined. -/
theorem kamal_math_marks
  (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ)
  (h_english : english = 76)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_biology : biology = 85)
  (h_average : average = 75)
  (h_subjects : 5 * average = english + physics + chemistry + biology + mathematics) :
  mathematics = 65 :=
by sorry

end kamal_math_marks_l4117_411779


namespace sqrt_inequality_increasing_function_inequality_l4117_411757

-- Part 1
theorem sqrt_inequality (x₁ x₂ : ℝ) (h1 : 0 ≤ x₁) (h2 : 0 ≤ x₂) (h3 : x₁ ≠ x₂) :
  (1/2) * (Real.sqrt x₁ + Real.sqrt x₂) < Real.sqrt ((x₁ + x₂) / 2) := by
  sorry

-- Part 2
theorem increasing_function_inequality {f : ℝ → ℝ} (h : Monotone f) 
  {a b : ℝ} (h1 : a + f a ≤ b + f b) : a ≤ b := by
  sorry

end sqrt_inequality_increasing_function_inequality_l4117_411757


namespace gabby_fruit_problem_l4117_411724

theorem gabby_fruit_problem (watermelons peaches plums : ℕ) : 
  peaches = watermelons + 12 →
  plums = 3 * peaches →
  watermelons + peaches + plums = 53 →
  watermelons = 1 := by
sorry

end gabby_fruit_problem_l4117_411724


namespace seventeen_to_fourteen_greater_than_thirtyone_to_eleven_l4117_411720

theorem seventeen_to_fourteen_greater_than_thirtyone_to_eleven :
  (17 : ℝ)^14 > (31 : ℝ)^11 := by sorry

end seventeen_to_fourteen_greater_than_thirtyone_to_eleven_l4117_411720


namespace abby_and_damon_weight_l4117_411780

theorem abby_and_damon_weight (a b c d : ℝ)
  (h1 : a + b = 265)
  (h2 : b + c = 250)
  (h3 : c + d = 280) :
  a + d = 295 := by
  sorry

end abby_and_damon_weight_l4117_411780


namespace b_share_of_earnings_l4117_411743

theorem b_share_of_earnings 
  (a_days b_days c_days : ℕ) 
  (total_earnings : ℚ) 
  (ha : a_days = 6)
  (hb : b_days = 8)
  (hc : c_days = 12)
  (htotal : total_earnings = 2340) :
  (1 / b_days) / ((1 / a_days) + (1 / b_days) + (1 / c_days)) * total_earnings = 780 := by
  sorry

end b_share_of_earnings_l4117_411743


namespace factorize_quadratic_l4117_411715

theorem factorize_quadratic (a : ℝ) : a^2 - 8*a + 15 = (a-3)*(a-5) := by
  sorry

end factorize_quadratic_l4117_411715


namespace circle_ratio_l4117_411793

theorem circle_ratio (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 5 := by
sorry

end circle_ratio_l4117_411793


namespace inequality_proof_l4117_411721

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃ ∧ a₃ > 0)
  (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃ ∧ b₃ > 0)
  (hab : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (hdiff : a₁ - a₃ ≤ b₁ - b₃) :
  a₁ + a₂ + a₃ ≤ 2 * (b₁ + b₂ + b₃) := by
  sorry

end inequality_proof_l4117_411721


namespace bacteria_count_l4117_411707

theorem bacteria_count (original : ℕ) (increase : ℕ) (current : ℕ) : 
  original = 600 → increase = 8317 → current = original + increase → current = 8917 := by
sorry

end bacteria_count_l4117_411707


namespace recommendation_plans_count_l4117_411731

/-- Represents the number of recommendation spots for each language --/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the number of male and female candidates --/
structure Candidates :=
  (male : Nat)
  (female : Nat)

/-- Calculates the number of different recommendation plans --/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

/-- Theorem stating that the number of recommendation plans is 36 --/
theorem recommendation_plans_count :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  countRecommendationPlans spots candidates = 36 :=
sorry

end recommendation_plans_count_l4117_411731


namespace fifteenth_term_of_sequence_l4117_411754

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence : arithmetic_sequence 3 4 15 = 59 := by
  sorry

end fifteenth_term_of_sequence_l4117_411754


namespace john_finishes_ahead_l4117_411770

/-- The distance John finishes ahead of Steve in a race --/
def distance_john_ahead (john_speed steve_speed initial_distance push_time : ℝ) : ℝ :=
  (john_speed * push_time - initial_distance) - (steve_speed * push_time)

/-- Theorem stating that John finishes 2 meters ahead of Steve --/
theorem john_finishes_ahead :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.7
  let initial_distance : ℝ := 12
  let push_time : ℝ := 28
  distance_john_ahead john_speed steve_speed initial_distance push_time = 2 := by
sorry


end john_finishes_ahead_l4117_411770


namespace distance_not_proportional_to_time_l4117_411747

/-- Uniform motion equation -/
def uniform_motion (a v t : ℝ) : ℝ := a + v * t

/-- Proportionality definition -/
def proportional (f : ℝ → ℝ) : Prop := ∀ (k t : ℝ), f (k * t) = k * f t

/-- Theorem: In uniform motion, distance is not generally proportional to time -/
theorem distance_not_proportional_to_time (a v : ℝ) (h : a ≠ 0) :
  ¬ proportional (uniform_motion a v) := by
  sorry

end distance_not_proportional_to_time_l4117_411747


namespace min_sum_of_squares_l4117_411742

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) :
  a^2 + b^2 + c^2 + d^2 ≥ 24/5 := by
  sorry

end min_sum_of_squares_l4117_411742


namespace publishing_break_even_l4117_411787

/-- A publishing company's break-even point calculation -/
theorem publishing_break_even 
  (fixed_cost : ℝ) 
  (variable_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : fixed_cost = 50000)
  (h2 : variable_cost = 4)
  (h3 : selling_price = 9) :
  ∃ x : ℝ, x = 10000 ∧ selling_price * x = fixed_cost + variable_cost * x :=
sorry

end publishing_break_even_l4117_411787


namespace expected_steps_l4117_411732

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The probability of moving in any direction -/
def moveProbability : ℚ := 1/4

/-- Roger's starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- Function to determine if a point can be reached more quickly by a different route -/
def canReachQuicker (path : List Point) : Bool :=
  sorry

/-- The expected number of additional steps after the initial step -/
def e₁ : ℚ := 8/3

/-- The expected number of additional steps after moving perpendicular -/
def e₂ : ℚ := 2

/-- The main theorem: The expected number of steps Roger takes before he stops is 11/3 -/
theorem expected_steps :
  let totalSteps := 1 + e₁
  totalSteps = 11/3 := by sorry

end expected_steps_l4117_411732


namespace circle_position_l4117_411726

def circle_center : ℝ × ℝ := (-3, 4)
def circle_radius : ℝ := 3

theorem circle_position :
  let (x, y) := circle_center
  let r := circle_radius
  (abs y > r) ∧ (abs x = r) := by sorry

end circle_position_l4117_411726


namespace carpet_breadth_calculation_l4117_411717

theorem carpet_breadth_calculation (b : ℝ) : 
  let first_length := 1.44 * b
  let second_length := 1.4 * first_length
  let second_breadth := 1.25 * b
  let second_area := second_length * second_breadth
  let cost_per_sqm := 45
  let total_cost := 4082.4
  second_area = total_cost / cost_per_sqm →
  b = 6.08 := by
sorry

end carpet_breadth_calculation_l4117_411717


namespace fraction_equality_l4117_411719

theorem fraction_equality (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end fraction_equality_l4117_411719


namespace continuity_at_two_l4117_411768

def f (x : ℝ) : ℝ := -5 * x^2 - 8

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

end continuity_at_two_l4117_411768


namespace rearranged_digits_subtraction_l4117_411796

theorem rearranged_digits_subtraction :
  ∀ h t u : ℕ,
  h ≠ t → h ≠ u → t ≠ u →
  h > 0 → t > 0 → u > 0 →
  h > u →
  h * 100 + t * 10 + u - (t * 100 + u * 10 + h) = 179 →
  h = 8 ∧ t = 7 ∧ u = 9 :=
by sorry

end rearranged_digits_subtraction_l4117_411796


namespace orthogonal_vectors_m_value_l4117_411729

/-- Given two vectors a and b in R², where a = (3, 2) and b = (m, -1),
    if a and b are orthogonal, then m = 2/3 -/
theorem orthogonal_vectors_m_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (m, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 2/3 :=
by sorry

end orthogonal_vectors_m_value_l4117_411729


namespace volume_cylinder_from_square_rotation_l4117_411740

/-- The volume of a cylinder formed by rotating a square around its vertical line of symmetry -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (volume : ℝ) :
  side_length = 20 →
  volume = π * (side_length / 2)^2 * side_length →
  volume = 2000 * π := by
sorry

end volume_cylinder_from_square_rotation_l4117_411740


namespace total_hats_bought_l4117_411737

theorem total_hats_bought (blue_cost green_cost total_price green_count : ℕ) 
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 530)
  (h4 : green_count = 20) :
  ∃ (blue_count : ℕ), blue_count * blue_cost + green_count * green_cost = total_price ∧
                      blue_count + green_count = 85 := by
  sorry

end total_hats_bought_l4117_411737


namespace exists_x_in_interval_equivalence_of_statements_l4117_411748

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Theorem 1
theorem exists_x_in_interval : ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ (1/2)^x > log_half x := by sorry

-- Theorem 2
theorem equivalence_of_statements :
  (∀ x : ℝ, x ∈ Set.Ioo 1 5 → x + 1/x ≥ 2) ↔
  (∀ x : ℝ, x ∈ Set.Iic 1 ∪ Set.Ici 5 → x + 1/x < 2) := by sorry

end exists_x_in_interval_equivalence_of_statements_l4117_411748


namespace sequence_property_l4117_411788

def strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def gcd_property (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, gcd (a m) (a n) = a (gcd m n)

def least_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
  (∃ r s : ℕ, r < k ∧ k < s ∧ a k ^ 2 = a r * a s) ∧
  (∀ k' : ℕ, k' < k → ¬∃ r s : ℕ, r < k' ∧ k' < s ∧ a k' ^ 2 = a r * a s)

theorem sequence_property (a : ℕ → ℕ) (k r s : ℕ) :
  strictly_increasing a →
  gcd_property a →
  least_k a k →
  r < k →
  k < s →
  a k ^ 2 = a r * a s →
  r ∣ k ∧ k ∣ s :=
by sorry

end sequence_property_l4117_411788


namespace book_sale_profit_l4117_411764

theorem book_sale_profit (total_cost : ℝ) (loss_percentage : ℝ) (gain_percentage1 : ℝ) (gain_percentage2 : ℝ) 
  (h1 : total_cost = 1080)
  (h2 : loss_percentage = 0.1)
  (h3 : gain_percentage1 = 0.15)
  (h4 : gain_percentage2 = 0.25)
  (h5 : (1 - loss_percentage) * (total_cost / 2) = 
        (1 + gain_percentage1) * (total_cost * 2 / 6) + 
        (1 + gain_percentage2) * (total_cost / 6)) :
  total_cost / 2 = 784 := by
  sorry

end book_sale_profit_l4117_411764


namespace specific_line_equation_l4117_411784

/-- A line parameterized by real t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific parametric line from the problem -/
def specificLine : ParametricLine where
  x := λ t => 3 * t + 2
  y := λ t => 5 * t - 3

/-- The equation of a line in slope-intercept form -/
structure LineEquation where
  slope : ℝ
  intercept : ℝ

/-- Theorem stating that the specific parametric line has the given equation -/
theorem specific_line_equation :
  ∃ (t : ℝ), specificLine.y t = (5/3) * specificLine.x t - 19/3 := by
  sorry

end specific_line_equation_l4117_411784


namespace todd_total_gum_l4117_411795

-- Define the initial number of gum pieces Todd had
def initial_gum : ℕ := 38

-- Define the number of gum pieces Steve gave to Todd
def steve_gum : ℕ := 16

-- Theorem statement
theorem todd_total_gum : initial_gum + steve_gum = 54 := by
  sorry

end todd_total_gum_l4117_411795


namespace special_rectangle_area_l4117_411718

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  x : ℝ  -- Length
  y : ℝ  -- Width
  perimeter_eq : x + y = 30  -- Half perimeter equals 30
  side_diff : x = y + 3

/-- The area of a SpecialRectangle -/
def area (r : SpecialRectangle) : ℝ := r.x * r.y

/-- Theorem stating the area of the SpecialRectangle -/
theorem special_rectangle_area :
  ∀ r : SpecialRectangle, area r = 222.75 := by
  sorry

end special_rectangle_area_l4117_411718


namespace dual_polyhedra_equal_radii_l4117_411750

/-- Represents a regular polyhedron -/
structure RegularPolyhedron where
  inscribed_radius : ℝ
  circumscribed_radius : ℝ
  face_circumscribed_radius : ℝ

/-- Represents a pair of dual regular polyhedra -/
structure DualRegularPolyhedra where
  original : RegularPolyhedron
  dual : RegularPolyhedron

/-- Theorem: For dual regular polyhedra with equal inscribed sphere radii,
    their circumscribed sphere radii and face circumscribed circle radii are equal -/
theorem dual_polyhedra_equal_radii (p : DualRegularPolyhedra) 
    (h : p.original.inscribed_radius = p.dual.inscribed_radius) : 
    p.original.circumscribed_radius = p.dual.circumscribed_radius ∧ 
    p.original.face_circumscribed_radius = p.dual.face_circumscribed_radius := by
  sorry


end dual_polyhedra_equal_radii_l4117_411750


namespace arithmetic_sequence_properties_l4117_411777

/-- An arithmetic sequence with sum S_n and common difference d -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  d : ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Main theorem about properties of an arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
  (h : seq.S 7 > seq.S 6 ∧ seq.S 6 > seq.S 8) :
  seq.d < 0 ∧ 
  seq.S 14 < 0 ∧
  (∀ n, seq.S n ≤ seq.S 7) :=
by sorry

end arithmetic_sequence_properties_l4117_411777


namespace triangle_point_movement_l4117_411708

theorem triangle_point_movement (AB BC : ℝ) (v_P v_Q : ℝ) (area_PBQ : ℝ) : 
  AB = 6 →
  BC = 8 →
  v_P = 1 →
  v_Q = 2 →
  area_PBQ = 5 →
  ∃ t : ℝ, t = 1 ∧ 
    (1/2) * (AB - t * v_P) * (t * v_Q) = area_PBQ ∧
    t * v_Q ≤ BC :=
by sorry

end triangle_point_movement_l4117_411708


namespace intersection_equals_three_l4117_411722

theorem intersection_equals_three :
  ∃ a : ℝ, ({1, 3, a^2 + 3*a - 4} : Set ℝ) ∩ ({0, 6, a^2 + 4*a - 2, a + 3} : Set ℝ) = {3} :=
by
  sorry

end intersection_equals_three_l4117_411722


namespace smallest_fraction_between_l4117_411706

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by
sorry

end smallest_fraction_between_l4117_411706


namespace elizabeth_study_time_l4117_411704

/-- Calculates the study time for math test given total study time and science test study time -/
def math_study_time (total_time science_time : ℕ) : ℕ :=
  total_time - science_time

/-- Theorem stating that given the total study time of 60 minutes and science test study time of 25 minutes, 
    the math test study time is 35 minutes -/
theorem elizabeth_study_time : 
  math_study_time 60 25 = 35 := by
  sorry

end elizabeth_study_time_l4117_411704


namespace interest_rate_calculation_l4117_411798

/-- Calculate the interest rate given principal, final amount, and time -/
theorem interest_rate_calculation (P A t : ℝ) (h1 : P = 1200) (h2 : A = 1344) (h3 : t = 2.4) :
  (A - P) / (P * t) = 0.05 := by
  sorry

end interest_rate_calculation_l4117_411798


namespace product_of_binary_and_ternary_l4117_411728

/-- Converts a binary number represented as a list of digits to its decimal equivalent -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary_num := [1, 1, 1, 0]
  let ternary_num := [1, 0, 2]
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 154 := by
  sorry

end product_of_binary_and_ternary_l4117_411728


namespace dog_to_hamster_lifespan_ratio_l4117_411741

/-- The average lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a fish in years -/
def fish_lifespan : ℝ := 12

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := fish_lifespan - 2

theorem dog_to_hamster_lifespan_ratio :
  dog_lifespan / hamster_lifespan = 4 := by sorry

end dog_to_hamster_lifespan_ratio_l4117_411741


namespace jacket_price_calculation_l4117_411725

theorem jacket_price_calculation (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (sales_tax : ℝ) :
  initial_price = 120 ∧
  discount1 = 0.20 ∧
  discount2 = 0.25 ∧
  sales_tax = 0.05 →
  initial_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax) = 75.60 :=
by sorry

end jacket_price_calculation_l4117_411725


namespace teacher_worksheets_l4117_411733

theorem teacher_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) : 
  problems_per_worksheet = 3 →
  graded_worksheets = 7 →
  remaining_problems = 24 →
  graded_worksheets + (remaining_problems / problems_per_worksheet) = 15 := by
  sorry

end teacher_worksheets_l4117_411733


namespace xyz_equals_one_l4117_411786

theorem xyz_equals_one
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (2 * b + 3 * c) / (x - 3))
  (eq_b : b = (3 * a + 2 * c) / (y - 3))
  (eq_c : c = (2 * a + 2 * b) / (z - 3))
  (sum_product : x * y + x * z + y * z = -1)
  (sum : x + y + z = 1) :
  x * y * z = 1 := by
sorry


end xyz_equals_one_l4117_411786


namespace work_completion_time_l4117_411774

/-- Given:
  - A can finish the work in 6 days
  - B worked for 10 days and left the job
  - A alone can finish the remaining work in 2 days
  Prove that B can finish the work in 15 days -/
theorem work_completion_time
  (total_work : ℝ)
  (a_completion_time : ℝ)
  (b_work_days : ℝ)
  (a_remaining_time : ℝ)
  (h1 : a_completion_time = 6)
  (h2 : b_work_days = 10)
  (h3 : a_remaining_time = 2)
  (h4 : total_work > 0) :
  ∃ (b_completion_time : ℝ),
    b_completion_time = 15 ∧
    (total_work / a_completion_time) * a_remaining_time =
      total_work - (total_work / b_completion_time) * b_work_days :=
by
  sorry


end work_completion_time_l4117_411774


namespace clock_hands_separation_l4117_411761

/-- Represents the angle between clock hands at a given time -/
def clockHandAngle (m : ℕ) : ℝ :=
  |6 * m - 0.5 * m|

/-- Checks if the angle between clock hands is 1° (or equivalent) -/
def isOneDegreeSeparation (m : ℕ) : Prop :=
  ∃ k : ℤ, clockHandAngle m = 1 + 360 * k ∨ clockHandAngle m = 1 - 360 * k

theorem clock_hands_separation :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 720 →
    (isOneDegreeSeparation m ↔ m = 262 ∨ m = 458) :=
by sorry

end clock_hands_separation_l4117_411761


namespace smallest_root_of_g_l4117_411700

def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g :
  ∃ (r : ℝ), r = -Real.sqrt (3/7) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → |x| ≥ |r| :=
sorry

end smallest_root_of_g_l4117_411700


namespace average_equation_l4117_411775

theorem average_equation (a : ℝ) : ((2 * a + 16) + (3 * a - 8)) / 2 = 69 → a = 26 := by
  sorry

end average_equation_l4117_411775


namespace max_sum_abcd_l4117_411753

theorem max_sum_abcd (a c d : ℤ) (b : ℕ+) 
  (h1 : a + b = c) 
  (h2 : b + c = d) 
  (h3 : c + d = a) : 
  (∃ (a' c' d' : ℤ) (b' : ℕ+), 
    a' + b' = c' ∧ 
    b' + c' = d' ∧ 
    c' + d' = a' ∧ 
    a' + b' + c' + d' = -5) ∧ 
  (∀ (a' c' d' : ℤ) (b' : ℕ+), 
    a' + b' = c' → 
    b' + c' = d' → 
    c' + d' = a' → 
    a' + b' + c' + d' ≤ -5) :=
sorry

end max_sum_abcd_l4117_411753


namespace parallel_resistors_l4117_411782

theorem parallel_resistors (x y r : ℝ) (hx : x = 3) (hy : y = 5) 
  (hr : 1 / r = 1 / x + 1 / y) : r = 1.875 := by
  sorry

end parallel_resistors_l4117_411782


namespace min_value_a_plus_b_l4117_411765

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2 / (2 + a)) + (1 / (a + 2 * b)) = 1) :
  (a + b ≥ Real.sqrt 2 + 1/2) ∧ 
  (a + b = Real.sqrt 2 + 1/2 ↔ a = Real.sqrt 2) := by
  sorry

end min_value_a_plus_b_l4117_411765


namespace sum_equality_exists_l4117_411701

theorem sum_equality_exists (a : Fin 16 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ i, a i > 0) 
  (h_bound : ∀ i, a i ≤ 100) : 
  ∃ i j k l : Fin 16, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l :=
sorry

end sum_equality_exists_l4117_411701


namespace parabola_properties_l4117_411778

/-- A parabola with equation x² = 3y is symmetric with respect to the y-axis and passes through
    the intersection points of x - y = 0 and x² + y² - 6y = 0 -/
theorem parabola_properties (x y : ℝ) :
  (x^2 = 3*y) →
  (∀ (x₀ : ℝ), (x₀^2 = 3*y) ↔ ((-x₀)^2 = 3*y)) ∧
  (∃ (x₁ y₁ : ℝ), x₁ - y₁ = 0 ∧ x₁^2 + y₁^2 - 6*y₁ = 0 ∧ x₁^2 = 3*y₁) :=
by sorry

end parabola_properties_l4117_411778


namespace angle_bak_is_right_angle_l4117_411727

-- Define the tetrahedron and its points
variable (A B C D K : EuclideanSpace ℝ (Fin 3))

-- Define the angles
def angle (p q r : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

-- State the conditions
variable (h1 : angle B A C + angle B A D = Real.pi)
variable (h2 : angle C A K = angle K A D)

-- State the theorem
theorem angle_bak_is_right_angle : angle B A K = Real.pi / 2 := by sorry

end angle_bak_is_right_angle_l4117_411727


namespace total_cost_calculation_l4117_411714

def regular_admission : ℚ := 8
def early_discount_percentage : ℚ := 25 / 100
def student_discount_percentage : ℚ := 10 / 100
def total_people : ℕ := 6
def students : ℕ := 2

def discounted_price : ℚ := regular_admission * (1 - early_discount_percentage)
def student_price : ℚ := discounted_price * (1 - student_discount_percentage)

theorem total_cost_calculation :
  let non_student_cost := (total_people - students) * discounted_price
  let student_cost := students * student_price
  non_student_cost + student_cost = 348 / 10 := by sorry

end total_cost_calculation_l4117_411714


namespace geometric_sequence_property_l4117_411789

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = 16) :
  a 3^2 + 2 * a 2 * a 6 + a 3 * a 7 = 400 :=
sorry

end geometric_sequence_property_l4117_411789


namespace hexagon_sectors_perimeter_l4117_411767

/-- The perimeter of a shape formed by removing three equal sectors from a regular hexagon -/
def shaded_perimeter (sector_perimeter : ℝ) : ℝ :=
  3 * sector_perimeter

theorem hexagon_sectors_perimeter :
  ∀ (sector_perimeter : ℝ),
  sector_perimeter = 18 →
  shaded_perimeter sector_perimeter = 54 := by
sorry

end hexagon_sectors_perimeter_l4117_411767


namespace ryn_to_nikki_ratio_l4117_411705

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ

/-- The conditions of the movie lengths problem -/
def movie_conditions (m : MovieLengths) : Prop :=
  m.joyce = m.michael + 2 ∧
  m.nikki = 3 * m.michael ∧
  m.nikki = 30 ∧
  m.michael + m.joyce + m.nikki + m.ryn = 76

/-- The theorem stating the ratio of Ryn's movie length to Nikki's movie length -/
theorem ryn_to_nikki_ratio (m : MovieLengths) :
  movie_conditions m → m.ryn / m.nikki = 4 / 5 := by
  sorry

end ryn_to_nikki_ratio_l4117_411705


namespace product_of_roots_l4117_411755

theorem product_of_roots (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (4 * y) * Real.sqrt (25 * y) = 50) : 
  x * y = Real.sqrt (25 / 24) := by
sorry

end product_of_roots_l4117_411755


namespace trolley_passengers_count_l4117_411751

/-- Calculates the number of people on a trolley after three stops -/
def trolley_passengers : ℕ :=
  let initial := 1  -- driver
  let first_stop := initial + 10
  let second_stop := first_stop - 3 + (2 * 10)
  let third_stop := second_stop - 18 + 2
  third_stop

theorem trolley_passengers_count : trolley_passengers = 12 := by
  sorry

end trolley_passengers_count_l4117_411751


namespace chocolate_bars_per_box_l4117_411799

theorem chocolate_bars_per_box (total_bars : ℕ) (num_boxes : ℕ) 
  (h1 : total_bars = 442) (h2 : num_boxes = 17) :
  total_bars / num_boxes = 26 := by
  sorry

end chocolate_bars_per_box_l4117_411799


namespace percentage_difference_difference_is_twelve_l4117_411794

theorem percentage_difference : ℝ → Prop :=
  let percent_of_40 := (80 / 100) * 40
  let fraction_of_25 := (4 / 5) * 25
  λ x => percent_of_40 - fraction_of_25 = x

theorem difference_is_twelve : percentage_difference 12 := by
  sorry

end percentage_difference_difference_is_twelve_l4117_411794


namespace number_of_students_l4117_411746

/-- Proves the number of students in a class given certain conditions about average ages --/
theorem number_of_students (n : ℕ) (student_avg : ℚ) (new_avg : ℚ) (staff_age : ℕ) 
  (h1 : student_avg = 16)
  (h2 : new_avg = student_avg + 1)
  (h3 : staff_age = 49)
  (h4 : (n * student_avg + staff_age) / (n + 1) = new_avg) : 
  n = 32 := by
  sorry

#check number_of_students

end number_of_students_l4117_411746


namespace symmetry_proof_l4117_411797

/-- A point in the 2D Cartesian coordinate system -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Symmetry with respect to the y-axis -/
def symmetric_wrt_y_axis (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = q.y

/-- The given point -/
def given_point : Point :=
  { x := -1, y := -2 }

/-- The symmetric point to be proved -/
def symmetric_point : Point :=
  { x := 1, y := -2 }

theorem symmetry_proof : symmetric_wrt_y_axis given_point symmetric_point := by
  sorry

end symmetry_proof_l4117_411797


namespace sqrt_meaningful_range_l4117_411783

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 4 - 2 * x) ↔ x ≤ 2 := by
  sorry

end sqrt_meaningful_range_l4117_411783


namespace exists_point_P_trajectory_G_l4117_411791

/-- Definition of the ellipse C -/
def is_on_ellipse_C (x y : ℝ) : Prop := x^2/36 + y^2/20 = 1

/-- Definition of point A -/
def point_A : ℝ × ℝ := (-6, 0)

/-- Definition of point F -/
def point_F : ℝ × ℝ := (4, 0)

/-- Definition of vector AP -/
def vector_AP (x y : ℝ) : ℝ × ℝ := (x + 6, y)

/-- Definition of vector FP -/
def vector_FP (x y : ℝ) : ℝ × ℝ := (x - 4, y)

/-- Theorem stating the existence of point P -/
theorem exists_point_P :
  ∃ (x y : ℝ), 
    is_on_ellipse_C x y ∧ 
    y > 0 ∧ 
    (vector_AP x y).1 * (vector_FP x y).1 + (vector_AP x y).2 * (vector_FP x y).2 = 0 ∧
    x = 3/2 ∧ 
    y = 5 * Real.sqrt 3 / 2 :=
sorry

/-- Definition of point M on ellipse C -/
def point_M (x₀ y₀ : ℝ) : Prop := is_on_ellipse_C x₀ y₀

/-- Definition of midpoint G of MF -/
def point_G (x y : ℝ) (x₀ y₀ : ℝ) : Prop :=
  x = (x₀ + 2) / 2 ∧ y = y₀ / 2

/-- Theorem stating the trajectory equation of G -/
theorem trajectory_G :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), point_M x₀ y₀ ∧ point_G x y x₀ y₀) ↔ 
    (x - 1)^2 / 9 + y^2 / 5 = 1 :=
sorry

end exists_point_P_trajectory_G_l4117_411791


namespace product_of_solutions_l4117_411758

theorem product_of_solutions (y : ℝ) : (|y| = 3 * (|y| - 2)) → ∃ z : ℝ, (|z| = 3 * (|z| - 2)) ∧ (y * z = -9) := by
  sorry

end product_of_solutions_l4117_411758


namespace inverse_f_composition_l4117_411730

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

theorem inverse_f_composition : 
  ∃ (f_inv : ℤ → ℤ), 
    (∀ (y : ℤ), f (f_inv y) = y) ∧ 
    (∀ (x : ℤ), f_inv (f x) = x) ∧
    f_inv (f_inv 122 / f_inv 18 + f_inv 50) = 4 := by
  sorry

end inverse_f_composition_l4117_411730


namespace remainder_theorem_l4117_411785

/-- A polynomial p(x) satisfying p(2) = 4 and p(5) = 10 -/
def p : Polynomial ℝ :=
  sorry

theorem remainder_theorem (h1 : p.eval 2 = 4) (h2 : p.eval 5 = 10) :
  ∃ q : Polynomial ℝ, p = q * ((X - 2) * (X - 5)) + (2 * X) :=
sorry

end remainder_theorem_l4117_411785


namespace inequality_theorem_equality_conditions_l4117_411739

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

theorem equality_conditions (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2)) ↔
  (x₁ * y₁ - z₁^2 = x₂ * y₂ - z₂^2 ∧ x₁ = x₂ ∧ z₁ = z₂) :=
by sorry

end inequality_theorem_equality_conditions_l4117_411739


namespace bag_of_balls_l4117_411709

theorem bag_of_balls (total : ℕ) (blue : ℕ) (green : ℕ) : 
  blue = 6 →
  blue + green = total →
  (blue : ℚ) / total = 1 / 4 →
  green = 18 := by
sorry

end bag_of_balls_l4117_411709


namespace cupcake_count_l4117_411734

theorem cupcake_count (initial : ℕ) (sold : ℕ) (additional : ℕ) : 
  initial ≥ sold → initial - sold + additional = (initial - sold) + additional := by
  sorry

end cupcake_count_l4117_411734


namespace simple_interest_calculation_l4117_411713

/-- Calculate simple interest for a loan where the time period equals the interest rate -/
theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) : 
  principal = 1800 →
  rate = 5.93 →
  let interest := principal * rate * rate / 100
  ∃ ε > 0, |interest - 632.61| < ε :=
by
  sorry

end simple_interest_calculation_l4117_411713


namespace austin_robot_purchase_l4117_411759

theorem austin_robot_purchase (num_robots : ℕ) (cost_per_robot tax change : ℚ) : 
  num_robots = 7 →
  cost_per_robot = 8.75 →
  tax = 7.22 →
  change = 11.53 →
  (num_robots : ℚ) * cost_per_robot + tax + change = 80 := by
  sorry

end austin_robot_purchase_l4117_411759


namespace polynomial_division_remainder_l4117_411760

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^4 : Polynomial ℝ) + 3 * X^3 - 4 = (X^2 + X - 3) * q + (5 * X - 1) := by
  sorry

end polynomial_division_remainder_l4117_411760


namespace complement_intersection_M_N_l4117_411781

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 + 2) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≠ p.1 - 4}

-- Theorem statement
theorem complement_intersection_M_N : 
  (U \ M) ∩ (U \ N) = {(2, -2)} := by sorry

end complement_intersection_M_N_l4117_411781


namespace polyhedral_angle_sum_lt_360_l4117_411749

/-- A polyhedral angle is represented by a list of planar angles (in degrees) -/
def PolyhedralAngle := List Float

/-- The sum of planar angles in a polyhedral angle is less than 360° -/
theorem polyhedral_angle_sum_lt_360 (pa : PolyhedralAngle) : 
  pa.sum < 360 := by sorry

end polyhedral_angle_sum_lt_360_l4117_411749


namespace pizza_slices_per_child_l4117_411736

/-- Calculates the number of pizza slices each child wants given the following conditions:
  * There are 2 adults and 6 children
  * Each adult wants 3 slices
  * They order 3 pizzas with 4 slices each
-/
theorem pizza_slices_per_child 
  (num_adults : Nat) 
  (num_children : Nat) 
  (slices_per_adult : Nat) 
  (num_pizzas : Nat) 
  (slices_per_pizza : Nat) 
  (h1 : num_adults = 2) 
  (h2 : num_children = 6) 
  (h3 : slices_per_adult = 3) 
  (h4 : num_pizzas = 3) 
  (h5 : slices_per_pizza = 4) : 
  (num_pizzas * slices_per_pizza - num_adults * slices_per_adult) / num_children = 1 := by
  sorry

end pizza_slices_per_child_l4117_411736


namespace bobby_blocks_l4117_411756

theorem bobby_blocks (initial_blocks : ℕ) (given_blocks : ℕ) : 
  initial_blocks = 2 → given_blocks = 6 → initial_blocks + given_blocks = 8 :=
by sorry

end bobby_blocks_l4117_411756


namespace junior_senior_ratio_l4117_411710

theorem junior_senior_ratio (j s : ℕ) 
  (h1 : j > 0) (h2 : s > 0)
  (h3 : (j / 3 : ℚ) = (2 * s / 3 : ℚ)) : 
  j = 2 * s := by
sorry

end junior_senior_ratio_l4117_411710


namespace sqrt2_irrational_bound_l4117_411712

theorem sqrt2_irrational_bound (p q : ℤ) (hq : q ≠ 0) :
  |Real.sqrt 2 - (p : ℝ) / (q : ℝ)| > 1 / (3 * (q : ℝ)^2) := by
  sorry

end sqrt2_irrational_bound_l4117_411712


namespace quadratic_equation_roots_condition_l4117_411769

theorem quadratic_equation_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2 * x + 3 = 0 ∧ m * y^2 - 2 * y + 3 = 0) ↔ 
  (m < 1/3 ∧ m ≠ 0) :=
sorry

end quadratic_equation_roots_condition_l4117_411769


namespace number_problem_l4117_411702

theorem number_problem (x : ℝ) : 
  (15 / 100 * 40 = 25 / 100 * x + 2) → x = 16 := by
  sorry

end number_problem_l4117_411702


namespace xiao_ding_distance_to_school_l4117_411762

/-- Proof that Xiao Ding's distance to school is 60 meters -/
theorem xiao_ding_distance_to_school : 
  ∀ (xw xd xc xz : ℝ),
  xw + xd + xc + xz = 705 →  -- Total distance condition
  xw = 4 * xd →              -- Xiao Wang's distance condition
  xc = xw / 2 + 20 →         -- Xiao Chen's distance condition
  xz = 2 * xc - 15 →         -- Xiao Zhang's distance condition
  xd = 60 := by              -- Conclusion: Xiao Ding's distance is 60 meters
sorry

end xiao_ding_distance_to_school_l4117_411762


namespace angle_adjustment_l4117_411711

def are_complementary (a b : ℝ) : Prop := a + b = 90

theorem angle_adjustment (x y : ℝ) 
  (h1 : are_complementary x y)
  (h2 : x / y = 1 / 2)
  (h3 : x < y) :
  let new_x := x * 1.2
  let new_y := 90 - new_x
  (y - new_y) / y = 0.1 := by sorry

end angle_adjustment_l4117_411711
