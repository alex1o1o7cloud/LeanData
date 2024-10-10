import Mathlib

namespace roots_are_eccentricities_l3328_332817

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 4 * x + 1 = 0

def is_ellipse_eccentricity (e : ℝ) : Prop := 0 < e ∧ e < 1

def is_parabola_eccentricity (e : ℝ) : Prop := e = 1

theorem roots_are_eccentricities :
  ∃ (e₁ e₂ : ℝ),
    quadratic_equation e₁ ∧
    quadratic_equation e₂ ∧
    e₁ ≠ e₂ ∧
    ((is_ellipse_eccentricity e₁ ∧ is_parabola_eccentricity e₂) ∨
     (is_ellipse_eccentricity e₂ ∧ is_parabola_eccentricity e₁)) :=
by sorry

end roots_are_eccentricities_l3328_332817


namespace highest_common_factor_l3328_332804

/- Define the polynomials f and g -/
def f (n : ℕ) (x : ℝ) : ℝ := n * x^(n+1) - (n+1) * x^n + 1

def g (n : ℕ) (x : ℝ) : ℝ := x^n - n*x + n - 1

/- State the theorem -/
theorem highest_common_factor (n : ℕ) (h : n ≥ 2) :
  ∃ (p q : ℝ → ℝ), 
    (∀ x, f n x = (x - 1)^2 * p x) ∧ 
    (∀ x, g n x = (x - 1) * q x) ∧
    (∀ r : ℝ → ℝ, (∀ x, f n x = r x * (p x)) → (∀ x, g n x = r x * (q x)) → 
      ∃ (s : ℝ → ℝ), ∀ x, r x = (x - 1)^2 * s x) :=
sorry

end highest_common_factor_l3328_332804


namespace complex_equation_solution_l3328_332852

theorem complex_equation_solution (z : ℂ) : (1 + 2 * Complex.I) * z = 3 - Complex.I → z = 1/5 - 7/5 * Complex.I := by
  sorry

end complex_equation_solution_l3328_332852


namespace division_simplification_l3328_332826

theorem division_simplification (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (8 * a^3 * b - 4 * a^2 * b^2) / (4 * a * b) = 2 * a^2 - a * b :=
by sorry

end division_simplification_l3328_332826


namespace ariel_fencing_years_l3328_332870

theorem ariel_fencing_years (birth_year : ℕ) (fencing_start : ℕ) (current_age : ℕ) : 
  birth_year = 1992 → fencing_start = 2006 → current_age = 30 → 
  fencing_start - birth_year - current_age = 16 := by
  sorry

end ariel_fencing_years_l3328_332870


namespace fold_square_problem_l3328_332891

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let distAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  distAB = 8 ∧ 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2

-- Define point E as the midpoint of AD
def Midpoint (E A D : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + D.1) / 2 ∧ E.2 = (A.2 + D.2) / 2

-- Define point F on BD such that BF = EF
def PointF (F B D E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  F.1 = B.1 + t * (D.1 - B.1) ∧ 
  F.2 = B.2 + t * (D.2 - B.2) ∧
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = (F.1 - E.1)^2 + (F.2 - E.2)^2

-- Theorem statement
theorem fold_square_problem (A B C D E F : ℝ × ℝ) :
  Square A B C D → Midpoint E A D → PointF F B D E →
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 3^2 := by
  sorry

end fold_square_problem_l3328_332891


namespace equality_of_ratios_l3328_332822

theorem equality_of_ratios (a b c d : ℕ) 
  (h1 : a / c = b / d) 
  (h2 : a / c = (a * b + 1) / (c * d + 1)) 
  (h3 : b / d = (a * b + 1) / (c * d + 1)) : 
  a = c ∧ b = d := by
  sorry

end equality_of_ratios_l3328_332822


namespace tennis_tournament_matches_l3328_332811

theorem tennis_tournament_matches (n : ℕ) (b : ℕ) (h1 : n = 120) (h2 : b = 40) :
  let total_matches := n - 1
  total_matches = 119 ∧ total_matches % 7 = 0 := by
  sorry

end tennis_tournament_matches_l3328_332811


namespace calories_burned_proof_l3328_332888

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 40

/-- The number of stairs climbed in one direction -/
def stairs_one_way : ℕ := 32

/-- The number of calories burned per stair -/
def calories_per_stair : ℕ := 2

/-- Calculates the total calories burned during the exercise -/
def total_calories_burned : ℕ :=
  num_runs * (2 * stairs_one_way) * calories_per_stair

/-- Theorem stating that the total calories burned is 5120 -/
theorem calories_burned_proof : total_calories_burned = 5120 := by
  sorry

end calories_burned_proof_l3328_332888


namespace probability_one_defective_l3328_332814

def total_items : ℕ := 6
def good_items : ℕ := 4
def defective_items : ℕ := 2
def selected_items : ℕ := 3

theorem probability_one_defective :
  (Nat.choose good_items (selected_items - 1) * Nat.choose defective_items 1) /
  Nat.choose total_items selected_items = 3 / 5 := by sorry

end probability_one_defective_l3328_332814


namespace triangle_inequality_equality_condition_l3328_332805

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point P
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the sine of an angle in a triangle
def sine (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Define whether a point lies on the circumcircle of a triangle
def onCircumcircle (t : Triangle) (p : Point) : Prop := sorry

theorem triangle_inequality (t : Triangle) (p : Point) :
  distance p t.A * sine t t.A ≤ distance p t.B * sine t t.B + distance p t.C * sine t t.C :=
sorry

theorem equality_condition (t : Triangle) (p : Point) :
  distance p t.A * sine t t.A = distance p t.B * sine t t.B + distance p t.C * sine t t.C ↔
  onCircumcircle t p :=
sorry

end triangle_inequality_equality_condition_l3328_332805


namespace root_implies_k_value_l3328_332841

theorem root_implies_k_value (k : ℚ) : 
  (∃ x : ℚ, x^2 - 2*x + 2*k = 0) ∧ (1^2 - 2*1 + 2*k = 0) → k = 1/2 := by
  sorry

end root_implies_k_value_l3328_332841


namespace farmer_apples_final_apple_count_l3328_332829

theorem farmer_apples (initial : ℝ) (given_away : ℝ) (harvested : ℝ) :
  initial - given_away + harvested = initial + harvested - given_away :=
by sorry

theorem final_apple_count (initial : ℝ) (given_away : ℝ) (harvested : ℝ) :
  initial = 5708 → given_away = 2347.5 → harvested = 1526.75 →
  initial - given_away + harvested = 4887.25 :=
by sorry

end farmer_apples_final_apple_count_l3328_332829


namespace inequality_proof_l3328_332899

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2 * Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := by
  sorry

end inequality_proof_l3328_332899


namespace mayo_bottle_size_l3328_332828

/-- Proves the size of a mayo bottle at a normal store given bulk pricing information -/
theorem mayo_bottle_size 
  (costco_price : ℝ) 
  (normal_store_price : ℝ) 
  (savings : ℝ) 
  (gallon_in_ounces : ℝ) 
  (h1 : costco_price = 8) 
  (h2 : normal_store_price = 3) 
  (h3 : savings = 16) 
  (h4 : gallon_in_ounces = 128) : 
  (gallon_in_ounces / ((savings + costco_price) / normal_store_price)) = 16 :=
by sorry

end mayo_bottle_size_l3328_332828


namespace abcd_mod_11_l3328_332863

theorem abcd_mod_11 (a b c d : ℕ) : 
  a < 11 → b < 11 → c < 11 → d < 11 →
  (a + 3*b + 4*c + 2*d) % 11 = 3 →
  (3*a + b + 2*c + d) % 11 = 5 →
  (2*a + 4*b + c + 3*d) % 11 = 7 →
  (a + b + c + d) % 11 = 2 →
  (a * b * c * d) % 11 = 9 := by
sorry

end abcd_mod_11_l3328_332863


namespace scaled_roots_polynomial_l3328_332896

theorem scaled_roots_polynomial (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 10 = 0) → 
  (r₂^3 - 4*r₂^2 + 10 = 0) → 
  (r₃^3 - 4*r₃^2 + 10 = 0) → 
  (∀ x : ℂ, x^3 - 12*x^2 + 270 = (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃)) := by
sorry

end scaled_roots_polynomial_l3328_332896


namespace middle_number_problem_l3328_332860

theorem middle_number_problem (x y z : ℤ) 
  (sum_xy : x + y = 15)
  (sum_xz : x + z = 18)
  (sum_yz : y + z = 22) :
  y = (19 : ℚ) / 2 := by
sorry

end middle_number_problem_l3328_332860


namespace smallest_value_fraction_achievable_value_l3328_332864

theorem smallest_value_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 0) : 
  (a + 2*b : ℚ)/(a - b) + (a - b : ℚ)/(a + 2*b) ≥ 10/3 :=
sorry

theorem achievable_value (a b : ℕ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (a b : ℕ), a > b ∧ b > 0 ∧ (a + 2*b : ℚ)/(a - b) + (a - b : ℚ)/(a + 2*b) = 10/3 :=
sorry

end smallest_value_fraction_achievable_value_l3328_332864


namespace cubic_root_series_sum_l3328_332806

/-- Given a positive real number s satisfying s³ + (1/4)s - 1 = 0,
    the series s² + 2s⁵ + 3s⁸ + 4s¹¹ + ... converges to 16 -/
theorem cubic_root_series_sum (s : ℝ) (hs : 0 < s) (heq : s^3 + (1/4) * s - 1 = 0) :
  ∑' n, (n + 1) * s^(3*n + 2) = 16 := by
  sorry

end cubic_root_series_sum_l3328_332806


namespace triangle_third_side_length_l3328_332842

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 10) (h3 : θ = 135 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * Real.cos θ ∧ c = Real.sqrt (181 + 90 * Real.sqrt 2) :=
sorry

end triangle_third_side_length_l3328_332842


namespace quadratic_root_difference_l3328_332819

theorem quadratic_root_difference (b : ℝ) : 
  (∃ (x y : ℝ), 2 * x^2 + b * x = 12 ∧ 
                 2 * y^2 + b * y = 12 ∧ 
                 y - x = 5.5 ∧ 
                 (∀ z : ℝ, 2 * z^2 + b * z = 12 → (z = x ∨ z = y))) →
  b = -5 := by
sorry

end quadratic_root_difference_l3328_332819


namespace bus_problem_l3328_332812

theorem bus_problem (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 5 → 
  (initial_students : ℚ) * (2/3)^num_stops = 640/81 := by
  sorry

end bus_problem_l3328_332812


namespace tom_carrot_consumption_l3328_332853

/-- Proves that Tom ate 1 pound of carrots given the conditions of the problem -/
theorem tom_carrot_consumption (C : ℝ) : 
  C > 0 →  -- Assuming C is positive (implicit in the original problem)
  51 * C + 2 * C * (51 / 3) = 85 →
  C = 1 := by
  sorry

end tom_carrot_consumption_l3328_332853


namespace not_all_same_probability_l3328_332834

def roll_five_eight_sided_dice : ℕ := 8^5

def same_number_outcomes : ℕ := 8

theorem not_all_same_probability :
  (roll_five_eight_sided_dice - same_number_outcomes) / roll_five_eight_sided_dice = 4095 / 4096 :=
by sorry

end not_all_same_probability_l3328_332834


namespace perpendicular_parallel_transitivity_l3328_332827

-- Define the types for lines and planes
def Line : Type := Real × Real × Real → Prop
def Plane : Type := Real × Real × Real → Prop

-- Define the relations
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular_line_plane m α) 
  (h3 : parallel m n) : 
  perpendicular_line_plane n α :=
sorry

end perpendicular_parallel_transitivity_l3328_332827


namespace arc_length_45_degrees_l3328_332807

/-- Given a circle with circumference 72 meters and an arc subtended by a 45° central angle,
    the length of the arc is 9 meters. -/
theorem arc_length_45_degrees (D : ℝ) (EF : ℝ) : 
  D = 72 →  -- circumference of the circle
  EF = (45 / 360) * D →  -- arc length as a fraction of the circumference
  EF = 9 := by sorry

end arc_length_45_degrees_l3328_332807


namespace fraction_zero_implies_x_equals_four_l3328_332815

theorem fraction_zero_implies_x_equals_four (x : ℝ) : 
  (16 - x^2) / (x + 4) = 0 ∧ x + 4 ≠ 0 → x = 4 := by
sorry

end fraction_zero_implies_x_equals_four_l3328_332815


namespace monotone_decreasing_range_l3328_332851

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x else -x^2 + a

theorem monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 1 := by sorry

end monotone_decreasing_range_l3328_332851


namespace units_digit_of_six_to_fifth_l3328_332882

theorem units_digit_of_six_to_fifth (n : ℕ) : n = 6^5 → n % 10 = 6 := by
  sorry

end units_digit_of_six_to_fifth_l3328_332882


namespace team_a_wins_l3328_332890

theorem team_a_wins (total_matches : ℕ) (team_a_points : ℕ) : 
  total_matches = 10 → 
  team_a_points = 22 → 
  ∃ (wins draws : ℕ), 
    wins + draws = total_matches ∧ 
    3 * wins + draws = team_a_points ∧ 
    wins = 6 :=
by sorry

end team_a_wins_l3328_332890


namespace least_addition_for_divisibility_l3328_332800

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1076 ∧ m = 23) :
  (5 : ℕ) = Nat.minFac ((n + 5) % m) :=
sorry

end least_addition_for_divisibility_l3328_332800


namespace polynomial_factorization_l3328_332816

theorem polynomial_factorization (m : ℝ) : 
  (∀ x, x^2 + m*x - 6 = (x - 2) * (x + 3)) → m = 1 := by
  sorry

end polynomial_factorization_l3328_332816


namespace years_ago_p_half_q_l3328_332839

/-- The number of years ago when p was half of q's age, given their current ages' ratio and sum. -/
theorem years_ago_p_half_q (p q : ℕ) (h1 : p * 4 = q * 3) (h2 : p + q = 28) : 
  ∃ y : ℕ, p - y = (q - y) / 2 ∧ y = 8 := by
  sorry

end years_ago_p_half_q_l3328_332839


namespace parabola_translation_l3328_332813

/-- The equation of a parabola after vertical translation -/
def translated_parabola (original : ℝ → ℝ) (translation : ℝ) : ℝ → ℝ :=
  fun x => original x + translation

/-- Theorem: Moving y = x^2 up 3 units results in y = x^2 + 3 -/
theorem parabola_translation :
  let original := fun x : ℝ => x^2
  translated_parabola original 3 = fun x => x^2 + 3 := by
  sorry

end parabola_translation_l3328_332813


namespace fifth_root_unity_sum_l3328_332894

theorem fifth_root_unity_sum (x : ℂ) : x^5 = 1 → 1 + x^4 + x^8 + x^12 + x^16 = 0 := by
  sorry

end fifth_root_unity_sum_l3328_332894


namespace simplify_expression_l3328_332850

theorem simplify_expression (a b : ℝ) : 
  (30*a + 70*b) + (15*a + 40*b) - (12*a + 55*b) + (5*a - 10*b) = 38*a + 45*b := by
  sorry

end simplify_expression_l3328_332850


namespace stating_remaining_pieces_l3328_332847

/-- The number of pieces on a standard chessboard at the start of the game. -/
def initial_pieces : ℕ := 32

/-- The number of pieces Audrey lost. -/
def audrey_lost : ℕ := 6

/-- The number of pieces Thomas lost. -/
def thomas_lost : ℕ := 5

/-- The total number of pieces lost by both players. -/
def total_lost : ℕ := audrey_lost + thomas_lost

/-- 
  Theorem stating that the number of pieces remaining on the chessboard is 21,
  given the initial number of pieces and the number of pieces lost by each player.
-/
theorem remaining_pieces :
  initial_pieces - total_lost = 21 := by sorry

end stating_remaining_pieces_l3328_332847


namespace cow_husk_consumption_l3328_332802

/-- If 55 cows eat 55 bags of husk in 55 days, then one cow will eat one bag of husk in 55 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 55 ∧ bags = 55 ∧ days = 55) :
  (1 : ℕ) * bags = (1 : ℕ) * cows * days := by
  sorry

end cow_husk_consumption_l3328_332802


namespace square_root_fraction_simplification_l3328_332876

theorem square_root_fraction_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 36)) = 17 / Real.sqrt 61 := by
  sorry

end square_root_fraction_simplification_l3328_332876


namespace square_greater_than_abs_square_l3328_332843

theorem square_greater_than_abs_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_greater_than_abs_square_l3328_332843


namespace probability_of_drawing_two_l3328_332897

/-- Represents a card with a number -/
structure Card where
  number : ℕ

/-- Represents the set of cards -/
def cardSet : Finset Card := sorry

/-- The total number of cards -/
def totalCards : ℕ := 5

/-- The number of cards with the number 2 -/
def cardsWithTwo : ℕ := 2

/-- The probability of drawing a card with the number 2 -/
def probabilityOfTwo : ℚ := cardsWithTwo / totalCards

theorem probability_of_drawing_two :
  probabilityOfTwo = 2 / 5 := by sorry

end probability_of_drawing_two_l3328_332897


namespace tangent_slope_implies_a_l3328_332862

/-- Given a function f(x) = ax^2 / (x+1), prove that if the slope of the tangent line
    at the point (1, f(1)) is 1, then a = 4/3 -/
theorem tangent_slope_implies_a (a : ℝ) :
  let f := fun x : ℝ => (a * x^2) / (x + 1)
  let f' := fun x : ℝ => ((a * x^2 + 2 * a * x) / (x + 1)^2)
  f' 1 = 1 → a = 4/3 := by
  sorry

end tangent_slope_implies_a_l3328_332862


namespace christmas_tree_lights_l3328_332831

theorem christmas_tree_lights (total : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : total = 95)
  (h2 : yellow = 37)
  (h3 : blue = 32) :
  total - (yellow + blue) = 26 := by
  sorry

end christmas_tree_lights_l3328_332831


namespace newspaper_photos_newspaper_photos_proof_l3328_332844

theorem newspaper_photos : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (pages_with_two_photos : ℕ) 
      (photos_per_page_first : ℕ) 
      (pages_with_three_photos : ℕ) 
      (photos_per_page_second : ℕ) 
      (total_photos : ℕ) =>
    pages_with_two_photos = 12 ∧ 
    photos_per_page_first = 2 ∧
    pages_with_three_photos = 9 ∧ 
    photos_per_page_second = 3 →
    total_photos = pages_with_two_photos * photos_per_page_first + 
                   pages_with_three_photos * photos_per_page_second ∧
    total_photos = 51

theorem newspaper_photos_proof : newspaper_photos 12 2 9 3 51 := by
  sorry

end newspaper_photos_newspaper_photos_proof_l3328_332844


namespace john_haircut_tip_percentage_l3328_332879

/-- Represents the growth rate of John's hair in inches per month -/
def hair_growth_rate : ℝ := 1.5

/-- Represents the length of John's hair in inches when he gets a haircut -/
def hair_length_at_cut : ℝ := 9

/-- Represents the length of John's hair in inches after a haircut -/
def hair_length_after_cut : ℝ := 6

/-- Represents the cost of a single haircut in dollars -/
def haircut_cost : ℝ := 45

/-- Represents the total amount John spends on haircuts in a year in dollars -/
def annual_haircut_spend : ℝ := 324

/-- Theorem stating that the percentage of the tip John gives for a haircut is 20% -/
theorem john_haircut_tip_percentage :
  let hair_growth_between_cuts := hair_length_at_cut - hair_length_after_cut
  let months_between_cuts := hair_growth_between_cuts / hair_growth_rate
  let haircuts_per_year := 12 / months_between_cuts
  let total_cost_per_haircut := annual_haircut_spend / haircuts_per_year
  let tip_amount := total_cost_per_haircut - haircut_cost
  let tip_percentage := (tip_amount / haircut_cost) * 100
  tip_percentage = 20 := by sorry

end john_haircut_tip_percentage_l3328_332879


namespace sum_coordinates_of_D_l3328_332818

/-- Given that N(3,5) is the midpoint of line segment CD and C has coordinates (1,10),
    prove that the sum of the coordinates of point D is 5. -/
theorem sum_coordinates_of_D (C D N : ℝ × ℝ) : 
  C = (1, 10) →
  N = (3, 5) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 5 := by
  sorry

end sum_coordinates_of_D_l3328_332818


namespace last_number_crossed_out_l3328_332866

/-- Represents the circular arrangement of numbers from 1 to 2016 -/
def CircularArrangement := Fin 2016

/-- The deletion process function -/
def deletionProcess (n : ℕ) : ℕ :=
  (n + 2) * (n - 1) / 2

/-- Theorem stating that 2015 is the last number to be crossed out -/
theorem last_number_crossed_out :
  ∃ (n : ℕ), deletionProcess n = 2015 ∧ 
  ∀ (m : ℕ), m > n → deletionProcess m > 2015 :=
sorry

end last_number_crossed_out_l3328_332866


namespace special_function_at_one_fifth_l3328_332821

/-- A monotonic function on (0, +∞) satisfying f(f(x) - 1/x) = 2 for all x > 0 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y ∨ f x > f y) ∧
  (∀ x, 0 < x → f (f x - 1/x) = 2)

/-- The value of f(1/5) for a special function f -/
theorem special_function_at_one_fifth
    (f : ℝ → ℝ) (h : special_function f) : f (1/5) = 6 := by
  sorry

end special_function_at_one_fifth_l3328_332821


namespace system_no_solution_l3328_332875

theorem system_no_solution (n : ℝ) : 
  (∀ x y z : ℝ, nx + y + z ≠ 2 ∨ x + ny + z ≠ 2 ∨ x + y + nz ≠ 2) ↔ n = -1 :=
by sorry

end system_no_solution_l3328_332875


namespace potato_ratio_l3328_332846

def potato_distribution (initial : ℕ) (gina : ℕ) (remaining : ℕ) : Prop :=
  ∃ (tom anne : ℕ),
    tom = 2 * gina ∧
    initial = gina + tom + anne + remaining ∧
    anne * 3 = tom

theorem potato_ratio (initial : ℕ) (gina : ℕ) (remaining : ℕ) 
  (h : potato_distribution initial gina remaining) :
  potato_distribution 300 69 47 :=
by sorry

end potato_ratio_l3328_332846


namespace quadratic_inequality_properties_l3328_332893

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < -1 ∨ x > 3}) :
  a < 0 ∧ 
  a + b + c > 0 ∧ 
  solution_set c (-b) a = {x : ℝ | -1/3 < x ∧ x < 1} :=
by sorry

end quadratic_inequality_properties_l3328_332893


namespace complex_modulus_equation_l3328_332840

theorem complex_modulus_equation (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end complex_modulus_equation_l3328_332840


namespace inequality_solution_set_l3328_332885

theorem inequality_solution_set (x : ℝ) :
  (2 * x) / (x - 2) ≤ 1 ↔ x ∈ Set.Icc (-2) 2 ∧ x ≠ 2 :=
by sorry

end inequality_solution_set_l3328_332885


namespace isosceles_trapezoid_right_triangle_l3328_332849

/-- 
Given an isosceles trapezoid with parallel sides a and c, non-parallel sides (legs) b, 
and diagonals e, prove that e² = b² + ac, which implies that the triangle formed by 
e, b, and √(ac) is a right triangle.
-/
theorem isosceles_trapezoid_right_triangle 
  (a c b e : ℝ) 
  (h_positive : a > 0 ∧ c > 0 ∧ b > 0 ∧ e > 0)
  (h_isosceles : ∃ m : ℝ, b^2 = ((a - c)/2)^2 + m^2 ∧ e^2 = ((a + c)/2)^2 + m^2) :
  e^2 = b^2 + a*c :=
sorry

end isosceles_trapezoid_right_triangle_l3328_332849


namespace cubic_sum_theorem_l3328_332881

theorem cubic_sum_theorem (p q r : ℝ) (hp : p ≠ q) (hq : q ≠ r) (hr : r ≠ p)
  (h : (p^3 + 8) / p = (q^3 + 8) / q ∧ (q^3 + 8) / q = (r^3 + 8) / r) :
  p^3 + q^3 + r^3 = -24 := by sorry

end cubic_sum_theorem_l3328_332881


namespace little_twelve_games_l3328_332889

/-- Represents a basketball conference with two divisions -/
structure BasketballConference :=
  (teams_per_division : ℕ)
  (divisions : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ)

/-- Calculates the total number of games in the conference -/
def total_games (conf : BasketballConference) : ℕ :=
  let total_teams := conf.teams_per_division * conf.divisions
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games + 
                        conf.teams_per_division * conf.inter_division_games
  total_teams * games_per_team / 2

/-- Theorem stating that the Little Twelve Basketball Conference schedules 96 games -/
theorem little_twelve_games : 
  ∀ (conf : BasketballConference), 
    conf.teams_per_division = 6 ∧ 
    conf.divisions = 2 ∧ 
    conf.intra_division_games = 2 ∧ 
    conf.inter_division_games = 1 → 
    total_games conf = 96 := by
  sorry

end little_twelve_games_l3328_332889


namespace avery_egg_cartons_l3328_332880

/-- The number of egg cartons that can be filled given the number of chickens,
    eggs per chicken, and eggs per carton. -/
def egg_cartons_filled (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / eggs_per_carton

/-- Theorem stating that with 20 chickens, each laying 6 eggs, and egg cartons
    that hold 12 eggs each, the number of egg cartons that can be filled is 10. -/
theorem avery_egg_cartons :
  egg_cartons_filled 20 6 12 = 10 := by
  sorry

end avery_egg_cartons_l3328_332880


namespace smallest_number_l3328_332861

theorem smallest_number (s : Set ℤ) (hs : s = {-2, 0, -1, 3}) : 
  ∃ m ∈ s, ∀ x ∈ s, m ≤ x ∧ m = -2 :=
by sorry

end smallest_number_l3328_332861


namespace triangle_area_l3328_332854

/-- The area of a triangle with side lengths 7, 7, and 5 is 2.5√42.75 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 7) (h2 : b = 7) (h3 : c = 5) :
  (1/2 : ℝ) * c * Real.sqrt ((a^2 - (c/2)^2) : ℝ) = 2.5 * Real.sqrt 42.75 := by
  sorry

end triangle_area_l3328_332854


namespace divisibility_problem_l3328_332884

theorem divisibility_problem (n a b c d : ℤ) 
  (hn : n > 0) 
  (h1 : n ∣ (a + b + c + d)) 
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) : 
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) :=
by sorry

end divisibility_problem_l3328_332884


namespace divisor_counts_of_N_l3328_332871

def N : ℕ := 10^40

/-- The number of natural divisors of N that are neither perfect squares nor perfect cubes -/
def count_non_square_non_cube_divisors (n : ℕ) : ℕ := sorry

/-- The number of natural divisors of N that cannot be represented as m^n where m and n are natural numbers and n > 1 -/
def count_non_power_divisors (n : ℕ) : ℕ := sorry

theorem divisor_counts_of_N :
  (count_non_square_non_cube_divisors N = 1093) ∧
  (count_non_power_divisors N = 981) := by sorry

end divisor_counts_of_N_l3328_332871


namespace total_marks_proof_l3328_332820

theorem total_marks_proof (keith_score larry_score danny_score : ℕ) : 
  keith_score = 3 →
  larry_score = 3 * keith_score →
  danny_score = larry_score + 5 →
  keith_score + larry_score + danny_score = 26 := by
sorry

end total_marks_proof_l3328_332820


namespace shaded_square_ratio_l3328_332877

/-- The ratio of the area of a shaded square to the area of a large square in a specific grid configuration -/
theorem shaded_square_ratio : 
  ∀ (n : ℕ) (large_square_area shaded_square_area : ℝ),
  n = 5 →
  large_square_area = n^2 →
  shaded_square_area = 4 * (1/2) →
  shaded_square_area / large_square_area = 2/25 := by
sorry

end shaded_square_ratio_l3328_332877


namespace supremum_of_expression_l3328_332868

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -1 / (2 * a) - 2 / b ≤ -9/2 ∧ 
  ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 1 ∧ -1 / (2 * a') - 2 / b' = -9/2 := by
  sorry

end supremum_of_expression_l3328_332868


namespace mathematician_daily_questions_l3328_332835

theorem mathematician_daily_questions 
  (project1_questions : ℕ) 
  (project2_questions : ℕ) 
  (days_in_week : ℕ) 
  (h1 : project1_questions = 518) 
  (h2 : project2_questions = 476) 
  (h3 : days_in_week = 7) :
  (project1_questions + project2_questions) / days_in_week = 142 :=
by sorry

end mathematician_daily_questions_l3328_332835


namespace track_length_is_24_l3328_332803

/-- Represents a circular ski track -/
structure SkiTrack where
  length : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : SkiTrack) : Prop :=
  track.downhill_speed = 4 * track.uphill_speed ∧
  track.length > 0 ∧
  ∃ (min_distance max_distance : ℝ),
    min_distance = 4 ∧
    max_distance = 13 ∧
    max_distance - min_distance = 9

/-- The theorem to be proved -/
theorem track_length_is_24 (track : SkiTrack) :
  problem_conditions track → track.length = 24 := by
  sorry

end track_length_is_24_l3328_332803


namespace cube_from_wire_l3328_332887

/-- Given a wire of length 60 cm formed into a cube frame, prove that the volume is 125 cm³ and the surface area is 150 cm². -/
theorem cube_from_wire (wire_length : ℝ) (h_wire : wire_length = 60) :
  let edge_length : ℝ := wire_length / 12
  let volume : ℝ := edge_length ^ 3
  let surface_area : ℝ := 6 * edge_length ^ 2
  volume = 125 ∧ surface_area = 150 := by sorry

end cube_from_wire_l3328_332887


namespace ellipse_eccentricity_l3328_332855

/-- Given an ellipse with point P and foci F₁ and F₂, if ∠PF₁F₂ = 60° and |PF₂| = √3|PF₁|,
    then the eccentricity of the ellipse is √3 - 1. -/
theorem ellipse_eccentricity (P F₁ F₂ : ℝ × ℝ) (a c : ℝ) :
  let e := c / a
  let angle_PF₁F₂ := Real.pi / 3  -- 60° in radians
  let dist_PF₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let dist_PF₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let dist_F₁F₂ := 2 * c
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 4 * a^2 →  -- P is on the ellipse
  dist_F₁F₂^2 = (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 →  -- Definition of distance between foci
  Real.cos angle_PF₁F₂ = (dist_PF₁^2 + dist_F₁F₂^2 - dist_PF₂^2) / (2 * dist_PF₁ * dist_F₁F₂) →  -- Cosine rule
  dist_PF₂ = Real.sqrt 3 * dist_PF₁ →
  e = Real.sqrt 3 - 1 :=
by sorry

end ellipse_eccentricity_l3328_332855


namespace sufficient_not_necessary_l3328_332865

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end sufficient_not_necessary_l3328_332865


namespace largest_factorable_n_l3328_332848

/-- The largest value of n for which 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 271

/-- A function that checks if a quadratic expression 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def is_factorable (n : ℤ) : Prop :=
  ∃ (a b : ℤ), (3 * a + b = n) ∧ (a * b = 90)

theorem largest_factorable_n :
  (is_factorable largest_n) ∧ 
  (∀ m : ℤ, m > largest_n → ¬(is_factorable m)) :=
sorry

end largest_factorable_n_l3328_332848


namespace greatest_integer_with_nonpositive_product_l3328_332872

theorem greatest_integer_with_nonpositive_product (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_coprime : Nat.Coprime a b) :
  ∀ n : ℕ, n > a * b →
    ∃ x y : ℤ, (n : ℤ) = a * x + b * y ∧ x * y > 0 :=
by sorry

end greatest_integer_with_nonpositive_product_l3328_332872


namespace gcd_of_specific_squares_l3328_332858

theorem gcd_of_specific_squares : Nat.gcd (131^2 + 243^2 + 357^2) (130^2 + 242^2 + 358^2) = 1 := by
  sorry

end gcd_of_specific_squares_l3328_332858


namespace society_officer_selection_l3328_332808

theorem society_officer_selection (n : ℕ) (k : ℕ) : n = 12 ∧ k = 5 →
  (n.factorial / (n - k).factorial) = 95040 := by
  sorry

end society_officer_selection_l3328_332808


namespace point_coordinates_wrt_origin_l3328_332837

/-- The coordinates of a point (3, -2) with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let p : ℝ × ℝ := (3, -2)
  p = (3, -2) :=
by sorry

end point_coordinates_wrt_origin_l3328_332837


namespace probability_zero_l3328_332892

def P (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 20

theorem probability_zero :
  ∀ x : ℝ, 3 ≤ x ∧ x ≤ 10 →
  (⌊(P x)^(1/3)⌋ : ℝ) ≠ (P ⌊x⌋)^(1/3) :=
by sorry

end probability_zero_l3328_332892


namespace vector_points_to_line_and_parallel_l3328_332801

/-- The line is parameterized by x = 3t + 1, y = t + 1 -/
def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, t + 1)

/-- The direction vector -/
def direction : ℝ × ℝ := (3, 1)

/-- The vector we want to prove -/
def vector : ℝ × ℝ := (9, 3)

theorem vector_points_to_line_and_parallel :
  (∃ t : ℝ, line_param t = vector) ∧ 
  (∃ k : ℝ, vector = (k * direction.1, k * direction.2)) :=
sorry

end vector_points_to_line_and_parallel_l3328_332801


namespace inverse_B_cubed_l3328_332836

theorem inverse_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, 7; -2, -4]) : 
  (B⁻¹)^3 = !![11, 17; -10, -18] := by
  sorry

end inverse_B_cubed_l3328_332836


namespace product_tens_digit_is_nine_l3328_332874

theorem product_tens_digit_is_nine (x : ℤ) : 
  0 ≤ x ∧ x ≤ 9 → 
  ((200 + 10 * x + 7) * 39 ≡ 90 [ZMOD 100] ↔ x = 8) :=
by sorry

end product_tens_digit_is_nine_l3328_332874


namespace sequence_matches_first_five_terms_general_term_formula_l3328_332830

/-- The sequence a_n defined by the given first five terms and the general formula -/
def a : ℕ → ℕ := λ n => n^2 + 5

/-- The theorem stating that the sequence matches the given first five terms -/
theorem sequence_matches_first_five_terms :
  a 1 = 6 ∧ a 2 = 9 ∧ a 3 = 14 ∧ a 4 = 21 ∧ a 5 = 30 := by sorry

/-- The main theorem proving that a_n is the general term formula for the sequence -/
theorem general_term_formula (n : ℕ) (h : n > 0) : a n = n^2 + 5 := by sorry

end sequence_matches_first_five_terms_general_term_formula_l3328_332830


namespace l_shape_area_and_perimeter_l3328_332898

/-- Represents the dimensions of a rectangle -/
structure RectangleDimensions where
  length : Real
  width : Real

/-- Calculates the area of a rectangle -/
def rectangleArea (d : RectangleDimensions) : Real :=
  d.length * d.width

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (d : RectangleDimensions) : Real :=
  2 * (d.length + d.width)

/-- Represents an L-shaped region formed by two rectangles -/
structure LShape where
  rect1 : RectangleDimensions
  rect2 : RectangleDimensions

/-- Calculates the area of an L-shaped region -/
def lShapeArea (l : LShape) : Real :=
  rectangleArea l.rect1 + rectangleArea l.rect2

/-- Calculates the perimeter of an L-shaped region -/
def lShapePerimeter (l : LShape) : Real :=
  rectanglePerimeter l.rect1 + rectanglePerimeter l.rect2 - 2 * l.rect1.length

theorem l_shape_area_and_perimeter :
  let l : LShape := {
    rect1 := { length := 0.5, width := 0.3 },
    rect2 := { length := 0.2, width := 0.5 }
  }
  lShapeArea l = 0.25 ∧ lShapePerimeter l = 2.0 := by sorry

end l_shape_area_and_perimeter_l3328_332898


namespace K_is_perfect_square_l3328_332856

def K (n : ℕ) : ℚ :=
  (4 * (10^(2*n) - 1) / 9) - (8 * (10^n - 1) / 9)

theorem K_is_perfect_square (n : ℕ) :
  ∃ (m : ℚ), K n = m^2 := by
sorry

end K_is_perfect_square_l3328_332856


namespace morgan_change_l3328_332886

/-- Calculates the change received from a purchase given item costs and amount paid --/
def calculate_change (hamburger_cost onion_rings_cost smoothie_cost amount_paid : ℕ) : ℕ :=
  amount_paid - (hamburger_cost + onion_rings_cost + smoothie_cost)

/-- Theorem stating that Morgan receives $11 in change --/
theorem morgan_change : calculate_change 4 2 3 20 = 11 := by
  sorry

end morgan_change_l3328_332886


namespace exists_a_with_two_common_tangents_l3328_332825

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y a : ℝ) : Prop := (x - 4)^2 + (y + a)^2 = 64

/-- Condition for two circles to have exactly 2 common tangents -/
def has_two_common_tangents (a : ℝ) : Prop :=
  6 < Real.sqrt (16 + a^2) ∧ Real.sqrt (16 + a^2) < 10

/-- Theorem stating the existence of a positive integer a satisfying the conditions -/
theorem exists_a_with_two_common_tangents :
  ∃ a : ℕ+, has_two_common_tangents a.val := by sorry

end exists_a_with_two_common_tangents_l3328_332825


namespace parabola_c_value_l3328_332857

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value :
  ∀ p : Parabola,
    p.x_coord (-3) = 2 →  -- vertex at (2, -3)
    p.x_coord (-1) = 7 →  -- passes through (7, -1)
    p.c = 53/4 := by
  sorry

end parabola_c_value_l3328_332857


namespace pencil_price_theorem_l3328_332883

/-- Calculates the final price of a pencil after applying discounts and taxes -/
def final_price (initial_cost christmas_discount seasonal_discount final_discount tax_rate : ℚ) : ℚ :=
  let price_after_christmas := initial_cost - christmas_discount
  let price_after_seasonal := price_after_christmas * (1 - seasonal_discount)
  let price_after_final := price_after_seasonal * (1 - final_discount)
  price_after_final * (1 + tax_rate)

/-- The final price of the pencil is approximately $3.17 -/
theorem pencil_price_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_price 4 0.63 0.07 0.05 0.065 - 3.17| < ε :=
sorry

end pencil_price_theorem_l3328_332883


namespace a_plus_b_value_l3328_332823

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (A ∪ B a b = univ) → (A ∩ B a b = Ioc 3 4) → a + b = -7 :=
by sorry

end a_plus_b_value_l3328_332823


namespace chord_length_l3328_332869

theorem chord_length (r : ℝ) (h : r = 15) : 
  ∃ (c : ℝ), c = 26 * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt (r^2 - (r/2)^2) := by
sorry

end chord_length_l3328_332869


namespace ratio_x_to_y_l3328_332809

theorem ratio_x_to_y (x y : ℝ) (h : (14*x - 5*y) / (17*x - 3*y) = 4/6) : x/y = 1/23 := by
  sorry

end ratio_x_to_y_l3328_332809


namespace product_from_sum_and_difference_l3328_332824

theorem product_from_sum_and_difference (a b : ℝ) : 
  a + b = 60 ∧ a - b = 10 → a * b = 875 := by
  sorry

end product_from_sum_and_difference_l3328_332824


namespace existence_of_equal_segments_l3328_332867

/-- An acute-angled triangle is a triangle where all angles are less than 90 degrees -/
def AcuteTriangle (A B C : Point) : Prop := sorry

/-- Point X is on line segment AB -/
def OnSegment (X A B : Point) : Prop := sorry

/-- AX = XY = YC -/
def EqualSegments (A X Y C : Point) : Prop := sorry

/-- Theorem: In any acute-angled triangle, there exist points X and Y on its sides
    such that AX = XY = YC -/
theorem existence_of_equal_segments (A B C : Point) 
  (h : AcuteTriangle A B C) : 
  ∃ X Y, OnSegment X A B ∧ OnSegment Y B C ∧ EqualSegments A X Y C := by
  sorry

end existence_of_equal_segments_l3328_332867


namespace factor_sum_l3328_332845

/-- If x^2 + 3x + 4 is a factor of x^4 + Px^2 + Q, then P + Q = 15 -/
theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end factor_sum_l3328_332845


namespace tutor_schedule_lcm_l3328_332878

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := by
  sorry

end tutor_schedule_lcm_l3328_332878


namespace jade_handled_80_transactions_l3328_332833

/-- Calculates the number of transactions Jade handled given the conditions of the problem. -/
def jade_transactions (mabel_transactions : ℕ) : ℕ :=
  let anthony_transactions := mabel_transactions + mabel_transactions / 10
  let cal_transactions := anthony_transactions * 2 / 3
  cal_transactions + 14

/-- Theorem stating that Jade handled 80 transactions given the conditions of the problem. -/
theorem jade_handled_80_transactions : jade_transactions 90 = 80 := by
  sorry

end jade_handled_80_transactions_l3328_332833


namespace daily_harvest_l3328_332873

/-- The number of sections in the orchard -/
def num_sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

theorem daily_harvest : total_sacks = 360 := by
  sorry

end daily_harvest_l3328_332873


namespace calculation_proof_l3328_332810

theorem calculation_proof : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end calculation_proof_l3328_332810


namespace remaining_quadrilateral_perimeter_l3328_332838

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perimeter of a quadrilateral -/
def Quadrilateral.perimeter (q : Quadrilateral) : ℝ :=
  q.a + q.b + q.c + q.d

/-- Given an equilateral triangle ABC with side length 5 and a triangular section DBE cut from it
    with DB = EB = 2, the perimeter of the remaining quadrilateral ACED is 13 -/
theorem remaining_quadrilateral_perimeter :
  ∀ (abc : Triangle) (dbe : Triangle) (aced : Quadrilateral),
    abc.a = 5 ∧ abc.b = 5 ∧ abc.c = 5 →  -- ABC is equilateral with side length 5
    dbe.a = 2 ∧ dbe.b = 2 ∧ dbe.c = 2 →  -- DBE is equilateral with side length 2
    aced.a = 5 ∧                         -- AC remains untouched
    aced.b = abc.b - dbe.b ∧             -- CE = AB - DB
    aced.c = dbe.c ∧                     -- ED is a side of DBE
    aced.d = abc.c - dbe.c →             -- DA = BC - BE
    aced.perimeter = 13 :=
by sorry

end remaining_quadrilateral_perimeter_l3328_332838


namespace range_a_theorem_l3328_332895

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 2, x^2 - a ≥ 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, 2*x^2 + a*x + 1 > 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2*Real.sqrt 2 ∨ (0 < a ∧ a < 2*Real.sqrt 2)

-- State the theorem
theorem range_a_theorem (a : ℝ) : 
  (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) → range_of_a a :=
by sorry

end range_a_theorem_l3328_332895


namespace equal_areas_of_same_side_lengths_l3328_332859

/-- A polygon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  sides : Fin n → ℝ
  inscribed : Bool

/-- The area of an inscribed polygon -/
noncomputable def area (p : InscribedPolygon n) : ℝ := sorry

/-- Two polygons have the same set of side lengths -/
def same_side_lengths (p1 p2 : InscribedPolygon n) : Prop :=
  ∃ (σ : Equiv (Fin n) (Fin n)), ∀ i, p1.sides i = p2.sides (σ i)

theorem equal_areas_of_same_side_lengths (n : ℕ) (p1 p2 : InscribedPolygon n) 
  (h1 : p1.inscribed) (h2 : p2.inscribed) (h3 : same_side_lengths p1 p2) : 
  area p1 = area p2 := by sorry

end equal_areas_of_same_side_lengths_l3328_332859


namespace students_catching_up_on_homework_l3328_332832

theorem students_catching_up_on_homework (total : ℕ) (silent_reading : ℕ) (board_games : ℕ) :
  total = 24 →
  silent_reading = total / 2 →
  board_games = total / 3 →
  total - (silent_reading + board_games) = 4 :=
by
  sorry

end students_catching_up_on_homework_l3328_332832
