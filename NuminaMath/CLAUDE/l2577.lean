import Mathlib

namespace probability_of_car_Z_winning_l2577_257761

/-- Given a race with 15 cars, prove that the probability of car Z winning is 1/12 -/
theorem probability_of_car_Z_winning (total_cars : ℕ) (prob_X prob_Y prob_XYZ : ℚ) :
  total_cars = 15 →
  prob_X = 1/4 →
  prob_Y = 1/8 →
  prob_XYZ = 458333333333333333/1000000000000000000 →
  prob_XYZ = prob_X + prob_Y + (1/12) :=
by sorry

end probability_of_car_Z_winning_l2577_257761


namespace greatest_divisor_of_product_l2577_257753

def S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {t | let (a, b, c, d, e, f) := t
       a^2 + b^2 + c^2 + d^2 + e^2 = f^2}

theorem greatest_divisor_of_product (k : ℕ) : k = 24 ↔ 
  (∀ t ∈ S, let (a, b, c, d, e, f) := t
            (k : ℤ) ∣ a * b * c * d * e * f) ∧
  (∀ m > k, ∃ t ∈ S, let (a, b, c, d, e, f) := t
            ¬((m : ℤ) ∣ a * b * c * d * e * f)) := by
  sorry

end greatest_divisor_of_product_l2577_257753


namespace negation_of_proposition_l2577_257739

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ (x : ℝ), x ≥ 0 → Real.log (x^2 + 1) ≥ 0)) ↔ 
  (∃ (x : ℝ), x ≥ 0 ∧ Real.log (x^2 + 1) < 0) :=
by sorry

end negation_of_proposition_l2577_257739


namespace students_suggesting_pasta_l2577_257799

theorem students_suggesting_pasta 
  (total_students : ℕ) 
  (mashed_potatoes : ℕ) 
  (bacon : ℕ) 
  (h1 : total_students = 470) 
  (h2 : mashed_potatoes = 230) 
  (h3 : bacon = 140) : 
  total_students - (mashed_potatoes + bacon) = 100 := by
sorry

end students_suggesting_pasta_l2577_257799


namespace total_new_emails_formula_l2577_257704

/-- Represents the number of new emails received in one deletion cycle -/
def new_emails_per_cycle : ℕ := 15 + 5

/-- Represents the final batch of emails received -/
def final_batch : ℕ := 10

/-- Calculates the total number of new emails after n cycles and a final batch -/
def total_new_emails (n : ℕ) : ℕ := n * new_emails_per_cycle + final_batch

/-- Theorem stating the total number of new emails after n cycles and a final batch -/
theorem total_new_emails_formula (n : ℕ) : 
  total_new_emails n = 20 * n + 10 := by
  sorry

#eval total_new_emails 5  -- Example evaluation

end total_new_emails_formula_l2577_257704


namespace rupert_weight_l2577_257723

/-- Proves that Rupert weighs 35 kilograms given the conditions -/
theorem rupert_weight (antoinette_weight rupert_weight : ℕ) : 
  antoinette_weight = 63 → 
  antoinette_weight = 2 * rupert_weight - 7 → 
  rupert_weight = 35 := by
  sorry

end rupert_weight_l2577_257723


namespace polynomial_factorization_l2577_257752

theorem polynomial_factorization (x : ℝ) : 
  2 * x^3 - 4 * x^2 + 2 * x = 2 * x * (x - 1)^2 := by
  sorry

end polynomial_factorization_l2577_257752


namespace parallel_transitivity_perpendicular_plane_implies_parallel_l2577_257735

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)

-- Axiom for transitivity of parallel lines
axiom parallel_trans (a b c : Line) : parallel a b → parallel b c → parallel a c

-- Axiom for perpendicular lines to the same plane being parallel
axiom perpendicular_plane_parallel (a b : Line) (γ : Plane) : 
  perpendicularToPlane a γ → perpendicularToPlane b γ → parallel a b

-- Theorem 1: If two lines are parallel to a third line, then they are parallel to each other
theorem parallel_transitivity (a b c : Line) : 
  parallel a b → parallel b c → parallel a c :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel to each other
theorem perpendicular_plane_implies_parallel (a b : Line) (γ : Plane) :
  perpendicularToPlane a γ → perpendicularToPlane b γ → parallel a b :=
sorry

end parallel_transitivity_perpendicular_plane_implies_parallel_l2577_257735


namespace triangle_angle_measure_l2577_257780

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 6 →
  c = 3 →
  C = 60 * π / 180 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a / Real.sin A = c / Real.sin C) →
  A + B + C = π →
  A = 75 * π / 180 :=
by
  sorry

end triangle_angle_measure_l2577_257780


namespace beth_bought_ten_cans_of_corn_l2577_257727

/-- The number of cans of corn Beth bought -/
def cans_of_corn : ℕ := sorry

/-- The number of cans of peas Beth bought -/
def cans_of_peas : ℕ := 35

/-- The relationship between cans of peas and cans of corn -/
axiom peas_corn_relation : cans_of_peas = 15 + 2 * cans_of_corn

theorem beth_bought_ten_cans_of_corn : cans_of_corn = 10 := by sorry

end beth_bought_ten_cans_of_corn_l2577_257727


namespace drinking_speed_ratio_l2577_257788

theorem drinking_speed_ratio 
  (total_volume : ℝ) 
  (mala_volume : ℝ) 
  (usha_volume : ℝ) 
  (drinking_time : ℝ) 
  (h1 : drinking_time > 0) 
  (h2 : total_volume > 0) 
  (h3 : mala_volume + usha_volume = total_volume) 
  (h4 : usha_volume = 2 / 10 * total_volume) : 
  (mala_volume / drinking_time) / (usha_volume / drinking_time) = 4 := by
sorry

end drinking_speed_ratio_l2577_257788


namespace count_lambs_l2577_257702

def farmer_cunningham_lambs : Nat → Nat → Prop
  | white_lambs, black_lambs =>
    ∀ (total_lambs : Nat),
      (white_lambs = 193) →
      (black_lambs = 5855) →
      (total_lambs = white_lambs + black_lambs) →
      (total_lambs = 6048)

theorem count_lambs :
  farmer_cunningham_lambs 193 5855 := by
  sorry

end count_lambs_l2577_257702


namespace coefficient_properties_l2577_257755

-- Define the polynomial coefficients
variable (a : Fin 7 → ℝ)

-- Define the given equation
def equation (x : ℝ) : Prop :=
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + 
              a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6

-- State the theorem
theorem coefficient_properties (a : Fin 7 → ℝ) 
  (h : ∀ x, equation a x) : 
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 := by
  sorry

end coefficient_properties_l2577_257755


namespace cube_volume_from_surface_area_l2577_257777

/-- Theorem: Volume of a cube with surface area 150 square inches --/
theorem cube_volume_from_surface_area :
  let surface_area : ℝ := 150  -- Surface area in square inches
  let edge_length : ℝ := Real.sqrt (surface_area / 6)  -- Edge length in inches
  let volume_cubic_inches : ℝ := edge_length ^ 3  -- Volume in cubic inches
  let cubic_inches_per_cubic_foot : ℝ := 1728  -- Conversion factor
  let volume_cubic_feet : ℝ := volume_cubic_inches / cubic_inches_per_cubic_foot
  ∃ ε > 0, |volume_cubic_feet - 0.0723| < ε :=
by
  sorry

end cube_volume_from_surface_area_l2577_257777


namespace complex_equation_solution_l2577_257778

theorem complex_equation_solution (z : ℂ) :
  z * (1 + Complex.I) = -2 * Complex.I → z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l2577_257778


namespace altitude_sum_of_triangle_l2577_257724

/-- The line equation --/
def line_equation (x y : ℝ) : Prop := 15 * x + 6 * y = 90

/-- A point is on the x-axis if its y-coordinate is 0 --/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A point is on the y-axis if its x-coordinate is 0 --/
def on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- The triangle vertices --/
def triangle_vertices : Set (ℝ × ℝ) := {(0, 0), (6, 0), (0, 15)}

/-- The sum of altitudes of the triangle --/
noncomputable def altitude_sum : ℝ := 21 + 10 * Real.sqrt (1 / 29)

/-- The main theorem --/
theorem altitude_sum_of_triangle :
  ∀ (p : ℝ × ℝ), p ∈ triangle_vertices →
  (on_x_axis p ∨ on_y_axis p ∨ line_equation p.1 p.2) →
  altitude_sum = 21 + 10 * Real.sqrt (1 / 29) :=
sorry

end altitude_sum_of_triangle_l2577_257724


namespace sum_of_cubes_of_sums_l2577_257745

-- Define the polynomial
def p (x : ℝ) : ℝ := 10 * x^3 + 101 * x + 210

-- Define the roots
def roots_of_p (a b c : ℝ) : Prop := p a = 0 ∧ p b = 0 ∧ p c = 0

-- Theorem statement
theorem sum_of_cubes_of_sums (a b c : ℝ) :
  roots_of_p a b c → (a + b)^3 + (b + c)^3 + (c + a)^3 = 63 := by
  sorry

end sum_of_cubes_of_sums_l2577_257745


namespace paint_area_is_134_l2577_257766

/-- The area to be painted on a wall with a window and door -/
def areaToPaint (wallHeight wallLength windowSide doorWidth doorHeight : ℝ) : ℝ :=
  wallHeight * wallLength - windowSide * windowSide - doorWidth * doorHeight

/-- Theorem: The area to be painted is 134 square feet -/
theorem paint_area_is_134 :
  areaToPaint 10 15 3 1 7 = 134 := by
  sorry

end paint_area_is_134_l2577_257766


namespace circle_radius_proof_l2577_257729

-- Define the circle and its properties
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the chord length
def chord_length : ℝ := 10

-- Define the internal segment of the secant
def secant_internal : ℝ := 12

-- Theorem statement
theorem circle_radius_proof (r : ℝ) (h1 : r > 0) :
  ∃ (A B C : ℝ × ℝ),
    A ∈ Circle r ∧ 
    B ∈ Circle r ∧ 
    C ∈ Circle r ∧
    ‖A - B‖ = chord_length ∧
    ‖B - C‖ = secant_internal ∧
    (∃ (D : ℝ × ℝ), D ∈ Circle r ∧ (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0) →
    r = 25 / 4 := by
  sorry

end circle_radius_proof_l2577_257729


namespace calculation_proof_l2577_257769

theorem calculation_proof : 72 / (6 / 3) * 2 = 72 := by
  sorry

end calculation_proof_l2577_257769


namespace no_common_elements_l2577_257772

theorem no_common_elements : ¬∃ (n m : ℕ), n^2 - 1 = m^2 + 1 := by sorry

end no_common_elements_l2577_257772


namespace bingley_has_six_bracelets_l2577_257770

/-- The number of bracelets Bingley has remaining after the exchanges -/
def bingleys_remaining_bracelets (bingley_initial : ℕ) (kelly_initial : ℕ) : ℕ :=
  let bingley_after_kelly := bingley_initial + kelly_initial / 4
  bingley_after_kelly - bingley_after_kelly / 3

/-- Theorem stating that Bingley will have 6 bracelets remaining -/
theorem bingley_has_six_bracelets : 
  bingleys_remaining_bracelets 5 16 = 6 := by
  sorry

end bingley_has_six_bracelets_l2577_257770


namespace quadratic_intersection_l2577_257741

theorem quadratic_intersection
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hcd : c ≠ d) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * x^2 - b * x + d
  let x_intersect := (d - c) / (2 * b)
  let y_intersect := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  ∃ (x y : ℝ), f x = g x ∧ f x = y ∧ x = x_intersect ∧ y = y_intersect :=
by
  sorry


end quadratic_intersection_l2577_257741


namespace remaining_terms_geometric_l2577_257710

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

theorem remaining_terms_geometric (a q : ℝ) (k : ℕ) :
  let original_seq := geometric_sequence a q
  let remaining_seq := fun n => original_seq (n + k)
  ∃ a', remaining_seq = geometric_sequence a' q :=
sorry

end remaining_terms_geometric_l2577_257710


namespace profit_40_percent_l2577_257754

/-- Calculates the profit percentage when selling a certain number of articles at a price equal to the cost of a different number of articles. -/
def profit_percentage (sold : ℕ) (cost_equivalent : ℕ) : ℚ :=
  ((cost_equivalent - sold) / sold) * 100

/-- Theorem stating that selling 50 articles at the cost price of 70 articles results in a 40% profit. -/
theorem profit_40_percent :
  profit_percentage 50 70 = 40 := by
  sorry

end profit_40_percent_l2577_257754


namespace origin_inside_ellipse_k_range_l2577_257721

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- A point (x,y) is inside the ellipse if the left side of the equation is negative -/
def inside_ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 < 0

/-- The theorem stating the range of k for which the origin (0,0) is inside the ellipse -/
theorem origin_inside_ellipse_k_range :
  ∀ k : ℝ, (inside_ellipse k 0 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end origin_inside_ellipse_k_range_l2577_257721


namespace inequality_proof_l2577_257760

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (x^6 / y^2) + (y^6 / x^2) ≥ x^4 + y^4 := by
sorry

end inequality_proof_l2577_257760


namespace fermat_like_prime_condition_l2577_257726

theorem fermat_like_prime_condition (a n : ℕ) (ha : a ≥ 2) (hn : n ≥ 2) 
  (h_prime : Nat.Prime (a^n - 1)) : a = 2 ∧ Nat.Prime n := by
  sorry

end fermat_like_prime_condition_l2577_257726


namespace cades_marbles_l2577_257751

theorem cades_marbles (initial_marbles : ℕ) (marbles_given : ℕ) : 
  initial_marbles = 87 → marbles_given = 8 → initial_marbles - marbles_given = 79 := by
  sorry

end cades_marbles_l2577_257751


namespace line_passes_through_P_triangle_perimeter_l2577_257740

/-- The equation of line l is (a+1)x + y - 5 - 2a = 0, where a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y - 5 - 2 * a = 0

/-- Point P that the line passes through -/
def point_P : ℝ × ℝ := (2, 3)

/-- The area of triangle AOB -/
def triangle_area : ℝ := 12

theorem line_passes_through_P (a : ℝ) : line_equation a (point_P.1) (point_P.2) := by sorry

theorem triangle_perimeter : 
  ∃ (a x_A y_B : ℝ), 
    line_equation a x_A 0 ∧ 
    line_equation a 0 y_B ∧ 
    x_A * y_B / 2 = triangle_area ∧ 
    x_A + y_B + Real.sqrt (x_A^2 + y_B^2) = 10 + 2 * Real.sqrt 13 := by sorry

end line_passes_through_P_triangle_perimeter_l2577_257740


namespace inequality_solution_set_l2577_257709

theorem inequality_solution_set : 
  {x : ℝ | (3 : ℝ) / (5 - 3 * x) > 1} = {x : ℝ | 2/3 < x ∧ x < 5/3} := by
  sorry

end inequality_solution_set_l2577_257709


namespace all_terms_are_integers_l2577_257782

/-- An infinite increasing arithmetic progression where the product of any two distinct terms is also a term in the progression. -/
structure SpecialArithmeticProgression where
  -- The sequence of terms in the progression
  sequence : ℕ → ℚ
  -- The common difference of the progression
  common_difference : ℚ
  -- The progression is increasing
  increasing : ∀ n : ℕ, sequence n < sequence (n + 1)
  -- The progression follows the arithmetic sequence formula
  is_arithmetic : ∀ n : ℕ, sequence (n + 1) = sequence n + common_difference
  -- The product of any two distinct terms is also a term
  product_is_term : ∀ m n : ℕ, m ≠ n → ∃ k : ℕ, sequence m * sequence n = sequence k

/-- All terms in a SpecialArithmeticProgression are integers. -/
theorem all_terms_are_integers (ap : SpecialArithmeticProgression) : 
  ∀ n : ℕ, ∃ k : ℤ, ap.sequence n = k :=
sorry

end all_terms_are_integers_l2577_257782


namespace distribute_five_balls_four_boxes_l2577_257791

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end distribute_five_balls_four_boxes_l2577_257791


namespace line_equation_from_intercept_and_angle_l2577_257767

/-- The equation of a line with given x-intercept and inclination angle -/
theorem line_equation_from_intercept_and_angle (x_intercept : ℝ) (angle : ℝ) :
  x_intercept = 2 ∧ angle = 135 * π / 180 →
  ∀ x y : ℝ, (x + y - 2 = 0) ↔ (y = (x - x_intercept) * Real.tan angle) :=
by sorry

end line_equation_from_intercept_and_angle_l2577_257767


namespace max_distance_theorem_l2577_257717

/-- Given points in a 2D Cartesian coordinate system, prove the maximum distance -/
theorem max_distance_theorem (x y : ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let C : ℝ × ℝ := (3, 0)
  let D : ℝ × ℝ := (x, y)
  (x - 3)^2 + y^2 = 1 →
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + y₀^2 = 1 ∧ 
    ∀ (x' y' : ℝ), (x' - 3)^2 + y'^2 = 1 → 
      ((x' - 1)^2 + (y' + Real.sqrt 3)^2) ≤ ((x₀ - 1)^2 + (y₀ + Real.sqrt 3)^2)) ∧
  ((x₀ - 1)^2 + (y₀ + Real.sqrt 3)^2) = (Real.sqrt 7 + 1)^2 :=
by sorry

end max_distance_theorem_l2577_257717


namespace lunch_price_with_gratuity_l2577_257725

theorem lunch_price_with_gratuity 
  (num_people : ℕ) 
  (avg_price : ℝ) 
  (gratuity_rate : ℝ) : 
  num_people = 15 →
  avg_price = 12 →
  gratuity_rate = 0.15 →
  num_people * avg_price * (1 + gratuity_rate) = 207 := by
  sorry

end lunch_price_with_gratuity_l2577_257725


namespace volume_right_prism_isosceles_base_l2577_257700

/-- Volume of a right prism with isosceles triangular base -/
theorem volume_right_prism_isosceles_base 
  (a : ℝ) (α : ℝ) (S : ℝ) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_S : S > 0) : 
  ∃ V : ℝ, V = (a * S / 2) * Real.sin (α / 2) * Real.tan ((π - α) / 4) ∧ 
  V = (Real.sin α * a^2 / 2) * (S / (2 * a * (1 + Real.sin (α / 2)))) :=
sorry

end volume_right_prism_isosceles_base_l2577_257700


namespace circle_properties_l2577_257790

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 36 = -y^2 + 14*x + 4

def is_center_and_radius (c d s : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2

theorem circle_properties :
  ∃ c d s : ℝ, is_center_and_radius c d s ∧ c = 7 ∧ d = 2 ∧ s^2 = 93 ∧ c + d + s = 9 + Real.sqrt 93 :=
sorry

end circle_properties_l2577_257790


namespace question_mark_value_l2577_257781

theorem question_mark_value : ∃ (x : ℕ), x * 40 = 173 * 240 ∧ x = 1036 := by
  sorry

end question_mark_value_l2577_257781


namespace range_of_4a_minus_2b_l2577_257789

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  5 ≤ 4*a - 2*b ∧ 4*a - 2*b ≤ 10 :=
by sorry

end range_of_4a_minus_2b_l2577_257789


namespace two_digit_number_pair_exists_l2577_257720

theorem two_digit_number_pair_exists : ∃ x y : ℤ,
  10 ≤ x ∧ x < 100 ∧
  10 ≤ y ∧ y < 100 ∧
  x + 15 < 100 ∧
  y - 20 ≥ 10 ∧
  (x + 15) * (y - 20) = x * y :=
by sorry

end two_digit_number_pair_exists_l2577_257720


namespace rectangular_plot_poles_l2577_257714

/-- The number of poles needed to enclose a rectangular plot -/
def num_poles (length width pole_distance : ℕ) : ℕ :=
  2 * (length + width) / pole_distance + 4

theorem rectangular_plot_poles :
  num_poles 90 50 4 = 74 :=
by sorry

end rectangular_plot_poles_l2577_257714


namespace complex_fraction_simplification_l2577_257713

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  let z : ℂ := 5 * i / (1 - 2 * i)
  z = -2 + i := by
  sorry

end complex_fraction_simplification_l2577_257713


namespace min_value_theorem_l2577_257773

/-- The function f(x) = |2x - 1| -/
def f (x : ℝ) : ℝ := |2 * x - 1|

/-- The function g(x) = f(x) + f(x - 1) -/
def g (x : ℝ) : ℝ := f x + f (x - 1)

/-- The minimum value of g(x) -/
def a : ℝ := 2

/-- Theorem: The minimum value of (m^2 + 2)/m + (n^2 + 1)/n is (7 + 2√2)/2,
    given m + n = a and m, n > 0 -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = a) :
  (m^2 + 2)/m + (n^2 + 1)/n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_theorem_l2577_257773


namespace competition_result_l2577_257728

/-- Represents the scores of contestants in a mathematics competition. -/
structure Scores where
  ann : ℝ
  bill : ℝ
  carol : ℝ
  dick : ℝ
  nonnegative : 0 ≤ ann ∧ 0 ≤ bill ∧ 0 ≤ carol ∧ 0 ≤ dick

/-- Conditions of the mathematics competition. -/
def CompetitionConditions (s : Scores) : Prop :=
  s.bill + s.dick = 2 * s.ann ∧
  s.ann + s.carol < s.bill + s.dick ∧
  s.ann < s.bill + s.carol

/-- The order of contestants from highest to lowest score. -/
def CorrectOrder (s : Scores) : Prop :=
  s.dick > s.bill ∧ s.bill > s.ann ∧ s.ann > s.carol

theorem competition_result (s : Scores) (h : CompetitionConditions s) : CorrectOrder s := by
  sorry

end competition_result_l2577_257728


namespace cupcakes_eaten_l2577_257785

theorem cupcakes_eaten (total_baked : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : 
  total_baked = 68 →
  packages = 6 →
  cupcakes_per_package = 6 →
  total_baked - (packages * cupcakes_per_package) = 
    total_baked - (total_baked - (packages * cupcakes_per_package)) := by
  sorry

end cupcakes_eaten_l2577_257785


namespace yellow_marble_probability_l2577_257744

/-- Represents a bag of marbles with two colors -/
structure Bag where
  color1 : ℕ
  color2 : ℕ

/-- Calculate the probability of drawing a specific color from a bag -/
def probColor (bag : Bag) (color : ℕ) : ℚ :=
  color / (bag.color1 + bag.color2)

/-- The probability of drawing a yellow marble as the second marble -/
def probYellowSecond (bagX bagY bagZ : Bag) : ℚ :=
  probColor bagX bagX.color1 * probColor bagY bagY.color1 +
  probColor bagX bagX.color2 * probColor bagZ bagZ.color1

theorem yellow_marble_probability :
  let bagX : Bag := ⟨4, 5⟩  -- 4 white, 5 black
  let bagY : Bag := ⟨7, 3⟩  -- 7 yellow, 3 blue
  let bagZ : Bag := ⟨3, 6⟩  -- 3 yellow, 6 blue
  probYellowSecond bagX bagY bagZ = 67 / 135 := by
  sorry


end yellow_marble_probability_l2577_257744


namespace remaining_episodes_l2577_257737

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) 
  (watched_fraction : ℚ) : 
  total_seasons = 12 → 
  episodes_per_season = 20 → 
  watched_fraction = 1/3 →
  total_seasons * episodes_per_season - (watched_fraction * (total_seasons * episodes_per_season)).num = 160 := by
  sorry

end remaining_episodes_l2577_257737


namespace sisters_gift_l2577_257764

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def additional_money_needed : ℕ := 3214

def job_earnings : ℕ := hourly_wage * hours_worked
def cookie_earnings : ℕ := cookie_price * cookies_sold
def total_earnings : ℕ := job_earnings + cookie_earnings - lottery_ticket_cost + lottery_winnings

theorem sisters_gift (sisters_gift : ℕ) : sisters_gift = 1000 :=
by
  sorry

end sisters_gift_l2577_257764


namespace root_in_interval_l2577_257706

def f (x : ℝ) := 3*x + x - 3

theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by sorry

end root_in_interval_l2577_257706


namespace intersection_condition_subset_complement_condition_l2577_257748

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 2 ≤ x ∧ x ≤ m + 2}

theorem intersection_condition (m : ℝ) :
  A ∩ B m = {x : ℝ | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

theorem subset_complement_condition (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by sorry

end intersection_condition_subset_complement_condition_l2577_257748


namespace sequence_properties_l2577_257719

def sequence_a (n : ℕ) : ℝ := 2^n - 1

def S (n : ℕ) : ℝ := 2 * sequence_a n - n

theorem sequence_properties :
  (∀ n : ℕ, S n = 2 * sequence_a n - n) →
  (∀ n : ℕ, sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, sequence_a n = 2^n - 1) ∧
  (∀ k : ℕ, 2 * sequence_a (k + 1) ≠ sequence_a k + sequence_a (k + 2)) :=
by sorry

end sequence_properties_l2577_257719


namespace oil_depth_theorem_l2577_257758

/-- Represents a horizontal cylindrical tank with oil -/
structure OilTank where
  length : ℝ
  diameter : ℝ
  oil_surface_area : ℝ

/-- Calculates the possible depths of oil in the tank -/
def oil_depths (tank : OilTank) : Set ℝ :=
  { h | h = 3 - Real.sqrt 5 ∨ h = 3 + Real.sqrt 5 }

theorem oil_depth_theorem (tank : OilTank) 
  (h_length : tank.length = 10)
  (h_diameter : tank.diameter = 6)
  (h_area : tank.oil_surface_area = 40) :
  ∀ h ∈ oil_depths tank, 
    ∃ c : ℝ, 
      c = tank.oil_surface_area / tank.length ∧ 
      c ^ 2 = 2 * (tank.diameter / 2) * h - h ^ 2 :=
by sorry

end oil_depth_theorem_l2577_257758


namespace largest_power_of_three_dividing_A_l2577_257795

/-- Given that A is the largest product of natural numbers whose sum is 2011,
    this theorem states that the largest power of three that divides A is 3^669. -/
theorem largest_power_of_three_dividing_A : ∃ A : ℕ,
  (∀ (factors : List ℕ), (factors.sum = 2011 ∧ factors.prod ≤ A) → 
    ∃ (k : ℕ), A = 3^669 * k ∧ ¬(∃ m : ℕ, A = 3^(669 + 1) * m)) := by
  sorry

end largest_power_of_three_dividing_A_l2577_257795


namespace binary_calculation_l2577_257797

theorem binary_calculation : 
  (0b101010 + 0b11010) * 0b1110 = 0b11000000000 := by sorry

end binary_calculation_l2577_257797


namespace shoes_sold_l2577_257734

theorem shoes_sold (large medium small left : ℕ) 
  (h1 : large = 22)
  (h2 : medium = 50)
  (h3 : small = 24)
  (h4 : left = 13) : 
  large + medium + small - left = 83 := by
  sorry

end shoes_sold_l2577_257734


namespace talia_total_distance_l2577_257731

/-- Represents the total distance Talia drives in a day -/
def total_distance (home_to_park park_to_grocery grocery_to_friend friend_to_home : ℕ) : ℕ :=
  home_to_park + park_to_grocery + grocery_to_friend + friend_to_home

/-- Theorem stating that Talia drives 18 miles in total -/
theorem talia_total_distance :
  ∃ (home_to_park park_to_grocery grocery_to_friend friend_to_home : ℕ),
    home_to_park = 5 ∧
    park_to_grocery = 3 ∧
    grocery_to_friend = 6 ∧
    friend_to_home = 4 ∧
    total_distance home_to_park park_to_grocery grocery_to_friend friend_to_home = 18 :=
by
  sorry

end talia_total_distance_l2577_257731


namespace coefficient_x4_is_160_l2577_257786

/-- The coefficient of x^4 in the expansion of (1+x) * (1+2x)^5 -/
def coefficient_x4 : ℕ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^4 in the expansion of (1+x) * (1+2x)^5 is 160 -/
theorem coefficient_x4_is_160 : coefficient_x4 = 160 := by
  sorry

end coefficient_x4_is_160_l2577_257786


namespace twenty_four_game_solvable_l2577_257759

/-- Represents the basic arithmetic operations -/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Represents an expression in the 24 Game -/
inductive Expr
  | Num (n : ℕ)
  | Op (op : Operation) (e1 e2 : Expr)

/-- Evaluates an expression -/
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
  | Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2
  | Expr.Op Operation.Divide e1 e2 => eval e1 / eval e2

/-- Checks if an expression uses all given numbers exactly once -/
def usesAllNumbers (e : Expr) (numbers : List ℕ) : Prop := sorry

/-- The 24 Game theorem -/
theorem twenty_four_game_solvable (numbers : List ℕ := [2, 5, 11, 12]) :
  ∃ e : Expr, usesAllNumbers e numbers ∧ eval e = 24 := by sorry

end twenty_four_game_solvable_l2577_257759


namespace triangle_selection_probability_l2577_257776

theorem triangle_selection_probability (total_triangles shaded_triangles : ℕ) 
  (h1 : total_triangles = 6)
  (h2 : shaded_triangles = 3)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
  sorry

end triangle_selection_probability_l2577_257776


namespace i_pow_45_plus_345_l2577_257775

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- Define the properties of i
axiom i_pow_one : i^1 = i
axiom i_pow_two : i^2 = -1
axiom i_pow_three : i^3 = -i
axiom i_pow_four : i^4 = 1

-- Define the cyclic nature of i
axiom i_cyclic (n : ℕ) : i^(n + 4) = i^n

-- Theorem to prove
theorem i_pow_45_plus_345 : i^45 + i^345 = 2*i := by
  sorry

end i_pow_45_plus_345_l2577_257775


namespace fraction_is_positive_integer_iff_p_18_l2577_257722

theorem fraction_is_positive_integer_iff_p_18 (p : ℕ+) :
  (∃ (n : ℕ+), (5 * p + 40 : ℚ) / (3 * p - 7 : ℚ) = n) ↔ p = 18 := by
  sorry

end fraction_is_positive_integer_iff_p_18_l2577_257722


namespace tangent_line_y_intercept_l2577_257703

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  yIntercept : ℝ

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

/-- The main theorem -/
theorem tangent_line_y_intercept :
  ∀ (l : TangentLine),
    l.circle1 = { center := (3, 0), radius := 3 } →
    l.circle2 = { center := (7, 0), radius := 2 } →
    (∃ (p1 p2 : ℝ × ℝ), isFirstQuadrant p1 ∧ isFirstQuadrant p2) →
    l.yIntercept = 24 * Real.sqrt 55 / 55 := by
  sorry


end tangent_line_y_intercept_l2577_257703


namespace high_quality_seed_probability_l2577_257716

/-- Represents the composition of seeds in a batch -/
structure SeedBatch where
  second_grade : ℝ
  third_grade : ℝ
  fourth_grade : ℝ

/-- Represents the probabilities of producing high-quality products for each seed grade -/
structure QualityProbabilities where
  first_grade : ℝ
  second_grade : ℝ
  third_grade : ℝ
  fourth_grade : ℝ

/-- Calculates the probability of selecting a high-quality seed from a given batch -/
def high_quality_probability (batch : SeedBatch) (probs : QualityProbabilities) : ℝ :=
  let first_grade_proportion := 1 - (batch.second_grade + batch.third_grade + batch.fourth_grade)
  first_grade_proportion * probs.first_grade +
  batch.second_grade * probs.second_grade +
  batch.third_grade * probs.third_grade +
  batch.fourth_grade * probs.fourth_grade

/-- Theorem stating the probability of selecting a high-quality seed from the given batch -/
theorem high_quality_seed_probability :
  let batch := SeedBatch.mk 0.02 0.015 0.01
  let probs := QualityProbabilities.mk 0.5 0.15 0.1 0.05
  high_quality_probability batch probs = 0.4825 := by
  sorry


end high_quality_seed_probability_l2577_257716


namespace expression_evaluation_l2577_257792

theorem expression_evaluation (c d : ℝ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 := by
  sorry

end expression_evaluation_l2577_257792


namespace max_values_l2577_257756

theorem max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b^2 = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y^2 = 1 ∧ b * Real.sqrt a ≤ x * Real.sqrt y) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y^2 = 1 → b * Real.sqrt a ≤ x * Real.sqrt y) ∧
  b * Real.sqrt a ≤ 1/2 ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y^2 = 1 ∧ Real.sqrt x + y ≤ Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y^2 = 1 → Real.sqrt x + y ≤ Real.sqrt 2) ∧
  Real.sqrt a + b ≤ Real.sqrt 2 :=
by sorry


end max_values_l2577_257756


namespace perfect_square_difference_l2577_257705

theorem perfect_square_difference : ∃ (x a b : ℤ), 
  (x + 100 = a^2) ∧ 
  (x + 164 = b^2) ∧ 
  (x = 125 ∨ x = -64 ∨ x = -100) :=
by sorry

end perfect_square_difference_l2577_257705


namespace quadratic_minimum_l2577_257747

/-- 
Given a quadratic function y = 3x^2 + px + q, 
prove that the value of q that makes the minimum value of y equal to 1 is 1 + p^2/18
-/
theorem quadratic_minimum (p : ℝ) : 
  ∃ (q : ℝ), (∀ (x : ℝ), 3 * x^2 + p * x + q ≥ 1) ∧ 
  (∃ (x : ℝ), 3 * x^2 + p * x + q = 1) → 
  q = 1 + p^2 / 18 :=
by sorry

end quadratic_minimum_l2577_257747


namespace power_mod_twenty_l2577_257711

theorem power_mod_twenty : 17^2037 % 20 = 17 := by
  sorry

end power_mod_twenty_l2577_257711


namespace sum_of_binary_digits_160_l2577_257715

/-- The sum of the digits in the binary representation of 160 is 2. -/
theorem sum_of_binary_digits_160 : 
  (Nat.digits 2 160).sum = 2 := by sorry

end sum_of_binary_digits_160_l2577_257715


namespace ball_probability_comparison_l2577_257750

theorem ball_probability_comparison :
  let total_balls : ℕ := 3
  let red_balls : ℕ := 2
  let white_balls : ℕ := 1
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  p_red > p_white :=
by
  sorry

end ball_probability_comparison_l2577_257750


namespace larger_number_in_ratio_l2577_257742

theorem larger_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 40 → x / y = 3 → x = 30 :=
by sorry

end larger_number_in_ratio_l2577_257742


namespace factorization_sum_l2577_257771

theorem factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 17*x + 72 = (x + a) * (x + b)) →
  (∀ x, x^2 + 9*x - 90 = (x + b) * (x - c)) →
  a + b + c = 27 := by
sorry

end factorization_sum_l2577_257771


namespace parabola_focus_l2577_257730

/-- A parabola is defined by the equation y = x^2 -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The focus of a parabola is a point with specific properties -/
def IsFocus (f : ℝ × ℝ) (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (a : ℝ), p = {point : ℝ × ℝ | point.2 = point.1^2} ∧ f = (0, 1/(4*a))

/-- The theorem states that the focus of the parabola y = x^2 is at (0, 1/4) -/
theorem parabola_focus : IsFocus (0, 1/4) Parabola := by
  sorry

end parabola_focus_l2577_257730


namespace perfect_square_expression_l2577_257749

theorem perfect_square_expression : ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.4792 + 0.02 * 0.02) = y^2 := by
  sorry

end perfect_square_expression_l2577_257749


namespace least_multiple_of_15_greater_than_500_l2577_257746

theorem least_multiple_of_15_greater_than_500 : 
  (∃ (n : ℕ), 15 * n = 510 ∧ 
   510 > 500 ∧ 
   ∀ (m : ℕ), 15 * m > 500 → 15 * m ≥ 510) := by
sorry

end least_multiple_of_15_greater_than_500_l2577_257746


namespace hyperbola_eccentricity_l2577_257765

/-- Represents a hyperbola with its asymptotic equation coefficient -/
structure Hyperbola where
  k : ℝ
  asymptote_eq : ∀ (x y : ℝ), y = k * x ∨ y = -k * x

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (h_asymptote : h.k = 1/2) :
  (eccentricity h = Real.sqrt 5 / 2) ∨ (eccentricity h = Real.sqrt 5) :=
sorry

end hyperbola_eccentricity_l2577_257765


namespace lcm_of_16_27_35_l2577_257708

theorem lcm_of_16_27_35 : Nat.lcm (Nat.lcm 16 27) 35 = 15120 := by
  sorry

end lcm_of_16_27_35_l2577_257708


namespace king_requirement_requirement_met_for_6_requirement_not_met_for_1986_l2577_257738

/-- Represents a network of cities and roads -/
structure CityNetwork where
  n : ℕ                -- number of cities
  roads : ℕ             -- number of roads
  connected : Prop      -- any city can be reached from any other city
  distances : Finset ℕ  -- set of shortest distances between pairs of cities

/-- The condition for a valid city network -/
def validNetwork (net : CityNetwork) : Prop :=
  net.roads = net.n - 1 ∧
  net.connected ∧
  net.distances = Finset.range (net.n * (net.n - 1) / 2 + 1) \ {0}

/-- The condition for the network to meet the king's requirement -/
def meetsRequirement (n : ℕ) : Prop :=
  ∃ (net : CityNetwork), net.n = n ∧ validNetwork net

/-- The main theorem -/
theorem king_requirement (n : ℕ) :
  meetsRequirement n ↔ (∃ k : ℕ, n = k^2) ∨ (∃ k : ℕ, n = k^2 + 2) :=
sorry

/-- The requirement can be met for n = 6 -/
theorem requirement_met_for_6 : meetsRequirement 6 :=
sorry

/-- The requirement cannot be met for n = 1986 -/
theorem requirement_not_met_for_1986 : ¬meetsRequirement 1986 :=
sorry

end king_requirement_requirement_met_for_6_requirement_not_met_for_1986_l2577_257738


namespace line_direction_vector_b_l2577_257712

def point_1 : ℝ × ℝ := (-3, 1)
def point_2 : ℝ × ℝ := (1, 5)

def direction_vector (b : ℝ) : ℝ × ℝ := (3, b)

theorem line_direction_vector_b (b : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ direction_vector b = k • (point_2 - point_1)) → b = 3 :=
by sorry

end line_direction_vector_b_l2577_257712


namespace frog_escapes_in_18_days_l2577_257736

/-- Represents the depth of the well in meters -/
def well_depth : ℕ := 20

/-- Represents the distance the frog climbs up each day in meters -/
def climb_distance : ℕ := 3

/-- Represents the distance the frog slips down each day in meters -/
def slip_distance : ℕ := 2

/-- Represents the net distance the frog climbs each day in meters -/
def net_daily_progress : ℕ := climb_distance - slip_distance

/-- Theorem stating that the frog can climb out of the well in 18 days -/
theorem frog_escapes_in_18_days :
  ∃ (n : ℕ), n = 18 ∧ n * net_daily_progress + climb_distance ≥ well_depth :=
sorry

end frog_escapes_in_18_days_l2577_257736


namespace problem_solution_l2577_257718

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a * b = 2) 
  (h2 : a / (a + b^2) + b / (b + a^2) = 7/8) : 
  a^6 + b^6 = 128 := by
sorry

end problem_solution_l2577_257718


namespace quadratic_roots_bounds_l2577_257798

theorem quadratic_roots_bounds (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m < 0) 
  (hroots : x₁^2 - x₁ - 6 = m ∧ x₂^2 - x₂ - 6 = m) 
  (horder : x₁ < x₂) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 := by
  sorry

end quadratic_roots_bounds_l2577_257798


namespace zero_sequence_arithmetic_not_geometric_l2577_257796

def zero_sequence : ℕ → ℝ := λ _ => 0

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem zero_sequence_arithmetic_not_geometric :
  is_arithmetic_sequence zero_sequence ∧ ¬is_geometric_sequence zero_sequence := by
  sorry

end zero_sequence_arithmetic_not_geometric_l2577_257796


namespace negative_a_fourth_div_negative_a_l2577_257757

theorem negative_a_fourth_div_negative_a (a : ℝ) : (-a)^4 / (-a) = -a^3 := by
  sorry

end negative_a_fourth_div_negative_a_l2577_257757


namespace bob_salary_calculation_l2577_257774

def initial_salary : ℝ := 3000
def raise_percentage : ℝ := 0.15
def cut_percentage : ℝ := 0.10
def bonus : ℝ := 500

def final_salary : ℝ := 
  initial_salary * (1 + raise_percentage) * (1 - cut_percentage) + bonus

theorem bob_salary_calculation : final_salary = 3605 := by
  sorry

end bob_salary_calculation_l2577_257774


namespace number_of_divisors_of_30_l2577_257779

theorem number_of_divisors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end number_of_divisors_of_30_l2577_257779


namespace intersection_of_A_and_B_l2577_257733

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 3 * p.2 = 7}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = -1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1/2, 3/2)} := by sorry

end intersection_of_A_and_B_l2577_257733


namespace andrews_appetizers_l2577_257701

/-- The number of hotdogs on sticks Andrew brought -/
def hotdogs : ℕ := 30

/-- The number of bite-sized cheese pops Andrew brought -/
def cheese_pops : ℕ := 20

/-- The number of chicken nuggets Andrew brought -/
def chicken_nuggets : ℕ := 40

/-- The total number of appetizer portions Andrew brought -/
def total_appetizers : ℕ := hotdogs + cheese_pops + chicken_nuggets

theorem andrews_appetizers :
  total_appetizers = 90 :=
by sorry

end andrews_appetizers_l2577_257701


namespace addSecondsCorrect_l2577_257707

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define a function to add seconds to a time
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

-- Define the initial time
def initialTime : Time :=
  { hours := 7, minutes := 45, seconds := 0 }

-- Define the number of seconds to add
def secondsToAdd : Nat := 9999

-- Theorem to prove
theorem addSecondsCorrect : 
  addSeconds initialTime secondsToAdd = { hours := 10, minutes := 31, seconds := 39 } :=
sorry

end addSecondsCorrect_l2577_257707


namespace bookcase_weight_excess_l2577_257794

/-- Proves that the total weight of books and knick-knacks exceeds the bookcase weight limit by 33 pounds -/
theorem bookcase_weight_excess :
  let bookcase_limit : ℕ := 80
  let hardcover_count : ℕ := 70
  let hardcover_weight : ℚ := 1/2
  let textbook_count : ℕ := 30
  let textbook_weight : ℕ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℕ := 6
  let total_weight := (hardcover_count : ℚ) * hardcover_weight + 
                      (textbook_count * textbook_weight : ℚ) + 
                      (knickknack_count * knickknack_weight : ℚ)
  total_weight - bookcase_limit = 33
  := by sorry

end bookcase_weight_excess_l2577_257794


namespace notecard_problem_l2577_257732

theorem notecard_problem (N E : ℕ) : 
  N - E = 80 →  -- Bill used all envelopes and had 80 notecards left
  3 * E = N →   -- John used all notecards, each letter used 3 notecards
  N = 120       -- The number of notecards in each set is 120
:= by sorry

end notecard_problem_l2577_257732


namespace parabola_properties_l2577_257787

/-- A parabola that intersects the x-axis at (-3,0) and (1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  h_root1 : a * (-3)^2 + b * (-3) + c = 0
  h_root2 : a * 1^2 + b * 1 + c = 0

/-- Properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.b^2 - 4*p.a*p.c > 0) ∧ (3*p.b + 2*p.c = 0) := by
  sorry

end parabola_properties_l2577_257787


namespace arithmetic_calculations_l2577_257793

theorem arithmetic_calculations :
  (25 - (3 - (-30 - 2)) = -10) ∧
  ((-80) * (1/2 + 2/5 - 1) = 8) ∧
  (81 / (-3)^3 + (-1/5) * (-10) = -1) := by
  sorry

end arithmetic_calculations_l2577_257793


namespace class_average_age_l2577_257763

theorem class_average_age (original_students : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_students = 12 →
  new_students = 12 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ),
    (original_students * original_average + new_students * new_average_age) / (original_students + new_students) = original_average - average_decrease ∧
    original_average = 40 :=
by sorry

end class_average_age_l2577_257763


namespace complex_fraction_equals_negative_i_l2577_257743

theorem complex_fraction_equals_negative_i :
  ∀ (i : ℂ), i * i = -1 →
  (1 - i) / (1 + i) = -i := by sorry

end complex_fraction_equals_negative_i_l2577_257743


namespace motorcyclist_problem_l2577_257768

/-- The time taken by the first motorcyclist to travel the distance AB -/
def time_first : ℝ := 80

/-- The time taken by the second motorcyclist to travel the distance AB -/
def time_second : ℝ := 60

/-- The time taken by the third motorcyclist to travel the distance AB -/
def time_third : ℝ := 3240

/-- The head start of the first motorcyclist -/
def head_start : ℝ := 5

/-- The time difference between the third and second motorcyclist overtaking the first -/
def overtake_diff : ℝ := 10

/-- The distance between points A and B -/
def distance : ℝ := 1  -- We can set this to any positive real number

theorem motorcyclist_problem :
  ∃ (speed_first speed_second speed_third : ℝ),
    speed_first > 0 ∧ speed_second > 0 ∧ speed_third > 0 ∧
    speed_first ≠ speed_second ∧ speed_first ≠ speed_third ∧ speed_second ≠ speed_third ∧
    speed_first = distance / time_first ∧
    speed_second = distance / time_second ∧
    speed_third = distance / time_third ∧
    (time_third - head_start) * speed_third = time_first * speed_first ∧
    (time_second - head_start) * speed_second = (time_first + overtake_diff) * speed_first :=
by sorry

end motorcyclist_problem_l2577_257768


namespace stratified_sample_male_count_l2577_257783

/-- Represents a company with employees -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  male_employees : ℕ
  h_total : total_employees = female_employees + male_employees

/-- Represents a stratified sample from a company -/
structure StratifiedSample where
  company : Company
  sample_size : ℕ
  female_sample : ℕ
  male_sample : ℕ
  h_sample : sample_size = female_sample + male_sample
  h_proportion : female_sample * company.total_employees = company.female_employees * sample_size

theorem stratified_sample_male_count (c : Company) (s : StratifiedSample) 
    (h_company : c.total_employees = 300 ∧ c.female_employees = 160)
    (h_sample : s.company = c ∧ s.sample_size = 15) :
    s.male_sample = 7 := by
  sorry

#check stratified_sample_male_count

end stratified_sample_male_count_l2577_257783


namespace champion_is_team_d_l2577_257762

-- Define the teams
inductive Team : Type
| A | B | C | D

-- Define the rankings
structure Ranking :=
(first : Team)
(second : Team)
(third : Team)
(fourth : Team)

-- Define the predictions
structure Prediction :=
(first : Option Team)
(second : Option Team)
(third : Option Team)
(fourth : Option Team)

-- Define the function to check if a prediction is half correct
def isHalfCorrect (pred : Prediction) (actual : Ranking) : Prop :=
  (pred.first = some actual.first ∨ pred.second = some actual.second ∨ 
   pred.third = some actual.third ∨ pred.fourth = some actual.fourth) ∧
  (pred.first ≠ some actual.first ∨ pred.second ≠ some actual.second ∨
   pred.third ≠ some actual.third ∨ pred.fourth ≠ some actual.fourth)

-- Theorem statement
theorem champion_is_team_d :
  ∀ (actual : Ranking),
    let wang_pred : Prediction := ⟨some Team.D, some Team.B, none, none⟩
    let li_pred : Prediction := ⟨none, some Team.A, none, some Team.C⟩
    let zhang_pred : Prediction := ⟨none, some Team.D, some Team.C, none⟩
    isHalfCorrect wang_pred actual ∧
    isHalfCorrect li_pred actual ∧
    isHalfCorrect zhang_pred actual →
    actual.first = Team.D :=
by sorry


end champion_is_team_d_l2577_257762


namespace divisors_of_factorial_eight_l2577_257784

theorem divisors_of_factorial_eight (n : ℕ) : n = 8 → (Nat.divisors (Nat.factorial n)).card = 96 := by
  sorry

end divisors_of_factorial_eight_l2577_257784
