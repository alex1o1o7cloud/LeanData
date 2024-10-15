import Mathlib

namespace NUMINAMATH_GPT_selection_of_projects_l645_64589

-- Mathematical definitions
def numberOfWaysToSelect2ProjectsFrom4KeyAnd6General (key: Finset ℕ) (general: Finset ℕ) : ℕ :=
  (key.card.choose 2) * (general.card.choose 2)

def numberOfWaysToSelectAtLeastOneProjectAorB (key: Finset ℕ) (general: Finset ℕ) (A B: ℕ) : ℕ :=
  let total_ways := (key.card.choose 2) * (general.card.choose 2)
  let ways_without_A := ((key.erase A).card.choose 2) * (general.card.choose 2)
  let ways_without_B := (key.card.choose 2) * ((general.erase B).card.choose 2)
  let ways_without_A_and_B := ((key.erase A).card.choose 2) * ((general.erase B).card.choose 2)
  total_ways - ways_without_A_and_B

-- Theorem we need to prove
theorem selection_of_projects (key general: Finset ℕ) (A B: ℕ) (hA: A ∈ key) (hB: B ∈ general) (h_key_card: key.card = 4) (h_general_card: general.card = 6) :
  numberOfWaysToSelectAtLeastOneProjectAorB key general A B = 60 := 
sorry

end NUMINAMATH_GPT_selection_of_projects_l645_64589


namespace NUMINAMATH_GPT_find_coordinates_of_C_l645_64551

structure Point where
  x : Int
  y : Int

def isSymmetricalAboutXAxis (A B : Point) : Prop :=
  A.x = B.x ∧ A.y = -B.y

def isSymmetricalAboutOrigin (B C : Point) : Prop :=
  C.x = -B.x ∧ C.y = -B.y

theorem find_coordinates_of_C :
  ∃ C : Point, let A := Point.mk 2 (-3)
               let B := Point.mk 2 3
               isSymmetricalAboutXAxis A B →
               isSymmetricalAboutOrigin B C →
               C = Point.mk (-2) (-3) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_C_l645_64551


namespace NUMINAMATH_GPT_inequality_must_hold_l645_64538

theorem inequality_must_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_must_hold_l645_64538


namespace NUMINAMATH_GPT_total_texts_received_l645_64503

structure TextMessageScenario :=
  (textsBeforeNoon : Nat)
  (textsAtNoon : Nat)
  (textsAfterNoonDoubling : (Nat → Nat) → Nat)
  (textsAfter6pm : (Nat → Nat) → Nat)

def textsBeforeNoon := 21
def textsAtNoon := 2

-- Calculation for texts received from noon to 6 pm
def noonTo6pmTexts (textsAtNoon : Nat) : Nat :=
  let rec doubling (n : Nat) : Nat := match n with
    | 0 => textsAtNoon
    | n + 1 => 2 * (doubling n)
  (doubling 0) + (doubling 1) + (doubling 2) + (doubling 3) + (doubling 4) + (doubling 5)

def textsAfterNoonDoubling : (Nat → Nat) → Nat := λ doubling => noonTo6pmTexts 2

-- Calculation for texts received from 6 pm to midnight
def after6pmTexts (textsAt6pm : Nat) : Nat :=
  let rec decrease (n : Nat) : Nat := match n with
    | 0 => textsAt6pm
    | n + 1 => (decrease n) - 5
  (decrease 0) + (decrease 1) + (decrease 2) + (decrease 3) + (decrease 4) + (decrease 5) + (decrease 6)

def textsAfter6pm : (Nat → Nat) → Nat := λ decrease => after6pmTexts 64

theorem total_texts_received : textsBeforeNoon + (textsAfterNoonDoubling (λ x => x)) + (textsAfter6pm (λ x => x)) = 490 := by
  sorry
 
end NUMINAMATH_GPT_total_texts_received_l645_64503


namespace NUMINAMATH_GPT_inequality_correct_transformation_l645_64539

-- Definitions of the conditions
variables (a b : ℝ)

-- The equivalent proof problem
theorem inequality_correct_transformation (h : a > b) : -a < -b :=
by sorry

end NUMINAMATH_GPT_inequality_correct_transformation_l645_64539


namespace NUMINAMATH_GPT_career_preference_angles_l645_64572

theorem career_preference_angles (m f : ℕ) (total_degrees : ℕ) (one_fourth_males one_half_females : ℚ) (male_ratio female_ratio : ℚ) :
  total_degrees = 360 → male_ratio = 2/3 → female_ratio = 3/3 →
  m = 2 * f / 3 → one_fourth_males = 1/4 * m → one_half_females = 1/2 * f →
  (one_fourth_males + one_half_females) / (m + f) * total_degrees = 144 :=
by
  sorry

end NUMINAMATH_GPT_career_preference_angles_l645_64572


namespace NUMINAMATH_GPT_find_number_l645_64526

theorem find_number (x : ℝ) (h : (2 / 5) * x = 10) : x = 25 :=
sorry

end NUMINAMATH_GPT_find_number_l645_64526


namespace NUMINAMATH_GPT_P_of_7_l645_64530

noncomputable def P (x : ℝ) : ℝ := 12 * (x - 1) * (x - 2) * (x - 3) * (x - 4)^2 * (x - 5)^2 * (x - 6)

theorem P_of_7 : P 7 = 51840 :=
by
  sorry

end NUMINAMATH_GPT_P_of_7_l645_64530


namespace NUMINAMATH_GPT_trapezoid_median_l645_64544

theorem trapezoid_median
  (h : ℝ)
  (area_triangle : ℝ)
  (area_trapezoid : ℝ)
  (bt : ℝ)
  (bt_sum : ℝ)
  (ht_positive : h ≠ 0)
  (triangle_area : area_triangle = (1/2) * bt * h)
  (trapezoid_area : area_trapezoid = area_triangle)
  (trapezoid_bt_sum : bt_sum = 40)
  (triangle_bt : bt = 24)
  : (bt_sum / 2) = 20 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_median_l645_64544


namespace NUMINAMATH_GPT_factorize_expression_l645_64531

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l645_64531


namespace NUMINAMATH_GPT_quadrilateral_area_is_114_5_l645_64502

noncomputable def area_of_quadrilateral_114_5 
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) : ℝ :=
  114.5

theorem quadrilateral_area_is_114_5
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) :
  area_of_quadrilateral_114_5 AB BC CD AD angle_ABC h1 h2 h3 h4 h5 = 114.5 :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_is_114_5_l645_64502


namespace NUMINAMATH_GPT_value_of_b_l645_64565

theorem value_of_b (a c : ℝ) (b : ℝ) (h1 : a = 105) (h2 : c = 70) (h3 : a^4 = 21 * 25 * 15 * b * c^3) : b = 0.045 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l645_64565


namespace NUMINAMATH_GPT_hyperbola_foci_problem_l645_64512

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

noncomputable def foci_1 : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def foci_2 : ℝ × ℝ := (Real.sqrt 5, 0)

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

noncomputable def vector (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v1.1 + v2.2 * v2.2

noncomputable def orthogonal (P : ℝ × ℝ) : Prop :=
  dot_product (vector P foci_1) (vector P foci_2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def required_value (P : ℝ × ℝ) : ℝ :=
  distance P foci_1 * distance P foci_2

theorem hyperbola_foci_problem (P : ℝ × ℝ) : 
  point_on_hyperbola P → orthogonal P → required_value P = 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_foci_problem_l645_64512


namespace NUMINAMATH_GPT_solution_set_l645_64587

noncomputable def f : ℝ → ℝ := sorry

axiom deriv_f_pos (x : ℝ) : deriv f x > 1 - f x
axiom f_at_zero : f 0 = 3

theorem solution_set (x : ℝ) : e^x * f x > e^x + 2 ↔ x > 0 :=
by sorry

end NUMINAMATH_GPT_solution_set_l645_64587


namespace NUMINAMATH_GPT_largest_non_prime_sum_l645_64504

theorem largest_non_prime_sum (a b n : ℕ) (h1 : a ≥ 1) (h2 : b < 47) (h3 : n = 47 * a + b) (h4 : ∀ b, b < 47 → ¬Nat.Prime b → b = 43) : 
  n = 90 :=
by
  sorry

end NUMINAMATH_GPT_largest_non_prime_sum_l645_64504


namespace NUMINAMATH_GPT_sandra_beignets_l645_64573

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end NUMINAMATH_GPT_sandra_beignets_l645_64573


namespace NUMINAMATH_GPT_equation_of_line_AB_l645_64595

noncomputable def circle_center : ℝ × ℝ := (1, 0)  -- center of the circle (x-1)^2 + y^2 = 1
noncomputable def circle_radius : ℝ := 1          -- radius of the circle (x-1)^2 + y^2 = 1
noncomputable def point_P : ℝ × ℝ := (3, 1)       -- point P(3,1)

theorem equation_of_line_AB :
  ∃ (AB : ℝ → ℝ → Prop),
    (∀ x y, AB x y ↔ (2 * x + y - 3 = 0)) := sorry

end NUMINAMATH_GPT_equation_of_line_AB_l645_64595


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_necessary_but_not_sufficient_l645_64582

def M (x : ℝ) : Prop := (x + 3) * (x - 5) > 0
def P (x : ℝ) (a : ℝ) : Prop := x^2 + (a - 8)*x - 8*a ≤ 0
def I : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x, M x ∧ P x a ↔ x ∈ I) → a = 0 :=
sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, (M x ∧ P x a → x ∈ I) ∧ (∀ x, x ∈ I → M x ∧ P x a)) → a ≤ 3 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_necessary_but_not_sufficient_l645_64582


namespace NUMINAMATH_GPT_sale_in_third_month_l645_64528

def sales_in_months (m1 m2 m3 m4 m5 m6 : Int) : Prop :=
  m1 = 5124 ∧
  m2 = 5366 ∧
  m4 = 6124 ∧
  m6 = 4579 ∧
  (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 5400

theorem sale_in_third_month (m5 : Int) :
  (∃ m3 : Int, sales_in_months 5124 5366 m3 6124 m5 4579 → m3 = 11207) :=
sorry

end NUMINAMATH_GPT_sale_in_third_month_l645_64528


namespace NUMINAMATH_GPT_roots_are_positive_integers_implies_r_values_l645_64584

theorem roots_are_positive_integers_implies_r_values (r x : ℕ) (h : (r * x^2 - (2 * r + 7) * x + (r + 7) = 0) ∧ (x > 0)) :
  r = 7 ∨ r = 0 ∨ r = 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_are_positive_integers_implies_r_values_l645_64584


namespace NUMINAMATH_GPT_find_algebraic_expression_l645_64593

-- Definitions as per the conditions
variable (a b : ℝ)

-- Given condition
def given_condition (σ : ℝ) : Prop := σ * (2 * a * b) = 4 * a^2 * b

-- The statement to prove
theorem find_algebraic_expression (σ : ℝ) (h : given_condition a b σ) : σ = 2 * a := 
sorry

end NUMINAMATH_GPT_find_algebraic_expression_l645_64593


namespace NUMINAMATH_GPT_total_money_spent_l645_64588

def candy_bar_cost : ℕ := 14
def cookie_box_cost : ℕ := 39
def total_spent : ℕ := candy_bar_cost + cookie_box_cost

theorem total_money_spent : total_spent = 53 := by
  sorry

end NUMINAMATH_GPT_total_money_spent_l645_64588


namespace NUMINAMATH_GPT_problem_solution_l645_64560

theorem problem_solution (s t : ℕ) (hpos_s : 0 < s) (hpos_t : 0 < t) (h_eq : s * (s - t) = 29) : s + t = 57 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l645_64560


namespace NUMINAMATH_GPT_polynomial_not_factorable_l645_64523

theorem polynomial_not_factorable (b c d : Int) (h₁ : (b * d + c * d) % 2 = 1) : 
  ¬ ∃ p q r : Int, (x + p) * (x^2 + q * x + r) = x^3 + b * x^2 + c * x + d :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_not_factorable_l645_64523


namespace NUMINAMATH_GPT_xiaopangs_score_is_16_l645_64594

-- Define the father's score
def fathers_score : ℕ := 48

-- Define Xiaopang's score in terms of father's score
def xiaopangs_score (fathers_score : ℕ) : ℕ := fathers_score / 2 - 8

-- The theorem to prove that Xiaopang's score is 16
theorem xiaopangs_score_is_16 : xiaopangs_score fathers_score = 16 := 
by
  sorry

end NUMINAMATH_GPT_xiaopangs_score_is_16_l645_64594


namespace NUMINAMATH_GPT_convert_cylindrical_to_rectangular_l645_64563

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 7 (Real.pi / 4) 8 = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2, 8) :=
by
  sorry

end NUMINAMATH_GPT_convert_cylindrical_to_rectangular_l645_64563


namespace NUMINAMATH_GPT_geometric_series_first_term_l645_64536

theorem geometric_series_first_term (a : ℝ) (r : ℝ) (s : ℝ) 
  (h1 : r = -1/3) (h2 : s = 12) (h3 : s = a / (1 - r)) : a = 16 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l645_64536


namespace NUMINAMATH_GPT_elena_bouquet_petals_l645_64532

def num_petals (count : ℕ) (petals_per_flower : ℕ) : ℕ :=
  count * petals_per_flower

theorem elena_bouquet_petals :
  let num_lilies := 4
  let lilies_petal_count := num_petals num_lilies 6
  
  let num_tulips := 2
  let tulips_petal_count := num_petals num_tulips 3

  let num_roses := 2
  let roses_petal_count := num_petals num_roses 5
  
  let num_daisies := 1
  let daisies_petal_count := num_petals num_daisies 12
  
  lilies_petal_count + tulips_petal_count + roses_petal_count + daisies_petal_count = 52 := by
  sorry

end NUMINAMATH_GPT_elena_bouquet_petals_l645_64532


namespace NUMINAMATH_GPT_popsicle_total_l645_64549

def popsicle_count (g c b : Nat) : Nat :=
  g + c + b

theorem popsicle_total : 
  let g := 2
  let c := 13
  let b := 2
  popsicle_count g c b = 17 := by
  sorry

end NUMINAMATH_GPT_popsicle_total_l645_64549


namespace NUMINAMATH_GPT_circle_reflection_l645_64506

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end NUMINAMATH_GPT_circle_reflection_l645_64506


namespace NUMINAMATH_GPT_product_of_symmetric_complex_numbers_l645_64567

def z1 : ℂ := 1 + 2 * Complex.I

def z2 : ℂ := -1 + 2 * Complex.I

theorem product_of_symmetric_complex_numbers :
  z1 * z2 = -5 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_symmetric_complex_numbers_l645_64567


namespace NUMINAMATH_GPT_find_unknown_number_l645_64581

def unknown_number (x : ℝ) : Prop :=
  (0.5^3) - (0.1^3 / 0.5^2) + x + (0.1^2) = 0.4

theorem find_unknown_number : ∃ (x : ℝ), unknown_number x ∧ x = 0.269 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l645_64581


namespace NUMINAMATH_GPT_sum_of_ages_l645_64558

theorem sum_of_ages (a b c d e : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 9) 
  (h2 : 1 ≤ b ∧ b ≤ 9) 
  (h3 : 1 ≤ c ∧ c ≤ 9) 
  (h4 : 1 ≤ d ∧ d ≤ 9) 
  (h5 : 1 ≤ e ∧ e ≤ 9) 
  (h6 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h7 : a * b = 28 ∨ a * c = 28 ∨ a * d = 28 ∨ a * e = 28 ∨ b * c = 28 ∨ b * d = 28 ∨ b * e = 28 ∨ c * d = 28 ∨ c * e = 28 ∨ d * e = 28)
  (h8 : a * b = 20 ∨ a * c = 20 ∨ a * d = 20 ∨ a * e = 20 ∨ b * c = 20 ∨ b * d = 20 ∨ b * e = 20 ∨ c * d = 20 ∨ c * e = 20 ∨ d * e = 20)
  (h9 : a + b = 14 ∨ a + c = 14 ∨ a + d = 14 ∨ a + e = 14 ∨ b + c = 14 ∨ b + d = 14 ∨ b + e = 14 ∨ c + d = 14 ∨ c + e = 14 ∨ d + e = 14) 
  : a + b + c + d + e = 25 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l645_64558


namespace NUMINAMATH_GPT_arrangement_problem_l645_64509

def numWaysToArrangeParticipants : ℕ := 90

theorem arrangement_problem :
  ∃ (boys : ℕ) (girls : ℕ) (select_boys : ℕ → ℕ) (select_girls : ℕ → ℕ)
    (arrange : ℕ × ℕ × ℕ → ℕ),
  boys = 3 ∧ girls = 5 ∧
  select_boys boys = 3 ∧ select_girls girls = 5 ∧ 
  arrange (select_boys boys, select_girls girls, 2) = numWaysToArrangeParticipants :=
by
  sorry

end NUMINAMATH_GPT_arrangement_problem_l645_64509


namespace NUMINAMATH_GPT_intersection_eq_l645_64583

def setA : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def expectedIntersection : Set ℝ := { y | 0 < y }

theorem intersection_eq :
  setA ∩ setB = expectedIntersection :=
sorry

end NUMINAMATH_GPT_intersection_eq_l645_64583


namespace NUMINAMATH_GPT_root_quad_eqn_l645_64542

theorem root_quad_eqn (a : ℝ) (h : a^2 - a - 50 = 0) : a^3 - 51 * a = 50 :=
sorry

end NUMINAMATH_GPT_root_quad_eqn_l645_64542


namespace NUMINAMATH_GPT_douglas_votes_in_county_y_l645_64577

variable (V : ℝ) -- Number of voters in County Y
variable (A B : ℝ) -- Votes won by Douglas in County X and County Y respectively

-- Conditions
axiom h1 : A = 0.74 * 2 * V
axiom h2 : A + B = 0.66 * 3 * V
axiom ratio : (2 * V) / V = 2

-- Proof Statement
theorem douglas_votes_in_county_y :
  (B / V) * 100 = 50 := by
sorry

end NUMINAMATH_GPT_douglas_votes_in_county_y_l645_64577


namespace NUMINAMATH_GPT_range_of_a_l645_64557

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x ^ 2 + (a - 1) * x + 1 / 2 ≤ 0) → (-1 < a ∧ a < 3) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l645_64557


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_eleven_l645_64571

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_eleven_l645_64571


namespace NUMINAMATH_GPT_point_on_line_and_in_first_quadrant_l645_64546

theorem point_on_line_and_in_first_quadrant (x y : ℝ) (hline : y = -2 * x + 3) (hfirst_quadrant : x > 0 ∧ y > 0) :
    (x, y) = (1, 1) :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_and_in_first_quadrant_l645_64546


namespace NUMINAMATH_GPT_completing_the_square_l645_64550

theorem completing_the_square (x : ℝ) : x^2 + 2 * x - 5 = 0 → (x + 1)^2 = 6 := by
  intro h
  -- Starting from h and following the steps outlined to complete the square.
  sorry

end NUMINAMATH_GPT_completing_the_square_l645_64550


namespace NUMINAMATH_GPT_find_positive_integers_l645_64591

theorem find_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^2 - Nat.factorial y = 2019 ↔ x = 45 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_l645_64591


namespace NUMINAMATH_GPT_find_a_range_l645_64514

-- Definitions of sets A and B
def A (a x : ℝ) : Prop := a + 1 ≤ x ∧ x ≤ 2 * a - 1
def B (x : ℝ) : Prop := x ≤ 3 ∨ x > 5

-- Condition p: A ⊆ B
def p (a : ℝ) : Prop := ∀ x, A a x → B x

-- The function f(x) = x^2 - 2ax + 1
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Condition q: f(x) is increasing on (1/2, +∞)
def q (a : ℝ) : Prop := ∀ x y, 1/2 < x → x < y → f a x ≤ f a y

-- The given propositions
def prop1 (a : ℝ) : Prop := p a
def prop2 (a : ℝ) : Prop := q a

-- Given conditions
def given_conditions (a : ℝ) : Prop := ¬ (prop1 a ∧ prop2 a) ∧ (prop1 a ∨ prop2 a)

-- Proof statement: Find the range of values for 'a' according to the given conditions
theorem find_a_range (a : ℝ) :
  given_conditions a →
  (1/2 < a ∧ a ≤ 2) ∨ (4 < a) :=
sorry

end NUMINAMATH_GPT_find_a_range_l645_64514


namespace NUMINAMATH_GPT_proportional_b_value_l645_64507

theorem proportional_b_value (b : ℚ) : (∃ k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, x + 2 - 3 * b = k * x)) ↔ b = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_proportional_b_value_l645_64507


namespace NUMINAMATH_GPT_commercials_count_l645_64513

-- Given conditions as definitions
def total_airing_time : ℤ := 90         -- 1.5 hours in minutes
def commercial_time : ℤ := 10           -- each commercial lasts 10 minutes
def show_time : ℤ := 60                 -- TV show (without commercials) lasts 60 minutes

-- Statement: Prove that the number of commercials is 3
theorem commercials_count :
  (total_airing_time - show_time) / commercial_time = 3 :=
sorry

end NUMINAMATH_GPT_commercials_count_l645_64513


namespace NUMINAMATH_GPT_gamma_max_success_ratio_l645_64537

theorem gamma_max_success_ratio :
  ∀ (x y z w : ℕ),
    x > 0 → z > 0 →
    (5 * x < 3 * y) →
    (5 * z < 3 * w) →
    (y + w = 600) →
    (x + z ≤ 359) :=
by
  intros x y z w hx hz hxy hzw hyw
  sorry

end NUMINAMATH_GPT_gamma_max_success_ratio_l645_64537


namespace NUMINAMATH_GPT_sequence_term_a_1000_eq_2340_l645_64585

theorem sequence_term_a_1000_eq_2340
  (a : ℕ → ℤ)
  (h1 : a 1 = 2007)
  (h2 : a 2 = 2008)
  (h_rec : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = n) :
  a 1000 = 2340 :=
sorry

end NUMINAMATH_GPT_sequence_term_a_1000_eq_2340_l645_64585


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_18_l645_64564

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_18 : ∃ p : ℕ, Prime p ∧ 18 = sum_of_digits p ∧ (∀ q : ℕ, (Prime q ∧ 18 = sum_of_digits q) → p ≤ q) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_18_l645_64564


namespace NUMINAMATH_GPT_solve_for_y_l645_64540

theorem solve_for_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l645_64540


namespace NUMINAMATH_GPT_total_jelly_beans_l645_64541

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end NUMINAMATH_GPT_total_jelly_beans_l645_64541


namespace NUMINAMATH_GPT_crayons_ratio_l645_64570

theorem crayons_ratio (K B G J : ℕ) 
  (h1 : K = 2 * B)
  (h2 : B = 2 * G)
  (h3 : G = J)
  (h4 : K = 128)
  (h5 : J = 8) : 
  G / J = 4 :=
by
  sorry

end NUMINAMATH_GPT_crayons_ratio_l645_64570


namespace NUMINAMATH_GPT_olivia_quarters_left_l645_64598

-- Define the initial condition and action condition as parameters
def initial_quarters : ℕ := 11
def quarters_spent : ℕ := 4
def quarters_left : ℕ := initial_quarters - quarters_spent

-- The theorem to state the result
theorem olivia_quarters_left : quarters_left = 7 := by
  sorry

end NUMINAMATH_GPT_olivia_quarters_left_l645_64598


namespace NUMINAMATH_GPT_tom_gave_2_seashells_to_jessica_l645_64518

-- Conditions
def original_seashells : Nat := 5
def current_seashells : Nat := 3

-- Question as a proposition
def seashells_given (x : Nat) : Prop :=
  original_seashells - current_seashells = x

-- The proof problem
theorem tom_gave_2_seashells_to_jessica : seashells_given 2 :=
by 
  sorry

end NUMINAMATH_GPT_tom_gave_2_seashells_to_jessica_l645_64518


namespace NUMINAMATH_GPT_problem1_problem2_l645_64524

def f (x a : ℝ) : ℝ := abs (1 - x - a) + abs (2 * a - x)

theorem problem1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
  sorry

theorem problem2 (a x : ℝ) (h : a ≥ 2/3) : f x a ≥ 1 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l645_64524


namespace NUMINAMATH_GPT_distance_to_nearest_edge_l645_64586

theorem distance_to_nearest_edge (wall_width picture_width : ℕ) (h1 : wall_width = 19) (h2 : picture_width = 3) (h3 : 2 * x + picture_width = wall_width) :
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_nearest_edge_l645_64586


namespace NUMINAMATH_GPT_cos_225_eq_l645_64592

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end NUMINAMATH_GPT_cos_225_eq_l645_64592


namespace NUMINAMATH_GPT_tiles_needed_l645_64520

-- Definitions of the given conditions
def side_length_smaller_tile : ℝ := 0.3
def number_smaller_tiles : ℕ := 500
def side_length_larger_tile : ℝ := 0.5

-- Statement to prove the required number of larger tiles
theorem tiles_needed (x : ℕ) :
  side_length_larger_tile * side_length_larger_tile * x =
  side_length_smaller_tile * side_length_smaller_tile * number_smaller_tiles →
  x = 180 :=
by
  sorry

end NUMINAMATH_GPT_tiles_needed_l645_64520


namespace NUMINAMATH_GPT_min_xy_l645_64566

variable {x y : ℝ}

theorem min_xy (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : x * y ≥ 180 := 
sorry

end NUMINAMATH_GPT_min_xy_l645_64566


namespace NUMINAMATH_GPT_number_of_ways_to_choose_officers_l645_64559

-- Define the number of boys and girls.
def num_boys : ℕ := 12
def num_girls : ℕ := 13

-- Define the total number of boys and girls.
def num_members : ℕ := num_boys + num_girls

-- Calculate the number of ways to choose the president, vice-president, and secretary with given conditions.
theorem number_of_ways_to_choose_officers : 
  (num_boys * num_girls * (num_boys - 1)) + (num_girls * num_boys * (num_girls - 1)) = 3588 :=
by
  -- The first part calculates the ways when the president is a boy.
  -- The second part calculates the ways when the president is a girl.
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_officers_l645_64559


namespace NUMINAMATH_GPT_speed_difference_l645_64552

noncomputable def park_distance : ℝ := 10
noncomputable def kevin_time_hours : ℝ := 1 / 4
noncomputable def joel_time_hours : ℝ := 2

theorem speed_difference : (10 / kevin_time_hours) - (10 / joel_time_hours) = 35 := by
  sorry

end NUMINAMATH_GPT_speed_difference_l645_64552


namespace NUMINAMATH_GPT_decode_plaintext_l645_64555

theorem decode_plaintext (a x y : ℕ) (h1 : y = a^x - 2) (h2 : 6 = a^3 - 2) (h3 : y = 14) : x = 4 := by
  sorry

end NUMINAMATH_GPT_decode_plaintext_l645_64555


namespace NUMINAMATH_GPT_find_value_l645_64548

-- Define the mean, standard deviation, and the number of standard deviations
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5
def num_std_dev : ℝ := 2.7

-- The theorem to prove that the value is exactly 10.75
theorem find_value : mean - (num_std_dev * std_dev) = 10.75 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l645_64548


namespace NUMINAMATH_GPT_quotient_of_sum_of_remainders_div_16_eq_0_l645_64527

-- Define the set of distinct remainders of squares modulo 16 for n in 1 to 15
def distinct_remainders_mod_16 : Finset ℕ :=
  {1, 4, 9, 0}

-- Define the sum of the distinct remainders
def sum_of_remainders : ℕ :=
  distinct_remainders_mod_16.sum id

-- Proposition to prove the quotient when sum_of_remainders is divided by 16 is 0
theorem quotient_of_sum_of_remainders_div_16_eq_0 :
  (sum_of_remainders / 16) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_sum_of_remainders_div_16_eq_0_l645_64527


namespace NUMINAMATH_GPT_negation_statement_l645_64556

variable {α : Type} 
variable (student prepared : α → Prop)

theorem negation_statement :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by 
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_negation_statement_l645_64556


namespace NUMINAMATH_GPT_ratio_cubed_eq_27_l645_64568

theorem ratio_cubed_eq_27 : (81000^3) / (27000^3) = 27 := 
by
  sorry

end NUMINAMATH_GPT_ratio_cubed_eq_27_l645_64568


namespace NUMINAMATH_GPT_cos_double_angle_from_sin_shift_l645_64534

theorem cos_double_angle_from_sin_shift (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_from_sin_shift_l645_64534


namespace NUMINAMATH_GPT_total_bill_l645_64553

theorem total_bill (n : ℝ) (h : 9 * (n / 10 + 3) = n) : n = 270 := 
sorry

end NUMINAMATH_GPT_total_bill_l645_64553


namespace NUMINAMATH_GPT_product_of_possible_x_values_l645_64519

theorem product_of_possible_x_values : 
  (∃ x1 x2 : ℚ, 
    (|15 / x1 + 4| = 3 ∧ |15 / x2 + 4| = 3) ∧
    -15 * -(15 / 7) = (225 / 7)) :=
sorry

end NUMINAMATH_GPT_product_of_possible_x_values_l645_64519


namespace NUMINAMATH_GPT_find_x_coord_of_N_l645_64590

theorem find_x_coord_of_N
  (M N : ℝ × ℝ)
  (hM : M = (3, -5))
  (hN : N = (x, 2))
  (parallel : M.1 = N.1) :
  x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_coord_of_N_l645_64590


namespace NUMINAMATH_GPT_leos_current_weight_l645_64576

theorem leos_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 180) : L = 104 := 
by 
  sorry

end NUMINAMATH_GPT_leos_current_weight_l645_64576


namespace NUMINAMATH_GPT_range_of_x_satisfies_conditions_l645_64543

theorem range_of_x_satisfies_conditions (x : ℝ) (h : x^2 - 4 < 0 ∨ |x| = 2) : -2 ≤ x ∧ x ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_x_satisfies_conditions_l645_64543


namespace NUMINAMATH_GPT_cone_height_ratio_l645_64521

theorem cone_height_ratio (circumference : ℝ) (orig_height : ℝ) (short_volume : ℝ)
  (h_circumference : circumference = 20 * Real.pi)
  (h_orig_height : orig_height = 40)
  (h_short_volume : short_volume = 400 * Real.pi) :
  let r := circumference / (2 * Real.pi)
  let h_short := (3 * short_volume) / (Real.pi * r^2)
  (h_short / orig_height) = 3 / 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_cone_height_ratio_l645_64521


namespace NUMINAMATH_GPT_min_distance_curveC1_curveC2_l645_64547

-- Definitions of the conditions
def curveC1 (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 3 + Real.cos θ ∧ P.2 = 4 + Real.sin θ

def curveC2 (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

-- Proof statement
theorem min_distance_curveC1_curveC2 :
  (∀ A B : ℝ × ℝ,
    curveC1 A →
    curveC2 B →
    ∃ m : ℝ, m = 3 ∧ ∀ d : ℝ, (d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) → d ≥ m) := 
  sorry

end NUMINAMATH_GPT_min_distance_curveC1_curveC2_l645_64547


namespace NUMINAMATH_GPT_eight_percent_is_64_l645_64569

-- Definition of the condition
variable (x : ℝ)

-- The theorem that states the problem to be proven
theorem eight_percent_is_64 (h : (8 / 100) * x = 64) : x = 800 :=
sorry

end NUMINAMATH_GPT_eight_percent_is_64_l645_64569


namespace NUMINAMATH_GPT_sqrt_of_1_5625_eq_1_25_l645_64554

theorem sqrt_of_1_5625_eq_1_25 : Real.sqrt 1.5625 = 1.25 :=
  sorry

end NUMINAMATH_GPT_sqrt_of_1_5625_eq_1_25_l645_64554


namespace NUMINAMATH_GPT_primes_with_large_gap_exists_l645_64529

noncomputable def exists_primes_with_large_gap_and_composites_between : Prop :=
  ∃ p q : ℕ, p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p > 2015 ∧ (∀ n : ℕ, p < n ∧ n < q → ¬Nat.Prime n)

theorem primes_with_large_gap_exists : exists_primes_with_large_gap_and_composites_between := sorry

end NUMINAMATH_GPT_primes_with_large_gap_exists_l645_64529


namespace NUMINAMATH_GPT_pizzas_difference_l645_64522

def pizzas (craig_first_day craig_second_day heather_first_day heather_second_day total_pizzas: ℕ) :=
  heather_first_day = 4 * craig_first_day ∧
  heather_second_day = craig_second_day - 20 ∧
  craig_first_day = 40 ∧
  craig_first_day + heather_first_day + craig_second_day + heather_second_day = total_pizzas

theorem pizzas_difference :
  ∀ (craig_first_day craig_second_day heather_first_day heather_second_day : ℕ),
  pizzas craig_first_day craig_second_day heather_first_day heather_second_day 380 →
  craig_second_day - craig_first_day = 60 :=
by
  intros craig_first_day craig_second_day heather_first_day heather_second_day h
  sorry

end NUMINAMATH_GPT_pizzas_difference_l645_64522


namespace NUMINAMATH_GPT_grandpa_max_pieces_l645_64517

theorem grandpa_max_pieces (m n : ℕ) (h : (m - 3) * (n - 3) = 9) : m * n = 112 :=
sorry

end NUMINAMATH_GPT_grandpa_max_pieces_l645_64517


namespace NUMINAMATH_GPT_beef_stew_duration_l645_64515

noncomputable def original_portions : ℕ := 14
noncomputable def your_portion : ℕ := 1
noncomputable def roommate_portion : ℕ := 3
noncomputable def guest_portion : ℕ := 4
noncomputable def total_daily_consumption : ℕ := your_portion + roommate_portion + guest_portion
noncomputable def days_stew_lasts : ℕ := original_portions / total_daily_consumption

theorem beef_stew_duration : days_stew_lasts = 2 :=
by
  sorry

end NUMINAMATH_GPT_beef_stew_duration_l645_64515


namespace NUMINAMATH_GPT_impossible_network_of_triangles_l645_64575

-- Define the conditions of the problem, here we could define vertices and properties of the network
structure Vertex :=
(triangles_meeting : Nat)

def five_triangles_meeting (v : Vertex) : Prop :=
v.triangles_meeting = 5

-- The main theorem statement - it's impossible to cover the entire plane with such a network
theorem impossible_network_of_triangles :
  ¬ (∀ v : Vertex, five_triangles_meeting v) :=
sorry

end NUMINAMATH_GPT_impossible_network_of_triangles_l645_64575


namespace NUMINAMATH_GPT_grocery_delivery_amount_l645_64533

theorem grocery_delivery_amount (initial_savings final_price trips : ℕ) 
(fixed_charge : ℝ) (percent_charge : ℝ) (total_saved : ℝ) : 
  initial_savings = 14500 →
  final_price = 14600 →
  trips = 40 →
  fixed_charge = 1.5 →
  percent_charge = 0.05 →
  total_saved = final_price - initial_savings →
  60 + percent_charge * G = total_saved →
  G = 800 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_grocery_delivery_amount_l645_64533


namespace NUMINAMATH_GPT_fewest_apples_l645_64510

-- Definitions based on the conditions
def Yoongi_apples : Nat := 4
def Jungkook_initial_apples : Nat := 6
def Jungkook_additional_apples : Nat := 3
def Jungkook_apples : Nat := Jungkook_initial_apples + Jungkook_additional_apples
def Yuna_apples : Nat := 5

-- Main theorem based on the question and the correct answer
theorem fewest_apples : Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end NUMINAMATH_GPT_fewest_apples_l645_64510


namespace NUMINAMATH_GPT_value_of_expression_when_x_is_2_l645_64579

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_when_x_is_2_l645_64579


namespace NUMINAMATH_GPT_triangle_angle_extension_l645_64599

theorem triangle_angle_extension :
  ∀ (BAC ABC BCA CDB DBC : ℝ),
  180 = BAC + ABC + BCA →
  CDB = BAC + ABC →
  DBC = BAC + BCA →
  (CDB + DBC) / (BAC + ABC) = 2 :=
by
  intros BAC ABC BCA CDB DBC h1 h2 h3
  sorry

end NUMINAMATH_GPT_triangle_angle_extension_l645_64599


namespace NUMINAMATH_GPT_spherical_to_rectangular_l645_64597

theorem spherical_to_rectangular
  (ρ θ φ : ℝ)
  (ρ_eq : ρ = 10)
  (θ_eq : θ = 5 * Real.pi / 4)
  (φ_eq : φ = Real.pi / 4) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_l645_64597


namespace NUMINAMATH_GPT_circle_tangent_x_axis_at_origin_l645_64525

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → (∃ r : ℝ, r^2 = x^2 + y^2) ∧ y = 0) →
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 := 
sorry

end NUMINAMATH_GPT_circle_tangent_x_axis_at_origin_l645_64525


namespace NUMINAMATH_GPT_quadratic_value_at_6_l645_64500

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 3

theorem quadratic_value_at_6 
  (a b : ℝ) (h : a ≠ 0) 
  (h_eq : f a b 2 = f a b 4) : 
  f a b 6 = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_value_at_6_l645_64500


namespace NUMINAMATH_GPT_clea_ride_down_time_l645_64596

theorem clea_ride_down_time (c s d : ℝ) (h1 : d = 70 * c) (h2 : d = 28 * (c + s)) :
  (d / s) = 47 := by
  sorry

end NUMINAMATH_GPT_clea_ride_down_time_l645_64596


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l645_64578

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ - k^2 = 0) ∧ (x₂^2 - x₂ - k^2 = 0) :=
by
  -- The proof is omitted as requested.
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l645_64578


namespace NUMINAMATH_GPT_midpoint_of_AB_l645_64505

theorem midpoint_of_AB (xA xB : ℝ) (p : ℝ) (h_parabola : ∀ y, y^2 = 4 * xA → y^2 = 4 * xB)
  (h_focus : (2 : ℝ) = p)
  (h_length_AB : (abs (xB - xA)) = 5) :
  (xA + xB) / 2 = 3 / 2 :=
sorry

end NUMINAMATH_GPT_midpoint_of_AB_l645_64505


namespace NUMINAMATH_GPT_max_product_areas_l645_64508

theorem max_product_areas (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h : a + b + c + d = 1) :
  a * b * c * d ≤ 1 / 256 :=
sorry

end NUMINAMATH_GPT_max_product_areas_l645_64508


namespace NUMINAMATH_GPT_c_plus_d_is_even_l645_64501

-- Define the conditions
variables {c d : ℕ}
variables (m n : ℕ) (hc : c = 6 * m) (hd : d = 9 * n)

-- State the theorem to be proven
theorem c_plus_d_is_even : 
  (c = 6 * m) → (d = 9 * n) → Even (c + d) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_c_plus_d_is_even_l645_64501


namespace NUMINAMATH_GPT_horner_eval_v3_at_minus4_l645_64516

def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

def horner_form (x : ℤ) : ℤ :=
  let a6 := 3
  let a5 := 5
  let a4 := 6
  let a3 := 79
  let a2 := -8
  let a1 := 35
  let a0 := 12
  let v := a6
  let v1 := v * x + a5
  let v2 := v1 * x + a4
  let v3 := v2 * x + a3
  let v4 := v3 * x + a2
  let v5 := v4 * x + a1
  let v6 := v5 * x + a0
  v3

theorem horner_eval_v3_at_minus4 :
  horner_form (-4) = -57 :=
by
  sorry

end NUMINAMATH_GPT_horner_eval_v3_at_minus4_l645_64516


namespace NUMINAMATH_GPT_cost_of_each_green_hat_l645_64580

theorem cost_of_each_green_hat
  (total_hats : ℕ) (cost_blue_hat : ℕ) (total_price : ℕ) (green_hats : ℕ) (blue_hats : ℕ) (cost_green_hat : ℕ)
  (h1 : total_hats = 85) 
  (h2 : cost_blue_hat = 6) 
  (h3 : total_price = 550) 
  (h4 : green_hats = 40) 
  (h5 : blue_hats = 45) 
  (h6 : green_hats + blue_hats = total_hats) 
  (h7 : total_price = green_hats * cost_green_hat + blue_hats * cost_blue_hat) :
  cost_green_hat = 7 := 
sorry

end NUMINAMATH_GPT_cost_of_each_green_hat_l645_64580


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l645_64535

theorem equation1_solution (x : ℝ) (h : 3 * x - 1 = x + 7) : x = 4 := by
  sorry

theorem equation2_solution (x : ℝ) (h : (x + 1) / 2 - 1 = (1 - 2 * x) / 3) : x = 5 / 7 := by
  sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l645_64535


namespace NUMINAMATH_GPT_gcd_840_1764_gcd_459_357_l645_64562

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := sorry

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := sorry

end NUMINAMATH_GPT_gcd_840_1764_gcd_459_357_l645_64562


namespace NUMINAMATH_GPT_union_set_A_set_B_l645_64545

def set_A : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }
def set_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def set_union (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∨ x ∈ B }

theorem union_set_A_set_B : set_union set_A set_B = { x | -3 < x ∧ x < 6 } := 
by sorry

end NUMINAMATH_GPT_union_set_A_set_B_l645_64545


namespace NUMINAMATH_GPT_multiple_of_1984_exists_l645_64511

theorem multiple_of_1984_exists (a : Fin 97 → ℕ) (h_distinct: Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
  1984 ∣ (a i - a j) * (a k - a l) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_1984_exists_l645_64511


namespace NUMINAMATH_GPT_second_and_third_shooters_cannot_win_or_lose_simultaneously_l645_64561

-- Define the conditions C1, C2, and C3
variables (C1 C2 C3 : Prop)

-- The first shooter bets that at least one of the second or third shooters will miss
def first_shooter_bet : Prop := ¬ (C2 ∧ C3)

-- The second shooter bets that if the first shooter hits, then at least one of the remaining shooters will miss
def second_shooter_bet : Prop := C1 → ¬ (C2 ∧ C3)

-- The third shooter bets that all three will hit the target on the first attempt
def third_shooter_bet : Prop := C1 ∧ C2 ∧ C3

-- Prove that it is impossible for both the second and third shooters to either win or lose their bets concurrently
theorem second_and_third_shooters_cannot_win_or_lose_simultaneously :
  ¬ ((second_shooter_bet C1 C2 C3 ∧ third_shooter_bet C1 C2 C3) ∨ (¬ second_shooter_bet C1 C2 C3 ∧ ¬ third_shooter_bet C1 C2 C3)) :=
by
  sorry

end NUMINAMATH_GPT_second_and_third_shooters_cannot_win_or_lose_simultaneously_l645_64561


namespace NUMINAMATH_GPT_sequence_inequality_l645_64574

theorem sequence_inequality (a : ℕ → ℕ)
  (h1 : a 0 > 0) -- Ensure all entries are positive integers.
  (h2 : ∀ k l m n : ℕ, k * l = m * n → a k + a l = a m + a n)
  {p q : ℕ} (hpq : p ∣ q) :
  a p ≤ a q :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l645_64574
