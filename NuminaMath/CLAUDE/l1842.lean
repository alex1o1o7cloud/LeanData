import Mathlib

namespace fibonacci_closed_form_l1842_184227

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_closed_form (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) (h3 : a > b) :
  ∀ n : ℕ, fibonacci n = (a^(n+1) - b^(n+1)) / Real.sqrt 5 :=
by sorry

end fibonacci_closed_form_l1842_184227


namespace download_speed_scientific_notation_l1842_184225

/-- The download speed of a 5G network in KB per second -/
def download_speed : ℝ := 1300000

/-- Scientific notation representation of the download speed -/
def scientific_notation : ℝ := 1.3 * (10 ^ 6)

theorem download_speed_scientific_notation : 
  download_speed = scientific_notation := by
  sorry

end download_speed_scientific_notation_l1842_184225


namespace expansion_coefficient_remainder_counts_l1842_184205

/-- 
Given a natural number n, Tᵣ(n) represents the number of coefficients in the expansion of (1+x)ⁿ 
that give a remainder of r when divided by 3, where r ∈ {0,1,2}.
-/
def T (r n : ℕ) : ℕ := sorry

/-- The theorem states the values of T₀(2006), T₁(2006), and T₂(2006) for the expansion of (1+x)²⁰⁰⁶. -/
theorem expansion_coefficient_remainder_counts : 
  T 0 2006 = 1764 ∧ T 1 2006 = 122 ∧ T 2 2006 = 121 := by sorry

end expansion_coefficient_remainder_counts_l1842_184205


namespace book_selection_combinations_l1842_184222

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of books in the library -/
def total_books : ℕ := 15

/-- The number of books to be selected -/
def selected_books : ℕ := 3

/-- Theorem: The number of ways to choose 3 books from 15 books is 455 -/
theorem book_selection_combinations :
  choose total_books selected_books = 455 := by sorry

end book_selection_combinations_l1842_184222


namespace simplify_sqrt_expression_l1842_184290

theorem simplify_sqrt_expression (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end simplify_sqrt_expression_l1842_184290


namespace max_y_coord_sin_3theta_l1842_184293

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 1 -/
theorem max_y_coord_sin_3theta :
  let r : ℝ → ℝ := λ θ ↦ Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ (θ : ℝ), y θ = 1 ∧ ∀ (φ : ℝ), y φ ≤ 1 := by
  sorry

end max_y_coord_sin_3theta_l1842_184293


namespace female_salmon_count_l1842_184217

theorem female_salmon_count (male_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : total_salmon = 971639) :
  total_salmon - male_salmon = 259378 := by
  sorry

end female_salmon_count_l1842_184217


namespace range_of_a_l1842_184285

theorem range_of_a (a : ℝ) : (¬ ∃ x, x < 2023 ∧ x > a) → a ≥ 2023 := by
  sorry

end range_of_a_l1842_184285


namespace vote_difference_is_40_l1842_184212

-- Define the committee and voting scenario
def CommitteeVoting (total_members : ℕ) (initial_for initial_against revote_for revote_against : ℕ) : Prop :=
  -- Total members condition
  total_members = initial_for + initial_against ∧
  total_members = revote_for + revote_against ∧
  -- Initially rejected condition
  initial_against > initial_for ∧
  -- Re-vote margin condition
  (revote_for - revote_against) = 3 * (initial_against - initial_for) ∧
  -- Re-vote for vs initial against condition
  revote_for * 12 = initial_against * 13

-- Theorem statement
theorem vote_difference_is_40 :
  ∀ (initial_for initial_against revote_for revote_against : ℕ),
    CommitteeVoting 500 initial_for initial_against revote_for revote_against →
    revote_for - initial_for = 40 := by
  sorry

end vote_difference_is_40_l1842_184212


namespace height_average_comparison_l1842_184219

theorem height_average_comparison 
  (h₁ : ℝ → ℝ → ℝ → ℝ → ℝ → Prop) 
  (a b c d : ℝ) 
  (h₂ : 3 * a + 2 * b = 2 * c + 3 * d) 
  (h₃ : a > d) : 
  |c + d| / 2 > |a + b| / 2 := by
sorry

end height_average_comparison_l1842_184219


namespace female_rainbow_count_l1842_184232

/-- Represents the number of trout in a fishery -/
structure Fishery where
  female_speckled : ℕ
  male_speckled : ℕ
  female_rainbow : ℕ
  male_rainbow : ℕ

/-- The conditions of the fishery problem -/
def fishery_conditions (f : Fishery) : Prop :=
  f.female_speckled + f.male_speckled = 645 ∧
  f.male_speckled = 2 * f.female_speckled + 45 ∧
  4 * f.male_rainbow = 3 * f.female_speckled ∧
  3 * (f.female_speckled + f.male_speckled + f.female_rainbow + f.male_rainbow) = 20 * f.male_rainbow

/-- The theorem stating that under the given conditions, there are 205 female rainbow trout -/
theorem female_rainbow_count (f : Fishery) :
  fishery_conditions f → f.female_rainbow = 205 := by
  sorry


end female_rainbow_count_l1842_184232


namespace nested_average_equals_seven_ninths_l1842_184280

def average2 (a b : ℚ) : ℚ := (a + b) / 2

def average3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_seven_ninths :
  average3 (average3 2 2 0) (average2 0 2) 0 = 7/9 := by sorry

end nested_average_equals_seven_ninths_l1842_184280


namespace alok_mixed_veg_order_l1842_184296

/-- Represents the order and payment details of Alok's meal --/
structure MealOrder where
  chapatis : ℕ
  rice_plates : ℕ
  ice_cream_cups : ℕ
  chapati_cost : ℕ
  rice_cost : ℕ
  mixed_veg_cost : ℕ
  total_paid : ℕ

/-- Calculates the number of mixed vegetable plates ordered --/
def mixed_veg_plates (order : MealOrder) : ℕ :=
  ((order.total_paid - (order.chapatis * order.chapati_cost + order.rice_plates * order.rice_cost)) / order.mixed_veg_cost)

/-- Theorem stating that Alok ordered 9 plates of mixed vegetable --/
theorem alok_mixed_veg_order : 
  ∀ (order : MealOrder), 
    order.chapatis = 16 ∧ 
    order.rice_plates = 5 ∧ 
    order.ice_cream_cups = 6 ∧
    order.chapati_cost = 6 ∧ 
    order.rice_cost = 45 ∧ 
    order.mixed_veg_cost = 70 ∧ 
    order.total_paid = 961 → 
    mixed_veg_plates order = 9 := by
  sorry


end alok_mixed_veg_order_l1842_184296


namespace rectangle_area_change_l1842_184274

theorem rectangle_area_change (initial_short : ℝ) (initial_long : ℝ) 
  (h1 : initial_short = 5)
  (h2 : initial_long = 7)
  (h3 : ∃ x, initial_short * (initial_long - x) = 24) :
  (initial_short * (initial_long - 2) = 25) := by
  sorry

end rectangle_area_change_l1842_184274


namespace point_N_coordinates_l1842_184201

def M : ℝ × ℝ := (3, -4)
def a : ℝ × ℝ := (1, -2)

theorem point_N_coordinates : 
  ∀ N : ℝ × ℝ, 
  (N.1 - M.1, N.2 - M.2) = (-2 * a.1, -2 * a.2) → 
  N = (1, 0) := by
  sorry

end point_N_coordinates_l1842_184201


namespace average_fuel_efficiency_l1842_184282

/-- Calculates the average fuel efficiency for a trip with multiple segments and different vehicles. -/
theorem average_fuel_efficiency 
  (total_distance : ℝ)
  (sedan_distance : ℝ)
  (truck_distance : ℝ)
  (detour_distance : ℝ)
  (sedan_efficiency : ℝ)
  (truck_efficiency : ℝ)
  (detour_efficiency : ℝ)
  (h1 : total_distance = sedan_distance + truck_distance + detour_distance)
  (h2 : sedan_distance = 150)
  (h3 : truck_distance = 150)
  (h4 : detour_distance = 50)
  (h5 : sedan_efficiency = 25)
  (h6 : truck_efficiency = 15)
  (h7 : detour_efficiency = 10) :
  ∃ (ε : ℝ), abs (total_distance / (sedan_distance / sedan_efficiency + 
                                    truck_distance / truck_efficiency + 
                                    detour_distance / detour_efficiency) - 16.67) < ε :=
by sorry

end average_fuel_efficiency_l1842_184282


namespace hyperbolas_same_asymptotes_implies_M_64_l1842_184260

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2 / 16 - y^2 / 25 = 1
def hyperbola2 (x y M : ℝ) : Prop := y^2 / 100 - x^2 / M = 1

-- Define the asymptotes of the hyperbolas
def asymptote1 (x y : ℝ) : Prop := y = (5/4) * x ∨ y = -(5/4) * x
def asymptote2 (x y M : ℝ) : Prop := y = (10/Real.sqrt M) * x ∨ y = -(10/Real.sqrt M) * x

-- Theorem statement
theorem hyperbolas_same_asymptotes_implies_M_64 :
  ∀ M : ℝ, (∀ x y : ℝ, asymptote1 x y ↔ asymptote2 x y M) → M = 64 := by
  sorry

end hyperbolas_same_asymptotes_implies_M_64_l1842_184260


namespace partial_fraction_sum_zero_l1842_184268

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end partial_fraction_sum_zero_l1842_184268


namespace point_movement_theorem_l1842_184220

theorem point_movement_theorem (A : ℝ) : 
  (A + 7 - 4 = 0) → A = -3 := by
  sorry

end point_movement_theorem_l1842_184220


namespace absolute_value_equation_a_l1842_184252

theorem absolute_value_equation_a (x : ℝ) : |x - 5| = 2 ↔ x = 3 ∨ x = 7 := by sorry

end absolute_value_equation_a_l1842_184252


namespace weight_of_replaced_person_l1842_184272

theorem weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  let replaced_person_weight := new_person_weight - initial_count * average_increase
  replaced_person_weight

#check weight_of_replaced_person 8 5 75 -- Should evaluate to 35

end weight_of_replaced_person_l1842_184272


namespace leftover_books_l1842_184278

/-- The number of leftover books when repacking from boxes of 45 to boxes of 47 -/
theorem leftover_books (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1500 →
  books_per_initial_box = 45 →
  books_per_new_box = 47 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 13 := by
  sorry

#eval (1500 * 45) % 47  -- This should output 13

end leftover_books_l1842_184278


namespace not_all_greater_than_one_l1842_184292

theorem not_all_greater_than_one (a b c : Real) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) := by
  sorry

end not_all_greater_than_one_l1842_184292


namespace hawks_score_l1842_184270

theorem hawks_score (total_points eagles_margin hawks_min_score : ℕ) 
  (h1 : total_points = 82)
  (h2 : eagles_margin = 18)
  (h3 : hawks_min_score = 9)
  (h4 : ∃ (hawks_score : ℕ), 
    hawks_score ≥ hawks_min_score ∧ 
    hawks_score + (hawks_score + eagles_margin) = total_points) :
  ∃ (hawks_score : ℕ), hawks_score = 32 :=
by sorry

end hawks_score_l1842_184270


namespace fraction_to_decimal_l1842_184215

theorem fraction_to_decimal : (29 : ℚ) / 160 = 0.18125 := by
  sorry

end fraction_to_decimal_l1842_184215


namespace prob_at_most_one_red_l1842_184238

/-- The probability of drawing at most 1 red ball from a bag of 8 balls (3 red, 2 white, 3 black) when randomly selecting 3 balls. -/
theorem prob_at_most_one_red (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ) 
  (h_total : total = 8)
  (h_red : red = 3)
  (h_white : white = 2)
  (h_black : black = 3)
  (h_sum : red + white + black = total)
  (draw : ℕ)
  (h_draw : draw = 3) :
  (Nat.choose (total - red) draw + Nat.choose red 1 * Nat.choose (total - red) (draw - 1)) / Nat.choose total draw = 5 / 7 := by
  sorry

end prob_at_most_one_red_l1842_184238


namespace polynomial_coefficients_l1842_184211

theorem polynomial_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (x + 2) * (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = 241 ∧ a₂ = -70) := by
sorry

end polynomial_coefficients_l1842_184211


namespace curve_and_intersection_l1842_184204

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 3))^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

-- Define the line that intersects C
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem statement
theorem curve_and_intersection :
  ∃ (k : ℝ),
    (∀ x y, C x y ↔ x^2 + y^2/4 = 1) ∧
    (∃ x₁ y₁ x₂ y₂,
      C x₁ y₁ ∧ C x₂ y₂ ∧
      intersecting_line k x₁ y₁ ∧
      intersecting_line k x₂ y₂ ∧
      perpendicular x₁ y₁ x₂ y₂ ∧
      (k = 1/2 ∨ k = -1/2)) :=
by sorry

end curve_and_intersection_l1842_184204


namespace condition_relationship_l1842_184277

theorem condition_relationship (x : ℝ) : 
  (∀ x, x = 1 → x^2 - 3*x + 2 = 0) ∧ 
  (∃ x, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) := by
  sorry

end condition_relationship_l1842_184277


namespace smallest_product_factors_l1842_184288

/-- A structure representing an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ -- first term
  d : ℕ -- common difference

/-- A structure representing a geometric sequence -/
structure GeometricSequence where
  b : ℕ -- first term
  r : ℕ -- common ratio

/-- The product of the first four terms of an arithmetic sequence -/
def arithProduct (seq : ArithmeticSequence) : ℕ :=
  seq.a * (seq.a + seq.d) * (seq.a + 2*seq.d) * (seq.a + 3*seq.d)

/-- The product of the first four terms of a geometric sequence -/
def geoProduct (seq : GeometricSequence) : ℕ :=
  seq.b * (seq.b * seq.r) * (seq.b * seq.r^2) * (seq.b * seq.r^3)

/-- The number of positive factors of a natural number -/
def numPositiveFactors (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem smallest_product_factors : 
  ∃ (n : ℕ) (arith : ArithmeticSequence) (geo : GeometricSequence), 
    n > 500000 ∧ 
    n = arithProduct arith ∧ 
    n = geoProduct geo ∧
    (∀ m, m > 500000 → m = arithProduct arith → m = geoProduct geo → m ≥ n) ∧
    numPositiveFactors n = 56 := by
  sorry

end smallest_product_factors_l1842_184288


namespace beaus_age_is_42_l1842_184259

/-- Beau's age today given his triplet sons' ages and a condition from the past -/
def beaus_age_today (sons_age_today : ℕ) : ℕ :=
  let sons_age_past := sons_age_today - 3
  let beaus_age_past := 3 * sons_age_past
  beaus_age_past + 3

theorem beaus_age_is_42 :
  beaus_age_today 16 = 42 := by sorry

end beaus_age_is_42_l1842_184259


namespace game_lives_per_player_l1842_184254

theorem game_lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) :
  initial_players = 8 →
  additional_players = 2 →
  total_lives = 60 →
  (total_lives / (initial_players + additional_players) : ℚ) = 6 := by
  sorry

end game_lives_per_player_l1842_184254


namespace vector_sum_squared_l1842_184289

variable (a b c m : ℝ × ℝ)

/-- m is the midpoint of a and b -/
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The squared norm of a 2D vector -/
def norm_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem vector_sum_squared (a b : ℝ × ℝ) :
  is_midpoint m a b →
  m = (4, 5) →
  dot_product a b = 12 →
  dot_product c (a.1 + b.1, a.2 + b.2) = 0 →
  norm_squared a + norm_squared b = 140 := by
  sorry

end vector_sum_squared_l1842_184289


namespace garage_sale_ratio_l1842_184246

theorem garage_sale_ratio (treadmill_price chest_price tv_price total_sale : ℚ) : 
  treadmill_price = 100 →
  chest_price = treadmill_price / 2 →
  total_sale = 600 →
  total_sale = treadmill_price + chest_price + tv_price →
  tv_price / treadmill_price = 9 / 2 := by
  sorry

end garage_sale_ratio_l1842_184246


namespace speeding_ticket_percentage_l1842_184241

theorem speeding_ticket_percentage
  (total_motorists : ℝ)
  (exceed_limit_percentage : ℝ)
  (no_ticket_percentage : ℝ)
  (h1 : exceed_limit_percentage = 0.5)
  (h2 : no_ticket_percentage = 0.2)
  (h3 : total_motorists > 0) :
  let speeding_motorists := total_motorists * exceed_limit_percentage
  let no_ticket_motorists := speeding_motorists * no_ticket_percentage
  let ticket_motorists := speeding_motorists - no_ticket_motorists
  ticket_motorists / total_motorists = 0.4 :=
sorry

end speeding_ticket_percentage_l1842_184241


namespace second_attempt_score_l1842_184250

/-- Represents the number of points scored in each attempt -/
structure Attempts where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The minimum and maximum possible points for a single dart throw -/
def min_points : ℕ := 3
def max_points : ℕ := 9

/-- The number of darts thrown in each attempt -/
def num_darts : ℕ := 8

theorem second_attempt_score (a : Attempts) : 
  (a.second = 2 * a.first) → 
  (a.third = 3 * a.first) → 
  (a.first ≥ num_darts * min_points) → 
  (a.third ≤ num_darts * max_points) → 
  a.second = 48 := by
  sorry

end second_attempt_score_l1842_184250


namespace tan_product_identity_l1842_184295

theorem tan_product_identity : (1 + Real.tan (3 * π / 180)) * (1 + Real.tan (42 * π / 180)) = 2 := by
  sorry

end tan_product_identity_l1842_184295


namespace triangle_circumcircle_intersection_l1842_184284

theorem triangle_circumcircle_intersection (PQ QR RP : ℝ) (h1 : PQ = 39) (h2 : QR = 15) (h3 : RP = 50) : 
  ∃ (PS : ℝ), PS = 5 * Real.sqrt 61 ∧ 
  ⌊5 + Real.sqrt 61⌋ = 13 :=
by sorry

end triangle_circumcircle_intersection_l1842_184284


namespace negation_of_universal_proposition_l1842_184214

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end negation_of_universal_proposition_l1842_184214


namespace vector_orthogonality_l1842_184243

theorem vector_orthogonality (a b c : ℝ × ℝ) (x : ℝ) : 
  a = (1, 2) → b = (1, 0) → c = (3, 4) → 
  (b.1 + x * a.1, b.2 + x * a.2) • c = 0 → 
  x = -3/11 := by
  sorry

end vector_orthogonality_l1842_184243


namespace smallest_consecutive_odd_divisibility_l1842_184299

theorem smallest_consecutive_odd_divisibility (n : ℕ+) :
  ∃ (u_n : ℕ+),
    (∀ (d : ℕ+) (a : ℕ),
      (∀ k : Fin u_n, ∃ m : ℕ, a + 2 * k.val = d * m) →
      (∀ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * m)) ∧
    (∀ (v : ℕ+),
      v < u_n →
      ∃ (d : ℕ+) (a : ℕ),
        (∀ k : Fin v, ∃ m : ℕ, a + 2 * k.val = d * m) ∧
        ¬(∀ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * m)) ∧
    u_n = 2 * n - 1 := by
  sorry

end smallest_consecutive_odd_divisibility_l1842_184299


namespace multiplication_division_equality_l1842_184283

theorem multiplication_division_equality : (3.6 * 0.25) / 0.5 = 1.8 := by
  sorry

end multiplication_division_equality_l1842_184283


namespace complex_equation_sum_squares_l1842_184240

theorem complex_equation_sum_squares (a b : ℝ) :
  (a + Complex.I) / Complex.I = b + Complex.I * Real.sqrt 2 →
  a^2 + b^2 = 3 := by
sorry

end complex_equation_sum_squares_l1842_184240


namespace interest_calculation_l1842_184230

/-- Represents the interest calculation problem -/
theorem interest_calculation (x y z : ℝ) 
  (h1 : x * y / 100 * 2 = 800)  -- Simple interest condition
  (h2 : x * ((1 + y / 100)^2 - 1) = 820)  -- Compound interest condition
  : x = 8000 := by
  sorry

end interest_calculation_l1842_184230


namespace p_less_than_q_l1842_184237

theorem p_less_than_q (a : ℝ) (h : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 5) < Real.sqrt (a + 2) + Real.sqrt (a + 3) := by
sorry

end p_less_than_q_l1842_184237


namespace jeds_speed_jeds_speed_is_66_l1842_184228

def fine_per_mph : ℝ := 16
def total_fine : ℝ := 256
def speed_limit : ℝ := 50

theorem jeds_speed : ℝ :=
  speed_limit + total_fine / fine_per_mph

theorem jeds_speed_is_66 : jeds_speed = 66 := by sorry

end jeds_speed_jeds_speed_is_66_l1842_184228


namespace flag_raising_arrangements_l1842_184262

/-- The number of classes in the first year of high school -/
def first_year_classes : ℕ := 8

/-- The number of classes in the second year of high school -/
def second_year_classes : ℕ := 6

/-- The total number of possible arrangements for selecting one class for flag-raising duty -/
def total_arrangements : ℕ := first_year_classes + second_year_classes

/-- Theorem stating that the total number of possible arrangements is 14 -/
theorem flag_raising_arrangements :
  total_arrangements = 14 := by sorry

end flag_raising_arrangements_l1842_184262


namespace problem_solution_l1842_184267

theorem problem_solution (t : ℝ) (x y : ℝ) 
    (h1 : x = 3 - 2*t) 
    (h2 : y = 3*t + 6) 
    (h3 : x = 0) : 
  y = 21/2 := by
sorry

end problem_solution_l1842_184267


namespace integral_problem_l1842_184249

theorem integral_problem : ∫ x in (0)..(2 * Real.arctan (1/2)), (1 - Real.sin x) / (Real.cos x * (1 + Real.cos x)) = 2 * Real.log (3/2) - 1/2 := by
  sorry

end integral_problem_l1842_184249


namespace current_speed_l1842_184264

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 8.6) :
  ∃ (current_speed : ℝ), current_speed = 3.2 :=
by
  sorry

end current_speed_l1842_184264


namespace polynomial_division_remainder_l1842_184266

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^2 - 22 * X + 64 = (X - 3) * q + 25 := by sorry

end polynomial_division_remainder_l1842_184266


namespace area_of_similar_pentagons_l1842_184223

/-- Theorem: Area of similar pentagons
  Given two similar pentagons with perimeters K₁ and K₂, and areas L₁ and L₂,
  if K₁ = 18, K₂ = 24, and L₁ = 8 7/16, then L₂ = 15.
-/
theorem area_of_similar_pentagons (K₁ K₂ L₁ L₂ : ℝ) : 
  K₁ = 18 → K₂ = 24 → L₁ = 8 + 7/16 → 
  (K₁ / K₂)^2 = L₁ / L₂ → 
  L₂ = 15 := by
  sorry


end area_of_similar_pentagons_l1842_184223


namespace opposites_and_reciprocals_problem_l1842_184271

theorem opposites_and_reciprocals_problem 
  (a b x y : ℝ) 
  (h1 : a + b = 0)      -- a and b are opposites
  (h2 : x * y = 1)      -- x and y are reciprocals
  : 5 * |a + b| - 5 * x * y = -5 := by
  sorry

end opposites_and_reciprocals_problem_l1842_184271


namespace aunt_gemma_dog_food_duration_l1842_184210

/-- Calculates the number of days dog food will last given the number of dogs, 
    feeding frequency, food consumption per meal, and amount of food bought. -/
def dogFoodDuration (numDogs : ℕ) (feedingsPerDay : ℕ) (gramsPerMeal : ℕ) 
                    (numSacks : ℕ) (kgPerSack : ℕ) : ℕ :=
  let dailyConsumptionGrams := numDogs * feedingsPerDay * gramsPerMeal
  let totalFoodKg := numSacks * kgPerSack
  totalFoodKg * 1000 / dailyConsumptionGrams

theorem aunt_gemma_dog_food_duration :
  dogFoodDuration 4 2 250 2 50 = 50 := by
  sorry

#eval dogFoodDuration 4 2 250 2 50

end aunt_gemma_dog_food_duration_l1842_184210


namespace solution_equations_solution_inequalities_l1842_184297

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def equation2 (x y : ℝ) : Prop := 5 * x + 2 * y = 6

-- Define the system of inequalities
def inequality1 (x : ℝ) : Prop := 3 * x + 6 > 0
def inequality2 (x : ℝ) : Prop := x - 2 < -x

-- Theorem for the system of equations
theorem solution_equations :
  ∃! (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 1 ∧ y = 1/2 := by sorry

-- Theorem for the system of inequalities
theorem solution_inequalities :
  ∀ x : ℝ, inequality1 x ∧ inequality2 x ↔ -2 < x ∧ x < 1 := by sorry

end solution_equations_solution_inequalities_l1842_184297


namespace second_group_size_l1842_184200

theorem second_group_size (sum_first : ℕ) (count_first : ℕ) (avg_second : ℚ) (avg_total : ℚ) 
  (h1 : sum_first = 84)
  (h2 : count_first = 7)
  (h3 : avg_second = 21)
  (h4 : avg_total = 18) :
  ∃ (count_second : ℕ), 
    (sum_first + count_second * avg_second) / (count_first + count_second) = avg_total ∧ 
    count_second = 14 := by
  sorry

end second_group_size_l1842_184200


namespace triangle_area_l1842_184224

theorem triangle_area (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 6) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3 * Real.sqrt 7 := by
  sorry

end triangle_area_l1842_184224


namespace solution_implies_m_value_l1842_184203

theorem solution_implies_m_value (m : ℚ) : 
  (m * (-3) - 8 = 15 + m) → m = -23/4 := by
  sorry

end solution_implies_m_value_l1842_184203


namespace polynomial_correction_l1842_184216

/-- If a polynomial P(x) satisfies P(x) - 3x² = x² - 2x + 1, 
    then -3x² * P(x) = -12x⁴ + 6x³ - 3x² -/
theorem polynomial_correction (x : ℝ) (P : ℝ → ℝ) 
  (h : P x - 3 * x^2 = x^2 - 2*x + 1) : 
  -3 * x^2 * P x = -12 * x^4 + 6 * x^3 - 3 * x^2 := by
  sorry

end polynomial_correction_l1842_184216


namespace symmetrical_line_equation_l1842_184281

/-- Given two lines in the plane, this function returns the equation of a line symmetrical to the first line with respect to the second line. -/
def symmetricalLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line with equation 2x - y - 2 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y ↦ 2 * x - y - 2 = 0

/-- The line with equation x + y - 4 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - 4 = 0

/-- The theorem stating that the symmetrical line has the equation x - 2y + 2 = 0 -/
theorem symmetrical_line_equation :
  symmetricalLine line1 line2 = fun x y ↦ x - 2 * y + 2 = 0 :=
sorry

end symmetrical_line_equation_l1842_184281


namespace positive_real_inequality_l1842_184239

theorem positive_real_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a - b) * (a - c) / (2 * a^2 + (b + c)^2) +
  (b - c) * (b - a) / (2 * b^2 + (c + a)^2) +
  (c - a) * (c - b) / (2 * c^2 + (a + b)^2) ≥ 0 := by
  sorry

end positive_real_inequality_l1842_184239


namespace square_side_length_l1842_184207

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 7 → 
  rectangle_width = 5 → 
  4 * square_side = 2 * (rectangle_length + rectangle_width) → 
  square_side = 6 := by
sorry

end square_side_length_l1842_184207


namespace marble_draw_probability_l1842_184218

/-- Represents a bag of marbles -/
structure MarbleBag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0
  red : ℕ := 0
  green : ℕ := 0

/-- Calculate the total number of marbles in a bag -/
def MarbleBag.total (bag : MarbleBag) : ℕ :=
  bag.white + bag.black + bag.yellow + bag.blue + bag.red + bag.green

/-- Definition of Bag A -/
def bagA : MarbleBag := { white := 5, black := 5 }

/-- Definition of Bag B -/
def bagB : MarbleBag := { yellow := 8, blue := 7 }

/-- Definition of Bag C -/
def bagC : MarbleBag := { yellow := 3, blue := 7 }

/-- Definition of Bag D -/
def bagD : MarbleBag := { red := 4, green := 6 }

/-- Probability of drawing a yellow marble from a bag -/
def probYellow (bag : MarbleBag) : ℚ :=
  bag.yellow / bag.total

/-- Probability of drawing a green marble from a bag -/
def probGreen (bag : MarbleBag) : ℚ :=
  bag.green / bag.total

/-- Main theorem: Probability of drawing yellow as second and green as third marble -/
theorem marble_draw_probability : 
  (1/2 * probYellow bagB + 1/2 * probYellow bagC) * probGreen bagD = 17/50 := by
  sorry

end marble_draw_probability_l1842_184218


namespace min_sum_of_digits_3n2_plus_n_plus_1_l1842_184247

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest sum of digits of 3n^2 + n + 1 for positive integer n is 3 -/
theorem min_sum_of_digits_3n2_plus_n_plus_1 :
  (∀ n : ℕ+, sum_of_digits (3 * n^2 + n + 1) ≥ 3) ∧
  (∃ n : ℕ+, sum_of_digits (3 * n^2 + n + 1) = 3) :=
sorry

end min_sum_of_digits_3n2_plus_n_plus_1_l1842_184247


namespace lane_length_correct_l1842_184255

/-- Represents the length of a swimming lane in meters -/
def lane_length : ℝ := 100

/-- Represents the number of round trips swum -/
def round_trips : ℕ := 3

/-- Represents the total distance swum in meters -/
def total_distance : ℝ := 600

/-- Theorem stating that the lane length is correct given the conditions -/
theorem lane_length_correct : 
  lane_length * (2 * round_trips) = total_distance :=
by sorry

end lane_length_correct_l1842_184255


namespace min_value_expression1_min_value_expression2_min_value_expression3_l1842_184242

/-- The minimum value of x^2 + y^2 + xy + x + y for real x and y is -1/3 -/
theorem min_value_expression1 :
  ∃ (m : ℝ), m = -1/3 ∧ ∀ (x y : ℝ), x^2 + y^2 + x*y + x + y ≥ m :=
sorry

/-- The minimum value of x^2 + y^2 + z^2 + xy + yz + zx + x + y + z for real x, y, and z is -3/8 -/
theorem min_value_expression2 :
  ∃ (m : ℝ), m = -3/8 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + x*y + y*z + z*x + x + y + z ≥ m :=
sorry

/-- The minimum value of x^2 + y^2 + z^2 + r^2 + xy + xz + xr + yz + yr + zr + x + y + z + r for real x, y, z, and r is -2/5 -/
theorem min_value_expression3 :
  ∃ (m : ℝ), m = -2/5 ∧ ∀ (x y z r : ℝ),
    x^2 + y^2 + z^2 + r^2 + x*y + x*z + x*r + y*z + y*r + z*r + x + y + z + r ≥ m :=
sorry

end min_value_expression1_min_value_expression2_min_value_expression3_l1842_184242


namespace system_solution_existence_l1842_184235

/-- The system of equations has at least one solution for some b iff a ≥ -√2 - 1/4 -/
theorem system_solution_existence (a : ℝ) : 
  (∃ (b x y : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8*b^2 = 4*b*(y - x) + 1) ↔ 
  a ≥ -Real.sqrt 2 - 1/4 := by sorry

end system_solution_existence_l1842_184235


namespace interest_rate_calculation_l1842_184206

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Problem statement -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h1 : principal = 500)
  (h2 : time = 4)
  (h3 : interest = 90) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.045 := by
  sorry

end interest_rate_calculation_l1842_184206


namespace circles_intersection_l1842_184253

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ → ℝ × ℝ := λ m ↦ (m, -1)

-- Define the line equation
def line_equation (x y c : ℝ) : Prop := x - y + c = 0

-- Theorem statement
theorem circles_intersection (m c : ℝ) 
  (h1 : ∃ (center1 center2 : ℝ × ℝ), 
    line_equation center1.1 center1.2 c ∧ 
    line_equation center2.1 center2.2 c) : 
  m + c = 3 := by
sorry

end circles_intersection_l1842_184253


namespace a_minus_b_value_l1842_184248

theorem a_minus_b_value (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) (hab : a + b > 0) :
  a - b = 2 ∨ a - b = 8 := by
sorry

end a_minus_b_value_l1842_184248


namespace existence_of_equal_modulus_unequal_squares_l1842_184275

theorem existence_of_equal_modulus_unequal_squares : ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end existence_of_equal_modulus_unequal_squares_l1842_184275


namespace xy_equals_twelve_l1842_184298

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_equals_twelve_l1842_184298


namespace solution_triples_l1842_184294

theorem solution_triples : 
  ∀ (x y : ℤ) (m : ℝ),
    x < 0 ∧ y > 0 ∧ 
    -2 * x + 3 * y = 2 * m ∧
    x - 5 * y = -11 →
    ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) :=
by sorry

end solution_triples_l1842_184294


namespace shanghai_masters_matches_l1842_184287

/-- Represents the tournament structure described in the problem -/
structure Tournament :=
  (num_players : Nat)
  (num_groups : Nat)
  (players_per_group : Nat)
  (advancing_per_group : Nat)

/-- Calculates the number of matches in a round-robin tournament -/
def round_robin_matches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : Nat :=
  let group_matches := t.num_groups * round_robin_matches t.players_per_group
  let elimination_matches := t.num_groups * t.advancing_per_group / 2
  let final_matches := 2
  group_matches + elimination_matches + final_matches

/-- Theorem stating that the total number of matches in the given tournament format is 16 -/
theorem shanghai_masters_matches :
  ∃ t : Tournament, t.num_players = 8 ∧ t.num_groups = 2 ∧ t.players_per_group = 4 ∧ t.advancing_per_group = 2 ∧ total_matches t = 16 :=
by
  sorry

end shanghai_masters_matches_l1842_184287


namespace first_pipe_fill_time_l1842_184244

/-- Given two pipes that can fill a tank, an outlet pipe that can empty it, 
    and the time it takes to fill the tank when all pipes are open, 
    this theorem proves the time it takes for the first pipe to fill the tank. -/
theorem first_pipe_fill_time (t : ℝ) (h1 : t > 0) 
  (h2 : 1/t + 1/30 - 1/45 = 1/15) : t = 18 := by
  sorry

end first_pipe_fill_time_l1842_184244


namespace otimes_equation_solution_l1842_184234

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + 1 + Real.sqrt (b + 2 + Real.sqrt (b + 3 + Real.sqrt (b + 4)))))

-- Theorem statement
theorem otimes_equation_solution (h : ℝ) :
  otimes 3 h = 15 → h = 20 := by
  sorry

end otimes_equation_solution_l1842_184234


namespace exact_two_support_probability_l1842_184258

/-- The probability of a voter supporting the law -/
def p_support : ℝ := 0.6

/-- The probability of a voter not supporting the law -/
def p_oppose : ℝ := 1 - p_support

/-- The number of voters selected -/
def n : ℕ := 5

/-- The number of voters supporting the law in our target scenario -/
def k : ℕ := 2

/-- The binomial coefficient for choosing k items from n items -/
def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k out of n voters supporting the law -/
def prob_exact_support (n k : ℕ) (p : ℝ) : ℝ :=
  (binom_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exact_two_support_probability :
  prob_exact_support n k p_support = 0.2304 := by sorry

end exact_two_support_probability_l1842_184258


namespace abc_fraction_simplification_l1842_184236

theorem abc_fraction_simplification 
  (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (sum_condition : a + b + c = 1) :
  let s := a * b + b * c + c * a
  (a^2 + b^2 + c^2) ≠ 0 ∧ 
  (ab+bc+ca) / (a^2+b^2+c^2) = s / (1 - 2*s) := by
sorry

end abc_fraction_simplification_l1842_184236


namespace negation_of_existence_negation_of_quadratic_inequality_l1842_184213

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1842_184213


namespace dan_youngest_l1842_184269

def ages (a b c d e : ℕ) : Prop :=
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105

theorem dan_youngest (a b c d e : ℕ) (h : ages a b c d e) : 
  d < a ∧ d < b ∧ d < c ∧ d < e := by
  sorry

end dan_youngest_l1842_184269


namespace complex_expression_equality_l1842_184221

theorem complex_expression_equality : ∀ (a b : ℂ), 
  a = 3 - 2*I ∧ b = -2 + 3*I → 3*a + 4*b = 1 + 6*I :=
by
  sorry

end complex_expression_equality_l1842_184221


namespace magazines_read_in_five_hours_l1842_184257

/-- 
Proves that given a reading rate of 1 magazine per 20 minutes, 
the number of magazines that can be read in 5 hours is equal to 15.
-/
theorem magazines_read_in_five_hours 
  (reading_rate : ℚ) -- Reading rate in magazines per minute
  (hours : ℕ) -- Number of hours
  (h1 : reading_rate = 1 / 20) -- Reading rate is 1 magazine per 20 minutes
  (h2 : hours = 5) -- Time period is 5 hours
  : ⌊hours * 60 * reading_rate⌋ = 15 := by
  sorry

#check magazines_read_in_five_hours

end magazines_read_in_five_hours_l1842_184257


namespace parabola_equation_l1842_184251

/-- Parabola with focus F and point M -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  M : ℝ × ℝ
  h_p_pos : p > 0
  h_F : F = (p/2, 0)
  h_M_on_C : M.2^2 = 2 * p * M.1
  h_MF_dist : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 5

/-- Circle with diameter MF passing through (0,2) -/
def circle_passes_through (P : Parabola) : Prop :=
  let center := ((P.M.1 + P.F.1)/2, (P.M.2 + P.F.2)/2)
  Real.sqrt (center.1^2 + (center.2 - 2)^2) = Real.sqrt ((P.M.1 - P.F.1)^2 + (P.M.2 - P.F.2)^2) / 2

/-- Main theorem -/
theorem parabola_equation (P : Parabola) (h_circle : circle_passes_through P) :
  P.p = 2 ∨ P.p = 8 :=
sorry

end parabola_equation_l1842_184251


namespace probability_three_one_color_l1842_184202

/-- The probability of drawing 3 balls of one color and 1 of another color
    from a set of 20 balls (12 black, 8 white) when 4 are drawn at random -/
theorem probability_three_one_color (black_balls white_balls total_balls drawn : ℕ) 
  (h1 : black_balls = 12)
  (h2 : white_balls = 8)
  (h3 : total_balls = black_balls + white_balls)
  (h4 : drawn = 4) :
  (Nat.choose black_balls 3 * Nat.choose white_balls 1 +
   Nat.choose black_balls 1 * Nat.choose white_balls 3) / 
  Nat.choose total_balls drawn = 1 / 3 :=
sorry

end probability_three_one_color_l1842_184202


namespace smaller_number_proof_l1842_184231

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : 
  min x y = 4 := by
sorry

end smaller_number_proof_l1842_184231


namespace power_multiplication_equals_128_l1842_184256

theorem power_multiplication_equals_128 : 
  ∀ b : ℕ, b = 2 → b^3 * b^4 = 128 := by
  sorry

end power_multiplication_equals_128_l1842_184256


namespace base_3_to_base_9_first_digit_l1842_184265

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  Nat.log 9 n

theorem base_3_to_base_9_first_digit :
  let y : Nat := base_3_to_decimal [2, 0, 2, 2, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 0, 1, 1, 1]
  first_digit_base_9 y = 4 := by
  sorry

end base_3_to_base_9_first_digit_l1842_184265


namespace valleyball_club_members_l1842_184276

/-- The cost of a pair of knee pads in dollars -/
def knee_pad_cost : ℕ := 6

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := knee_pad_cost + 7

/-- The cost of a wristband in dollars -/
def wristband_cost : ℕ := jersey_cost + 3

/-- The total cost for one member's equipment (indoor and outdoor sets) -/
def member_cost : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)

/-- The total cost for all members' equipment -/
def total_cost : ℕ := 4080

/-- The number of members in the Valleyball Volleyball Club -/
def club_members : ℕ := total_cost / member_cost

theorem valleyball_club_members : club_members = 58 := by
  sorry

end valleyball_club_members_l1842_184276


namespace min_value_expression_l1842_184226

theorem min_value_expression (x y : ℝ) 
  (h1 : x * y + 3 * x = 3)
  (h2 : 0 < x)
  (h3 : x < 1/2) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x' y' : ℝ), 
    x' * y' + 3 * x' = 3 → 
    0 < x' → 
    x' < 1/2 → 
    3 / x' + 1 / (y' - 3) ≥ m :=
sorry

end min_value_expression_l1842_184226


namespace uncovered_area_l1842_184286

/-- The area not covered by a smaller square and a right triangle within a larger square -/
theorem uncovered_area (larger_side small_side triangle_base triangle_height : ℝ) 
  (h1 : larger_side = 10)
  (h2 : small_side = 4)
  (h3 : triangle_base = 3)
  (h4 : triangle_height = 3)
  : larger_side ^ 2 - (small_side ^ 2 + (triangle_base * triangle_height) / 2) = 79.5 := by
  sorry

#check uncovered_area

end uncovered_area_l1842_184286


namespace middle_frequency_is_32_l1842_184229

/-- Represents a frequency distribution histogram -/
structure Histogram where
  n : ℕ  -- number of rectangles
  middle_area : ℕ  -- area of the middle rectangle
  total_area : ℕ  -- total area of the histogram
  h_area_sum : middle_area + (n - 1) * middle_area = total_area  -- area sum condition
  h_total_area : total_area = 160  -- total area is 160

/-- The frequency of the middle group in the histogram is 32 -/
theorem middle_frequency_is_32 (h : Histogram) : h.middle_area = 32 := by
  sorry

end middle_frequency_is_32_l1842_184229


namespace max_value_fraction_l1842_184263

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (∀ a b : ℝ, -4 ≤ a ∧ a ≤ -2 → 2 ≤ b ∧ b ≤ 4 → (x + y) / x ≥ (a + b) / a) →
  (x + y) / x = 0 :=
by sorry

end max_value_fraction_l1842_184263


namespace inequality_holds_l1842_184279

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

-- State the theorem
theorem inequality_holds (a b : ℝ) (ha : a > 2) (hb : b > 0) :
  ∀ x, |x + 1| < b → |2 * f x - 4| < a := by
  sorry

end inequality_holds_l1842_184279


namespace parabola_circle_theorem_trajectory_theorem_l1842_184261

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line passing through (1,0)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the circle condition
def circle_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Define the vector equation
def vector_equation (x y x₁ y₁ x₂ y₂ : ℝ) : Prop := 
  x = x₁ + x₂ - 1/4 ∧ y = y₁ + y₂

-- Theorem 1
theorem parabola_circle_theorem (p : ℝ) :
  (∃ k x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_condition x₁ y₁ x₂ y₂) →
  p = 1/2 :=
sorry

-- Theorem 2
theorem trajectory_theorem (p : ℝ) (x y : ℝ) :
  p = 1/2 →
  (∃ k x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_condition x₁ y₁ x₂ y₂ ∧
    vector_equation x y x₁ y₁ x₂ y₂) →
  y^2 = x - 7/4 :=
sorry

end parabola_circle_theorem_trajectory_theorem_l1842_184261


namespace fraction_who_say_dislike_but_like_l1842_184291

/-- Represents the student population at Greendale College --/
structure StudentPopulation where
  total : ℝ
  likesSwimming : ℝ
  dislikesSwimming : ℝ
  likesSayLike : ℝ
  likesSayDislike : ℝ
  dislikesSayLike : ℝ
  dislikesSayDislike : ℝ

/-- Conditions of the problem --/
def greendaleCollege : StudentPopulation where
  total := 100
  likesSwimming := 70
  dislikesSwimming := 30
  likesSayLike := 0.75 * 70
  likesSayDislike := 0.25 * 70
  dislikesSayLike := 0.15 * 30
  dislikesSayDislike := 0.85 * 30

/-- The main theorem to prove --/
theorem fraction_who_say_dislike_but_like (ε : ℝ) (hε : ε > 0) :
  let totalSayDislike := greendaleCollege.likesSayDislike + greendaleCollege.dislikesSayDislike
  let fraction := greendaleCollege.likesSayDislike / totalSayDislike
  abs (fraction - 0.407) < ε := by
  sorry


end fraction_who_say_dislike_but_like_l1842_184291


namespace players_per_group_l1842_184245

theorem players_per_group (new_players returning_players total_groups : ℕ) : 
  new_players = 48 → 
  returning_players = 6 → 
  total_groups = 9 → 
  (new_players + returning_players) / total_groups = 6 :=
by sorry

end players_per_group_l1842_184245


namespace initial_value_theorem_l1842_184208

theorem initial_value_theorem :
  ∃ (n : ℤ) (initial_value : ℤ), 
    initial_value = 136 * n - 21 ∧ 
    ∃ (added_value : ℤ), initial_value + added_value = 136 * n ∧ 
    (added_value ≥ 20 ∧ added_value ≤ 22) := by
  sorry

end initial_value_theorem_l1842_184208


namespace geometric_sequence_sum_l1842_184273

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 5 = -6 →
  a 2 * a 6 = 8 →
  a 1 + a 7 = -9 := by
  sorry

end geometric_sequence_sum_l1842_184273


namespace opposite_reciprocal_abs_sum_l1842_184209

theorem opposite_reciprocal_abs_sum (x y m n a : ℝ) : 
  (x + y = 0) →  -- x and y are opposite numbers
  (m * n = 1) →  -- m and n are reciprocals
  (|a| = 3) →    -- absolute value of a is 3
  (a / (m * n) + 2018 * (x + y) = a) ∧ (a = 3 ∨ a = -3) := by
    sorry

end opposite_reciprocal_abs_sum_l1842_184209


namespace inequality_proof_l1842_184233

theorem inequality_proof (k l m n : ℕ) 
  (h1 : k < l) (h2 : l < m) (h3 : m < n) (h4 : l * m = k * n) : 
  ((n - k) / 2 : ℚ)^2 ≥ k + 2 := by sorry

end inequality_proof_l1842_184233
